import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightkurve as lk
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

FIXED_LENGTH = 2000
DATA_DIR = "data"
MODEL_PATH = "tess_pytorch_cnn.pth"

# ==========================================
# 1. Data Preprocessing & Training  (unchanged)
# ==========================================
def load_and_preprocess_fits(filepath):
    try:
        lc = lk.read(filepath)
        lc = lc.remove_nans().normalize()
        flux = lc.flux.value
        time = lc.time.value
        if len(flux) == 0:
            return None
        time_uniform = np.linspace(time[0], time[-1], FIXED_LENGTH)
        flux_interp = np.interp(time_uniform, time, flux)
        flux_mean = np.mean(flux_interp)
        flux_std = np.std(flux_interp)
        flux_scaled = (flux_interp - flux_mean) / (flux_std + 1e-8)
        return flux_scaled.astype(np.float32)
    except Exception:
        return None


class TESSDataset(Dataset):
    def __init__(self, csv_file, data_dir):
        df = pd.read_csv(csv_file)
        self.X, self.y = [], []
        print("Loading and normalizing FITS files...")
        for _, row in df.iterrows():
            filepath = os.path.join(data_dir, f"TIC_{row['TIC_ID']}.fits")
            if os.path.exists(filepath):
                flux = load_and_preprocess_fits(filepath)
                if flux is not None:
                    self.X.append(flux)
                    self.y.append(int(row["label"]))
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        print(f"Loaded {len(self.X)} samples.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]).unsqueeze(0),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


class TESS1DCNN(nn.Module):
    def __init__(self):
        super(TESS1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 5), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(32, 64, 5), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 497, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TESSDataset("resources.csv", DATA_DIR)

class_counts = np.bincount(dataset.y)
total_samples = len(dataset.y)
class_weights = total_samples / (len(class_counts) * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
model = TESS1DCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nStarting training...")
for epoch in range(15):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(
        f"Epoch {epoch+1}/15 - Loss: {running_loss/len(train_loader):.4f}"
        f" - Accuracy: {accuracy:.2f}%"
    )

torch.save(model.state_dict(), MODEL_PATH)


# ==========================================
# 2. Precise Exoplanet Parameter Extraction
# ==========================================

def detrend_lightcurve(lc):
    """
    Remove stellar variability with a Savitzky-Golay filter.
    Window is chosen as ~0.5 day in cadence units (minimum 11 points, odd).
    The filter smooths out slow stellar signals while preserving short transit dips.
    """
    time = lc.time.value
    flux = lc.flux.value
    cadence_days = np.nanmedian(np.diff(time))          # typical cadence
    window_points = int(0.5 / cadence_days)             # half-day smoothing window
    if window_points < 11:
        window_points = 11
    if window_points % 2 == 0:
        window_points += 1                              # must be odd for savgol

    trend = savgol_filter(flux, window_length=window_points, polyorder=2)
    flux_detrended = flux / trend                       # divide out the trend
    flux_detrended = flux_detrended / np.nanmedian(flux_detrended)   # re-normalise

    # Build a new LightCurve with the detrended flux.
    # assign_flux does not exist in lightkurve; construct directly instead.
    lc_detrended = lk.LightCurve(time=lc.time, flux=flux_detrended)
    return lc_detrended


def run_bls(lc_detrended, period_min=0.5, period_max=None):
    """
    Run BLS on the detrended light curve and return the periodogram.
    period_max defaults to 1/3 of the total baseline (need at least 3 transits).
    """
    baseline = lc_detrended.time.value[-1] - lc_detrended.time.value[0]
    if period_max is None:
        period_max = baseline / 3.0

    period_grid = np.linspace(period_min, period_max, 10_000)
    bls = lc_detrended.to_periodogram(
        method="bls",
        period=period_grid,
        frequency_factor=1.0,
    )
    return bls


def check_harmonics(bls, best_period):
    """
    Make sure the BLS peak is not a harmonic (½×, 2×, 3× …) of a stronger signal.
    If a sub-harmonic has higher power, return the sub-harmonic instead.
    """
    power = bls.power.value
    periods = bls.period.value

    # look at ½P, ⅓P, 2P, 3P
    for multiplier in [0.5, 1/3, 2.0, 3.0]:
        test_period = best_period * multiplier
        idx = np.argmin(np.abs(periods - test_period))
        if power[idx] > bls.max_power.value * 0.95:
            best_period = periods[idx]
    return best_period


def compute_sde(bls):
    """
    Signal Detection Efficiency (SDE) = (peak power - median) / std of power.
    SDE > 7 is a widely-used threshold for a credible detection.
    """
    power = bls.power.value
    sde = (bls.max_power.value - np.median(power)) / np.std(power)
    return float(sde)


def trapezoid_model(phase, t0_model, total_duration, ingress_duration, depth):
    """
    Trapezoid transit model evaluated at phase values (centred on 0).
    Parameters
    ----------
    phase            : array of phase values in days
    t0_model         : mid-transit phase offset (days) — should be ~0
    total_duration   : T14, first to fourth contact (days)
    ingress_duration : T12, ingress / egress duration (days)
    depth            : fractional flux drop (dimensionless, positive)
    """
    flux_model = np.ones_like(phase)
    shifted = phase - t0_model
    half_total = total_duration / 2.0
    half_flat  = half_total - ingress_duration

    # flat bottom
    in_flat = np.abs(shifted) <= half_flat
    flux_model[in_flat] = 1.0 - depth

    # ingress / egress slopes
    in_ingress = (np.abs(shifted) > half_flat) & (np.abs(shifted) <= half_total)
    ramp = (half_total - np.abs(shifted[in_ingress])) / ingress_duration
    flux_model[in_ingress] = 1.0 - depth * ramp

    return flux_model


def fit_transit_model(folded_lc, bls_duration_days, bls_depth):
    """
    Fit a trapezoid model to the phase-folded light curve.
    Returns refined (t0_offset, T14_duration, depth) — all in days.
    """
    phase = folded_lc.time.value          # already centred on 0 by lightkurve
    flux  = folded_lc.flux.value

    # Crop to a ±3× BLS-duration window to speed up fitting
    window = 3.0 * bls_duration_days
    mask = np.abs(phase) < window
    if mask.sum() < 10:
        return None

    phase_crop = phase[mask]
    flux_crop  = flux[mask]

    # Initial guesses
    p0 = [0.0,                   # t0_model offset ≈ 0
          bls_duration_days,     # total duration T14
          bls_duration_days / 4, # ingress ≈ 25 % of T14
          abs(bls_depth)]        # depth

    # Bounds: sensible astrophysical ranges
    lower = [-bls_duration_days, 1e-4, 1e-5, 1e-6]
    upper = [+bls_duration_days, window, bls_duration_days / 2, 1.0]

    try:
        popt, pcov = curve_fit(
            trapezoid_model,
            phase_crop,
            flux_crop,
            p0=p0,
            bounds=(lower, upper),
            maxfev=10_000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "t0_offset_days":       popt[0],
            "T14_duration_days":    popt[1],
            "ingress_duration_days":popt[2],
            "depth_fraction":       popt[3],
            "depth_ppm":            popt[3] * 1e6,
            "T14_hours":            popt[1] * 24.0,
            "t0_offset_err_days":   perr[0],
            "duration_err_hours":   perr[1] * 24.0,
            "depth_err_ppm":        perr[3] * 1e6,
        }
    except (RuntimeError, ValueError):
        return None


def precise_exoplanet_params(lc_raw):
    """
    Full pipeline:
      1. Detrend
      2. BLS with dense period grid
      3. Harmonic check
      4. SDE vetting
      5. Trapezoid model fit on folded curve
    Returns a dict of parameters (or None if no credible signal).
    """
    print("\n[1/5] Detrending light curve with Savitzky-Golay filter...")
    lc_dt = detrend_lightcurve(lc_raw)

    print("[2/5] Running BLS period search (dense grid, 0.5 – baseline/3 days)...")
    bls = run_bls(lc_dt)

    raw_period = bls.period_at_max_power.value
    print(f"      Raw BLS best period: {raw_period:.6f} days")

    print("[3/5] Checking for harmonic aliases...")
    best_period = check_harmonics(bls, raw_period)
    if best_period != raw_period:
        print(f"      Harmonic correction → {best_period:.6f} days")
    else:
        print(f"      No harmonic alias detected.")

    print("[4/5] Computing Signal Detection Efficiency (SDE)...")
    sde = compute_sde(bls)
    print(f"      SDE = {sde:.2f}  (threshold ≥ 7 for credible detection)")
    if sde < 7.0:
        print("  ⚠  Low SDE — transit signal may not be significant.")

    # Pull BLS best-fit parameters directly from the periodogram object.
    # (get_transit_mask / lk.units.day are not needed here.)
    bls_duration  = bls.duration_at_max_power.value     # days
    bls_depth     = bls.depth_at_max_power.value        # fractional
    bls_t0        = bls.transit_time_at_max_power.value # BJD

    print("[5/5] Fitting trapezoid model to phase-folded light curve...")
    folded = lc_dt.fold(period=best_period, epoch_time=bls_t0)
    fit    = fit_transit_model(folded, bls_duration, bls_depth)

    if fit is None:
        print("  ⚠  Trapezoid fit failed — falling back to BLS values.")
        fit = {
            "T14_hours":     bls_duration * 24.0,
            "depth_ppm":     bls_depth * 1e6,
            "depth_fraction":bls_depth,
            "T14_duration_days": bls_duration,
            "t0_offset_days": 0.0,
            "duration_err_hours": None,
            "depth_err_ppm":      None,
        }

    return {
        "period_days":       best_period,
        "t0_bjd":            bls_t0,
        "T14_hours":         fit["T14_hours"],
        "T14_duration_days": fit["T14_duration_days"],
        "depth_ppm":         fit["depth_ppm"],
        "depth_fraction":    fit["depth_fraction"],
        "sde":               sde,
        "duration_err_hours":fit.get("duration_err_hours"),
        "depth_err_ppm":     fit.get("depth_err_ppm"),
        "bls_period":        raw_period,
        "folded_lc":         folded,
        "lc_detrended":      lc_dt,
        "bls":               bls,
    }


# ==========================================
# 3. Inference Loop
# ==========================================
label_map = {0: "Noise", 1: "Exoplanet", 2: "Binary Star"}
model.eval()

while True:
    user_input = input("\nEnter a TIC ID to predict (or type 'exit'): ").strip()
    if user_input.lower() == "exit":
        break
    if not user_input.isdigit():
        continue

    target_tic = int(user_input)
    target_file = os.path.join(DATA_DIR, f"TIC_{target_tic}.fits")

    if not os.path.exists(target_file):
        print(f"Downloading TIC {target_tic}...")
        search = lk.search_lightcurve(f"TIC {target_tic}")
        if len(search) > 0:
            search[0].download().to_fits(target_file, overwrite=True)
        else:
            print("No data found for this TIC ID.")
            continue

    # --- CNN classification ---
    flux_interp = load_and_preprocess_fits(target_file)
    if flux_interp is None:
        print("Could not preprocess this light curve.")
        continue

    input_tensor = torch.tensor(flux_interp).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_class].item() * 100

    print(f"\n{'='*45}")
    print(f"  RESULTS FOR TIC {target_tic}")
    print(f"{'='*45}")

    # Confidence threshold: CNN must be ≥75% sure before committing to
    # Exoplanet.  Scores below this are too ambiguous — the model often
    # confuses low-SNR noise with shallow transit signals.
    CONFIDENCE_THRESHOLD = 75.0
    if pred_class == 1 and confidence < CONFIDENCE_THRESHOLD:
        print(f"  Classification : Uncertain  "
              f"(raw prediction: {label_map[pred_class]}, {confidence:.2f}%)")
        print(f"  ⚠  Confidence below {CONFIDENCE_THRESHOLD:.0f}% threshold — "
              f"treating as Noise / insufficient signal.")
        print(f"{'='*45}")
        # Fall through to the plain light-curve plot below
        pred_class = 0   # treat as Noise for plotting purposes
    else:
        print(f"  Classification : {label_map[pred_class]}")
        print(f"  Confidence     : {confidence:.2f}%")
    print(f"{'='*45}")

    # -------------------------------------------------------
    # Only run the precise parameter pipeline for exoplanets
    # -------------------------------------------------------
    if pred_class == 1:  # Exoplanet
        lc_raw = lk.read(target_file).remove_nans().normalize()
        params = precise_exoplanet_params(lc_raw)

        print(f"\n  ── Astrophysical Parameters ──")
        print(f"  Orbital Period   : {params['period_days']:.6f}  days")
        print(f"  Transit Epoch t0 : {params['t0_bjd']:.4f}  BJD")
        print(f"  Transit Duration : {params['T14_hours']:.4f}  hours", end="")
        if params["duration_err_hours"] is not None:
            print(f"  ±  {params['duration_err_hours']:.4f} hours")
        else:
            print()
        print(f"  Transit Depth    : {params['depth_ppm']:.2f}  ppm", end="")
        if params["depth_err_ppm"] is not None:
            print(f"  ±  {params['depth_err_ppm']:.2f} ppm")
        else:
            print()
        print(f"  SDE              : {params['sde']:.2f}"
              f"  {'(credible ✓)' if params['sde'] >= 7 else '(weak signal ⚠)'}")
        print(f"{'='*45}")

        # ---- 4-panel diagnostic plot ----
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"TIC {target_tic} — Exoplanet Candidate Analysis", fontsize=14)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax0 = fig.add_subplot(gs[0, :])       # full width: raw light curve
        ax1 = fig.add_subplot(gs[1, 0])       # BLS periodogram
        ax2 = fig.add_subplot(gs[1, 1])       # phase-folded + model

        # Raw (detrended) light curve
        lc_dt = params["lc_detrended"]
        ax0.scatter(lc_dt.time.value, lc_dt.flux.value,
                    s=1, c="steelblue", alpha=0.4, label="Detrended flux")
        ax0.set_xlabel("Time (BJD)")
        ax0.set_ylabel("Normalised Flux")
        ax0.set_title("Detrended Light Curve")
        ax0.legend(markerscale=4)

        # BLS periodogram
        bls = params["bls"]
        ax1.plot(bls.period.value, bls.power.value, lw=0.8, color="darkorange")
        ax1.axvline(params["period_days"], color="crimson", lw=1.5,
                    linestyle="--", label=f"P = {params['period_days']:.4f} d")
        ax1.set_xlabel("Period (days)")
        ax1.set_ylabel("BLS Power")
        ax1.set_title("BLS Periodogram")
        ax1.legend()

        # Phase-folded light curve + trapezoid model overlay
        folded = params["folded_lc"]
        ax2.scatter(folded.time.value, folded.flux.value,
                    s=2, c="gray", alpha=0.3, label="Folded data")

        # Overlay the fitted trapezoid
        phase_fit = np.linspace(folded.time.value.min(), folded.time.value.max(), 2000)
        trap_flux = trapezoid_model(
            phase_fit,
            t0_model=0.0,
            total_duration=params["T14_duration_days"],
            ingress_duration=params["T14_duration_days"] / 4,
            depth=params["depth_fraction"],
        )
        ax2.plot(phase_fit, trap_flux, color="crimson", lw=2, label="Trapezoid fit")
        ax2.set_xlabel("Phase (days)")
        ax2.set_ylabel("Normalised Flux")
        ax2.set_title(f"Phase-Folded  (P = {params['period_days']:.4f} d)")
        ax2.legend()

        plt.show()

    else:
        # For Binary Star / Noise: just show the raw light curve
        lc_raw = lk.read(target_file).remove_nans().normalize()
        fig, ax = plt.subplots(figsize=(10, 4))
        lc_raw.scatter(ax=ax, color="black", alpha=0.5, s=1)
        ax.set_title(f"TIC {target_tic} — {label_map[pred_class]} (Confidence: {confidence:.1f}%)")
        plt.tight_layout()
        plt.show()