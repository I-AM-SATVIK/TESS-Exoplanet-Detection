'''TIC IDs collection of exoplanets, binary stars and noise'''
# # This was used to make a resources.csv file which contains TIC_IDs in column 1 and Label in column 2
# # Label : 1 --> exoplanet
# # Label : 2 --> eclipsing binary stars
# # Label : 3 --> noise
# import pandas as pd
# exo = pd.read_csv("unique_stars.csv")
# exo_list = exo['TIC_ID'].tolist()
# exo_list = [str(item).replace('TIC ', '') for item in exo_list]
# output_df_exo = pd.DataFrame({'TIC_ID': exo_list, 'label': 1}).head(300)
# eb = pd.read_csv("unique_binary.csv")
# eb_list = eb['TIC_ID'].tolist()
# eb_list = [str(item).replace('TIC ', '') for item in eb_list]
# output_df_eb = pd.DataFrame({'TIC_ID': eb_list, 'label': 2}).head(300)
# noise = pd.read_csv("unique_noise.csv")
# noise_list = noise['TIC_ID'].tolist()
# noise_list = [str(item).replace('TIC ', '') for item in noise_list]
# output_df_noise = pd.DataFrame({'TIC_ID': noise_list, 'label': 0}).head(300)
# final_df = pd.concat([output_df_eb, output_df_exo, output_df_noise], ignore_index=True)
# final_df.to_csv("resources.csv", index=False)


'''DATA download of all TIC IDs from resources.csv'''
import os
import pandas as pd
import lightkurve as lk
df = pd.read_csv("resources.csv", )
tic_ids = df['TIC_ID'].tolist()
print(tic_ids)