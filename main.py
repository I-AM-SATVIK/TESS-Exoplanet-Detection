import lightkurve as lk

# Search for a star in Sector 1
search_result = lk.search_lightcurve("TIC 261136679", sector=1, author="SPOC")
print("Search Result:")
print(search_result)
lc = search_result.download()
lc = lc.remove_nans() # This is important as it is used to remove NaN values(to avoid model Crash)
print("\nLight Curve Data:")
print(lc)