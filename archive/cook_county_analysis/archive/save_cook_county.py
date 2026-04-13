# this script loads the entire df and then just saves the cook county data to a new csv
import pandas as pd

cook_county_chunks = []
for chunk in pd.read_csv('/nlp/scr/salilg/corelogic_census_2018_2023.csv', chunksize=100000):
    cook_county_chunks.append(chunk[chunk['fips'] == 17031])

df = pd.concat(cook_county_chunks, ignore_index=False)
df.to_csv('cook_county.csv', index=True)