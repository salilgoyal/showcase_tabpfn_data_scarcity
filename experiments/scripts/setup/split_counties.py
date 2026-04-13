"""
Script to split the large corelogic_census CSV by county (fips code).
Processes the file in chunks to avoid OOM errors.
"""
import pandas as pd
import os
import logging
from collections import defaultdict
from tqdm import tqdm

# Set up logging to file
log_file = 'save_counties_separately.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Output directory
# output_dir = '/nlp/scr/salilg/county_csvs/'
output_dir = '/scratch/users/salilg/property_tax/county_csvs/'
os.makedirs(output_dir, exist_ok=True)

# Dictionary to accumulate data for each county
county_data = defaultdict(list)

# Process file in chunks
chunk_size = 100000
# csv_file = '/nlp/scr/salilg/corelogic_census_2018_2023.csv'
csv_file = '/oak/stanford/groups/deho/proptax/clean/corelogic_census_2018_2023.csv'

logger.info(f"Processing {csv_file} in chunks of {chunk_size} rows...")
logger.info(f"Output directory: {output_dir}")

# First pass: collect data by county
chunk_count = 0
for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
    chunk_count += 1
    logger.info(f"Processing chunk {chunk_count}...")

    # Group by fips code and store
    for fips_code in chunk['fips'].unique():
        if pd.notna(fips_code):  # Skip NaN fips codes
            county_chunk = chunk[chunk['fips'] == fips_code]
            county_data[int(fips_code)].append(county_chunk)

    # Periodically write out counties that have accumulated enough data
    # This prevents memory from growing too large
    if chunk_count % 10 == 0:
        logger.info(f"  Intermediate save after chunk {chunk_count}...")
        for fips_code in list(county_data.keys()):
            if len(county_data[fips_code]) >= 5:  # If we have 5+ chunks for a county
                df_county = pd.concat(county_data[fips_code], ignore_index=False)
                output_file = os.path.join(output_dir, f'fips_{fips_code}.csv')

                # Append to existing file or create new one
                if os.path.exists(output_file):
                    df_county.to_csv(output_file, mode='a', header=False, index=True)
                else:
                    df_county.to_csv(output_file, index=True)

                # Clear this county's data from memory
                county_data[fips_code] = []
                logger.info(f"    Saved/appended data for fips {fips_code}")

logger.info("\nFinal save for remaining counties...")
# Write out any remaining data
for fips_code, chunks in county_data.items():
    if chunks:  # If there's any data left
        df_county = pd.concat(chunks, ignore_index=False)
        output_file = os.path.join(output_dir, f'fips_{fips_code}.csv')

        if os.path.exists(output_file):
            df_county.to_csv(output_file, mode='a', header=False, index=True)
        else:
            df_county.to_csv(output_file, index=True)

        logger.info(f"  Saved final data for fips {fips_code} ({len(df_county)} rows)")

# Log summary
logger.info("\n" + "="*50)
logger.info("SUMMARY")
logger.info("="*50)
county_files = sorted([f for f in os.listdir(output_dir) if f.startswith('fips_')])
logger.info(f"Total county files created: {len(county_files)}")
logger.info(f"\nFiles saved in: {output_dir}")

# Show file sizes
logger.info("\nSample of created files:")
for i, filename in enumerate(county_files[:10]):
    filepath = os.path.join(output_dir, filename)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logger.info(f"  {filename}: {size_mb:.2f} MB")
if len(county_files) > 10:
    logger.info(f"  ... and {len(county_files) - 10} more files")
