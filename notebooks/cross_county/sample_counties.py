"""
Script to sample counties for cross-county train/test split.

Strategy:
1. Sample 50 small counties randomly
2. Split each small county 50/50 into test/train
3. Fill up to 10K training rows by sampling 20% from other counties
4. Fill up to 10K test rows by sampling 50% from remaining counties
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# File paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
SMALL_COUNTY_FILE = DATA_DIR / "small_county_metadata.csv"
ALL_COUNTY_FILE = DATA_DIR / "county_row_counts.csv"

# Parameters
N_SMALL_COUNTIES = 50
SMALL_COUNTY_TEST_RATIO = 0.5
TARGET_TRAIN_SIZE = 10000
TARGET_TEST_SIZE = 10000
OTHER_COUNTY_SAMPLE_RATIO = 0.2
REMAINING_COUNTY_TEST_RATIO = 0.5


def main():
    # Read the CSV files
    print("Reading data files...")
    small_counties = pd.read_csv(SMALL_COUNTY_FILE)
    all_counties = pd.read_csv(ALL_COUNTY_FILE)

    print(f"Total small counties available: {len(small_counties)}")
    print(f"Total counties available: {len(all_counties)}")

    # Step 1: Randomly sample 50 small counties
    sampled_small_counties = small_counties.sample(n=N_SMALL_COUNTIES, random_state=42)
    sampled_small_fips = set(sampled_small_counties['fips'].values)

    print(f"\nSampled {N_SMALL_COUNTIES} small counties")

    # Step 2: Split each small county into test and train (50/50)
    test_allocation = []
    train_allocation = []

    for _, row in sampled_small_counties.iterrows():
        fips = row['fips']
        total_rows = row['row_count']

        # 50% for test, 50% for train
        test_rows = int(total_rows * SMALL_COUNTY_TEST_RATIO)
        train_rows = total_rows - test_rows

        test_allocation.append({
            'fips': fips,
            'rows': test_rows,
            'source': 'small_county'
        })

        train_allocation.append({
            'fips': fips,
            'rows': train_rows,
            'source': 'small_county'
        })

    # Calculate how many rows we have so far in training
    current_train_size = sum(item['rows'] for item in train_allocation)
    print(f"\nTraining rows from small counties: {current_train_size}")
    print(f"Test rows from small counties: {sum(item['rows'] for item in test_allocation)}")

    # Step 3: Fill up to 10K training rows by sampling from other counties
    remaining_train_needed = TARGET_TRAIN_SIZE - current_train_size
    print(f"\nNeed {remaining_train_needed} more training rows to reach {TARGET_TRAIN_SIZE}")

    # Filter out the 50 small counties we already sampled
    other_counties = all_counties[~all_counties['fips'].isin(sampled_small_fips)].copy()

    # Shuffle to randomize selection order
    other_counties = other_counties.sample(frac=1, random_state=42).reset_index(drop=True)

    # Track which counties we've used for training
    counties_used_for_train = set()

    if remaining_train_needed > 0:
        print(f"Sampling from {len(other_counties)} other counties for training...")

        accumulated_rows = 0
        for idx, row in other_counties.iterrows():
            if accumulated_rows >= remaining_train_needed:
                break

            fips = row['fips']
            total_rows = row['row_count']

            # Take 20% from this county
            sample_rows = int(total_rows * OTHER_COUNTY_SAMPLE_RATIO)

            # Don't exceed what we need
            sample_rows = min(sample_rows, remaining_train_needed - accumulated_rows)

            if sample_rows > 0:
                train_allocation.append({
                    'fips': fips,
                    'rows': sample_rows,
                    'source': 'other_county'
                })
                accumulated_rows += sample_rows
                counties_used_for_train.add(fips)

        print(f"Added {accumulated_rows} rows from {len(counties_used_for_train)} other counties to training")

    # Step 4: Fill up to 10K test rows by sampling 50% from remaining counties
    current_test_size = sum(item['rows'] for item in test_allocation)
    remaining_test_needed = TARGET_TEST_SIZE - current_test_size
    print(f"\nNeed {remaining_test_needed} more test rows to reach {TARGET_TEST_SIZE}")

    if remaining_test_needed > 0:
        # Get remaining counties (not used in small sample or training fill-up)
        all_used_fips = sampled_small_fips.union(counties_used_for_train)
        remaining_counties = all_counties[~all_counties['fips'].isin(all_used_fips)].copy()

        # Shuffle to randomize selection order
        remaining_counties = remaining_counties.sample(frac=1, random_state=43).reset_index(drop=True)

        print(f"Sampling from {len(remaining_counties)} remaining counties for test...")

        accumulated_test_rows = 0
        counties_used_for_test = 0

        # Cycle through remaining counties until we reach target
        while accumulated_test_rows < remaining_test_needed and len(remaining_counties) > 0:
            for idx, row in remaining_counties.iterrows():
                if accumulated_test_rows >= remaining_test_needed:
                    break

                fips = row['fips']
                total_rows = row['row_count']

                # Take 50% from this county
                sample_rows = int(total_rows * REMAINING_COUNTY_TEST_RATIO)

                # Don't exceed what we need
                sample_rows = min(sample_rows, remaining_test_needed - accumulated_test_rows)

                if sample_rows > 0:
                    test_allocation.append({
                        'fips': fips,
                        'rows': sample_rows,
                        'source': 'remaining_county'
                    })
                    accumulated_test_rows += sample_rows
                    counties_used_for_test += 1

            # If we still need more and cycled through all counties once, break to avoid infinite loop
            if accumulated_test_rows < remaining_test_needed:
                print(f"Warning: Exhausted all remaining counties. Test set has {current_test_size + accumulated_test_rows} rows.")
                break

        print(f"Added {accumulated_test_rows} rows from {counties_used_for_test} remaining counties to test")

    # Convert to DataFrames for better display
    test_df = pd.DataFrame(test_allocation)
    train_df = pd.DataFrame(train_allocation)

    # Summary statistics
    print("\n" + "="*60)
    print("FINAL ALLOCATION SUMMARY")
    print("="*60)

    print(f"\nTest Set:")
    print(f"  Total counties: {len(test_df)}")
    print(f"  Total rows: {test_df['rows'].sum()}")
    print(f"\n  Breakdown by source:")
    print(test_df.groupby('source')['rows'].agg(['count', 'sum']))

    print(f"\nTraining Set:")
    print(f"  Total counties: {len(train_df)}")
    print(f"  Total rows: {train_df['rows'].sum()}")
    print(f"\n  Breakdown by source:")
    print(train_df.groupby('source')['rows'].agg(['count', 'sum']))

    # Detailed listings
    print("\n" + "="*60)
    print("DETAILED TEST SET ALLOCATION")
    print("="*60)
    print(test_df.to_string(index=False))

    print("\n" + "="*60)
    print("DETAILED TRAINING SET ALLOCATION")
    print("="*60)
    print(train_df.to_string(index=False))

    # Save results
    output_dir = Path(__file__).parent
    test_df.to_csv(output_dir / "test_allocation.csv", index=False)
    train_df.to_csv(output_dir / "train_allocation.csv", index=False)

    print("\n" + "="*60)
    print(f"Results saved to:")
    print(f"  - {output_dir / 'test_allocation.csv'}")
    print(f"  - {output_dir / 'train_allocation.csv'}")
    print("="*60)


if __name__ == "__main__":
    main()
