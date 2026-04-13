"""
Analyze feature coverage across all county CSV files.

Usage:
    python analyze_county_features.py /scratch/users/salilg/property_tax/county_csvs/ output_dir/
"""

import pandas as pd
from pathlib import Path
import sys
from collections import defaultdict
from tqdm import tqdm


def analyze_county_features(county_csvs_dir: str, output_dir: str):
    """
    Analyze which features appear in which counties.

    Args:
        county_csvs_dir: Directory containing fips_*.csv files
        output_dir: Directory to save output files
    """
    county_csvs_dir = Path(county_csvs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all county files
    county_files = sorted(county_csvs_dir.glob("fips_*.csv"))
    print(f"Found {len(county_files)} county CSV files")

    if len(county_files) == 0:
        print(f"No files found in {county_csvs_dir}")
        return

    # Track which features appear in which counties
    feature_to_counties = defaultdict(set)
    county_to_features = {}

    print("Reading county file headers...")
    for county_file in tqdm(county_files):
        # Extract FIPS code from filename (fips_12345.csv -> 12345)
        fips = int(county_file.stem.split('_')[1])

        try:
            # Read just the header (no data rows)
            df = pd.read_csv(county_file, nrows=0)
            features = list(df.columns)

            county_to_features[fips] = features

            for feature in features:
                feature_to_counties[feature].add(fips)

        except Exception as e:
            print(f"Error reading {county_file}: {e}")
            continue

    print(f"Processed {len(county_to_features)} counties")

    # 1. Create feature_coverage.csv
    print("Creating feature_coverage.csv...")
    n_counties = len(county_to_features)

    feature_coverage = []
    for feature, counties in feature_to_counties.items():
        n_with_feature = len(counties)
        pct_coverage = (n_with_feature / n_counties) * 100

        # Assign coverage tier
        if pct_coverage == 100:
            tier = "100%"
        elif pct_coverage >= 90:
            tier = "90-99%"
        elif pct_coverage >= 75:
            tier = "75-89%"
        elif pct_coverage >= 50:
            tier = "50-74%"
        elif pct_coverage >= 25:
            tier = "25-49%"
        else:
            tier = "<25%"

        feature_coverage.append({
            'feature_name': feature,
            'n_counties': n_with_feature,
            'pct_counties': round(pct_coverage, 2),
            'coverage_tier': tier
        })

    df_coverage = pd.DataFrame(feature_coverage)
    df_coverage = df_coverage.sort_values('n_counties', ascending=False)
    df_coverage.to_csv(output_dir / 'feature_coverage.csv', index=False)
    print(f"  Saved: {output_dir / 'feature_coverage.csv'}")

    # 2. Create county_feature_matrix.csv (binary matrix)
    print("Creating county_feature_matrix.csv...")
    all_features = sorted(feature_to_counties.keys())

    matrix_data = []
    for fips in sorted(county_to_features.keys()):
        row = {'fips': fips}
        county_features = set(county_to_features[fips])
        for feature in all_features:
            row[feature] = 1 if feature in county_features else 0
        matrix_data.append(row)

    df_matrix = pd.DataFrame(matrix_data)
    df_matrix.to_csv(output_dir / 'county_feature_matrix.csv', index=False)
    print(f"  Saved: {output_dir / 'county_feature_matrix.csv'}")

    # 3. Create coverage_summary.txt
    print("Creating coverage_summary.txt...")
    with open(output_dir / 'coverage_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COUNTY FEATURE COVERAGE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total counties analyzed: {n_counties}\n")
        f.write(f"Total unique features: {len(feature_to_counties)}\n\n")

        # Coverage tier distribution
        f.write("Feature Distribution by Coverage Tier:\n")
        f.write("-" * 40 + "\n")
        tier_counts = df_coverage['coverage_tier'].value_counts()
        for tier in ["100%", "90-99%", "75-89%", "50-74%", "25-49%", "<25%"]:
            count = tier_counts.get(tier, 0)
            f.write(f"  {tier:10s}: {count:5d} features\n")

        f.write("\n")
        f.write("Features in ALL counties (100% coverage):\n")
        f.write("-" * 40 + "\n")
        universal_features = df_coverage[df_coverage['coverage_tier'] == '100%']['feature_name'].tolist()
        f.write(f"  Count: {len(universal_features)}\n")
        if len(universal_features) <= 50:
            for feat in universal_features:
                f.write(f"    - {feat}\n")
        else:
            f.write(f"  (Too many to list - see feature_coverage.csv)\n")

        f.write("\n")
        f.write("Features in very FEW counties (<25% coverage):\n")
        f.write("-" * 40 + "\n")
        rare_features = df_coverage[df_coverage['coverage_tier'] == '<25%']
        f.write(f"  Count: {len(rare_features)}\n")
        if len(rare_features) <= 50:
            for _, row in rare_features.iterrows():
                f.write(f"    - {row['feature_name']} ({row['n_counties']} counties)\n")
        else:
            f.write(f"  (Showing first 20 - see feature_coverage.csv for all)\n")
            for _, row in rare_features.head(20).iterrows():
                f.write(f"    - {row['feature_name']} ({row['n_counties']} counties)\n")

    print(f"  Saved: {output_dir / 'coverage_summary.txt'}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output files saved to: {output_dir}")
    print(f"  - feature_coverage.csv: Feature-level statistics")
    print(f"  - county_feature_matrix.csv: Binary matrix (counties x features)")
    print(f"  - coverage_summary.txt: Human-readable summary")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_county_features.py <county_csvs_dir> <output_dir>")
        print("Example: python analyze_county_features.py /scratch/users/salilg/property_tax/county_csvs/ ./output/")
        sys.exit(1)

    county_csvs_dir = sys.argv[1]
    output_dir = sys.argv[2]

    analyze_county_features(county_csvs_dir, output_dir)
