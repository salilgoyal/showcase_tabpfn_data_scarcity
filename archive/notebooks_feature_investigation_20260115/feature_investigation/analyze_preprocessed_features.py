"""
Analyze feature coverage after preprocessing is applied to each county.

This shows which features survive preprocessing on a per-county basis,
which is what actually matters for the cross-county experiment.

Usage:
    python analyze_preprocessed_features.py <county_csvs_dir> <output_dir> <config_file>
"""

import pandas as pd
from pathlib import Path
import sys
import yaml
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import CountyDataLoader


def load_config(config_path: str) -> dict:
    """Load experiment config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def analyze_preprocessed_features(county_csvs_dir: str, output_dir: str, config_path: str):
    """
    Analyze which features remain after preprocessing each county.

    Args:
        county_csvs_dir: Directory containing fips_*.csv files
        output_dir: Directory to save output files
        config_path: Path to experiment config (for preprocessing settings)
    """
    county_csvs_dir = Path(county_csvs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Initialize data loader with preprocessing config
    preprocessing_config = config.get('preprocessing')
    target_column = config['data']['target_column']

    print(f"Initializing data loader with preprocessing config...")
    data_loader = CountyDataLoader(
        county_csvs_dir=county_csvs_dir,
        target_column=target_column,
        preprocessing_config=preprocessing_config
    )

    # Get all county files
    county_files = sorted(county_csvs_dir.glob("fips_*.csv"))
    print(f"Found {len(county_files)} county CSV files")

    if len(county_files) == 0:
        print(f"No files found in {county_csvs_dir}")
        return

    # Track which features appear in which counties AFTER preprocessing
    feature_to_counties = defaultdict(set)
    county_to_features = {}
    county_to_n_samples = {}
    failed_counties = []

    print("Processing counties with preprocessing pipeline...")
    for county_file in tqdm(county_files):
        # Extract FIPS code from filename (fips_12345.csv -> 12345)
        fips = int(county_file.stem.split('_')[1])

        try:
            # Load and preprocess the county
            df = data_loader.load_county(fips, drop_missing_target=True)
            X, y = data_loader.preprocess_for_training(df)

            features = list(X.columns)
            county_to_features[fips] = features
            county_to_n_samples[fips] = len(X)

            for feature in features:
                feature_to_counties[feature].add(fips)

        except Exception as e:
            print(f"\nError processing county {fips}: {e}")
            failed_counties.append((fips, str(e)))
            continue

    print(f"\nSuccessfully processed {len(county_to_features)} counties")
    if failed_counties:
        print(f"Failed to process {len(failed_counties)} counties")

    n_counties = len(county_to_features)

    # 1. Create feature_coverage_preprocessed.csv
    print("Creating feature_coverage_preprocessed.csv...")
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
    df_coverage.to_csv(output_dir / 'feature_coverage_preprocessed.csv', index=False)
    print(f"  Saved: {output_dir / 'feature_coverage_preprocessed.csv'}")

    # 2. Create county_feature_matrix_preprocessed.csv (binary matrix)
    print("Creating county_feature_matrix_preprocessed.csv...")
    all_features = sorted(feature_to_counties.keys())

    matrix_data = []
    for fips in sorted(county_to_features.keys()):
        row = {'fips': fips, 'n_samples': county_to_n_samples[fips]}
        county_features = set(county_to_features[fips])
        for feature in all_features:
            row[feature] = 1 if feature in county_features else 0
        matrix_data.append(row)

    df_matrix = pd.DataFrame(matrix_data)
    df_matrix.to_csv(output_dir / 'county_feature_matrix_preprocessed.csv', index=False)
    print(f"  Saved: {output_dir / 'county_feature_matrix_preprocessed.csv'}")

    # 3. Create coverage_summary_preprocessed.txt
    print("Creating coverage_summary_preprocessed.txt...")
    with open(output_dir / 'coverage_summary_preprocessed.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PREPROCESSED FEATURE COVERAGE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total counties analyzed: {n_counties}\n")
        f.write(f"Total unique features (after preprocessing): {len(feature_to_counties)}\n\n")

        if failed_counties:
            f.write(f"Failed counties: {len(failed_counties)}\n")
            for fips, error in failed_counties[:10]:
                f.write(f"  - {fips}: {error}\n")
            if len(failed_counties) > 10:
                f.write(f"  ... and {len(failed_counties) - 10} more\n")
            f.write("\n")

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
        if len(universal_features) <= 100:
            for feat in universal_features:
                f.write(f"    - {feat}\n")
        else:
            f.write(f"  First 50:\n")
            for feat in universal_features[:50]:
                f.write(f"    - {feat}\n")
            f.write(f"  ... (see feature_coverage_preprocessed.csv for all)\n")

        f.write("\n")
        f.write("Features in 90%+ of counties:\n")
        f.write("-" * 40 + "\n")
        high_coverage = df_coverage[df_coverage['pct_counties'] >= 90.0]
        f.write(f"  Count: {len(high_coverage)}\n")
        if len(high_coverage) <= 100 and len(high_coverage) > len(universal_features):
            for _, row in high_coverage[high_coverage['coverage_tier'] != '100%'].iterrows():
                f.write(f"    - {row['feature_name']} ({row['n_counties']} counties, {row['pct_counties']:.1f}%)\n")

        f.write("\n")
        f.write("Features in very FEW counties (<25% coverage):\n")
        f.write("-" * 40 + "\n")
        rare_features = df_coverage[df_coverage['coverage_tier'] == '<25%']
        f.write(f"  Count: {len(rare_features)}\n")
        if len(rare_features) <= 50:
            for _, row in rare_features.iterrows():
                f.write(f"    - {row['feature_name']} ({row['n_counties']} counties, {row['pct_counties']:.1f}%)\n")
        else:
            f.write(f"  First 30:\n")
            for _, row in rare_features.head(30).iterrows():
                f.write(f"    - {row['feature_name']} ({row['n_counties']} counties, {row['pct_counties']:.1f}%)\n")
            f.write(f"  ... (see feature_coverage_preprocessed.csv for all)\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("KEY INSIGHT:\n")
        f.write("=" * 80 + "\n")
        f.write("In the current cross-county experiment, feature alignment uses the\n")
        f.write("intersection of (union of train county features) and (union of test county features).\n")
        f.write(f"\nThis means a feature present in only 1 train county (out of {n_counties}) would be\n")
        f.write("included if at least 1 test county also has it, leading to sparse data with many NaNs.\n")
        f.write("\n")
        f.write(f"Features with 100% coverage: {len(universal_features)}\n")
        f.write(f"Total features after alignment: {len(feature_to_counties)}\n")
        f.write(f"Difference: {len(feature_to_counties) - len(universal_features)} features NOT in all counties\n")

    print(f"  Saved: {output_dir / 'coverage_summary_preprocessed.txt'}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output files saved to: {output_dir}")
    print(f"  - feature_coverage_preprocessed.csv")
    print(f"  - county_feature_matrix_preprocessed.csv")
    print(f"  - coverage_summary_preprocessed.txt")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python analyze_preprocessed_features.py <county_csvs_dir> <output_dir> <config_file>")
        print("Example: python analyze_preprocessed_features.py /scratch/users/salilg/property_tax/county_csvs/ ./output/ experiments/configs/cross_county/small_in_context_10k.yaml")
        sys.exit(1)

    county_csvs_dir = sys.argv[1]
    output_dir = sys.argv[2]
    config_file = sys.argv[3]

    analyze_preprocessed_features(county_csvs_dir, output_dir, config_file)
