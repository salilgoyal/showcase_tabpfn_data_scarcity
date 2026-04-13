"""
Identify counties that have high-coverage features and calculate total data points.

This script:
1. Finds features with >= 50% coverage across counties
2. Identifies counties that have ALL of these high-coverage features
3. Calculates total data points available in those counties

Usage:
    python analyze_high_coverage_counties.py <feature_coverage_csv> <county_matrix_csv> <county_metadata_csv> <output_file>
"""

import pandas as pd
from pathlib import Path
import sys


def analyze_high_coverage_counties(
    feature_coverage_csv: str,
    county_matrix_csv: str,
    county_metadata_csv: str,
    output_file: str
):
    """
    Analyze counties with high-coverage features.

    Args:
        feature_coverage_csv: Path to feature_coverage_preprocessed.csv
        county_matrix_csv: Path to county_feature_matrix_preprocessed.csv
        county_metadata_csv: Path to county_row_counts.csv
        output_file: Path to save output summary
    """
    print("Loading data...")

    # Load feature coverage
    df_coverage = pd.read_csv(feature_coverage_csv)

    # Load county feature matrix
    df_matrix = pd.read_csv(county_matrix_csv)

    # Load county metadata (row counts)
    df_metadata = pd.read_csv(county_metadata_csv)

    print(f"Loaded {len(df_coverage)} features")
    print(f"Loaded {len(df_matrix)} counties")
    print(f"Loaded metadata for {len(df_metadata)} counties")

    # Identify high-coverage features (>= 50% coverage)
    high_coverage_features = df_coverage[df_coverage['pct_counties'] >= 75.0]['feature_name'].tolist()
    print(f"\nFeatures with >= 50% coverage: {len(high_coverage_features)}")

    # Get feature columns (exclude fips, n_samples, state if present)
    meta_cols = ['fips', 'n_samples', 'state']
    feature_cols = [col for col in df_matrix.columns if col not in meta_cols and col in high_coverage_features]

    print(f"Feature columns to check: {len(feature_cols)}")

    # Find counties that have ALL high-coverage features
    print("\nIdentifying counties with ALL high-coverage features...")
    counties_with_all = []

    for _, row in df_matrix.iterrows():
        fips = row['fips']
        # Check if this county has all high-coverage features
        has_all = all(row[feat] == 1 for feat in feature_cols)
        if has_all:
            counties_with_all.append(fips)

    print(f"Counties with ALL {len(feature_cols)} high-coverage features: {len(counties_with_all)}")

    # Merge with metadata to get row counts
    df_selected = df_metadata[df_metadata['fips'].isin(counties_with_all)].copy()

    if len(df_selected) == 0:
        print("\nWARNING: No counties found with all high-coverage features!")
        print("This might mean the 50% threshold is too strict.")
        print("\nTrying with features present in >= 75% of counties...")

        # Try more lenient threshold
        high_coverage_features = df_coverage[df_coverage['pct_counties'] >= 75.0]['feature_name'].tolist()
        feature_cols = [col for col in df_matrix.columns if col not in meta_cols and col in high_coverage_features]

        counties_with_all = []
        for _, row in df_matrix.iterrows():
            fips = row['fips']
            has_all = all(row[feat] == 1 for feat in feature_cols)
            if has_all:
                counties_with_all.append(fips)

        print(f"Counties with features present in >= 75% coverage: {len(counties_with_all)}")
        df_selected = df_metadata[df_metadata['fips'].isin(counties_with_all)].copy()

    # Calculate statistics
    total_rows = df_selected['row_count'].sum()
    mean_rows = df_selected['row_count'].mean()
    median_rows = df_selected['row_count'].median()
    min_rows = df_selected['row_count'].min()
    max_rows = df_selected['row_count'].max()

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HIGH-COVERAGE COUNTIES ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"High-coverage features (>= 50% coverage): {len(high_coverage_features)}\n")
        f.write(f"Counties with ALL these features: {len(counties_with_all)}\n")
        f.write(f"Percentage of total counties: {len(counties_with_all)/len(df_matrix)*100:.1f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("DATA AVAILABILITY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total data points across selected counties: {total_rows:,}\n")
        f.write(f"Mean data points per county: {mean_rows:,.0f}\n")
        f.write(f"Median data points per county: {median_rows:,.0f}\n")
        f.write(f"Min data points per county: {min_rows:,}\n")
        f.write(f"Max data points per county: {max_rows:,}\n\n")

        f.write("=" * 80 + "\n")
        f.write("HIGH-COVERAGE FEATURES\n")
        f.write("=" * 80 + "\n")
        for i, feat in enumerate(sorted(high_coverage_features), 1):
            coverage_row = df_coverage[df_coverage['feature_name'] == feat].iloc[0]
            f.write(f"{i:3d}. {feat:50s} ({coverage_row['n_counties']:4d} counties, {coverage_row['pct_counties']:5.1f}%)\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("COUNTY SIZE DISTRIBUTION\n")
        f.write("=" * 80 + "\n")

        # Size bins
        bins = [0, 100, 500, 1000, 5000, 10000, float('inf')]
        labels = ['<100', '100-500', '500-1K', '1K-5K', '5K-10K', '>10K']
        df_selected['size_bin'] = pd.cut(df_selected['row_count'], bins=bins, labels=labels)

        for label in labels:
            count = (df_selected['size_bin'] == label).sum()
            rows = df_selected[df_selected['size_bin'] == label]['row_count'].sum()
            f.write(f"  {label:10s}: {count:4d} counties, {rows:,} total rows\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SELECTED COUNTY FIPS CODES\n")
        f.write("=" * 80 + "\n")
        f.write("(First 50 counties shown, sorted by data availability)\n\n")

        df_selected_sorted = df_selected.sort_values('row_count', ascending=False)
        for _, row in df_selected_sorted.head(50).iterrows():
            f.write(f"  FIPS {row['fips']:5d}: {row['row_count']:8,} rows\n")

        if len(df_selected) > 50:
            f.write(f"\n  ... and {len(df_selected) - 50} more counties\n")

    print(f"\nSaved summary to: {output_path}")

    # Also save the list of FIPS codes to a CSV
    fips_output = output_path.parent / "high_coverage_county_list.csv"
    df_selected_sorted[['fips', 'row_count']].to_csv(fips_output, index=False)
    print(f"Saved county list to: {fips_output}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Counties with complete high-coverage features: {len(counties_with_all)}")
    print(f"Total data points available: {total_rows:,}")
    print(f"Average per county: {mean_rows:,.0f}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python analyze_high_coverage_counties.py <feature_coverage_csv> <county_matrix_csv> <county_metadata_csv> <output_file>")
        print("\nExample:")
        print("  python analyze_high_coverage_counties.py \\")
        print("    notebooks/feature_investigation/output_preprocessed/feature_coverage_preprocessed.csv \\")
        print("    notebooks/feature_investigation/output_preprocessed/county_feature_matrix_preprocessed.csv \\")
        print("    data/county_row_counts.csv \\")
        print("    notebooks/feature_investigation/output_preprocessed/high_coverage_analysis.txt")
        sys.exit(1)

    feature_coverage_csv = sys.argv[1]
    county_matrix_csv = sys.argv[2]
    county_metadata_csv = sys.argv[3]
    output_file = sys.argv[4]

    analyze_high_coverage_counties(feature_coverage_csv, county_matrix_csv, county_metadata_csv, output_file)
