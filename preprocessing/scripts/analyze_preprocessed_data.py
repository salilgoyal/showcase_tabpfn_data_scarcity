#!/usr/bin/env python
"""
Analyze preprocessed data: feature overlap, county statistics, data quality.

This script loads the preprocessed parquet file and generates analysis reports:
- Feature overlap across counties
- County-level statistics (rows, features, nulls)
- Feature value distributions
- Data quality checks

Usage:
    python preprocessing/scripts/analyze_preprocessed_data.py \
        --data_path /scratch/.../v1_no_onehot/data.parquet \
        --output_dir preprocessing/analysis/v1_no_onehot/
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_metadata(data_path: Path) -> dict:
    """Load metadata.json if it exists."""
    metadata_path = data_path.parent / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return {}


def analyze_county_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze statistics per county.

    Returns DataFrame with columns:
        - fips: County FIPS code
        - n_rows: Number of rows
        - n_features: Number of features (excluding fips, target)
        - target_mean: Mean of target
        - target_std: Std of target
        - null_fraction: Fraction of values that are null
    """
    if 'fips' not in df.columns:
        print("WARNING: No 'fips' column found, cannot analyze by county")
        return pd.DataFrame()

    stats = []

    # Identify target column (should be SALE_AMOUNT)
    target_col = 'SALE_AMOUNT'
    if target_col not in df.columns:
        print(f"WARNING: Target column {target_col} not found")
        target_col = None

    # Identify feature columns (exclude fips, CLIP, target)
    exclude_cols = ['fips', 'CLIP', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    for fips in sorted(df['fips'].unique()):
        county_df = df[df['fips'] == fips]

        stat = {
            'fips': fips,
            'n_rows': len(county_df),
            'n_features': len(feature_cols),
        }

        if target_col:
            stat['target_mean'] = county_df[target_col].mean()
            stat['target_std'] = county_df[target_col].std()
            stat['target_min'] = county_df[target_col].min()
            stat['target_max'] = county_df[target_col].max()

        # Null fraction across all feature columns
        null_count = county_df[feature_cols].isnull().sum().sum()
        total_values = len(county_df) * len(feature_cols)
        stat['null_fraction'] = null_count / total_values if total_values > 0 else 0

        stats.append(stat)

    return pd.DataFrame(stats)


def analyze_feature_overlap(df: pd.DataFrame) -> dict:
    """
    Analyze feature overlap across counties.

    Returns dict with:
        - common_features: Features present in all counties
        - county_feature_matrix: Binary matrix (counties x features)
        - feature_coverage: How many counties have each feature
    """
    if 'fips' not in df.columns:
        return {}

    # Identify feature columns
    exclude_cols = ['fips', 'CLIP', 'SALE_AMOUNT']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    counties = sorted(df['fips'].unique())

    # Build binary matrix: county x feature (1 = has non-null values, 0 = all null)
    matrix = []

    for fips in counties:
        county_df = df[df['fips'] == fips]
        # Feature "exists" if it has at least one non-null value in this county
        has_feature = (county_df[feature_cols].notnull().any()).astype(int).values
        matrix.append(has_feature)

    matrix = np.array(matrix)

    # Features present in all counties
    common_features = [feature_cols[i] for i in range(len(feature_cols)) if matrix[:, i].all()]

    # Feature coverage (how many counties have each feature)
    feature_coverage = pd.DataFrame({
        'feature': feature_cols,
        'n_counties': matrix.sum(axis=0),
        'coverage_pct': 100 * matrix.sum(axis=0) / len(counties)
    }).sort_values('n_counties', ascending=False)

    return {
        'common_features': common_features,
        'county_feature_matrix': pd.DataFrame(
            matrix,
            index=[f'fips_{fips}' for fips in counties],
            columns=feature_cols
        ),
        'feature_coverage': feature_coverage
    }


def analyze_feature_types(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Analyze feature types and distributions.

    Returns DataFrame with:
        - feature: Feature name
        - dtype: Data type
        - n_unique: Number of unique values
        - null_count: Number of nulls
        - null_pct: Percentage of nulls
        - category: Feature category (from metadata if available)
    """
    exclude_cols = ['fips', 'CLIP', 'SALE_AMOUNT']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Get feature categories from metadata if available
    feature_categories = {}
    if metadata:
        cols_info = metadata.get('columns', {})
        for cat in ['continuous', 'binary', 'categorical']:
            for col in cols_info.get(cat, []):
                feature_categories[col] = cat

    stats = []
    for col in feature_cols:
        stat = {
            'feature': col,
            'dtype': str(df[col].dtype),
            'n_unique': df[col].nunique(),
            'null_count': df[col].isnull().sum(),
            'null_pct': 100 * df[col].isnull().sum() / len(df),
            'category': feature_categories.get(col, 'unknown')
        }

        # Add min/max for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            stat['min'] = df[col].min()
            stat['max'] = df[col].max()
            stat['mean'] = df[col].mean()
            stat['std'] = df[col].std()

        stats.append(stat)

    return pd.DataFrame(stats)


def create_summary_report(
    df: pd.DataFrame,
    metadata: dict,
    county_stats: pd.DataFrame,
    feature_overlap: dict,
    feature_types: pd.DataFrame,
    output_dir: Path
):
    """Create a text summary report."""
    report_path = output_dir / "summary_report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PREPROCESSED DATA ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"Total columns: {len(df.columns)}\n")
        f.write(f"Total counties: {df['fips'].nunique() if 'fips' in df.columns else 'N/A'}\n")
        f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB\n\n")

        if metadata:
            f.write("Preprocessing info:\n")
            f.write(f"  Version: {metadata.get('version', 'unknown')}\n")
            f.write(f"  Target log-transformed: {metadata.get('preprocessing', {}).get('target_log_transformed', 'unknown')}\n")
            f.write(f"  Categorical handling: {metadata.get('preprocessing', {}).get('categorical_handling', 'unknown')}\n\n")

        # County statistics
        f.write("COUNTY-LEVEL STATISTICS\n")
        f.write("-" * 80 + "\n")
        if not county_stats.empty:
            f.write(f"Counties: {len(county_stats)}\n")
            f.write(f"Rows per county:\n")
            f.write(f"  Min: {county_stats['n_rows'].min():,}\n")
            f.write(f"  Max: {county_stats['n_rows'].max():,}\n")
            f.write(f"  Mean: {county_stats['n_rows'].mean():.0f}\n")
            f.write(f"  Median: {county_stats['n_rows'].median():.0f}\n\n")

            if 'target_mean' in county_stats.columns:
                f.write(f"Target statistics (log-transformed):\n")
                f.write(f"  Mean across counties: {county_stats['target_mean'].mean():.3f}\n")
                f.write(f"  Std across counties: {county_stats['target_std'].mean():.3f}\n\n")

        # Feature overlap
        f.write("FEATURE OVERLAP ANALYSIS\n")
        f.write("-" * 80 + "\n")
        if feature_overlap:
            f.write(f"Total features: {len(feature_overlap['feature_coverage'])}\n")
            f.write(f"Common features (in all counties): {len(feature_overlap['common_features'])}\n")

            coverage = feature_overlap['feature_coverage']
            f.write(f"\nFeature coverage distribution:\n")
            f.write(f"  100% coverage: {(coverage['coverage_pct'] == 100).sum()} features\n")
            f.write(f"   90%+ coverage: {(coverage['coverage_pct'] >= 90).sum()} features\n")
            f.write(f"   50%+ coverage: {(coverage['coverage_pct'] >= 50).sum()} features\n")
            f.write(f"   <50% coverage: {(coverage['coverage_pct'] < 50).sum()} features\n\n")

            # Show least common features
            least_common = coverage.tail(10)
            f.write("Least common features (bottom 10):\n")
            for _, row in least_common.iterrows():
                f.write(f"  {row['feature']}: {row['n_counties']} counties ({row['coverage_pct']:.1f}%)\n")
            f.write("\n")

        # Feature types
        f.write("FEATURE TYPE SUMMARY\n")
        f.write("-" * 80 + "\n")
        if not feature_types.empty:
            type_counts = feature_types['dtype'].value_counts()
            f.write("Data types:\n")
            for dtype, count in type_counts.items():
                f.write(f"  {dtype}: {count} features\n")
            f.write("\n")

            cat_counts = feature_types['category'].value_counts()
            f.write("Feature categories:\n")
            for cat, count in cat_counts.items():
                f.write(f"  {cat}: {count} features\n")
            f.write("\n")

            # Null analysis
            high_null = feature_types[feature_types['null_pct'] > 10].sort_values('null_pct', ascending=False)
            if not high_null.empty:
                f.write(f"Features with >10% nulls: {len(high_null)}\n")
                f.write("Top 10 features with highest null percentage:\n")
                for _, row in high_null.head(10).iterrows():
                    f.write(f"  {row['feature']}: {row['null_pct']:.1f}% nulls\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Summary report saved to {report_path}")


def create_visualizations(
    county_stats: pd.DataFrame,
    feature_overlap: dict,
    feature_types: pd.DataFrame,
    output_dir: Path
):
    """Create visualization plots."""

    # 1. County size distribution
    if not county_stats.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        axes[0].hist(county_stats['n_rows'], bins=30, edgecolor='black')
        axes[0].set_xlabel('Number of Rows')
        axes[0].set_ylabel('Number of Counties')
        axes[0].set_title('County Size Distribution')
        axes[0].set_yscale('log')

        # Box plot of target by county size
        if 'target_mean' in county_stats.columns:
            axes[1].scatter(county_stats['n_rows'], county_stats['target_mean'], alpha=0.5)
            axes[1].set_xlabel('Number of Rows (County Size)')
            axes[1].set_ylabel('Mean Target Value (log-transformed)')
            axes[1].set_title('Target vs County Size')
            axes[1].set_xscale('log')

        plt.tight_layout()
        plt.savefig(output_dir / 'county_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved county_statistics.png")

    # 2. Feature coverage histogram
    if feature_overlap and 'feature_coverage' in feature_overlap:
        coverage = feature_overlap['feature_coverage']

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(coverage['coverage_pct'], bins=20, edgecolor='black')
        ax.set_xlabel('Coverage (% of counties with feature)')
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Coverage Distribution')
        ax.axvline(100, color='red', linestyle='--', label='100% coverage')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'feature_coverage.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved feature_coverage.png")

    # 3. Feature types
    if not feature_types.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Null percentage distribution
        axes[0].hist(feature_types['null_pct'], bins=30, edgecolor='black')
        axes[0].set_xlabel('Null Percentage')
        axes[0].set_ylabel('Number of Features')
        axes[0].set_title('Feature Null Distribution')
        axes[0].axvline(10, color='red', linestyle='--', label='10% threshold')
        axes[0].legend()

        # Number of unique values (log scale)
        numeric_features = feature_types[feature_types['n_unique'] > 0]
        axes[1].hist(np.log10(numeric_features['n_unique']), bins=30, edgecolor='black')
        axes[1].set_xlabel('log10(Number of Unique Values)')
        axes[1].set_ylabel('Number of Features')
        axes[1].set_title('Feature Cardinality Distribution')

        plt.tight_layout()
        plt.savefig(output_dir / 'feature_quality.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved feature_quality.png")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze preprocessed data"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to preprocessed data.parquet file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample N rows for faster analysis (default: use all)"
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ANALYZING PREPROCESSED DATA")
    print("=" * 80)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    print("Loading data...")
    df = pd.read_parquet(data_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    if args.sample_size and len(df) > args.sample_size:
        print(f"  Sampling {args.sample_size:,} rows for analysis...")
        df = df.sample(n=args.sample_size, random_state=42)

    # Load metadata
    metadata = load_metadata(data_path)

    # Run analyses
    print("\n1. Analyzing county statistics...")
    county_stats = analyze_county_statistics(df)
    if not county_stats.empty:
        county_stats.to_csv(output_dir / 'county_statistics.csv', index=False)
        print(f"   Saved county_statistics.csv")

    print("\n2. Analyzing feature overlap...")
    feature_overlap = analyze_feature_overlap(df)
    if feature_overlap:
        feature_overlap['feature_coverage'].to_csv(
            output_dir / 'feature_coverage.csv', index=False
        )
        print(f"   Saved feature_coverage.csv")

        # Save common features list
        with open(output_dir / 'common_features.txt', 'w') as f:
            for feat in feature_overlap['common_features']:
                f.write(f"{feat}\n")
        print(f"   Saved common_features.txt ({len(feature_overlap['common_features'])} features)")

    print("\n3. Analyzing feature types...")
    feature_types = analyze_feature_types(df, metadata)
    if not feature_types.empty:
        feature_types.to_csv(output_dir / 'feature_types.csv', index=False)
        print(f"   Saved feature_types.csv")

    print("\n4. Creating summary report...")
    create_summary_report(
        df, metadata, county_stats, feature_overlap, feature_types, output_dir
    )

    print("\n5. Creating visualizations...")
    create_visualizations(
        county_stats, feature_overlap, feature_types, output_dir
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - summary_report.txt        : Text summary of key findings")
    print("  - county_statistics.csv     : Per-county statistics")
    print("  - feature_coverage.csv      : Feature coverage across counties")
    print("  - common_features.txt       : Features in all counties")
    print("  - feature_types.csv         : Feature types and distributions")
    print("  - *.png                     : Visualization plots")


if __name__ == "__main__":
    main()
