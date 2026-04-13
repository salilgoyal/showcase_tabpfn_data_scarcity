"""
Analyze feature coverage grouped by state.

This groups counties by state (using first 2 digits of FIPS code) and analyzes
which features are common within each state.

Usage:
    python analyze_by_state.py <county_feature_matrix_csv> <output_dir>
"""

import pandas as pd
from pathlib import Path
import sys
from collections import defaultdict


def get_state_from_fips(fips: int) -> str:
    """
    Extract state code from FIPS code.

    Args:
        fips: County FIPS code (4 or 5 digits)

    Returns:
        2-digit state code as string
    """
    # Convert to string and pad to 5 digits
    fips_str = str(fips).zfill(5)
    # First 2 digits are state code
    return fips_str[:2]


def analyze_by_state(matrix_csv: str, output_dir: str):
    """
    Analyze feature coverage within each state.

    Args:
        matrix_csv: Path to county_feature_matrix_preprocessed.csv
        output_dir: Directory to save output files
    """
    matrix_csv = Path(matrix_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {matrix_csv}...")
    df = pd.read_csv(matrix_csv)

    print(f"Loaded {len(df)} counties")

    # Add state column
    df['state'] = df['fips'].apply(get_state_from_fips)

    # Get feature columns (exclude fips, n_samples, state)
    meta_cols = ['fips', 'n_samples', 'state']
    feature_cols = [col for col in df.columns if col not in meta_cols]

    print(f"Analyzing {len(feature_cols)} features across states...")

    # Group by state
    states = sorted(df['state'].unique())
    print(f"Found {len(states)} states")

    # === 1. State-level feature coverage ===
    print("\nCalculating state-level feature coverage...")
    state_feature_data = []

    for state in states:
        state_df = df[df['state'] == state]
        n_counties = len(state_df)

        for feature in feature_cols:
            n_with_feature = state_df[feature].sum()
            pct_coverage = (n_with_feature / n_counties) * 100

            state_feature_data.append({
                'state': state,
                'feature': feature,
                'n_counties_total': n_counties,
                'n_counties_with_feature': int(n_with_feature),
                'pct_coverage': round(pct_coverage, 2)
            })

    df_state_features = pd.DataFrame(state_feature_data)
    df_state_features.to_csv(output_dir / 'state_feature_coverage.csv', index=False)
    print(f"  Saved: {output_dir / 'state_feature_coverage.csv'}")

    # === 2. Universal features per state ===
    print("\nIdentifying universal features per state...")
    state_universal = []

    for state in states:
        state_df = df[df['state'] == state]
        n_counties = len(state_df)

        # Features present in 100% of counties in this state
        universal_features = []
        for feature in feature_cols:
            if state_df[feature].sum() == n_counties:
                universal_features.append(feature)

        state_universal.append({
            'state': state,
            'n_counties': n_counties,
            'n_universal_features': len(universal_features),
            'universal_features': '|'.join(universal_features)  # pipe-separated
        })

    df_state_universal = pd.DataFrame(state_universal)
    df_state_universal = df_state_universal.sort_values('n_universal_features', ascending=False)
    df_state_universal.to_csv(output_dir / 'state_universal_features.csv', index=False)
    print(f"  Saved: {output_dir / 'state_universal_features.csv'}")

    # === 3. Summary statistics ===
    print("\nGenerating summary...")

    # Find features that are universal across ALL states
    universal_across_all_states = []
    for feature in feature_cols:
        # Check if feature is in 100% of counties in every state
        universal_in_all = True
        for state in states:
            state_df = df[df['state'] == state]
            if state_df[feature].sum() != len(state_df):
                universal_in_all = False
                break
        if universal_in_all:
            universal_across_all_states.append(feature)

    # States with most/least universal features
    best_states = df_state_universal.head(10)
    worst_states = df_state_universal.tail(10)

    # Write summary
    with open(output_dir / 'state_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATE-LEVEL FEATURE COVERAGE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total states analyzed: {len(states)}\n")
        f.write(f"Total counties: {len(df)}\n")
        f.write(f"Total features: {len(feature_cols)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("UNIVERSAL FEATURES (100% coverage across ALL states)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Count: {len(universal_across_all_states)}\n\n")
        if universal_across_all_states:
            for feat in universal_across_all_states:
                f.write(f"  - {feat}\n")
        else:
            f.write("  (None - no features are universal across all states)\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("STATES WITH MOST UNIVERSAL FEATURES\n")
        f.write("=" * 80 + "\n")
        for _, row in best_states.iterrows():
            f.write(f"  State {row['state']}: {row['n_universal_features']} features "
                   f"(out of {len(feature_cols)}) in all {row['n_counties']} counties\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("STATES WITH FEWEST UNIVERSAL FEATURES\n")
        f.write("=" * 80 + "\n")
        for _, row in worst_states.iterrows():
            f.write(f"  State {row['state']}: {row['n_universal_features']} features "
                   f"(out of {len(feature_cols)}) in all {row['n_counties']} counties\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("COVERAGE TIER DISTRIBUTION (across all state-feature pairs)\n")
        f.write("=" * 80 + "\n")

        # Count how many state-feature pairs fall into each coverage tier
        tier_counts = {
            '100%': 0,
            '90-99%': 0,
            '75-89%': 0,
            '50-74%': 0,
            '25-49%': 0,
            '<25%': 0
        }

        for _, row in df_state_features.iterrows():
            pct = row['pct_coverage']
            if pct == 100:
                tier_counts['100%'] += 1
            elif pct >= 90:
                tier_counts['90-99%'] += 1
            elif pct >= 75:
                tier_counts['75-89%'] += 1
            elif pct >= 50:
                tier_counts['50-74%'] += 1
            elif pct >= 25:
                tier_counts['25-49%'] += 1
            else:
                tier_counts['<25%'] += 1

        total_pairs = len(df_state_features)
        for tier, count in tier_counts.items():
            pct = (count / total_pairs) * 100
            f.write(f"  {tier:10s}: {count:6d} state-feature pairs ({pct:.1f}%)\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 80 + "\n")
        f.write("1. If universal features across all states is LOW:\n")
        f.write("   -> Feature heterogeneity is ACROSS states\n")
        f.write("   -> Solution: Train models per-state or use state-specific features\n")
        f.write("\n")
        f.write("2. If individual states have HIGH universal features:\n")
        f.write("   -> Feature consistency is WITHIN states\n")
        f.write("   -> Solution: State-based models would work well\n")
        f.write("\n")
        f.write("3. If even individual states have LOW universal features:\n")
        f.write("   -> Feature heterogeneity is WITHIN states (county-level)\n")
        f.write("   -> Solution: Need more robust handling of missing features\n")

    print(f"  Saved: {output_dir / 'state_summary.txt'}")

    # === 4. State statistics CSV ===
    print("\nCreating state statistics summary...")
    state_stats = []
    for state in states:
        state_df = df[df['state'] == state]
        n_counties = len(state_df)

        # Count features by coverage tier within this state
        features_100 = sum(1 for f in feature_cols if state_df[f].sum() == n_counties)
        features_90 = sum(1 for f in feature_cols if state_df[f].sum() >= 0.9 * n_counties and state_df[f].sum() < n_counties)
        features_75 = sum(1 for f in feature_cols if state_df[f].sum() >= 0.75 * n_counties and state_df[f].sum() < 0.9 * n_counties)
        features_50 = sum(1 for f in feature_cols if state_df[f].sum() >= 0.5 * n_counties and state_df[f].sum() < 0.75 * n_counties)

        state_stats.append({
            'state': state,
            'n_counties': n_counties,
            'features_100pct': features_100,
            'features_90pct': features_90,
            'features_75pct': features_75,
            'features_50pct': features_50,
            'features_high_coverage_90plus': features_100 + features_90
        })

    df_state_stats = pd.DataFrame(state_stats)
    df_state_stats = df_state_stats.sort_values('features_100pct', ascending=False)
    df_state_stats.to_csv(output_dir / 'state_statistics.csv', index=False)
    print(f"  Saved: {output_dir / 'state_statistics.csv'}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output files saved to: {output_dir}")
    print(f"  - state_feature_coverage.csv: Feature coverage % for each state-feature pair")
    print(f"  - state_universal_features.csv: List of universal features per state")
    print(f"  - state_statistics.csv: Summary stats per state")
    print(f"  - state_summary.txt: Human-readable summary with key insights")
    print()
    print(f"Quick summary:")
    print(f"  - Features universal across ALL states: {len(universal_across_all_states)}")
    print(f"  - Best state: {best_states.iloc[0]['state']} with {best_states.iloc[0]['n_universal_features']} universal features")
    print(f"  - Worst state: {worst_states.iloc[-1]['state']} with {worst_states.iloc[-1]['n_universal_features']} universal features")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_by_state.py <county_feature_matrix_csv> <output_dir>")
        print("Example: python analyze_by_state.py notebooks/feature_investigation/output_preprocessed/county_feature_matrix_preprocessed.csv notebooks/feature_investigation/output_by_state/")
        sys.exit(1)

    matrix_csv = sys.argv[1]
    output_dir = sys.argv[2]

    analyze_by_state(matrix_csv, output_dir)
