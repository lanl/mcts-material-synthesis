#!/usr/bin/env python3
"""
Calculate rewards for compounds based on DOSCAR peak data.

For each compound, the reward is calculated as:
    reward = sum((peak_height/peak_width) * exp(-0.5*(1/sigma)^2))

where sigma = 0.5 and the sum is over all peaks for that compound. The result
is left unscaled; any normalization is applied downstream via the composite
score's gamma/gamma_prefactor weight, not baked into the reward itself.

Usage:
    python calculate_doscar_rewards.py
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_compound_reward(peaks_df, sigma=0.5):
    """
    Calculate reward for a single compound based on its peaks.

    Args:
        peaks_df: DataFrame containing peaks for a single compound
        sigma: Gaussian width parameter (default: 0.5)

    Returns:
        Calculated reward value
    """
    # Calculate the exponential factor (constant for all peaks)
    exp_factor = np.exp(-0.5 * (1.0 / sigma) ** 2)

    # Calculate the sum of (peak_height / peak_width) * exp_factor
    peak_contributions = (peaks_df['PEAK_HEIGHT'] / peaks_df['PEAK_WIDTH']) * exp_factor
    return peak_contributions.sum()


def main():
    """Calculate rewards for all compounds in DOSCAR peaks data."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    logger.info("=" * 80)
    logger.info("DOSCAR PEAKS REWARD CALCULATOR")
    logger.info("=" * 80)

    # Step 1: Load DOSCAR peaks data
    logger.info("\n1. Loading DOSCAR peaks data...")
    input_file = Path(__file__).parent.parent / "doscar_peaks_data_with_U.csv"

    if not input_file.exists():
        logger.error(f"Error: Input file not found: {input_file}")
        return 1

    try:
        df = pd.read_csv(input_file)
        logger.info(f"   Loaded {len(df)} peak records")
        logger.info(f"   Columns: {', '.join(df.columns)}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return 1

    # Step 2: Filter compounds - prefer core (non-valence) over valence
    logger.info("\n2. Filtering compounds (prefer core over valence)...")

    # Separate core and valence compounds
    core_compounds = df[~df['COMPOUND_NAME'].str.endswith('_valence')]
    valence_compounds = df[df['COMPOUND_NAME'].str.endswith('_valence')]

    # Get base names from valence compounds (remove _valence suffix)
    valence_base_names = valence_compounds['COMPOUND_NAME'].str.replace('_valence', '').unique()
    core_names = core_compounds['COMPOUND_NAME'].unique()

    # Find valence compounds whose core version doesn't exist
    missing_core_bases = set(valence_base_names) - set(core_names)
    valence_to_include = valence_compounds[
        valence_compounds['COMPOUND_NAME'].str.replace('_valence', '').isin(missing_core_bases)
    ]

    # Combine core compounds with valence-only compounds
    filtered_df = pd.concat([core_compounds, valence_to_include])

    logger.info(f"   Core compounds: {len(core_names)}")
    logger.info(f"   Valence compounds (no core): {len(missing_core_bases)}")
    logger.info(f"   Total compounds to process: {len(core_names) + len(missing_core_bases)}")

    # Step 3: Calculate rewards for each compound
    logger.info("\n3. Calculating rewards for each compound...")
    sigma = 0.5
    logger.info(f"   Using sigma = {sigma}")
    logger.info(f"   Formula: sum((peak_height/peak_width) * exp(-0.5*(1/sigma)^2))")

    # Group by compound name
    compound_groups = filtered_df.groupby('COMPOUND_NAME')
    n_compounds = len(compound_groups)

    # Calculate reward for each compound (unscaled; normalization happens
    # downstream via gamma/gamma_prefactor, not here)
    results = []
    for compound_name, group_df in compound_groups:
        exp_factor = np.exp(-0.5 * (1.0 / sigma) ** 2)
        peak_contributions = (group_df['PEAK_HEIGHT'] / group_df['PEAK_WIDTH']) * exp_factor
        unscaled_sum = peak_contributions.sum()
        results.append({
            'compound_name': compound_name,
            'reward_raw': unscaled_sum,
            'reward_normalized': unscaled_sum
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    logger.info(f"   Calculated rewards for all {n_compounds} compounds")

    # Step 4: Display statistics
    logger.info("\n4. Reward Statistics:")
    logger.info(f"   Mean reward:   {results_df['reward_normalized'].mean():.6f}")
    logger.info(f"   Median reward: {results_df['reward_normalized'].median():.6f}")
    logger.info(f"   Min reward:    {results_df['reward_normalized'].min():.6f}")
    logger.info(f"   Max reward:    {results_df['reward_normalized'].max():.6f}")
    logger.info(f"   Std dev:       {results_df['reward_normalized'].std():.6f}")

    # Step 5: Show top compounds
    logger.info("\n5. Top 10 Compounds by Reward:")
    top_compounds = results_df.nlargest(10, 'reward_normalized')
    for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
        logger.info(f"   {i:2d}. {row['compound_name']:20s}  raw = {row['reward_raw']:10.2f}  normalized = {row['reward_normalized']:8.6f}")

    # Step 6: Save results
    logger.info("\n6. Saving results...")
    output_file = Path(__file__).parent.parent / "doscar_rewards.csv"

    try:
        results_df.to_csv(output_file, index=False)
        logger.info(f"   Results saved to: {output_file}")
        logger.info(f"   Total compounds: {len(results_df)}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return 1

    logger.info("\n" + "=" * 80)
    logger.info("REWARD CALCULATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput file: {output_file}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
