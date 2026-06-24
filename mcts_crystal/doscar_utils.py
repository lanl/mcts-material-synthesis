"""
Utility functions for DOSCAR reward lookup and formula conversion.
"""

import logging
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DoscarRewardLookup:
    """
    Handles loading and looking up DOSCAR rewards for compounds.
    """

    def __init__(self, csv_file: Optional[str] = None):
        """
        Initialize DOSCAR reward lookup.

        Args:
            csv_file: Path to doscar_rewards.csv file
        """
        self.rewards_dict = {}

        # Determine default CSV and peaks file locations
        repo_root = Path(__file__).parent.parent
        default_csv = repo_root / "doscar_rewards.csv"
        peaks_file = repo_root / "doscar_peaks_data_with_U.csv"

        # Allow user-specified csv_file path; otherwise prefer default CSV
        if csv_file is None:
            csv_path = default_csv
        else:
            csv_path = Path(csv_file)

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Create dictionary mapping compound_name -> reward_normalized
            if 'compound_name' in df.columns and 'reward_normalized' in df.columns:
                self.rewards_dict = dict(zip(df['compound_name'], df['reward_normalized']))
                logger.info(f"   Loaded {len(self.rewards_dict)} DOSCAR rewards from {csv_path.name}")
            else:
                logger.warning(f"   CSV file {csv_path} missing expected columns; ignoring")

        elif peaks_file.exists():
            # Compute rewards from raw peaks data on-the-fly (fall back when precomputed CSV absent)
            try:
                logger.info(f"   Precomputed DOSCAR CSV not found; computing rewards from peaks: {peaks_file.name}")
                peaks_df = pd.read_csv(peaks_file)
                # Prefer core compounds (no '_valence' suffix); include valence-only compounds if core missing
                core_compounds = peaks_df[~peaks_df['COMPOUND_NAME'].str.endswith('_valence')]
                valence_compounds = peaks_df[peaks_df['COMPOUND_NAME'].str.endswith('_valence')]
                valence_base_names = valence_compounds['COMPOUND_NAME'].str.replace('_valence', '').unique()
                core_names = core_compounds['COMPOUND_NAME'].unique()
                missing_core_bases = set(valence_base_names) - set(core_names)
                valence_to_include = valence_compounds[
                    valence_compounds['COMPOUND_NAME'].str.replace('_valence', '').isin(missing_core_bases)
                ]
                filtered_df = pd.concat([core_compounds, valence_to_include])

                # Compute reward for each compound. Left unscaled here; any
                # normalization is applied downstream via gamma/gamma_prefactor.
                sigma = 0.5
                exp_factor = np.exp(-0.5 * (1.0 / sigma) ** 2)
                results = {}
                for cname, group in filtered_df.groupby('COMPOUND_NAME'):
                    # sum (PEAK_HEIGHT / PEAK_WIDTH) * exp_factor
                    contrib = (group['PEAK_HEIGHT'] / group['PEAK_WIDTH']) * exp_factor
                    results[cname] = float(contrib.sum())

                self.rewards_dict = results
                logger.info(f"   Computed {len(self.rewards_dict)} DOSCAR rewards from peaks data")
            except Exception as e:
                logger.error(f"   Error computing DOSCAR rewards from peaks: {e}")
                logger.warning(f"   DOSCAR rewards will be set to 0.0")
        else:
            logger.warning(f"   DOSCAR rewards file not found: {csv_path}")
            logger.warning(f"   DOSCAR peaks file not found: {peaks_file}")
            logger.warning(f"   DOSCAR rewards will be set to 0.0")

    def convert_formula_to_doscar_format(self, formula: str) -> Optional[str]:
        """
        Convert MCTS formula (e.g., Ti6Si6Ce) to DOSCAR format (e.g., Ce-Si-Ti).

        The DOSCAR format is: fblock-groupIV-metal
        The MCTS format is: metal6groupIV6fblock (with counts)

        Args:
            formula: Chemical formula in MCTS format (e.g., "Ti6Si6Ce")

        Returns:
            Formula in DOSCAR format (e.g., "Ce-Si-Ti") or None if cannot convert
        """
        # Parse formula to extract elements and counts
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)

        elements = {}
        for element, count in matches:
            if element:  # Skip empty matches
                count = int(count) if count else 1
                elements[element] = count

        if len(elements) != 3:
            return None  # DOSCAR format expects exactly 3 elements

        # Define element categories
        group_iv = {'Si', 'Ge', 'Sn', 'Pb'}
        f_block = set()
        # Lanthanides: Ce (58) to Lu (71)
        lanthanides = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        # Actinides: Th (90) to Pu (94)
        actinides = ['Th', 'Pa', 'U', 'Np', 'Pu']
        f_block = set(lanthanides + actinides)

        # Transition metals: 3d, 4d, 5d
        transition_metals = {
            'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',  # 3d
            'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',  # 4d
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'    # 5d
        }

        # Identify element types
        f_elem = None
        g_iv_elem = None
        metal_elem = None

        for elem in elements.keys():
            if elem in f_block:
                f_elem = elem
            elif elem in group_iv:
                g_iv_elem = elem
            elif elem in transition_metals:
                metal_elem = elem

        # Check if we found all three types
        if f_elem is None or g_iv_elem is None or metal_elem is None:
            return None

        # Format as fblock-groupIV-metal
        doscar_format = f"{f_elem}-{g_iv_elem}-{metal_elem}"
        return doscar_format

    def get_reward(self, formula: str) -> float:
        """
        Get DOSCAR reward for a given compound formula.

        Args:
            formula: Chemical formula in MCTS format (e.g., "Ti6Si6Ce")

        Returns:
            Normalized DOSCAR reward, or 0.0 if not found
        """
        # Convert to DOSCAR format
        doscar_formula = self.convert_formula_to_doscar_format(formula)

        if doscar_formula is None:
            return 0.0

        # Look up reward
        return self.rewards_dict.get(doscar_formula, 0.0)
