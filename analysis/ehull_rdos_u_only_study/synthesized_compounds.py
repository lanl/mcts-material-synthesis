"""Shared metadata for the U-only ehull_rdos study.

Single source of truth for the four U-compounds our experimentalists have
successfully synthesized in the past, used to flag "Priority Match" rows in
generate_top10_report.py and the synthesized-compound overlay in
generate_figures.py. Dash "f_block-groupIV-metal" format - see
DoscarRewardLookup.convert_formula_to_doscar_format.
"""

SYNTHESIZED_COMPOUNDS = ['U-Sn-V', 'U-Sn-Nb', 'U-Ge-Cr', 'U-Ge-Co']
