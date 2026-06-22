#!/usr/bin/env python3
"""
Prepare data for gnuplot visualization - Space Group 191 only.
Generates top 10 MCTS predictions from this run's composite score ranking.
"""
import pandas as pd
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent

# Read shunshun reference CSV (all SG191 compounds as background)
reference_csv = script_dir.parent / 'shunshun_mace_predictions_with_elements.csv'
print(f"Reading shunshun reference compounds from: {reference_csv}")
df = pd.read_csv(reference_csv)
print(f"Total compounds in reference: {len(df)}")

# Filter to space group 191 only
df_sg191 = df[df['Space_group'] == 191].copy()
print(f"Compounds in space group 191: {len(df_sg191)}")

# Create output directory
output_dir = script_dir / 'gnuplot_data_sg191'
output_dir.mkdir(exist_ok=True)

# All SG191 compounds (grey dots)
output_file = output_dir / 'all_compounds_sg191.dat'
with open(output_file, 'w') as f:
    f.write("# e_form e_hull\n")
    for _, row in df_sg191.iterrows():
        e_form = row['Predicted_formation_energy (eV/atom)']
        e_hull = row['energy_above_hull']
        f.write(f"{e_form}\t{e_hull}\n")
print(f"Data file created: {output_file} ({len(df_sg191)} compounds)")

# Compute Pareto front for space group 191
e_forms = df_sg191['Predicted_formation_energy (eV/atom)'].values
e_hulls = df_sg191['energy_above_hull'].values

sorted_indices = np.argsort(e_forms)
sorted_e_forms = e_forms[sorted_indices]
sorted_e_hulls = e_hulls[sorted_indices]

pareto_e_forms = []
pareto_e_hulls = []
min_e_hull = float('inf')
for e_form, e_hull in zip(sorted_e_forms, sorted_e_hulls):
    if e_hull < min_e_hull:
        pareto_e_forms.append(e_form)
        pareto_e_hulls.append(e_hull)
        min_e_hull = e_hull

pareto_file = output_dir / 'pareto_front_sg191.dat'
with open(pareto_file, 'w') as f:
    f.write("# e_form e_hull\n")
    for e_form, e_hull in zip(pareto_e_forms, pareto_e_hulls):
        f.write(f"{e_form}\t{e_hull}\n")
print(f"Pareto front file created: {pareto_file} ({len(pareto_e_forms)} points)")

# Generate top 10 MCTS predictions from THIS run's composite score ranking
composite_csv = script_dir / 'all_compounds_by_composite_score.csv'
if composite_csv.exists():
    df_mcts = pd.read_csv(composite_csv)
    top10 = df_mcts.head(10)
    top10_file = output_dir / 'top10_mcts_composite.dat'
    with open(top10_file, 'w') as f:
        f.write("# e_form e_hull formula\n")
        for _, row in top10.iterrows():
            f.write(f"{row['e_form']}\t{row['e_above_hull']}\t{row['name']}\n")
    print(f"Top 10 MCTS predictions file created: {top10_file}")
else:
    print(f"Warning: {composite_csv} not found - run generate_top10_report.py first")

# Print statistics
print(f"\nSpace Group 191 Statistics:")
print(f"  E_form range: [{e_forms.min():.4f}, {e_forms.max():.4f}] eV/atom")
print(f"  E_hull range: [{e_hulls.min():.4f}, {e_hulls.max():.4f}] eV/atom")
print(f"  Thermodynamically stable (E_hull < 0): {sum(1 for e in e_hulls if e < 0)}")
print(f"  Near-stable (E_hull < 0.1): {sum(1 for e in e_hulls if e < 0.1)}")
