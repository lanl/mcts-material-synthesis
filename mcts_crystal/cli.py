"""
Simple MCTS Crystal Structure Optimization Runner.

This is the implementation behind `python run_mcts.py [OPTIONS]` (a thin wrapper
at the repo root, kept for backward compatibility) and the `mcts-run` console
command installed by `pip install -e .`.

Usage:
    python run_mcts.py [OPTIONS]

Local secrets/config:
    Instead of passing --mp-api-key on the command line, copy config.example.json
    to config.json and fill in your Materials Project API key. config.json is
    gitignored and is never pushed to the repo; any value loaded from it is used
    as the default and can still be overridden by a CLI flag.

Examples:
    python run_mcts.py                                    # Default: 1000 iterations, ehull rollout
    python run_mcts.py --iterations 100                   # 100 iterations
    python run_mcts.py --structure my_structure.cif        # Custom structure
    python run_mcts.py --rollout-method ehull --mp-api-key YOUR_KEY       # E_hull only (MACE + Materials Project, no DFT/DOSCAR data needed)
    python run_mcts.py --rollout-method ehull_rdos --beta 1.0 --gamma 2.5 --mp-api-key YOUR_KEY  # E_hull + rDOS (requires doscar_rewards.csv)
    python run_mcts.py --rollout-method rdos               # rDOS only (requires doscar_rewards.csv, no MACE/MP needed)
    python run_mcts.py --f-block-mode lanthanides_u_extended  # Lanthanides + U, extended moves
    python run_mcts.py --exploration-constant 0.2          # Higher exploration (default: 0.1)
    python run_mcts.py --no-labels                         # Turn off labels on radial tree visualization
    python run_mcts.py --iterations 200 --structure my_structure.cif --rollout-method ehull
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

from .node import MCTSTreeNode
from .mcts import MCTS
from .energy_calculator import MaceEnergyCalculator
from .visualization import TreeVisualizer
from .analysis import ResultsAnalyzer
from .doscar_utils import DoscarRewardLookup
from ase.io import read
from ase import Atoms
import pandas as pd


def load_config(config_path: str = 'config.json') -> dict:
    """
    Load local run configuration (e.g. Materials Project API key) if present.

    config.json is gitignored - it never gets pushed to the repo. See
    config.example.json for the expected keys. Values loaded here become
    argparse defaults, so any CLI flag still overrides them.
    """
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            config = json.load(f)
        print(f"Loaded local config from {path}")
        return config
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return {}


def override_composition(atoms: Atoms, transition_metal: str = None, group_iv: str = None) -> Atoms:
    """
    Override the composition of a structure while maintaining the crystal structure.

    Args:
        atoms: ASE Atoms object with template structure
        transition_metal: Symbol for transition metal (e.g., 'Rh', 'Pt', 'W')
        group_iv: Symbol for Group IV element (e.g., 'Si', 'Ge', 'Sn', 'Pb')

    Returns:
        New Atoms object with substituted elements
    """
    # Element symbol to atomic number mapping
    element_to_z = {
        'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
        'Si': 14, 'Ge': 32, 'Sn': 50, 'Pb': 82,
        'U': 92
    }

    # Define element groups
    transition_metals = list(range(22, 31)) + list(range(40, 49)) + list(range(72, 81))
    group_iv_elements = [14, 32, 50, 82]

    # Get target atomic numbers
    target_tm_z = element_to_z.get(transition_metal) if transition_metal else None
    target_giv_z = element_to_z.get(group_iv) if group_iv else None

    # Validate
    if target_tm_z is not None and target_tm_z not in transition_metals:
        raise ValueError(f"Invalid transition metal: {transition_metal}")
    if target_giv_z is not None and target_giv_z not in group_iv_elements:
        raise ValueError(f"Invalid Group IV element: {group_iv}")

    # Create new atoms with substitutions
    new_atoms = atoms.copy()
    atomic_numbers = new_atoms.get_atomic_numbers().copy()

    for i, z in enumerate(atomic_numbers):
        if z in transition_metals and target_tm_z is not None:
            atomic_numbers[i] = target_tm_z
        elif z in group_iv_elements and target_giv_z is not None:
            atomic_numbers[i] = target_giv_z
        # f-block elements (U) are not changed - kept as is

    new_atoms.set_atomic_numbers(atomic_numbers)
    return new_atoms


def build_parser(config: Optional[dict] = None) -> argparse.ArgumentParser:
    """
    Build the run_mcts.py argument parser.

    Args:
        config: Values loaded from config.json (or {}/None). These become argparse
            defaults; an explicit CLI flag always overrides them.
    """
    parser = argparse.ArgumentParser(description='Run MCTS crystal structure optimization')
    parser.add_argument('--iterations', '-n', type=int, default=1000,
                       help='Number of MCTS iterations (default: 1000)')
    parser.add_argument('--structure', '-s', type=str,
                       default='examples/mat_Pb6U1W6_sg191.cif',
                       help='Path to starting crystal structure CIF file')
    parser.add_argument('--output', '-o', type=str, default='mcts_results',
                       help='Output directory name (default: mcts_results)')
    parser.add_argument('--f-block-mode', type=str, default='u_only',
                       choices=['u_only', 'full_f_block', 'experimental', 'lanthanides_u', 'lanthanides_u_extended'],
                       help='F-block substitution mode: u_only (default), full_f_block, experimental (actinides except Ac), lanthanides_u (lanthanides + U, ±1 moves), or lanthanides_u_extended (lanthanides + U, ±3 moves for faster exploration of heavy lanthanides)')
    parser.add_argument('--exploration-constant', '-c', type=float, default=0.1,
                       help='Exploration constant for UCB calculation (default: 0.1)')
    parser.add_argument('--rollout-method', type=str, default='ehull',
                       choices=['ehull', 'ehull_rdos', 'rdos'],
                       help='Rollout method: ehull (tanh-transformed energy above hull only; MACE + Materials Project, no DFT/DOSCAR data needed), '
                            'ehull_rdos (ehull + rDOS, weighted by --beta/--gamma; requires doscar_rewards.csv), '
                            'or rdos (rDOS only, looked up from doscar_rewards.csv; no MACE/Materials Project needed). '
                            'Reward: ehull -> ehull_reward(e_above_hull); ehull_rdos -> beta*ehull_reward(e_above_hull) + gamma*r_DOS; rdos -> r_DOS')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for the E_hull reward when using ehull_rdos (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=2.5,
                       help='Weight for the rDOS reward when using ehull_rdos (default: 2.5)')
    parser.add_argument('--termination-limit', type=int, default=60,
                       help='Number of visits before terminating a node without improvement (default: 60)')
    parser.add_argument('--epsilon', '-e', type=float, default=0.2,
                       help='Exploration rate for epsilon-greedy selection (default: 0.2, range: 0.0-1.0)')
    parser.add_argument('--rollout-depth', type=int, default=1,
                       help='Number of random mutations per rollout (default: 1). Higher values create more random compounds.')
    parser.add_argument('--n-rollout', type=int, default=5,
                       help='Number of rollout simulations per expansion (default: 5). Higher values increase computation per iteration.')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    parser.add_argument('--mp-api-key', type=str, default=None,
                       help='Materials Project API key (required for rollout methods: ehull, ehull_rdos). Prefer setting this in config.json instead of passing it on the command line.')
    parser.add_argument('--no-labels', action='store_true',
                       help='Turn off labels on radial tree visualization (default: labels shown)')
    parser.add_argument('--transition-metal', type=str, default=None,
                       help='Override transition metal in starting structure (e.g., Rh, Pt, W). If specified, substitutes TM in loaded CIF file.')
    parser.add_argument('--group-iv', type=str, default=None,
                       help='Override Group IV element in starting structure (e.g., Si, Ge, Sn, Pb). If specified, substitutes Group IV in loaded CIF file.')

    if config:
        # Local config supplies defaults; explicit CLI flags still take precedence.
        known_dests = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in config.items() if k in known_dests})

    return parser


def main():
    """Main MCTS runner function."""
    config = load_config()
    parser = build_parser(config)
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")

    # Validate that MP API key is provided when needed
    methods_requiring_api_key = ['ehull', 'ehull_rdos']
    if args.rollout_method in methods_requiring_api_key and args.mp_api_key is None:
        print(f"❌ Error: --mp-api-key is required when using rollout method '{args.rollout_method}'")
        print(f"   Energy above hull calculations require Materials Project API access")
        print(f"   Get your API key from: https://materialsproject.org/api")
        print(f"   Then run with: --mp-api-key YOUR_KEY, or set \"mp_api_key\" in config.json")
        return 1

    # Check if DOSCAR rewards file exists for methods that need rDOS
    if args.rollout_method in ['ehull_rdos', 'rdos']:
        doscar_file = Path("doscar_rewards.csv")
        if not doscar_file.exists():
            print(f"❌ Error: doscar_rewards.csv not found for rollout method '{args.rollout_method}'")
            print(f"   This file is derived from local DFT/DOSCAR data and is not bundled with the repo")
            print(f"   pre-release. See the README's Data Availability section for the expected schema")
            print(f"   and how to generate it from your own DOSCAR data.")
            return 1

    print("=" * 80)
    print("MCTS CRYSTAL STRUCTURE OPTIMIZATION")
    print("=" * 80)

    # Step 1: Load starting crystal structure
    print(f"\n1. Loading starting crystal structure...")
    structure_path = Path(args.structure)

    if not structure_path.exists():
        print(f"❌ Error: Structure file not found: {structure_path}")
        print(f"   Please check the file path or use the default structure")
        return 1

    try:
        atoms = read(str(structure_path))
        print(f"   ✓ Loaded: {atoms.get_chemical_formula()}")
        print(f"   ✓ File: {structure_path}")
        print(f"   ✓ Atoms: {len(atoms)}")

        # Override composition if requested
        if args.transition_metal is not None or args.group_iv is not None:
            print(f"\n1.5. Overriding starting composition...")
            atoms = override_composition(atoms, args.transition_metal, args.group_iv)
            print(f"   ✓ New composition: {atoms.get_chemical_formula()}")
    except Exception as e:
        print(f"❌ Error loading structure: {e}")
        return 1

    # Step 2: Set up energy calculator
    print(f"\n2. Setting up energy calculator...")
    csv_file = Path("high_throughput_mace_results.full.csv")

    if not csv_file.exists():
        print(f"❌ Error: MACE calculations file not found: {csv_file}")
        print(f"   Please ensure high_throughput_mace_results.full.csv is in the working directory")
        print(f"   See the README's Data Availability section for the expected schema.")
        return 1

    try:
        df = pd.read_csv(csv_file)
        energy_calc = MaceEnergyCalculator(csv_file=str(csv_file), mp_api_key=args.mp_api_key)
        print(f"   ✓ Cached calculations: {len(df)} entries")
        print(f"   ✓ Energy range: {df['e_form'].min():.3f} to {df['e_form'].max():.3f} eV/atom")
        if args.mp_api_key:
            print(f"   ✓ Materials Project API key provided")
        else:
            print(f"   ⚠ No MP API key - energy above hull will be approximate (e_above_hull = e_form)")
    except Exception as e:
        print(f"❌ Error setting up energy calculator: {e}")
        return 1

    # Load DOSCAR rewards if this rollout method uses rDOS
    doscar_lookup = None
    if args.rollout_method in ['ehull_rdos', 'rdos']:
        print(f"\n2.5. Loading DOSCAR rewards...")
        try:
            doscar_lookup = DoscarRewardLookup()
        except Exception as e:
            print(f"❌ Error loading DOSCAR rewards: {e}")
            print(f"   Cannot continue without DOSCAR rewards for '{args.rollout_method}' method")
            return 1

    # Step 3: Initialize MCTS
    print(f"\n3. Initializing MCTS algorithm...")
    try:
        root_node = MCTSTreeNode(atoms, f_block_mode=args.f_block_mode,
                                exploration_constant=args.exploration_constant,
                                termination_limit=args.termination_limit)

        # Calculate energies for root node (tracked for reference even when not part of the reward)
        root_e_form, root_e_hull = energy_calc.calculate_energies(atoms)
        root_node.e_form = root_e_form
        root_node.e_above_hull = root_e_hull

        mcts = MCTS(root_node, epsilon=args.epsilon)

        print(f"   ✓ Root compound: {root_node.get_chemical_formula()}")
        print(f"   ✓ Root E_form: {root_e_form:.4f} eV/atom")
        print(f"   ✓ Root E_hull: {root_e_hull:.4f} eV/atom")
        print(f"   ✓ F-block mode: {args.f_block_mode}")
        print(f"   ✓ Exploration constant: {args.exploration_constant}")
        print(f"   ✓ Epsilon (exploration rate): {args.epsilon}")
        print(f"   ✓ Termination limit: {args.termination_limit}")

        # Show search space
        root_node.expand()
        print(f"   ✓ Search space: {len(root_node.expansion_list)} possible moves")
        print(f"   ✓ Transition metals: {len(root_node.metal_move)} options")
        print(f"   ✓ Group IV elements: {len(root_node.g_iv_move)} options")
        if hasattr(root_node, 'f_block_move'):
            if args.f_block_mode == 'u_only':
                print(f"   ✓ F-block elements: {len(root_node.f_block_move)} options (U-only mode)")
            else:
                print(f"   ✓ F-block elements: {len(root_node.f_block_move)} options (full f-block)")

    except Exception as e:
        print(f"❌ Error initializing MCTS: {e}")
        return 1

    # Step 4: Run MCTS optimization
    print(f"\n4. Running MCTS optimization...")
    print(f"   Iterations: {args.iterations}")
    print(f"   Rollout method: {args.rollout_method}")
    print(f"   Rollout depth: {args.rollout_depth}")
    print(f"   Number of rollouts: {args.n_rollout}")
    if args.rollout_method == 'ehull_rdos':
        print(f"   Beta (E_hull weight, ehull_reward = -tanh(300*(E_hull-0.05))): {args.beta}")
        print(f"   Gamma (rDOS weight): {args.gamma}")
    print(f"   This may take several minutes depending on cache hit rate...")

    try:
        results = mcts.run(
            n_iterations=args.iterations,
            energy_calculator=energy_calc,
            rollout_depth=args.rollout_depth,
            n_rollout=args.n_rollout,
            selection_mode='epsilon',
            rollout_method=args.rollout_method,
            beta=args.beta,
            gamma=args.gamma,
            doscar_lookup=doscar_lookup
        )

        print(f"   ✓ Completed: {results['iterations_completed']} iterations")
        print(f"   ✓ Compounds explored: {len(results['stat_dict'])}")
        print(f"   ✓ Search terminated: {results['terminated']}")

    except Exception as e:
        print(f"❌ Error during MCTS run: {e}")
        return 1

    # Step 5: Analyze results
    print(f"\n5. Analyzing results...")
    try:
        analyzer = ResultsAnalyzer(csv_file=str(csv_file))

        # Get efficiency metrics
        efficiency = analyzer.analyze_search_efficiency(mcts.stat_dict)

        print(f"   ✓ Best formation energy: {efficiency['best_formation_energy']:.4f} eV/atom")
        print(f"   ✓ Best compound: {results['best_node_formula']}")
        print(f"   ✓ Compounds within 100 meV of hull: {efficiency['compounds_near_hull_100meV']}")
        print(f"   ✓ Search efficiency: {efficiency['search_diversity']:.4f}")

    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return 1

    # Step 6: Save results
    print(f"\n6. Saving results...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save convergence history first (before visualizations that might fail)
    try:
        mcts.save_convergence_history(str(output_dir / "convergence_history.csv"))
    except Exception as e:
        print(f"Warning: Could not save convergence history: {e}")

    try:
        # Create visualizations
        visualizer = TreeVisualizer()

        # Radial tree visualization
        visualizer.plot_radial_tree_visualization(
            mcts,
            output_dir=str(output_dir),
            csv_file=str(csv_file),
            show_labels=not args.no_labels
        )

        # Energy distribution plot
        visualizer.plot_energy_distribution(
            mcts.stat_dict,
            top_n=15,
            save_path=output_dir / "energy_distribution.png",
            csv_file=str(csv_file)
        )

        # Iteration progress plot
        visualizer.plot_iteration_progress(
            mcts,
            save_path=output_dir / "iteration_progress.png",
            csv_file=str(csv_file)
        )

        # Energy above hull distribution plot
        visualizer.plot_energy_above_hull_distribution(
            mcts.stat_dict,
            top_n=15,
            save_path=output_dir / "energy_above_hull_distribution.png",
            csv_file=str(csv_file)
        )

        # Energy above hull iteration progress plot
        visualizer.plot_energy_above_hull_progress(
            mcts,
            save_path=output_dir / "energy_above_hull_progress.png",
            csv_file=str(csv_file)
        )

        # Formation energy by elements plot
        visualizer.plot_formation_energy_by_elements(
            mcts.stat_dict,
            csv_file=str(csv_file),
            save_path=output_dir / "formation_energy_by_elements.png"
        )

        # Energy above hull by elements plot
        visualizer.plot_energy_above_hull_by_elements(
            mcts.stat_dict,
            csv_file=str(csv_file),
            save_path=output_dir / "energy_above_hull_by_elements.png"
        )

        # Generate reports
        analyzer.create_summary_report(
            mcts,
            save_path=output_dir / "mcts_report.txt"
        )

        # Export data
        analyzer.export_results(
            mcts.stat_dict,
            output_dir / "all_compounds.csv"
        )

        # Get and display top compounds
        top_compounds = analyzer.get_top_compounds(mcts.stat_dict, n_top=10)

        print(f"   ✓ Results saved to: {output_dir.absolute()}")

    except Exception as e:
        print(f"❌ Error saving results: {e}")
        # Don't return early - still save MCTS pickle below

    # Save MCTS object for later visualization (outside try-except to ensure it always runs)
    try:
        import pickle
        mcts_pickle_path = output_dir / "mcts_object.pkl"
        with open(mcts_pickle_path, 'wb') as f:
            pickle.dump(mcts, f)
        print(f"   ✓ MCTS object saved to: {mcts_pickle_path}")
    except Exception as e:
        print(f"❌ Error saving MCTS pickle: {e}")

    # Step 7: Display summary
    print(f"\n" + "=" * 80)
    print("🎯 MCTS OPTIMIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"🥇 Best compound: {results['best_node_formula']}")
    print(f"⚡ Formation energy: {results['best_node_e_form']:.4f} eV/atom")
    print(f"🔍 Total compounds explored: {len(results['stat_dict'])}")
    print(f"📁 Results directory: {output_dir.absolute()}")

    print(f"\n🏆 TOP 10 COMPOUNDS DISCOVERED:")
    print("-" * 60)
    for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
        stability = "Stable" if row['e_above_hull'] < 0.1 else "Metastable"
        print(f"{i:2d}. {row['formula']:15s} | "
              f"E_form: {row['formation_energy']:8.4f} eV/atom | "
              f"{stability}")

    print(f"\n📊 FILES CREATED:")
    print(f"   • radial_tree_visualization.png - Tree structure with formation energies")
    print(f"   • energy_distribution.png - Formation energy distribution")
    print(f"   • iteration_progress.png - Search progress over iterations")
    print(f"   • energy_above_hull_distribution.png - Energy above hull distribution")
    print(f"   • energy_above_hull_progress.png - Energy above hull search progress")
    print(f"   • formation_energy_by_elements.png - Formation energy by transition metal/Group IV")
    print(f"   • energy_above_hull_by_elements.png - Energy above hull by transition metal/Group IV")
    print(f"   • mcts_report.txt - Detailed text report")
    print(f"   • all_compounds.csv - All discovered compounds data")

    print(f"\n💡 To run again:")
    print(f"   python run_mcts.py --iterations {args.iterations}")
    print(f"   python run_mcts.py --iterations 1000  # Longer search")
    print(f"   python run_mcts.py --structure my_file.cif  # Different starting material")
    print(f"   python run_mcts.py --f-block-mode lanthanides_u_extended  # Fast exploration of heavy lanthanides")
    print(f"   python run_mcts.py --rollout-method ehull_rdos --beta 1.0 --gamma 2.5  # E_hull + rDOS")
    print(f"   python run_mcts.py -c 0.2  # Higher exploration constant")

    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
