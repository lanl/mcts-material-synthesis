"""Generic convergence-sensitivity plot: mean best_reward vs. iteration, one
overlaid curve per swept value with a shaded +/-1 std-dev band across
replicate seeds. Single 3in x 3in panel, 10pt font (no subplots).

Usage: python plot_sweep.py <results_dir>/convergence_data.csv "x label" "title" [out.png]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 7,
    'figure.figsize': (3, 3),
})


def plot_convergence(csv_path, title, out_path):
    df = pd.read_csv(csv_path)
    # Iteration 0 is a pre-search sentinel (max_reward = -10.0, not a real
    # composite score), so the convergence curve starts at iteration 1.
    df = df[df['iteration'] > 0]

    stats = df.groupby(['value', 'iteration'])['best_reward'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots()
    # Preserve the order values were defined in the sweep script, not
    # alphabetical (pandas groupby sorts by default).
    value_order = list(dict.fromkeys(df['value']))
    for value in value_order:
        sub = stats[stats['value'] == value]
        ax.plot(sub['iteration'], sub['mean'], label=str(value), linewidth=1)
        ax.fill_between(sub['iteration'], sub['mean'] - sub['std'], sub['mean'] + sub['std'], alpha=0.2)

    ax.set_xscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best composite reward')
    ax.set_title(title)
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    csv_path = Path(sys.argv[1])
    title = sys.argv[2] if len(sys.argv) > 2 else ''
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else csv_path.with_suffix('.png')
    plot_convergence(csv_path, title, out_path)
