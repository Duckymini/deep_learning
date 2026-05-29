"""
Produces Fig. 3: HP tuning heatmap (K x threshold) for BERT and RoBERTa.
Reads from results/hp_tuning_grid.json and saves to results/fig_hp_heatmap.pdf
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path('..') / 'results'

# ── Load data ─────────────────────────────────────────────────────────────────
with open(RESULTS_DIR / 'hp_tuning_grid.json') as f:
    raw = json.load(f)

models     = ['bert', 'hatebert', 'roberta']
labels     = ['BERT', 'HateBERT', 'RoBERTa']
ks         = [3, 5, 10]
thresholds = [0.3, 0.4, 0.5, 0.6]

# Build 2D arrays: rows = K, cols = threshold
grids = {}
bests = {}
for model in models:
    grid = np.array([
        [raw[f'{model}_{k}_{t}'] for t in thresholds]
        for k in ks
    ])
    grids[model] = grid
    # best cell
    best_idx = np.unravel_index(np.argmax(grid), grid.shape)
    bests[model] = best_idx

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.6))
fig.subplots_adjust(wspace=0.38)

cmap = 'YlOrRd'

for ax, model, label in zip(axes, models, labels):
    grid = grids[model]
    vmin = grid.min() - 0.002
    vmax = grid.max() + 0.002

    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # Annotate each cell
    for i in range(len(ks)):
        for j in range(len(thresholds)):
            val = grid[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8.5, color='black')

    # Mark best cell with a bold rectangle
    bi, bj = bests[model]
    rect = mpatches.FancyBboxPatch(
        (bj - 0.48, bi - 0.48), 0.96, 0.96,
        boxstyle='round,pad=0.02',
        linewidth=2.0, edgecolor='#1a1a1a', facecolor='none',
    )
    ax.add_patch(rect)

    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([str(t) for t in thresholds], fontsize=9)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks], fontsize=9)
    ax.set_xlabel(r'Similarity threshold $\tau$', fontsize=9)
    ax.set_ylabel('$k$ (neighbours)', fontsize=9)
    ax.set_title(label, fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)

fig.suptitle('Macro F1 over HP grid (IHC, Full index)', fontsize=10, y=1.01)

out_path = RESULTS_DIR / 'fig_hp_heatmap.pdf'
plt.savefig(out_path, bbox_inches='tight', dpi=300)
print(f'Saved to {out_path}')

# Also save PNG for quick preview
png_path = RESULTS_DIR / 'fig_hp_heatmap.png'
plt.savefig(png_path, bbox_inches='tight', dpi=200)
print(f'Saved to {png_path}')
