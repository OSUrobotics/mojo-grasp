import os
import re
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
from mojograsp.simcore.data_gui_backend import *
import scikit_posthocs as sp

# Set up color palette and shape labels
colorful_grayscale_friendly_palette = [
    "#1f77b4", "#aec7e8", "#d62728", "#ff9896",
    "#2ca02c", "#98df8a", "#9467bd", "#c5b0d5",
    "#ff7f0e", "#ffbb78", "#17becf", "#9edae5"
]
keys = ['square', 'square25', 'circle', 'circle25', 'triangle', 'triangle25', 'square_circle', 'pentagon', 'trapazoid']
folders = ['/' + f + '_A' for f in keys]

# Define method paths
base_paths = {
    'Static': [
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_Set_90/Static90/Ast_Tests',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/Static_90_1/Ast_Tests',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/Static_90_2/Ast_Tests'
    ],
    'Dynamic': [
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_Set_90/Dynamic90/Ast_Tests',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/Dynamic_90_1/Ast_Tests',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/Dynamic_90_2/Ast_Tests'
    ],
    'Latent': [
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_Set_90/Latent90/Ast_Tests',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/Latent_90_1/Ast_Tests',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/Latent_90_2/Ast_Tests'
    ],
    'Static_Dynamic': [
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_Set_90/SD90/Ast_Tests',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/SD_90_1/Ast_Tests',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/SD_90_2/Ast_Tests'
    ],
    'Static_Latent': [
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_Set_90/SL90/Ast_Tests',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/SL_90_1/Ast_Tests',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/SL_90_2/Ast_Tests'
    ],
    'Static_Corrected': [
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Corrected_Static_Runs/Corrected_Static_90_1/Ast_Tests',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/New_Static_90/Ast_Tests'
    ]
}

# records = []

# for method, paths in base_paths.items():

#     print(method,paths)
#     backend = PlotBackend()

#     for root_path in paths:
#         # find all Ast_A_<shape>_<angle> dirs
#         for entry in os.listdir(root_path):
#             if not entry.startswith("Ast_A_"):
#                 continue

#             full_dir = os.path.join(root_path, entry)
#             if not os.path.isdir(full_dir):
#                 continue

#             parts = entry.split("_")
#             shape = "_".join(parts[2:-1])      # e.g. "square_circle"
#             try:
#                 angle = float(parts[-1])       # e.g. "3"
#             except ValueError:
#                 print(f"Skipping {entry}: bad angle '{parts[-1]}'")
#                 continue

#             try:
#                 Sx, Sy, Gx, Gy, Ex, Ey = backend.ast_return(full_dir)
#                 records.append({
#                     "method":   method,
#                     "shape":    shape,
#                     "angle":    angle,
#                     "start_x":  Sx,
#                     "start_y":  Sy,
#                     "goal_x":   Gx,
#                     "goal_y":   Gy,
#                     "end_x":    Ex,
#                     "end_y":    Ey,
#                 })
#             except Exception as e:
#                 print(f"Error in {full_dir}: {e}")
#             finally:
#                 backend.reset()

#     del backend
#     gc.collect()

# df = pd.DataFrame(records)

# df.to_csv('ast_test.csv', index=True)

# print(df)

df = pd.read_csv('/home/nigel/mojo-grasp/ast_test.csv')
df = df.drop(df.columns[0], axis=1)

# 2) Normalize column names
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(r'\s+', '_', regex=True)
)

# 3) Define a parser for the multi‐line Series repr
def parse_series_repr(s: str) -> list[float]:
    """
    Turn a string like
       "0    0.001174\n1    0.000991\n...Name: Start X, dtype: float64"
    into [0.001174, 0.000991, …].
    """
    lines = s.strip().splitlines()
    vals = []
    for line in lines:
        # skip the summary line
        if line.startswith('Name:') or 'dtype:' in line:
            continue
        # each data line is "index␣␣value"
        parts = line.strip().split()
        if len(parts) >= 2:
            vals.append(float(parts[-1]))
    return vals

# 4) Apply the parser to each of the six coord columns
list_cols = ['start_x','start_y','goal_x','goal_y','end_x','end_y']
for col in list_cols:
    df[col] = df[col].apply(parse_series_repr)

# 5) Cosine‐similarity helper
def cos_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# 6) Compute a list of 8 cosines per row
df['cosines'] = df.apply(
    lambda row: [
        cos_sim(
            np.array([gx - sx, gy - sy]),
            np.array([ex - sx, ey - sy])
        )
        for sx, sy, gx, gy, ex, ey
        in zip(
            row['start_x'], row['start_y'],
            row['goal_x'],  row['goal_y'],
            row['end_x'],   row['end_y']
        )
    ],
    axis=1
)

# 7) Explode those lists so each cosine is its own row
df_exploded = df.explode('cosines').copy()
df_exploded['cosines'] = df_exploded['cosines'].astype(float)

# 8) Group & aggregate
summary = (
    df_exploded
      .groupby(['method', 'angle'])['cosines']
      .agg(mean='mean', std='std')
      .reset_index()
)

print(summary)

df['mean_trial_cosine'] = df['cosines'].apply(np.mean)

# 2) group by method and get method‐level mean & std of those per‐row means
overall = (
    df
      .groupby('method')['mean_trial_cosine']
      .agg(overall_mean='mean', overall_std='std')
      .reset_index()
)

print(overall)

# --------------------------------------------------------
# 1)  Raw cosine similarity — one subplot per method
#     • x‑axis: angle
#     • y‑axis: cosine similarity
#     • outliers hidden (showfliers=False)
# --------------------------------------------------------
angles  = sorted(df_exploded['angle'].unique())
methods = sorted(df_exploded['method'].unique())
n_meth  = len(methods)

fig, axes = plt.subplots(n_meth, 1, sharex=True, figsize=(8, 4 * n_meth))

# if only one method, axes is not a list
if n_meth == 1:
    axes = [axes]

for i, m in enumerate(methods):
    ax      = axes[i]
    subset  = df_exploded[df_exploded['method'] == m]
    data    = [subset[subset['angle'] == a]['cosines'] for a in angles]

    ax.boxplot(data, labels=angles, showfliers=False)
    ax.set_title(f'Method: {m}')
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim(bottom=0.90)

axes[-1].set_xlabel('Angle')
#plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 2)  Mean trial cosine similarity — one boxplot per method
# --------------------------------------------------------
if 'mean_trial_cosine' not in df.columns:
    df['mean_trial_cosine'] = df['cosines'].apply(np.mean)  # safety

mean_data = [df[df['method'] == m]['mean_trial_cosine'] for m in methods]

plt.figure(figsize=(8, 5))
plt.boxplot(mean_data, labels=methods, showfliers=False)
plt.title('Mean Trial Cosine Similarity by Method')
plt.xlabel('Method')
plt.ylabel('Mean Trial Cosine')
plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

CM_PER_M = 100  # conversion factor

# ------------------------------------------------------------------
# 0)  Build net‑distance in **centimetres** if it’s not there yet
# ------------------------------------------------------------------
def scalar_projection_cm(v_end, v_goal):
    norm_goal = np.linalg.norm(v_goal)
    if norm_goal == 0:
        return np.nan
    # scalar projection in metres → convert to cm
    return (np.dot(v_end, v_goal) / norm_goal) * CM_PER_M

if 'net_distance_cm' not in df.columns:
    df['net_distance_cm'] = df.apply(
        lambda row: [
            scalar_projection_cm(
                np.array([ex - sx, ey - sy]),    # v_end
                np.array([gx - sx, gy - sy])     # v_goal
            )
            for sx, sy, gx, gy, ex, ey in zip(
                row['start_x'], row['start_y'],
                row['goal_x'],  row['goal_y'],
                row['end_x'],   row['end_y']
            )
        ],
        axis=1
    )

# ------------------------------------------------------------------
# 1)  Explode to one row per trial
# ------------------------------------------------------------------
df_dist = df.explode('net_distance_cm').copy()
df_dist['net_distance_cm'] = df_dist['net_distance_cm'].astype(float)

# ------------------------------------------------------------------
# 2)  Raw net‑distance (cm) boxplots — one subplot per method
# ------------------------------------------------------------------
angles  = sorted(df_dist['angle'].unique())
methods = sorted(df_dist['method'].unique())
n_meth  = len(methods)

fig, axes = plt.subplots(n_meth, 1, sharex=True, figsize=(8, 4 * n_meth))
axes = axes if n_meth > 1 else [axes]   # always iterable

for ax, m in zip(axes, methods):
    subset = df_dist[df_dist['method'] == m]
    data   = [subset[subset['angle'] == a]['net_distance_cm'] for a in angles]

    ax.boxplot(data, labels=angles, showfliers=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(f'Net Distance (cm) by Angle — Method {m}')
    ax.set_ylabel('Net Distance [cm]')

axes[-1].set_xlabel('Angle')
#plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 3)  Mean per‑trial net‑distance (cm) boxplot
# ------------------------------------------------------------------
if 'mean_net_distance_cm' not in df.columns:
    df['mean_net_distance_cm'] = df['net_distance_cm'].apply(np.nanmean)

mean_data = [df[df['method'] == m]['mean_net_distance_cm'] for m in methods]

plt.figure(figsize=(8, 5))
plt.boxplot(mean_data, labels=methods, showfliers=False)
plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right')
plt.title('Mean Net Distance per Trial (cm) — by Method')
plt.xlabel('Method')
plt.ylabel('Mean Net Distance [cm]')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 4)  Mean ± std of net‑distance (cm) **per method**  (all trials)
#     • Using the exploded DataFrame `df_dist`
# --------------------------------------------------------------
method_stats = (
    df_dist
      .groupby('method')['net_distance_cm']
      .agg(mean='mean', std='std')
      .reset_index()
)

print("\n=== Net‑distance (cm) — per method, all trials ===")
print(method_stats.to_string(index=False))

# --------------------------------------------------------------
# 5)  Mean ± std of **per‑row mean** net‑distance  (cm) per method
#     • Uses `mean_net_distance_cm` in the original `df`
# --------------------------------------------------------------
overall_stats = (
    df
      .groupby('method')['mean_net_distance_cm']
      .agg(mean='mean', std='std')
      .reset_index()
)

print("\n=== Mean trial net‑distance (cm) — per method ===")
print(overall_stats.to_string(index=False))