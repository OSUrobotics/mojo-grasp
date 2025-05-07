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
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_Set_90/Static90',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/Static_90_1',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/Static_90_2'
    ],
    'Dynamic': [
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_Set_90/Dynamic90',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/Dynamic_90_1',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/Dynamic_90_2'
    ],
    'Latent': [
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_Set_90/Latent90',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/Latent_90_1',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/Latent_90_2'
    ],
    'Static_Dynamic': [
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_Set_90/SD90',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/SD_90_1',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/SD_90_2'
    ],
    'Static_Latent': [
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_Set_90/SL90',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/SL_90_1',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Full_90_2/SL_90_2'
    ],
    'Static_Corrected': [
        '/home/nigel/mojo-grasp/demos/rl_demo/data/Corrected_Static_Runs/Corrected_Static_90_1',
        '/home/nigel/mojo-grasp/demos/rl_demo/data/New_Static_90'
    ]
}

# Function to save figures by title
def save_figure(fig, title):
    os.makedirs("figures", exist_ok=True)
    filename = re.sub(r'\W+', '_', title).strip('_') + ".png"
    fig.savefig(f"figures/{filename}", dpi=300)
    print(f"Saved figure as: figures/{filename}")

# Load data into DataFrame
records = []
for method, paths in base_paths.items():
    backend = PlotBackend()
    for root_path in paths:
        # only keep non-empty dirs ending in _0 or _A
        all_dirs = [
            d for d in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, d))
               and (d.endswith('_0') or d.endswith('_A'))
               and os.listdir(os.path.join(root_path, d))
        ]
        all_dirs.sort()

        for folder in all_dirs:
            shape = folder.rsplit('_A', 1)[0]
            full_folder = os.path.join(root_path, folder)
            try:
                distances = backend.end_distance_return(full_folder)
                for val in distances:
                    records.append({
                        'method': method,
                        'shape': shape,
                        'error_cm': val * 100
                    })
            except Exception as e:
                print(f"Error in {full_folder}: {e}")
            finally:
                backend.reset()

    del backend
    gc.collect()

df = pd.DataFrame(records)

grouped = df.groupby('method')['error_cm'].apply(list)

arrays = [group for group in grouped]

f_stat, p_value = stats.f_oneway(*arrays)

print(f"ANOVA F-statistic: {f_stat:.4f}")
print(f"ANOVA p-value     : {p_value:.4e}")

if p_value < 0.05:
    print("=> There is a statistically significant difference between methods (p < 0.05).")
else:
    print("=> No statistically significant difference detected between methods.")

# 1. Tukey’s HSD–style test
tukey_pvals = sp.posthoc_tukey(
    df,
    val_col='error_cm',
    group_col='method',
)
print("Tukey-style pairwise p-value matrix:")
print(tukey_pvals)
tukey_pvals.to_csv('Tukey-style pairwise p-value matrix.csv', index=True)


# Shape group mapping
group_map = {
    'square': 'Seen', 'circle': 'Seen', 'triangle': 'Seen',
    'square25': 'Unseen Aspect Ratio', 'circle25': 'Unseen Aspect Ratio', 'triangle25': 'Unseen Aspect Ratio',
    'trapazoid': 'Unseen Shape', 'pentagon': 'Unseen Shape', 'square_circle': 'Unseen Shape'
}
df['shape_group'] = df['shape'].map(group_map)

# --- Plot 1: Boxplot of All Shapes ---
plot_title_1 = "Ending Distance Error for Each Representation Method\n 0 Degree Starting Orientation"
num_groups = len(base_paths)
fig, axs = plt.subplots(1, num_groups, figsize=(5 * num_groups, 6), sharey=True, constrained_layout=True)
axs = axs if num_groups > 1 else [axs]
fig.suptitle(plot_title_1)
fig.supxlabel("Shape")
fig.supylabel("Ending Distance Error (cm)")

for idx, method in enumerate(base_paths.keys()):
    ax = axs[idx]
    color = colorful_grayscale_friendly_palette[idx % len(colorful_grayscale_friendly_palette)]
    subset = df[df['method'] == method]
    data_group = [subset[subset['shape'] == shape]['error_cm'].values for shape in keys]

    ax.boxplot(data_group, positions=np.arange(len(keys)), widths=0.6, patch_artist=True,
               boxprops=dict(facecolor=color), medianprops=dict(color="black"), showfliers=False)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=45)
    ax.set_xlabel(method)
    ax.set_ylim(-0.25, 13.25)
    ax.grid(axis='y', linestyle='dashed', alpha=0.85)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax.axhline(1, color='red', linewidth=0.5, label='1 cm threshold')
    ax.grid(which='minor', axis='y', linestyle='dotted', alpha=0.55)


save_figure(fig, plot_title_1)
plt.show()

agg = (
    df
    .groupby(['method', 'shape'])['error_cm']
    .agg(['mean', 'std'])
    .round(2)
    .reset_index()
)

# Create a formatted "mean ± std" column
agg['mean_std'] = agg['mean'].astype(str) + ' ± ' + agg['std'].astype(str)

# Pivot so methods are rows, shapes are columns
table = agg.pivot(index='method', columns='shape', values='mean_std')

# (Optional) reorder columns to your original shape order
table = table[keys]

# Show it
print(table)
table.to_csv('shape_table_results.csv', index=True)

# --- Plot 2: Boxplot of Error Distributions by Method, Ordered by Mean ---
plot_title_2 = "Ending Distance Error Distribution by Representation Method\n0 Degree Starting Orientation"

# 1. Compute mean error per method and get methods sorted by that mean
mean_errors = df.groupby("method")["error_cm"].mean()
sorted_methods = mean_errors.sort_values().index.tolist()

# 2. Gather data in that order
data_by_method = [
    df[df['method'] == method]['error_cm'].values
    for method in sorted_methods
]

stats = pd.DataFrame({
    'method':    sorted_methods,
    'mean_err':  [np.mean(errors) for errors in data_by_method],
    'std_err':   [np.std(errors, ddof=0) for errors in data_by_method]
})

print(stats)

# 3. Pick colors in the same order
colors = [
    colorful_grayscale_friendly_palette[i % len(colorful_grayscale_friendly_palette)]
    for i, _ in enumerate(sorted_methods)
]

# 4. Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

bp = ax2.boxplot(
    data_by_method,
    positions=np.arange(len(sorted_methods)),
    widths=0.6,
    patch_artist=True,
    boxprops=dict(facecolor='white'),
    medianprops=dict(color="black"),
    showfliers=False
)

# 5. Color each box
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# 6. Format axes with the new ordering
ax2.set_xticks(np.arange(len(sorted_methods)))
ax2.set_xticklabels(sorted_methods, rotation=0, ha='right')
ax2.set_ylabel("Ending Distance Error (cm)")
ax2.set_title(plot_title_2)

# 7. Grids and minor ticks
ax2.grid(axis='y', linestyle='dashed', alpha=0.7)
ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
ax2.grid(which='minor', axis='y', linestyle='dotted', alpha=0.5)
ax2.axhline(1, color='red', linewidth=0.5, label='1 cm threshold')

plt.tight_layout()
save_figure(fig2, plot_title_2)
plt.show()

# --- Plot 3: Boxplot by Shape Group ---
plot_title_3 = "Ending Distance Error for Each Representation Method\n Grouped By Seen or Unseen\n 0 Degree Starting Orientation"
fig3, axs3 = plt.subplots(1, len(base_paths), figsize=(5 * len(base_paths), 6), sharey=True, constrained_layout=True)
axs3 = axs3 if len(base_paths) > 1 else [axs3]
fig3.suptitle(plot_title_3)
fig3.supylabel("Ending Distance Error (cm)")
group_order = ['Seen', 'Unseen Aspect Ratio', 'Unseen Shape']

for idx, method in enumerate(base_paths.keys()):
    ax = axs3[idx]
    color = colorful_grayscale_friendly_palette[idx % len(colorful_grayscale_friendly_palette)]
    subset = df[df['method'] == method]
    data_grouped = [subset[subset['shape_group'] == g]['error_cm'].values for g in group_order]

    ax.boxplot(data_grouped, positions=np.arange(len(group_order)), widths=0.6, patch_artist=True,
               boxprops=dict(facecolor=color), medianprops=dict(color="black"), showfliers=False)
    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels(group_order, rotation=45)
    ax.set_xlabel(method)
    ax.axhline(1, color='red', linewidth=0.5, label='1 cm threshold')
    ax.grid(axis='y', linestyle='dashed', alpha=0.85)

save_figure(fig3, plot_title_3)
plt.show()

# --- Plot 4: Scatter Plot of Mean by Shape Group ---
plot_title_4 = "Mean Ending Distance Error by Shape Group and Method\n 0 Degree Starting Orientation"
mean_grouped = df.groupby(['method', 'shape_group'])['error_cm'].mean().reset_index()
group_pos = {group: i for i, group in enumerate(group_order)}
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.set_title(plot_title_4)
ax4.set_ylabel("Mean Ending Distance Error (cm)")
ax4.set_xlabel("Shape Group")
ax4.set_xticks(range(len(group_order)))
ax4.set_xticklabels(group_order)
ax4.set_ylim(0, 2.38)

offset_map = {method: (i - len(base_paths)/2) * 0.1 for i, method in enumerate(base_paths.keys())}
marker_styles = ['o', 's', 'D', '^', 'v', 'P', '*']

for idx, method in enumerate(base_paths.keys()):
    subset = mean_grouped[mean_grouped['method'] == method]
    x_positions = [group_pos[g] + offset_map[method] for g in subset['shape_group']]
    y_values = subset['error_cm'].values
    marker = marker_styles[idx % len(marker_styles)]
    ax4.scatter(
        x_positions,
        y_values,
        label=method,
        marker=marker,
        color=colorful_grayscale_friendly_palette[idx % len(colorful_grayscale_friendly_palette)],
        s=150,
        edgecolors='black',
        linewidths=0.5
    )

ax4.grid(axis='y', linestyle='dashed', alpha=0.7)
ax4.legend(title="Method")
plt.tight_layout()
save_figure(fig4, plot_title_4)
plt.show()

