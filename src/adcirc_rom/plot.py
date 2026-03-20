import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits.axes_grid1.inset_locator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


df_fwd = pd.read_csv('directory to the test_result.csv generated from test.py')
df_fwd['Model'] = 'ANN'

df_true = pd.DataFrame({
    "True Values": df_fwd["True Values"].values,
    "Predicted Values": df_fwd["True Values"].values,
    "Model": 'Ground Truth'
})


df_combined = pd.concat([df_fwd, df_true], axis=0, ignore_index=True)
df_combined["Model"] = df_combined["Model"].astype("category")

def downsample_data(df, total_samples=20000, seed=42):
    if not pd.api.types.is_categorical_dtype(df["Model"]):
        raise ValueError("Column 'Model' must be categorical before sampling.")

    original_categories = df["Model"].cat.categories.tolist()
    num_categories = len(original_categories)
    samples_per_category = total_samples // num_categories

    df_sampled = df.groupby("Model", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), samples_per_category), random_state=seed)
    ).reset_index(drop=True)

    df_sampled["Model"] = pd.Categorical(df_sampled["Model"], categories=original_categories, ordered=True)
    return df_sampled

df_combined = downsample_data(df_combined, total_samples=20000)


df_combined["Residuals_log"] = np.log1p(abs(df_combined["True Values"] - df_combined["Predicted Values"]))
palette = ["C0", 'k'] 
hue_order = ['ANN', 'Ground Truth']  


g = sns.jointplot(
    data=df_combined,
    kind="scatter",
    x="Predicted Values",
    y="True Values",
    hue="Model",
    alpha=0.4,
    joint_kws=dict(s=10),
    palette=palette,
    hue_order=hue_order,
    height=9
)


g.fig.set_figwidth(12)
g.fig.set_figheight(6)


max_val = max(df_combined["True Values"].max(), df_combined["Predicted Values"].max())

sns.lineplot(
    x=[0, max_val], 
    y=[0, max_val], 
    color="#A9A9A9",  
    lw=1.5,           
    linestyle="--",   
    alpha=0.15,        
    ax=g.ax_joint
)

g.ax_joint.grid(which="major", axis="both", linestyle="--", alpha=0.15, zorder=0)
g.ax_joint.set(
    xlabel="Predicted Values",
    ylabel="True Values",
    ylim=(0, df_combined["True Values"].max()),
    xlim=(0, df_combined["Predicted Values"].max())
)


handles, labels = g.ax_joint.get_legend_handles_labels()
g.ax_joint.legend(handles=handles, labels=labels, fontsize=8, loc="upper left")

g.plot_marginals(sns.kdeplot, bw_adjust=0.2, lw=0, alpha=0.5, hue=hue_order, fill=False, palette=["C0", 'k'])
g.ax_marg_y.set_visible(False)
res_ax_bbox = (0.15, 0.9, 0.3, 0.1)  
res_ax = mpl_toolkits.axes_grid1.inset_locator.inset_axes(
    g.ax_joint,
    width="100%",
    height="70%",
    bbox_to_anchor=res_ax_bbox,
    bbox_transform=g.ax_joint.transAxes
)


sns.boxplot(
    data=df_combined[df_combined["Model"] == "ANN"],
    x="Residuals_log",
    palette=["C0"],
    linewidth=1,
    fliersize=1,
    orient="h",
    ax=res_ax
)

for spine in res_ax.spines.values():
    spine.set_edgecolor("#D3D3D3")
    spine.set_linewidth(1)

res_ax.set_ylabel("")
res_ax.set_yticklabels([])
res_ax.set_yticks([])
res_ax.tick_params(axis="x", labelsize=8)
res_ax.set_xlabel("Residuals (log)", fontsize=8)

plt.savefig("ANN.png", dpi=800, bbox_inches="tight")
plt.show()
