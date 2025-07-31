import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# --- Paper-like styling (style only; content unchanged) ---
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.usetex": False,

    # Typography
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    # Clean look
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.linewidth": 1.2,
})

# Number of shots
shots = [0, 3, 5, 10]

# Performance data for all models and sampling methods (unchanged)
data = {
    "Random": {
        "Gemini": {
            "Accuracy": [0.4583, 0.5417, 0.5, 0.625],
            "MCC": [-0.2085, 0.0917, 0.0, 0.2582],
            "ROC-AUC": [0.4514, 0.5972, 0.5174, 0.5729],
        },
        "LLaVA": {
            "Accuracy": [0.5833, 0.5, 0.5833, 0.6667],
            "MCC": [0.1768, 0.0, 0.1667, 0.3536],
            "ROC-AUC": [0.6042, 0.667, 0.6042, 0.691],
        },
        "Med-LLaVA": {
            "Accuracy": [0.4583, 0.4583, 0.4583, 0.375],
            "MCC": [-0.1026, -0.0917, -0.0917, -0.3078],
            "ROC-AUC": [0.5694, 0.3194, 0.3194, 0.4097],
        }
    },
    "kNN": {
        "Gemini": {
            "Accuracy": [0.4583, 0.5833, 0.625, 0.6667],
            "MCC": [-0.2085, 0.1690, 0.2509, 0.333],
            "ROC-AUC": [0.4514, 0.5972, 0.6146, 0.667],
        },
        "LLaVA": {
            "Accuracy": [0.5, 0.5, 0.5, 0.5417],
            "MCC": [0.0, 0.0, 0.0, 0.0836],
            "ROC-AUC": [0.4792, 0.6354, 0.3819, 0.4236],
        },
        "Med-LLaVA": {
            "Accuracy": [0.4167, 0.3333, 0.4167, 0.5417],
            "MCC": [-0.1690, -0.3536, -0.1925, 0.1026],
            "ROC-AUC": [0.4375, 0.2951, 0.2917, 0.5417],
        }
    }
}

# Colorblind-safe, paper-like palette
colors = {
    "Gemini":   "#0072B2",  # blue
    "LLaVA":    "#56B4E9",  # sky blue
    "Med-LLaVA":"#6E6E6E",  # neutral gray
}

# --- Plotting function (content identical; style tweaked) ---
def plot_combined_metrics(sampling, save_path=None):
    metrics = ["Accuracy", "MCC", "ROC-AUC"]

    # Wider, flatter figure; room on right for legend
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharex=True)
    fig.subplots_adjust(right=0.82, wspace=0.35)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Minimalist axes: hide top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        # Outward ticks; no grid
        ax.tick_params(direction="out", length=4, width=1, pad=2)

        # Series style: thicker lines, hollow circle markers
        for model in data[sampling]:
            ax.plot(
                shots,
                data[sampling][model][metric],
                marker="o",
                markersize=5.5,
                markerfacecolor="none",          # hollow
                markeredgecolor=colors[model],
                markeredgewidth=1.5,
                label=model,
                color=colors[model],
                linewidth=2.2,
                solid_capstyle="round"
            )

        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 10)
        ax.set_xticks(shots)
        ax.set_yticks(np.linspace(0, 1, 6))  # 0.0 to 1.0 in steps of 0.2

        if idx == 0:
            ax.set_ylabel("Performance")
        ax.set_xlabel("Number of Shots")

    # Shared legend in floating box on the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Models",
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        frameon=True, fancybox=True, borderaxespad=0.0,
        handlelength=2.2, handletextpad=0.8, borderpad=0.8
    )

    fig.suptitle(f"Few-Shot Performance ({sampling} Sampling)", fontsize=13)
    fig.tight_layout(rect=[0.02, 0.02, 0.80, 0.92])

    # Save as PNG and PDF (unchanged)
    if save_path:
        fig.savefig(f"{save_path}_{sampling.lower()}.png")
        fig.savefig(f"{save_path}_{sampling.lower()}.pdf")

    plt.show()

# Generate both plots and save (unchanged)
plot_combined_metrics("Random", save_path="few_shot_metrics")
plot_combined_metrics("kNN", save_path="few_shot_metrics")