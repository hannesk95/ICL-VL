import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Clean publication-ready style
mpl.rcParams.update({
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.usetex": False
})

# Number of shots
shots = [0, 3, 5, 10]

# Performance data for all models and sampling methods
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

# Colors per model
colors = {
    "Gemini": "#1f77b4",
    "LLaVA": "#2ca02c",
    "Med-LLaVA": "#d62728"
}

# Plotting function
# Plotting function
def plot_combined_metrics(sampling, save_path=None):
    metrics = ["Accuracy", "MCC", "ROC-AUC"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 6), sharex=True)  # Taller, narrower

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for model in data[sampling]:
            ax.plot(
                shots,
                data[sampling][model][metric],
                marker='o',
                label=model,
                color=colors[model],
                linewidth=2
            )
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 10)
        ax.set_xticks(shots)
        ax.grid(True, linestyle='--', alpha=0.5)
        # ax.set_aspect(2.5)  # ← REMOVE or reduce this
        if idx == 0:
            ax.set_ylabel("Performance")
        ax.set_xlabel("Number of Shots")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(f"Few-Shot Performance ({sampling} Sampling)", fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Save as PNG and PDF
    if save_path:
        fig.savefig(f"{save_path}_{sampling.lower()}.png")
        fig.savefig(f"{save_path}_{sampling.lower()}.pdf")

    plt.show()

# Generate both plots and save
plot_combined_metrics("Random", save_path="few_shot_metrics")
plot_combined_metrics("kNN", save_path="few_shot_metrics")