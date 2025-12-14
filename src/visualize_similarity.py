"""Calculate pairwise text similarity and generate heatmap visualization with multi-model support."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from cal_similarity import cosine_similarity
from embedding import get_embedding

# Supported embedding models
EMBEDDING_MODELS = [
    "data/model/Qwen--Qwen3-Embedding-0.6B",
    # "data/output/lca-qwen3-st-finetuned",
    "data/model/BAAI--bge-m3",
    # "data/output/lca-bge-m3-finetuned"
]

# Scientific styling defaults
SCIENTIFIC_PALETTE = [
    "#081831",  # deep navy
    "#123865",
    "#1f5c8c",
    "#2e82a7",
    "#48a4b6",
    "#6bc7c5",
    "#9fe5d1",  # light teal
]
CMAP = LinearSegmentedColormap.from_list("deep_teal_scientific", SCIENTIFIC_PALETTE, N=256)
ANNOTATION_THRESHOLD = 0.65
ANNOTATION_FONTSIZE = 9
FIG_DPI = 300

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "font.size": 10,
        "axes.grid": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def _annotate_heatmap(ax: plt.Axes, matrix: np.ndarray) -> None:
    """Add similarity values inside each heatmap cell."""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "#fdfdfd" if value >= ANNOTATION_THRESHOLD else "#1b1b1b"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=ANNOTATION_FONTSIZE,
            )


def _style_heatmap_axis(
    ax: plt.Axes,
    tick_labels: list[str],
    axis_label: str = "Text Index",
    rotation: int = 35,
    show_y_label: bool = True,
) -> None:
    """Unify axis formatting for the heatmaps."""
    positions = np.arange(len(tick_labels))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(tick_labels, rotation=rotation, ha="right")
    ax.set_yticklabels(tick_labels)

    ax.tick_params(length=0)
    ax.grid(False)
    ax.set_xlabel(axis_label)
    if show_y_label:
        ax.set_ylabel(axis_label)
    else:
        ax.set_ylabel("")

    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


def compute_similarity_matrix(
    texts: list[str], model: str = "Qwen/Qwen3-Embedding-0.6B"
) -> np.ndarray:
    """计算文本列表两两之间的相似度矩阵。

    Args:
        texts: 文本列表
        model: 嵌入模型名称

    Returns:
        相似度矩阵，shape 为 (n, n)
    """
    embeddings = []
    for text in texts:
        emb = get_embedding(text, model=model)
        if emb is None:
            raise ValueError(f"获取文本嵌入失败: {text[:50]}...")
        embeddings.append(emb)

    n = len(texts)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

    return similarity_matrix


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: list[str],
    output_path: str = "similarity_heatmap.png",
    title: str = "Text Similarity Heatmap",
) -> None:
    """Plot similarity heatmap for a single model.

    Args:
        similarity_matrix: Similarity matrix
        labels: Text labels for axis display
        output_path: Output image path
        title: Chart title
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=FIG_DPI)
    fig.patch.set_facecolor("white")

    im = ax.imshow(
        similarity_matrix,
        cmap=CMAP,
        aspect="equal",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    short_labels = [f"Text {i+1}" for i in range(len(labels))]
    _style_heatmap_axis(ax, short_labels, axis_label="Text Index")
    _annotate_heatmap(ax, similarity_matrix)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout(pad=1.2)
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Heatmap saved to: {output_path}")


def plot_multi_model_comparison(
    texts: list[str],
    models: list[str] = EMBEDDING_MODELS,
    output_path: str = "multi_model_comparison.png",
) -> dict[str, np.ndarray]:
    """Compare similarity results across multiple models with side-by-side heatmaps.

    Args:
        texts: List of texts
        models: List of model names
        output_path: Output image path

    Returns:
        Dictionary of similarity matrices for each model
    """
    results = {}
    n_models = len(models)
    n_texts = len(texts)

    # Compute all similarity matrices first
    for model in models:
        print(f"\nProcessing model: {model}")
        similarity_matrix = compute_similarity_matrix(texts, model=model)
        results[model] = similarity_matrix

    # Create figure with GridSpec for better colorbar placement
    fig = plt.figure(figsize=(5.5 * n_models + 1, 5.5), dpi=FIG_DPI)
    gs = GridSpec(1, n_models + 1, width_ratios=[1] * n_models + [0.05], wspace=0.3)

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_models)]

    for idx, model in enumerate(models):
        similarity_matrix = results[model]
        ax = axes[idx]

        im = ax.imshow(
            similarity_matrix,
            cmap=CMAP,
            aspect="equal",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )

        short_labels = [f"T{i+1}" for i in range(n_texts)]
        _style_heatmap_axis(
            ax,
            short_labels,
            axis_label="Text Index",
            rotation=0,
            show_y_label=idx == 0,
        )
        _annotate_heatmap(ax, similarity_matrix)

        model_short = model.split("/")[-1]
        ax.set_title(f"{model_short}", fontsize=11, fontweight="bold", pad=10)

    # Add colorbar on the right side
    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Cosine Similarity", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    fig.suptitle(
        "Multi-Model Text Similarity Comparison",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(pad=1.2)
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nComparison chart saved to: {output_path}")

    return results


if __name__ == "__main__":
    # Example text list
#     texts = [
#         "steel production global",
#         """# steel production, electric, low-alloyed

# **UUID:** `76a406a4-7470-3688-b4d7-7e56887f9804`  
# **Dataset ID:** `d9c9894c-51c3-3cc4-9e85-1109341dcbfc`  
# **Flow Name:** steel, low-alloyed  
# **Unit:** kg  
# **Version:** 3.10.0  
# **Location Name:** Canada,Québec  
# **Location Code:** CA-QC  
# **Start Date:** 2010  
# **End Date:** 2023  
# **GWP100:** 0.8420535868100618 kg CO2 eq  
# **System Model:** Cutoff, U  
# **Activity Type:** ORDINARY_TRANSFORMING_ACTIVITY  
# **GWP Method:** IPCC 2021  
# **Industry Name:** Steels  

# ## Dataset Description

# This dataset represents the manufacturing of large alloyed steel slabs (2 to 25 tonnes). Iron scrap is used as the  main iron bearing input. Iron based alloys are added to acheived desired steel alloy composition. This dataset is based on data from the foundry department of an integrated forge.

# ## Technology Comment

# Quebec technology. An average for different alloys is represented.

# ## Location Description

# Dataset is representative of the Québec region.

# ## Activities Start

# - Activity starts with the arrival of iron scrap at the electric arc furnace

# ## Activities End

# - This activity ends with the casting in ingot molds and cooling of alloyed steel slabs. The dataset includes: the foundry processes (including scrap metal input, EAF melting, ladle oven, alloying, degassing, and ingot casting)""",
#         """# chromium steel pipe production

# **UUID:** `8c35ec3a-5a6d-390f-8ef9-d1ce74f6ded4`  
# **Dataset ID:** `2cb40347-e798-3610-8f14-21f86df7cda3`  
# **Flow Name:** chromium steel pipe  
# **Unit:** kg  
# **Version:** 3.10.0  
# **Location Name:** Global  
# **Location Code:** GLO  
# **Start Date:** 2007  
# **End Date:** 2023  
# **GWP100:** 5.513941735370028 kg CO2 eq  
# **System Model:** Cutoff, U  
# **Activity Type:** ORDINARY_TRANSFORMING_ACTIVITY  
# **GWP Method:** IPCC 2021  
# **Industry Name:** Steels  

# ## Dataset Description

# This is a proxy dataset for the production of chromium steel pipes.

# ## Activities Start

# - This dataset is a first proxy for the production of chromium steel pipes, but has to be revised by specialists.

# ## Activities End

# - This dataset doesn't include any specifications, but only produced chromium steel and it's processing as proxy inputs. Besides these, it doesn't include any energy or material use.""",
#         "steel production",
#         "chromium steel",
#     ]
    texts = [
        "steel",
        "steel made in China",
        "treatment of waste steel",
        "steel production",
        "steel and iron",
    ]
    # Multi-model comparison
    print("=" * 50)
    print("Multi-Model Similarity Comparison")
    print("=" * 50)
    results = plot_multi_model_comparison(texts, models=EMBEDDING_MODELS)

    # Print similarity matrices for each model
    for model, matrix in results.items():
        print(f"\n{model} similarity matrix:")
        print(matrix.round(3))
