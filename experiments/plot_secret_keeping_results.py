import os
from pathlib import Path
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Import sibling plotting utilities (run from repo root: `python experiments/plotSecretKeepingResults.py`)
import plot_taboo_eval_results as taboo
import plot_gender_eval_results as gender
import plot_ssc_results as ssc


# ---------- Configuration (edit as needed) ----------

# Data locations
TABOO_JSON_DIR = "experiments/taboo_eval_results/gemma-2-9b-it_open_ended_all_direct"
GENDER_JSON_DIR = "experiments/gender_results/gemma-2-9b-it_open_ended_all_direct"
SSC_JSON_DIR = "experiments/ssc_eval_results/Llama-3_3-70B-Instruct_open_ended_all_direct"

# Sequence/token-level choice per task (must match how you evaluated)
TABOO_SEQUENCE = False
GENDER_SEQUENCE = True
SSC_SEQUENCE = True

# Highlight keywords identifying the single LoRA/bar to feature per task
TABOO_HIGHLIGHT = "latentqa_cls_past_lens"
GENDER_HIGHLIGHT = "latentqa_cls_past_lens"
SSC_HIGHLIGHT = "act_cls_latentqa"

# Extra baseline bars to show in the second figure (per task)
TABOO_EXTRAS = [
        {"label": "Best Interp Method (SAEs)", "value": 0.0413, "error": 0.0038},
        {"label": "Best Black Box Method (Prefill)", "value": 0.0717, "error": 0.0055},
    ]

GENDER_EXTRAS = [
    {"label": "Best Interp Method", "value": 0.8695, "error": 0.0094},
    {"label": "Best Black Box Method", "value": 0.9765, "error": 0.0068},
]

SSC_EXTRAS = [
    {"label": "Best Interp Method", "value": 0.5224, "error": 0.0077},
    {"label": "Best Black Box Method", "value": 0.9676, "error": 0.0004},
]

# Output
IMAGE_FOLDER = "images"
PAPER_IMAGE_FOLDER = f"{IMAGE_FOLDER}/secretkeeping"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(PAPER_IMAGE_FOLDER, exist_ok=True)


# ---------- Small helpers ----------
INTERP_BAR_COLOR = "#FDB813"   # used for Our Method (interp) and Best Interp Method
BLACKBOX_BAR_COLOR = "#1f77b4"  # used for Best Black Box Method


CHOSEN_TABOO_PROMPT = "What is the secret word in this text?"

def ci95(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = np.mean(values)
    std_err = np.std(values, ddof=1) / np.sqrt(n)
    return 1.96 * std_err


def _collect_stats(results_by_lora: dict[str, list[float]], highlight_keyword: str):
    lora_names = []
    means = []
    cis = []

    for lora_path, accs in results_by_lora.items():
        name = lora_path.split("/")[-1]
        lora_names.append(name)
        means.append(sum(accs) / len(accs))
        cis.append(ci95(accs))

    matches = [i for i, name in enumerate(lora_names) if highlight_keyword in name]
    assert len(matches) == 1, f"Keyword '{highlight_keyword}' matched {len(matches)}: {[lora_names[i] for i in matches]}"
    m = matches[0]
    order = [m] + [i for i in range(len(lora_names)) if i != m]

    lora_names = [lora_names[i] for i in order]
    means = [means[i] for i in order]
    cis = [cis[i] for i in order]
    return lora_names, means, cis


def _legend_labels(names: list[str], label_map: dict[str, str] | None) -> list[str]:
    if label_map is None:
        return names
    out = []
    for n in names:
        if n in label_map and label_map[n]:
            out.append(label_map[n])
        else:
            out.append(n)
    return out


def _style_highlight(bar, color=INTERP_BAR_COLOR, hatch="////"):
    bar.set_color(color)
    bar.set_hatch(hatch)
    bar.set_edgecolor("black")
    bar.set_linewidth(2.0)


def _plot_results_panel(
    ax,
    names: list[str],
    labels: list[str],
    means: list[float],
    cis: list[float],
    title: str,
    palette: dict[str, tuple],
):
    colors = [palette[label] for label in labels]
    bars = ax.bar(range(len(names)), means, color=colors, yerr=cis, capsize=5, error_kw={"linewidth": 2})
    # Keep palette color; only add hatch and stroke for the highlighted (index 0)
    _style_highlight(bars[0], color=bars[0].get_facecolor())

    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Average Accuracy", fontsize=12)

    for bar, mean, err in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + err + 0.02, f"{mean:.3f}", ha="center", va="bottom", fontsize=10)

    # Legend handled at figure-level for shared colors


def _plot_selected_with_extras_panel(ax, selected_name: str, selected_mean: float, selected_ci: float, extras: list[dict], title: str, label_map: dict[str, str] | None):
    labels = [selected_name] + [e["label"] for e in extras]
    values = [selected_mean] + [e["value"] for e in extras]
    errors = [selected_ci] + [e["error"] for e in extras]

    # Color mapping: our method + best interp share interp color; black box has its own
    colormap = [INTERP_BAR_COLOR, INTERP_BAR_COLOR, BLACKBOX_BAR_COLOR]
    bars = ax.bar(range(len(labels)), values, color=colormap[: len(labels)], yerr=errors, capsize=5, error_kw={"linewidth": 2})
    _style_highlight(bars[0])

    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Average Accuracy", fontsize=12)

    for bar, v, err in zip(bars, values, errors):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + err + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    # Legend handled at figure-level; no per-axes legend here


async def main():
    # Set sequence-level configuration to match each task's eval
    taboo.SEQUENCE = TABOO_SEQUENCE
    gender.SEQUENCE = GENDER_SEQUENCE
    ssc.SEQUENCE = SSC_SEQUENCE

    # Load raw results
    taboo_results, _ = taboo.load_results(TABOO_JSON_DIR, required_verbalizer_prompt=CHOSEN_TABOO_PROMPT)
    gender_results, _ = gender.load_results(GENDER_JSON_DIR)
    ssc_results, _ = await ssc.load_results(SSC_JSON_DIR)

    # Pull pretty label maps if present
    taboo_labels = getattr(taboo, "CUSTOM_LABELS", None)
    gender_labels = getattr(gender, "CUSTOM_LABELS", None)
    ssc_labels = getattr(ssc, "CUSTOM_LABELS", None)

    # ----- Figure 1: Overall results (3 panels) with shared colors & legend -----
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    t_names, t_means, t_cis = _collect_stats(taboo_results, TABOO_HIGHLIGHT)
    g_names, g_means, g_cis = _collect_stats(gender_results, GENDER_HIGHLIGHT)
    s_names, s_means, s_cis = _collect_stats(ssc_results, SSC_HIGHLIGHT)

    # Resolve human-readable labels for consistent coloring across panels
    t_labels = _legend_labels(t_names, taboo_labels)
    g_labels = _legend_labels(g_names, gender_labels)
    s_labels = _legend_labels(s_names, ssc_labels)

    # Build a shared palette keyed by label (robust to different bar counts)
    unique_labels = sorted(set(t_labels) | set(g_labels) | set(s_labels))
    tab20 = plt.get_cmap("tab20").colors
    shared_palette = {lab: tab20[i % len(tab20)] for i, lab in enumerate(unique_labels)}
    shared_palette["Past Lens + LatentQA + Classification"] = INTERP_BAR_COLOR

    _plot_results_panel(axes1[0], t_names, t_labels, t_means, t_cis, title="Taboo", palette=shared_palette)
    _plot_results_panel(axes1[1], g_names, g_labels, g_means, g_cis, title="Gender", palette=shared_palette)
    _plot_results_panel(axes1[2], s_names, s_labels, s_means, s_cis, title="Secret Keeping", palette=shared_palette)

    # Single shared legend mapping label -> color
    handles = [Patch(facecolor=shared_palette[lab], edgecolor="black", label=lab) for lab in unique_labels]
    fig1.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False, fontsize=10)

    fig1.suptitle("Results by Dataset Mix", fontsize=15, y=1.02)
    plt.tight_layout()
    out1 = f"{PAPER_IMAGE_FOLDER}/secret_keeping_combined_results_dataset_comparison.pdf"
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    print(f"Saved: {out1}")

    # ----- Figure 2: Selected LoRA vs Extras (3 panels) -----
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Taboo panel
    t_selected_name = t_names[0]
    t_selected_mean = t_means[0]
    t_selected_ci = t_cis[0]
    _plot_selected_with_extras_panel(
        axes2[0], t_selected_name, t_selected_mean, t_selected_ci, TABOO_EXTRAS, title="Taboo", label_map=taboo_labels
    )

    # Gender panel
    g_selected_name = g_names[0]
    g_selected_mean = g_means[0]
    g_selected_ci = g_cis[0]
    _plot_selected_with_extras_panel(
        axes2[1], g_selected_name, g_selected_mean, g_selected_ci, GENDER_EXTRAS, title="Gender", label_map=gender_labels
    )

    # SSC panel
    s_selected_name = s_names[0]
    s_selected_mean = s_means[0]
    s_selected_ci = s_cis[0]
    _plot_selected_with_extras_panel(
        axes2[2], s_selected_name, s_selected_mean, s_selected_ci, SSC_EXTRAS, title="Secret Keeping", label_map=ssc_labels
    )

    fig2.suptitle("Talkative Probe vs. Baselines", fontsize=15, y=1.02)

    # Single shared legend for the 3 panels
    our_method_patch = Patch(facecolor=INTERP_BAR_COLOR, edgecolor="black", hatch="////", label="Our Method (Interp)")
    best_interp_patch = Patch(facecolor=INTERP_BAR_COLOR, edgecolor="black", label="Best Interp Method")
    best_blackbox_patch = Patch(facecolor=BLACKBOX_BAR_COLOR, edgecolor="black", label="Best Black Box Method")
    fig2.legend(
        handles=[our_method_patch, best_interp_patch, best_blackbox_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    plt.tight_layout()
    out2 = f"{PAPER_IMAGE_FOLDER}/secret_keeping_combined_selected_with_baselines.pdf"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    asyncio.run(main())
