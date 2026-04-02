"""
Analisi della distribuzione background vs foreground nel dataset EGTEA.
Esegui con: python explore_dataset.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from data.dataset import EGTEADataset, load_action_labels
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
LMDB_RGB     = r"D:\egtea\TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd"
ANNOTATION_DIR = r".\action_annotation"
SPLIT_FILE   = "train_split1.txt"
SEQ_LEN      = 256
N_SAMPLES    = 300   # quanti clip analizzare (tutti = len(ds), ma è lento)
# ──────────────────────────────────────────────────────────────────────────────

def load_class_names(annotation_dir: str) -> dict[int, str]:
    cls_path = Path(annotation_dir) / "raw_annotations" / "cls_label_index.csv"
    mapping = {}
    with open(cls_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) >= 2:
                try:
                    mapping[int(parts[0])] = parts[1]
                except ValueError:
                    continue
    return mapping


def main():
    print("Caricamento dataset...")
    ds = EGTEADataset(
        lmdb_rgb_path  = LMDB_RGB,
        annotation_dir = ANNOTATION_DIR,
        split_file     = SPLIT_FILE,
        seq_len        = SEQ_LEN,
    )
    class_names = load_class_names(ANNOTATION_DIR)
    n = min(N_SAMPLES, len(ds))
    print(f"Analisi su {n} clip...")

    bg_pcts      = []
    fg_pcts      = []
    class_counts = Counter()

    for i in range(n):
        _, labels = ds[i]
        labels_np = labels.numpy()

        bg = (labels_np == 0).sum()
        fg = (labels_np != 0).sum()
        bg_pcts.append(bg / SEQ_LEN)
        fg_pcts.append(fg / SEQ_LEN)

        for c in labels_np:
            if c != 0:
                class_counts[int(c)] += 1

    bg_pcts = np.array(bg_pcts)
    fg_pcts = np.array(fg_pcts)

    # ── Stampa statistiche ────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Distribuzione Background vs Foreground")
    print(f"{'='*50}")
    print(f"  Media background per clip: {bg_pcts.mean():.1%}")
    print(f"  Media foreground per clip: {fg_pcts.mean():.1%}")
    print(f"  Min background: {bg_pcts.min():.1%}  Max: {bg_pcts.max():.1%}")
    print(f"\n  Top 10 classi più frequenti:")
    for cls_id, count in class_counts.most_common(10):
        name = class_names.get(cls_id, f"class_{cls_id}")
        print(f"    [{cls_id:3d}] {name:<35} {count:5d} frame")
    print(f"{'='*50}\n")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("EGTEA Gaze+ — Distribuzione Background vs Foreground", fontsize=14)

    # 1. Pie chart globale
    ax = axes[0]
    total_bg = bg_pcts.sum()
    total_fg = fg_pcts.sum()
    ax.pie(
        [total_bg, total_fg],
        labels   = [f"Background\n{total_bg/(total_bg+total_fg):.1%}",
                    f"Foreground\n{total_fg/(total_bg+total_fg):.1%}"],
        colors   = ["#B0C4DE", "#E07B54"],
        startangle = 90,
        wedgeprops = {"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("Distribuzione globale")

    # 2. Istogramma % background per clip
    ax = axes[1]
    ax.hist(bg_pcts * 100, bins=20, color="#B0C4DE", edgecolor="white", linewidth=0.8)
    ax.axvline(bg_pcts.mean() * 100, color="#333", linestyle="--", linewidth=1.5,
               label=f"Media {bg_pcts.mean():.1%}")
    ax.set_xlabel("% Background per clip")
    ax.set_ylabel("Numero di clip")
    ax.set_title("Distribuzione background per clip")
    ax.legend()

    # 3. Top 15 classi foreground
    ax = axes[2]
    top15 = class_counts.most_common(15)
    cls_ids  = [class_names.get(c, str(c))[:20] for c, _ in top15]
    counts   = [cnt for _, cnt in top15]
    bars = ax.barh(cls_ids[::-1], counts[::-1], color="#E07B54", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Frame totali")
    ax.set_title("Top 15 classi più frequenti")
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    out_path = "dataset_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato in: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()