"""E2 plots (spec E2.5): quadrant figure and per-level surprisal curves."""
import argparse
import json
from pathlib import Path

from src.common import io as io_mod

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

NEEDLE_COLORS = {
    "n1_weird_benign": "#d62728",
    "n2_contradiction": "#9467bd",
    "n3_novel": "#2ca02c",
    "n4_control": "#7f7f7f",
}


def _l2_id_by_l0id(levels_raw: dict) -> dict:
    out = {}
    for l2_unit in levels_raw["l2"]:
        for l0id in l2_unit["source_l0_ids"]:
            out[l0id] = l2_unit["id"]
    return out


def plot_quadrant(seed: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = json.loads((io_mod.RESULTS_DIR / f"e2_scores_seed{seed}.json").read_text())
    ground_truth = json.loads((DATA_DIR / f"ground_truth_seed{seed}.json").read_text())
    levels_raw = json.loads((DATA_DIR / f"levels_seed{seed}.json").read_text())

    l2_id_by_l0id = _l2_id_by_l0id(levels_raw)
    l0_by_id = {u["unit_id"]: u for u in scores["levels"]["l0"]}
    l2_scores_by_id = {u["unit_id"]: u for u in scores["levels"]["l2"]}

    all_x = [u["z_mean"] for u in scores["levels"]["l0"]]
    all_y = [l2_scores_by_id[l2_id_by_l0id[u["unit_id"]]]["z_mean"] for u in scores["levels"]["l0"]]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(all_x, all_y, color="lightgrey", s=25, label="ordinary clauses")

    for needle in ground_truth["needles"]:
        cid = needle["id"]
        ntype = needle["type"]
        z_l0 = l0_by_id[cid]["z_mean"]
        z_l2 = l2_scores_by_id[l2_id_by_l0id[cid]]["z_mean"]
        ax.scatter([z_l0], [z_l2], color=NEEDLE_COLORS[ntype], s=160, marker="*",
                   edgecolor="black", linewidth=0.5, label=ntype, zorder=5)

    ax.axvline(1.0, color="black", linestyle=":", alpha=0.5)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("z_L0 (clause-level surprisal)")
    ax.set_ylabel("z_L2 (section-level surprisal)")
    ax.set_title(f"E2 seed {seed}: cross-scale surprisal quadrant")
    ax.legend()
    fig.tight_layout()
    out_path = io_mod.RESULTS_DIR / f"e2_seed{seed}_quadrant.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_curves(seed: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = json.loads((io_mod.RESULTS_DIR / f"e2_scores_seed{seed}.json").read_text())
    ground_truth = json.loads((DATA_DIR / f"ground_truth_seed{seed}.json").read_text())
    levels_raw = json.loads((DATA_DIR / f"levels_seed{seed}.json").read_text())
    l2_id_by_l0id = _l2_id_by_l0id(levels_raw)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for ax, level_name in zip(axes, ["l0", "l2"]):
        units = scores["levels"][level_name]
        xs = [u["unit_id"] for u in units]
        ys = [u["z_mean"] for u in units]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1, color="#3b6ea5")
        ax.axhline(1.0, color="black", linestyle=":", alpha=0.5)
        ax.set_ylabel(f"z_mean ({level_name})")
        for needle in ground_truth["needles"]:
            cid = needle["id"]
            target_id = cid if level_name == "l0" else l2_id_by_l0id[cid]
            ax.axvline(target_id, color=NEEDLE_COLORS[needle["type"]], linestyle="--", alpha=0.8)
    axes[-1].set_xlabel("unit id (document order)")
    axes[0].set_title(f"E2 seed {seed}: per-level surprisal curves")
    fig.tight_layout()
    out_path = io_mod.RESULTS_DIR / f"e2_seed{seed}_curves.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    p1 = plot_quadrant(args.seed)
    p2 = plot_curves(args.seed)
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
