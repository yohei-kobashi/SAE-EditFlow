"""Plot exact vs intervention count k (final protocol, 04 S9v).

Reads runs/tables/ksweep_final.json (built by run_ksweep_final.sh) and
writes runs/tables/ksweep_final.png: net exact (solid) and true/random
(faint) for both directions over log2 k.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open("runs/tables/ksweep_final.json"))
ks = [r["k"] for r in rows]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for d, color, label in (("abl", "tab:blue", "ablation"),
                        ("enh", "tab:red", "enhancement")):
    ax.plot(ks, [r[f"{d}_net"] for r in rows], "-o", color=color,
            label=f"{label} net")
    ax.plot(ks, [r[f"{d}_true"] for r in rows], "--", color=color,
            alpha=0.35, label=f"{label} true")
    ax.plot(ks, [r[f"{d}_rand"] for r in rows], ":", color=color,
            alpha=0.35, label=f"{label} random")
ax.set_xscale("log", base=2)
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel("intervention count k (top-k latents of the feature spec)")
ax.set_ylabel("exact (eval-500)")
ax.set_title("Exact vs intervention count — adopted config (L12)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("runs/tables/ksweep_final.png", dpi=200)
print("wrote runs/tables/ksweep_final.png")

import os
if os.path.exists("runs/tables/ksweep_final_cat.json"):
    crows = json.load(open("runs/tables/ksweep_final_cat.json"))
    cats = ("morphology", "syntax", "semantics", "pragmatics")
    fig2, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True)
    for ax2, cat in zip(axes.flat, cats):
        for d, color, label in (("abl", "tab:blue", "ablation"),
                                ("enh", "tab:red", "enhancement")):
            ys = [r.get(f"{d}_{cat}_net") for r in crows]
            if any(y is not None for y in ys):
                ax2.plot(ks, ys, "-o", color=color, label=f"{label} net")
        n = crows[0].get(f"abl_{cat}_n", "?")
        ax2.set_title(f"{cat} (n≈{n}/dir)", fontsize=10)
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(ks)
        ax2.set_xticklabels([str(k) for k in ks], fontsize=7)
        ax2.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    for ax2 in axes[-1]:
        ax2.set_xlabel("intervention count k")
    for ax2 in axes[:, 0]:
        ax2.set_ylabel("exact net")
    fig2.suptitle("Exact net vs k by linguistic category (L12, adopted)")
    fig2.tight_layout()
    fig2.savefig("runs/tables/ksweep_final_cat.png", dpi=200)
    print("wrote runs/tables/ksweep_final_cat.png")
