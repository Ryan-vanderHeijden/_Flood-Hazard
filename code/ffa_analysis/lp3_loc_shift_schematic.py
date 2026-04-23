"""
Schematic figure: LP3 loc shift under unit transformation Q → ω.

When ω = k·Q (constant k = ρgS/w per site), fitting LP3 to log10(ω) vs
log10(Q) produces distributions that differ only by a horizontal shift of
log10(k). The threshold also shifts by the same amount, so the AEP is
identical in both spaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearson3
from pathlib import Path

# ── illustrative LP3 parameters ──────────────────────────────────────────────
skew  = 0.35
loc_q = 0.0
scale = 0.40
log10_k = -1.7          # log10(ρgS/w) — arbitrary but visually clear shift
loc_w   = loc_q + log10_k

# threshold at ~10 % AEP on the Q distribution
aep     = 0.10
q_thr   = float(pearson3.isf(aep, skew, loc_q, scale))
w_thr   = q_thr + log10_k  # identical shift

# verify AEP is numerically the same
assert abs(pearson3.sf(w_thr, skew, loc_w, scale) - aep) < 1e-12

# ── x grid ────────────────────────────────────────────────────────────────────
x = np.linspace(-5.0, 2.0, 3000)
pdf_q = pearson3.pdf(x, skew, loc_q, scale)
pdf_w = pearson3.pdf(x, skew, loc_w, scale)

# ── colours ───────────────────────────────────────────────────────────────────
C_Q = "#1b6fe4"
C_W = "#e8331b"

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3.8))

# PDFs
ax.plot(x, pdf_q, color=C_Q, lw=2.0, label=r"$Q$  (discharge)")
ax.plot(x, pdf_w, color=C_W, lw=2.0, label=r"$\omega$  (specific stream power)")

# shaded AEP tails
ax.fill_between(x, pdf_q, where=(x >= q_thr), color=C_Q, alpha=0.22)
ax.fill_between(x, pdf_w, where=(x >= w_thr), color=C_W, alpha=0.22)

# threshold lines
ax.axvline(q_thr, color=C_Q, lw=1.2, ls="--", alpha=0.85)
ax.axvline(w_thr, color=C_W, lw=1.2, ls="--", alpha=0.85)

# threshold labels (top of dashed lines)
y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.1
pdf_max = max(pdf_q.max(), pdf_w.max())
ax.text(q_thr + 0.06, pdf_max * 0.97,
        r"$\log_{10}(Q^*)$", color=C_Q, fontsize=9, va="top")
ax.text(w_thr + 0.06, pdf_max * 0.97,
        r"$\log_{10}(\omega^*)$", color=C_W, fontsize=9, va="top")

# AEP labels inside the shaded tails
ax.text(q_thr + 0.15, pdf_max * 0.30,
        f"AEP = {aep:.0%}", color=C_Q, fontsize=8.5, ha="left", va="center")
ax.text(w_thr + 0.15, pdf_max * 0.30,
        f"AEP = {aep:.0%}", color=C_W, fontsize=8.5, ha="left", va="center")

# double-headed arrow between distribution peaks
y_arr = pdf_max * 0.72
ax.annotate(
    "", xy=(loc_w, y_arr), xytext=(loc_q, y_arr),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.4, mutation_scale=14),
)
ax.text(
    (loc_q + loc_w) / 2, y_arr + pdf_max * 0.06,
    r"$\log_{10}(k) = \log_{10}\!\left(\dfrac{\rho g S}{w}\right)$",
    ha="center", va="bottom", fontsize=9.5,
)

# axes
ax.set_xlim(-3.5, x[-1])
ax.set_ylim(0, pdf_max * 1.30)
ax.set_xlabel(r"$\log_{10}(\mathrm{value})$", fontsize=11)
ax.set_ylabel("Probability density", fontsize=11)
ax.legend(loc="upper right", frameon=False, fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = Path(__file__).parent / "lp3_loc_shift_schematic.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
