"""
Title-slide background: the moderate-threshold site map, blurred into a color field.

Inspired by Kupla's "Mirror Universe" cover: a saturated, out-of-focus inset
offset inside a flat frame. Every QC site is splatted at its Albers map
position in its return-period class color (make_rp_maps.py COLORS). Each
class's site density is blurred separately and the locally dominant class
takes the pixel through a sharpened (power) weighting, so the field shows
distinct saturated blobs instead of averaging to mud; empty regions fall back
to a teal base. The inset sits in the
lower part of the slide so the title text stays on the plain frame.

Usage (from the repo root, hydro env):
  python presentations/make_title_bg.py

Writes presentations/figures/title_bg.png (1600 × 1000, the deck's slide size).
"""

import sys
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "ffa_analysis"))
from make_rp_maps import COLORS, NORM, PROJ, load_sites, split_sites  # noqa: E402

W, H = 1600, 1000
# FRAME = to_rgb(COLORS[0])               # pale yellow, the cover's lime frame
FRAME = to_rgb('#ffffff')
BASE = np.array(to_rgb(COLORS[2]))      # teal where no sites are near
INSET = (0.07, 0.52, 0.96, 0.94)        # x0, y0, x1, y1 as fractions of the slide
LEVEL = "moderate"
SIGMA = 34                              # blur radius in inset pixels
SPLAT = 9                               # half-width of each site's square, px
SHARPEN = 4                             # power on class densities: higher = crisper blobs
ZOOM = 1.35                             # > 1 crops into CONUS so blobs read larger
OUT = Path(__file__).parent / "figures" / "title_bg.png"


def main() -> None:
    pts, bnd = split_sites(load_sites(False), LEVEL)
    sites = np.concatenate([pts[["longitude", "latitude", "v"]].to_numpy(),
                            bnd[["longitude", "latitude", "v"]].to_numpy()])
    xy = PROJ.transform_points(ccrs.PlateCarree(), sites[:, 0], sites[:, 1])[:, :2]
    cls = np.asarray(NORM(sites[:, 2]), dtype=int)

    x0, y0, x1, y1 = (int(INSET[0] * W), int(INSET[1] * H),
                      int(INSET[2] * W), int(INSET[3] * H))
    iw, ih = x1 - x0, y1 - y0
    # Fit CONUS to the inset, cropped a little so the coasts bleed off the edge.
    (mx0, my0), (mx1, my1) = xy.min(0), xy.max(0)
    s = max(iw / (mx1 - mx0), ih / (my1 - my0)) * ZOOM
    px = ((xy[:, 0] - (mx0 + mx1) / 2) * s + iw / 2).astype(int)
    py = (ih / 2 - (xy[:, 1] - (my0 + my1) / 2) * s).astype(int)

    dens = np.zeros((len(COLORS), ih, iw))
    for x, y, k in zip(px, py, cls):
        if 0 <= x < iw and 0 <= y < ih:
            dens[k, max(y - SPLAT, 0):y + SPLAT, max(x - SPLAT, 0):x + SPLAT] += 1
    dens = np.stack([gaussian_filter(d, SIGMA) for d in dens])

    # Locally dominant class wins, softly: weights ∝ density^SHARPEN.
    w = dens ** SHARPEN
    wsum = w.sum(0)
    pal = np.array([to_rgb(c) for c in COLORS])
    field = np.einsum("kyx,kc->yxc", w, pal) / np.maximum(wsum, 1e-30)[..., None]
    total = dens.sum(0)
    cover = np.clip(total / np.quantile(total, 0.5), 0, 1)[..., None]
    inset = cover * field + (1 - cover) * BASE
    inset = gaussian_filter(inset, sigma=(SIGMA * 0.5, SIGMA * 0.5, 0))   # soften the seams

    img = np.empty((H, W, 3))
    img[:] = FRAME
    img[y0:y1, x0:x1] = inset
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(OUT)
    print(f"{len(sites):,} sites → {OUT}")


if __name__ == "__main__":
    main()
