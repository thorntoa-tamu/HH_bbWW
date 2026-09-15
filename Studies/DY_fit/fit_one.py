#!/usr/bin/env python3
"""Fit ExpPolyLogY2D (N × pdf) on an adaptive cell grid.

Input: grid JSON from ``grid_merge.py``.  ROOT Minuit2 on a Python χ² with
Gauss–Legendre cell integrals of the continuous density:

  s(x,y) = exp(poly on xn, log-mapped yn)
  pdf    = s / ∬_window s
  μ_cell = N · ∬_cell pdf

**Empty meta-cluster (default):** all zero-yield cells → one χ² term
(Garwood σ).  Disable with ``--no-empty-meta``.

**Accepted uncertainty thr** (default max 0.1)::

  σ_eff² = σ_stat² + (thr · |content|)²

**Automatic selection (default):** if ``--degree`` is omitted, scan
``degree = 1, 2, …, --max-degree``.  For each degree, scan
``thr = 0, thr_step, …, thr`` (1% steps by default) and stop at the first
configuration with p≥0.05 and a usable covariance.  Returns fitted
parameters plus the chosen ``degree`` and ``thr_min``.

If ``--degree N`` is set, thr is fixed to ``--thr`` (use ``--auto-thr`` to
still scan thr at that degree).

**Presets** (``--preset``, default ``v1``): ``v1`` reproduces the published
fits bit-for-bit (window-GL normalisation, Neyman χ², cold starts).  ``v2``
switches on ``--norm tiling`` (fine-bin tiling normalisation → ∑μ == N),
``--stat poisson_eff`` (Bohm–Zech scaled Poisson −2lnL with effective MC
weights) and ``--warm-start`` unless those flags are given explicitly.
``--y-log-eps`` sets the HME log-map offset (default 1e-3).

**Acceptance GoF for poisson_eff:** a Pearson χ² on deterministically
coarsened *super-cells* built from the observations before fitting
(``--gof-band`` fine bins per DNN band, ``--gof-neff-min`` effective events
per super-cell; see ``build_gof_supercells``).  It is reported in
``chi2``/``ndf``/``p_value`` (PASS ⇔ converged ∧ covariance ∧ p ≥ 0.05); the
−2lnL minimum is kept in ``objective`` and the calibrated deviance in
``gof_deviance`` (diagnostic only).

Example::

  # Auto degree + thr ≤ 10%
  python3 -u fit_one.py --grid grid_m500.json --output m500.json

  # Fixed degree, thr = 10%
  python3 -u fit_one.py --grid grid_m500.json --output m500.json --degree 4 --thr 0.1

  # v2 preset (tiling + poisson_eff + warm start)
  python3 -u fit_one.py --grid grid_m500.json --output m500_v2.json --preset v2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import ROOT

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from core import ContinuousDensity2D, ExpPolyLogY2D  # noqa: E402

P_MIN = 0.05

# ROOT TH1 kPoisson alpha (Garwood)
_POISSON_ALPHA = 1.0 - 0.682689492137085897
_SIGMA_N0 = None

STATS = ("chi2_neyman", "poisson_eff")
NORMS = ("window_gl", "tiling")
PRESETS = ("v1", "v2")


class CellObs(object):
    """One χ² observation: a single rectangle or a multi-rect meta-cluster.

    For a normal cell, ``rects`` is None and (xlo,xhi,ylo,yhi) define the
    integral domain.  For the empty meta-cluster, ``rects`` is a list of
    (xlo,xhi,ylo,yhi) (not necessarily connected); content is the summed
    yield (0) and μ is the sum of continuous integrals over every sub-rect.
    """

    __slots__ = (
        "id",
        "xlo",
        "xhi",
        "ylo",
        "yhi",
        "content",
        "error",
        "error_stat",
        "x",
        "y",
        "rects",
        "n_subrects",
        "is_empty_meta",
        "w_eff",
        "n_eff",
        "w_class",
    )

    def __init__(
        self,
        cell_id,
        xlo,
        xhi,
        ylo,
        yhi,
        content,
        error,
        rects=None,
        is_empty_meta=False,
    ):
        self.id = int(cell_id)
        self.content = float(content)
        # error_stat = pure statistical (hist / Garwood); error = χ² sigma
        # after optional acceptable_unc_thr (set via apply_acceptable_thr).
        self.error_stat = float(error)
        self.error = float(error)
        self.is_empty_meta = bool(is_empty_meta)
        # Effective MC weight (poisson_eff stat; set by annotate_weights)
        self.w_eff = None
        self.n_eff = None
        self.w_class = None
        if rects is not None:
            self.rects = [
                (float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in rects
            ]
            self.n_subrects = len(self.rects)
            # Bbox of the union (diagnostic only; integral uses rects)
            self.xlo = min(r[0] for r in self.rects)
            self.xhi = max(r[1] for r in self.rects)
            self.ylo = min(r[2] for r in self.rects)
            self.yhi = max(r[3] for r in self.rects)
        else:
            self.rects = None
            self.n_subrects = 1
            self.xlo = float(xlo)
            self.xhi = float(xhi)
            self.ylo = float(ylo)
            self.yhi = float(yhi)
        self.x = 0.5 * (self.xlo + self.xhi)
        self.y = 0.5 * (self.ylo + self.yhi)


def _garwood_poisson_sigma0():
    return 0.5 * float(ROOT.TMath.ChisquareQuantile(1.0 - 0.5 * _POISSON_ALPHA, 2))


def _sigma0_from_th1():
    h = ROOT.TH1D("_sig0_ref", "", 1, 0.0, 1.0)
    h.SetDirectory(0)
    h.SetBinContent(1, 0.0)
    h.SetBinErrorOption(ROOT.TH1.kPoisson)
    return float(h.GetBinErrorUp(1))


def _cell_sigma(content, error, cell_id=None):
    """Statistical χ² bin error: hist error if n>0; Garwood Up for n=0."""
    global _SIGMA_N0
    c = float(content)
    e = float(error)
    if c < 0:
        raise ValueError(
            "cell id=%s has negative content %g (grid must be non-negative)"
            % (cell_id, c)
        )
    if c == 0.0:
        if _SIGMA_N0 is None:
            try:
                _SIGMA_N0 = _sigma0_from_th1()
            except Exception:
                _SIGMA_N0 = _garwood_poisson_sigma0()
        return _SIGMA_N0
    if e > 0.0 and math.isfinite(e):
        return e
    raise ValueError("cell id=%s n=%g has non-positive error %g" % (cell_id, c, e))


def effective_sigma(content, error_stat, thr):
    """σ_eff = sqrt(σ_stat² + (thr · |content|)²).

    ``thr`` is the overall accepted relative uncertainty (e.g. 0.1 = 10%).
    ``thr=0`` recovers pure statistical errors (legacy).  Empty bins
    (content=0) keep σ_stat only (Garwood).
    """
    e = float(error_stat)
    t = float(thr)
    if t <= 0.0 or not math.isfinite(t):
        return e
    c = abs(float(content))
    return math.sqrt(e * e + (t * c) * (t * c))


def apply_acceptable_thr(obs, thr):
    """Set each observation's χ² error from error_stat + thr (in place)."""
    t = float(thr)
    for b in obs:
        b.error = effective_sigma(b.content, b.error_stat, t)
    return obs


def load_grid_json(path):
    with open(path) as fh:
        return json.load(fh)


def fit_range_from_positive_cells(cells):
    """Bounding box of cells with content > 0 (outer edges)."""
    pos = [c for c in cells if float(c.get("content", 0)) > 0.0]
    if not pos:
        raise ValueError("no cells with content > 0 — cannot define fit range")
    return (
        min(float(c["xmin"]) for c in pos),
        max(float(c["xmax"]) for c in pos),
        min(float(c["ymin"]) for c in pos),
        max(float(c["ymax"]) for c in pos),
    )


def _coalesce_index_rects(index_rects):
    """Greedy merge of axis-aligned integer index boxes [ix0,ix1)×[iy0,iy1).

    1) Horizontal runs on each row-band of equal (iy0,iy1).
    2) Vertical merge of identical-x strips with adjacent y.
    Returns a (usually much smaller) list of index rects covering the same area.
    """
    if not index_rects:
        return []
    # Expand multi-row/col boxes to unit strips for clean RLE, then re-merge.
    # For general rects: group by (iy0, iy1) and merge along x.
    by_y = {}
    for ix0, ix1, iy0, iy1 in index_rects:
        key = (int(iy0), int(iy1))
        by_y.setdefault(key, []).append((int(ix0), int(ix1)))
    hmerged = []
    for (iy0, iy1), xs in by_y.items():
        xs.sort()
        a, b = xs[0]
        for c, d in xs[1:]:
            if c <= b:  # overlap or touch
                b = max(b, d)
            else:
                hmerged.append((a, b, iy0, iy1))
                a, b = c, d
        hmerged.append((a, b, iy0, iy1))
    # Vertical merge: group by (ix0, ix1)
    by_x = {}
    for ix0, ix1, iy0, iy1 in hmerged:
        key = (ix0, ix1)
        by_x.setdefault(key, []).append((iy0, iy1))
    out = []
    for (ix0, ix1), ys in by_x.items():
        ys.sort()
        a, b = ys[0]
        for c, d in ys[1:]:
            if c <= b:
                b = max(b, d)
            else:
                out.append((ix0, ix1, a, b))
                a, b = c, d
        out.append((ix0, ix1, a, b))
    return out


def _empty_rects_from_cells(empty_cells, x_edges, y_edges):
    """Build physical empty rects, coalescing via fine-bin indices when possible.

    Falls back to one rect per cell if indices / edges are unavailable.
    """
    if not empty_cells:
        return []
    have_idx = all(
        ("ix0" in c and "ix1" in c and "iy0" in c and "iy1" in c) for c in empty_cells
    )
    if not have_idx or x_edges is None or y_edges is None:
        return [
            (
                float(c["xmin"]),
                float(c["xmax"]),
                float(c["ymin"]),
                float(c["ymax"]),
            )
            for c in empty_cells
        ]
    idx = [
        (int(c["ix0"]), int(c["ix1"]), int(c["iy0"]), int(c["iy1"]))
        for c in empty_cells
    ]
    merged = _coalesce_index_rects(idx)
    xe = list(x_edges)
    ye = list(y_edges)
    rects = []
    for ix0, ix1, iy0, iy1 in merged:
        rects.append((float(xe[ix0]), float(xe[ix1]), float(ye[iy0]), float(ye[iy1])))
    return rects


def cells_to_observations(grid, dnn_min, dnn_max, hme_min, hme_max, empty_meta=True):
    """Build CellObs list: centres inside the fit window, content ≥ 0.

    If ``empty_meta`` is True (default), all zero-yield cells are merged into
    **one** meta-cluster observation (disconnected rectangles allowed):
    content=0, Garwood σ once, μ = Σ_empty ∬_rect pdf.  Empty fine-bin
    rectangles are **coalesced** via index RLE so GL cost scales with the
    number of maximal empty blocks, not the empty-cell count.
    """
    pos = []
    empty_cells = []
    for c in grid["cells"]:
        xlo, xhi = float(c["xmin"]), float(c["xmax"])
        ylo, yhi = float(c["ymin"]), float(c["ymax"])
        cx = 0.5 * (xlo + xhi)
        cy = 0.5 * (ylo + yhi)
        if cx < dnn_min or cx > dnn_max or cy < hme_min or cy > hme_max:
            continue
        cont = float(c["content"])
        if cont < 0:
            raise ValueError(
                "grid cell id=%s has negative content %g" % (c.get("id"), cont)
            )
        if cont == 0.0:
            empty_cells.append(c)
            continue
        err = _cell_sigma(cont, float(c.get("error") or 0.0), c.get("id"))
        pos.append(CellObs(c.get("id", len(pos)), xlo, xhi, ylo, yhi, cont, err))

    out = list(pos)
    n_empty_cells = len(empty_cells)
    x_edges = grid.get("x_edges_fine") or grid.get("x_edges")
    y_edges = grid.get("y_edges_fine") or grid.get("y_edges")
    empty_rects = _empty_rects_from_cells(empty_cells, x_edges, y_edges)
    n_empty_rects = len(empty_rects)
    if empty_rects and empty_meta:
        # Single χ² term; coalesced rects for cheap multi-rect GL.
        sig0 = _cell_sigma(0.0, 0.0, cell_id="empty_meta")
        meta = CellObs(
            cell_id=-1,
            xlo=empty_rects[0][0],
            xhi=empty_rects[0][1],
            ylo=empty_rects[0][2],
            yhi=empty_rects[0][3],
            content=0.0,
            error=sig0,
            rects=empty_rects,
            is_empty_meta=True,
        )
        out.append(meta)
    elif empty_rects:
        for i, (xlo, xhi, ylo, yhi) in enumerate(empty_rects):
            cid = int(empty_cells[i].get("id", i)) if i < len(empty_cells) else i
            err = _cell_sigma(0.0, 0.0, cid)
            out.append(CellObs(cid, xlo, xhi, ylo, yhi, 0.0, err))

    return out, {
        "n_pos_obs": len(pos),
        "n_empty_cells": n_empty_cells,
        "n_empty_rects": n_empty_rects,
        "n_empty_obs": (1 if (empty_meta and n_empty_cells) else n_empty_cells),
        "empty_meta": bool(empty_meta and n_empty_cells > 0),
        "n_obs": len(out),
    }


def _reseed_N(model, obs, inits, gl_pack=None):
    """Closed-form N* for fixed shape (χ² with free yield)."""
    unit = [1.0] + [float(v) for v in inits[1:]]
    if gl_pack is not None:
        mu = model.expected_cells_batch(unit, gl_pack)
        y = np.array([b.content for b in obs], dtype=float)
        e = np.array([b.error for b in obs], dtype=float)
        w = 1.0 / (e * e)
        mask = (mu > 0) & np.isfinite(mu)
        if not np.any(mask):
            return inits
        num = float(np.sum(y[mask] * mu[mask] * w[mask]))
        den = float(np.sum(mu[mask] * mu[mask] * w[mask]))
    else:
        num = den = 0.0
        for b in obs:
            try:
                mu1 = model.expected_bin(unit, b.xlo, b.xhi, b.ylo, b.yhi)
            except Exception:
                return inits
            if not (mu1 == mu1 and mu1 > 0 and abs(mu1) < 1e300):
                return inits
            w = 1.0 / (b.error * b.error)
            num += b.content * mu1 * w
            den += mu1 * mu1 * w
    if den > 0 and math.isfinite(num / den) and (num / den) > 0:
        inits = list(inits)
        inits[0] = num / den
    return inits


def _reseed_N_general(inits, mu_of, y_data, inv_e2=None, w_prime=None):
    """Closed-form N* for a fixed shape (tiling and/or poisson_eff paths).

    χ²:      N* = Σ y·μ₁/σ² / Σ μ₁²/σ²
    poisson: N* = Σ n / Σ μ₁/w'   (n = c/w';  d(−2lnL)/dN = 0)
    where μ₁ is the prediction at N=1.
    """
    unit = [1.0] + [float(v) for v in inits[1:]]
    try:
        mu1 = np.asarray(mu_of(unit), dtype=float)
    except Exception:
        return inits
    mask = (mu1 > 0) & np.isfinite(mu1)
    if not np.any(mask):
        return inits
    if w_prime is not None:
        n_scaled = y_data / w_prime
        num = float(np.sum(n_scaled[mask]))
        den = float(np.sum(mu1[mask] / w_prime[mask]))
    else:
        num = float(np.sum(y_data[mask] * mu1[mask] * inv_e2[mask]))
        den = float(np.sum(mu1[mask] * mu1[mask] * inv_e2[mask]))
    if den > 0 and math.isfinite(num / den) and (num / den) > 0:
        inits = list(inits)
        inits[0] = num / den
    return inits


def _read_cov(mini, npar):
    try:
        mat = [[float(mini.CovMatrix(i, j)) for j in range(npar)] for i in range(npar)]
        diags = [mat[i][i] for i in range(npar)]
        if all(math.isfinite(d) and d >= 0 for d in diags) and max(diags) > 0.0:
            return mat
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- effective MC weights (poisson_eff)


def _fine_index_of(edges, v):
    """Fine-bin index containing coordinate v (clipped to the axis)."""
    i = int(np.searchsorted(edges, v, side="right")) - 1
    return min(max(i, 0), len(edges) - 2)


def annotate_weights(
    obs,
    x_edges,
    y_edges,
    weight_map="local",
    neff_min=10.0,
    super_cell=(10, 10),
):
    """Assign an effective MC weight ``w_eff`` to every observation (in place).

    Per positive cell: n_eff = c²/e², w_cell = e²/c.  Global
    w_glob = Σe²/Σc over the positive cells.  Local map: super-cells of
    NX×NY fine bins (index-aligned from ix=iy=0); each aggregates Σc, Σe² of
    the positive cells whose centre falls in it (w_sc = Σe²/Σc,
    n_eff_sc = (Σc)²/Σe²).  Assignment: w_cell if n_eff ≥ neff_min; else w_sc
    if (weight_map == "local" and n_eff_sc ≥ neff_min); else w_glob.  The
    empty meta observation (and any zero-content obs) gets w_glob.
    Returns the ``weight_map_stats`` dict.
    """
    xe = np.asarray(x_edges, dtype=float)
    ye = np.asarray(y_edges, dtype=float)
    NX = max(int(super_cell[0]), 1)
    NY = max(int(super_cell[1]), 1)
    neff_min = float(neff_min)
    pos = [b for b in obs if (not b.is_empty_meta) and b.content > 0.0]
    sum_c = float(sum(b.content for b in pos))
    sum_e2 = float(sum(b.error_stat * b.error_stat for b in pos))
    if not (sum_c > 0.0 and sum_e2 > 0.0):
        raise ValueError("annotate_weights: no positive cells in the window")
    w_glob = sum_e2 / sum_c

    sc_acc = {}
    sc_key = {}
    for b in pos:
        key = (_fine_index_of(xe, b.x) // NX, _fine_index_of(ye, b.y) // NY)
        sc_key[id(b)] = key
        acc = sc_acc.setdefault(key, [0.0, 0.0])
        acc[0] += b.content
        acc[1] += b.error_stat * b.error_stat
    sc_w = {}
    sc_neff = {}
    for key, (c, e2) in sc_acc.items():
        sc_w[key] = e2 / c
        sc_neff[key] = c * c / e2

    counts = {"cell": 0, "local": 0, "global": 0}
    yields = {"cell": 0.0, "local": 0.0, "global": 0.0}
    for b in obs:
        c = b.content
        if b.is_empty_meta or c <= 0.0:
            b.w_eff = w_glob
            b.n_eff = 0.0
            b.w_class = "global"
            continue
        e2 = b.error_stat * b.error_stat
        b.n_eff = c * c / e2
        key = sc_key.get(id(b))
        if b.n_eff >= neff_min:
            b.w_eff = e2 / c
            b.w_class = "cell"
        elif weight_map == "local" and key in sc_neff and sc_neff[key] >= neff_min:
            b.w_eff = sc_w[key]
            b.w_class = "local"
        else:
            b.w_eff = w_glob
            b.w_class = "global"
        counts[b.w_class] += 1
        yields[b.w_class] += c

    ok_keys = [k for k in sc_neff if sc_neff[k] >= neff_min]
    w_ok = np.array([sc_w[k] for k in ok_keys], dtype=float)
    return {
        "weight_map": weight_map,
        "neff_min": neff_min,
        "super_cell": [NX, NY],
        "w_glob": w_glob,
        "n_eff_total": sum_c * sum_c / sum_e2,
        "n_pos_obs": len(pos),
        "sum_content": sum_c,
        "sum_err2": sum_e2,
        "n_cell": counts["cell"],
        "n_local": counts["local"],
        "n_global": counts["global"],
        "yield_frac_cell": yields["cell"] / sum_c,
        "yield_frac_local": yields["local"] / sum_c,
        "yield_frac_global": yields["global"] / sum_c,
        "n_supercells": len(sc_acc),
        "n_supercells_ok": len(ok_keys),
        "w_sc_min": float(w_ok.min()) if w_ok.size else None,
        "w_sc_median": float(np.median(w_ok)) if w_ok.size else None,
        "w_sc_max": float(w_ok.max()) if w_ok.size else None,
    }


# --------------------------------------------------------------------------- fine-bin tiling (norm=tiling)


def _edge_index(edges, v, what):
    i = int(np.argmin(np.abs(edges - v)))
    tol = 1e-6 * max(1.0, abs(float(edges[1] - edges[0])))
    if abs(float(edges[i]) - float(v)) > tol:
        raise ValueError("%s edge %g is not on the fine grid" % (what, v))
    return i


def build_fine_to_obs(obs, pack):
    """Map every in-window fine bin of ``pack`` to the observation owning it.

    Each observation's sub-rectangles (cell box, or the coalesced rects of
    the empty meta-cluster) are converted to fine-index boxes; fine bins
    outside the window are skipped.  Returns (fine_to_obs, info) where
    unmapped bins carry -1 (excluded from the tiling window integral).
    """
    xe = pack["x_edges"]
    ye = pack["y_edges"]
    fine_pos = pack["fine_pos"]
    f2o = -np.ones(pack["n_fine_in"], dtype=np.intp)
    n_multi = 0
    for k, b in enumerate(obs):
        for xlo, xhi, ylo, yhi in ContinuousDensity2D._cell_subrects(b):
            ix0 = _edge_index(xe, xlo, "x")
            ix1 = _edge_index(xe, xhi, "x")
            iy0 = _edge_index(ye, ylo, "y")
            iy1 = _edge_index(ye, yhi, "y")
            sub = fine_pos[ix0:ix1, iy0:iy1].ravel()
            sub = sub[sub >= 0]
            if b.content > 0 and sub.size != (ix1 - ix0) * (iy1 - iy0):
                # A positive cell partly outside the window would keep its whole
                # content but only the in-window part of its expectation.
                raise ValueError(
                    "observation %d (content %.4g, DNN [%g, %g], HME [%g, %g]) "
                    "extends outside the tiling window: use the automatic window "
                    "or align --dnn-min/--dnn-max/--hme-min/--hme-max with the "
                    "cell edges" % (k, b.content, xlo, xhi, ylo, yhi)
                )
            if sub.size == 0:
                continue
            n_multi += int(np.count_nonzero(f2o[sub] >= 0))
            f2o[sub] = k
    unmapped = np.nonzero(f2o < 0)[0]
    info = {
        "n_fine_in": int(pack["n_fine_in"]),
        "n_nodes": int(pack["n_nodes"]),
        "n_quad_fine": int(pack["n_quad_fine"]),
        "n_mapped": int(pack["n_fine_in"] - unmapped.size),
        "n_unmapped": int(unmapped.size),
        "n_multi_assigned": int(n_multi),
        "unmapped": [[int(pack["ix"][p]), int(pack["iy"][p])] for p in unmapped],
    }
    return f2o, info


# --------------------------------------------------------------------------- calibrated deviance GoF (poisson_eff)


def deviance_calibration(lam, tail=1e-12, chunk=256):
    """E[D_c], Var[D_c] for D_c(k) = 2[λ − k + k ln(k/λ)], k ~ Poisson(λ).

    Exact sums over k = 0..k_max (Poisson tail < ``tail``), vectorised in
    chunks of observations sorted by λ.  λ ≤ 0 / non-finite → E = V = 0.
    """
    from scipy.stats import poisson

    lam = np.asarray(lam, dtype=float)
    E = np.zeros_like(lam)
    V = np.zeros_like(lam)
    ok = np.isfinite(lam) & (lam > 0.0)
    idx_all = np.nonzero(ok)[0]
    order = idx_all[np.argsort(lam[idx_all])]
    for s in range(0, order.size, chunk):
        idx = order[s : s + chunk]
        l = lam[idx]
        kmax = int(np.max(poisson.isf(tail, l))) + 2
        k = np.arange(kmax + 1, dtype=float)
        pmf = poisson.pmf(k[None, :], l[:, None])
        klogk = np.zeros_like(k)
        klogk[1:] = k[1:] * np.log(k[1:])
        D = 2.0 * (
            l[:, None] - k[None, :] + klogk[None, :] - k[None, :] * np.log(l)[:, None]
        )
        Ec = np.sum(pmf * D, axis=1)
        E2 = np.sum(pmf * D * D, axis=1)
        E[idx] = Ec
        V[idx] = np.maximum(E2 - Ec * Ec, 0.0)
    return E, V


def deviance_gof(D, mu, w_prime, npar, ndf):
    """Calibrated-deviance GoF: z = (D − E[D] + npar) / sqrt(Var[D])."""
    lam = np.asarray(mu, dtype=float) / np.asarray(w_prime, dtype=float)
    E, V = deviance_calibration(lam)
    E_tot = float(np.sum(E)) - float(npar)
    V_tot = float(np.sum(V))
    if V_tot > 0.0 and math.isfinite(D):
        z = (float(D) - E_tot) / math.sqrt(V_tot)
        p = 0.5 * math.erfc(z / math.sqrt(2.0))
    else:
        z = float("nan")
        p = 0.0
    p_naive = (
        float(ROOT.TMath.Prob(float(D), int(ndf)))
        if ndf > 0 and math.isfinite(D) and D < 1e20
        else 0.0
    )
    return {
        "stat": "deviance_calib",
        "D": float(D),
        "E_D": E_tot,
        "V_D": V_tot,
        "z": z,
        "p": p,
        "p_naive": p_naive,
        "n_obs": int(lam.size),
        "npar": int(npar),
    }


# --------------------------------------------------------------------------- super-cell Pearson χ² GoF (poisson_eff)


def _merge_weak_bands(band_stats, neff_min):
    """Merge DNN bands whose total n_eff < ``neff_min`` into a neighbour.

    ``band_stats`` = sorted list of (band, Σc, Σe²) over bands that hold at
    least one positive cell.  Greedy + deterministic: repeatedly take the
    weakest group (smallest n_eff below threshold; tie → lowest index) and
    merge it with its adjacent group of smaller n_eff (tie → the lower one).
    Stops when every group has n_eff ≥ neff_min or one group is left.
    Returns a list of ``[bands, Σc, Σe²]`` with ``bands`` ascending.
    """
    groups = [[[int(b)], float(c), float(e2)] for b, c, e2 in band_stats]

    def _neff(g):
        return g[1] * g[1] / g[2] if g[2] > 0.0 else 0.0

    while len(groups) > 1:
        weak = [(_neff(g), i) for i, g in enumerate(groups) if _neff(g) < neff_min]
        if not weak:
            break
        _, i = min(weak)
        cands = []
        if i > 0:
            cands.append((_neff(groups[i - 1]), i - 1))
        if i + 1 < len(groups):
            cands.append((_neff(groups[i + 1]), i + 1))
        _, j = min(cands)
        lo, hi = min(i, j), max(i, j)
        merged = [
            groups[lo][0] + groups[hi][0],
            groups[lo][1] + groups[hi][1],
            groups[lo][2] + groups[hi][2],
        ]
        groups[lo : hi + 1] = [merged]
    return groups


def _sweep_supercells(members, neff_min):
    """Low→high HME sweep: close a super-cell once n_eff = (Σc)²/Σe² ≥ neff_min.

    ``members`` = list of (obs_index, c, e2, y, x, id) already sorted.  A
    trailing group below threshold joins the previous super-cell; if there is
    none (the whole set is below threshold) it stays as one flagged group.
    """
    groups = []
    cur = []
    C = E2 = 0.0
    for m in members:
        cur.append(m)
        C += m[1]
        E2 += m[2]
        if E2 > 0.0 and C * C / E2 >= neff_min:
            groups.append(cur)
            cur = []
            C = E2 = 0.0
    if cur:
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return groups


def build_gof_supercells(obs, x_edges, y_edges, band_nx=10, neff_min=10.0):
    """Deterministic, fit-independent super-cells for the poisson_eff GoF.

    Partition rule: every positive observation is assigned by its centre to a
    DNN band of ``band_nx`` fine bins (index-aligned from ix=0).  Bands whose
    total n_eff = (Σc)²/Σe² is below ``neff_min`` are merged with a
    neighbouring band first (``_merge_weak_bands``).  Inside each (merged)
    band the cells are ordered by HME centre (ties: DNN centre, id) and swept
    from low to high HME, accumulating cells into the current super-cell
    until its n_eff ≥ neff_min, which closes it; a trailing group below
    threshold is merged into the previous super-cell of the same band.  All
    zero-content observations (the empty meta-observation, or the per-cell
    empties with ``--no-empty-meta``) form one extra GoF term (content 0,
    μ = Σμ_empty, variance μ·w_glob).

    Returns a dict with ``sc_index`` (obs → super-cell id, -1 for
    non-members), ``C``/``E2`` per super-cell (Σc, Σe²_stat), ``empty_mask``,
    ``w_glob`` (Σe²/Σc over all positive cells) and the ``supercells`` list
    (band, ix0, ix1, xlo, xhi, ylo, yhi, n_cells, C, E2, n_eff).
    """
    xe = np.asarray(x_edges, dtype=float)
    NX = max(int(band_nx), 1)
    neff_min = float(neff_min)
    nx_fine = len(xe) - 1
    n_obs = len(obs)
    content = np.array([b.content for b in obs], dtype=float)
    e2 = np.array([b.error_stat * b.error_stat for b in obs], dtype=float)
    empty_mask = content <= 0.0
    pos_idx = [k for k, b in enumerate(obs) if b.content > 0.0 and not b.is_empty_meta]
    if not pos_idx:
        raise ValueError("build_gof_supercells: no positive cells in the window")
    sum_c = float(content[pos_idx].sum())
    sum_e2 = float(e2[pos_idx].sum())
    if not (sum_c > 0.0 and sum_e2 > 0.0):
        raise ValueError("build_gof_supercells: non-positive Σc or Σe²")
    w_glob = sum_e2 / sum_c

    by_band = {}
    for k in pos_idx:
        b = obs[k]
        band = _fine_index_of(xe, b.x) // NX
        by_band.setdefault(band, []).append(
            (k, b.content, float(e2[k]), b.y, b.x, b.id)
        )
    band_stats = sorted(
        (band, sum(m[1] for m in ms), sum(m[2] for m in ms))
        for band, ms in by_band.items()
    )
    groups = _merge_weak_bands(band_stats, neff_min)

    sc_index = -np.ones(n_obs, dtype=np.intp)
    cells = []
    for bands, _gC, _gE2 in groups:
        members = []
        for band in bands:
            members.extend(by_band[band])
        members.sort(key=lambda m: (m[3], m[4], m[5]))
        b_lo, b_hi = int(min(bands)), int(max(bands))
        ix0 = b_lo * NX
        ix1 = min((b_hi + 1) * NX, nx_fine)
        for grp in _sweep_supercells(members, neff_min):
            sid = len(cells)
            idx = [m[0] for m in grp]
            sc_index[idx] = sid
            C = float(sum(m[1] for m in grp))
            E2 = float(sum(m[2] for m in grp))
            neff = C * C / E2 if E2 > 0.0 else 0.0
            cells.append(
                {
                    "band": b_lo,
                    "band_hi": b_hi,
                    "ix0": int(ix0),
                    "ix1": int(ix1),
                    "xlo": float(xe[ix0]),
                    "xhi": float(xe[ix1]),
                    "ylo": float(min(obs[k].ylo for k in idx)),
                    "yhi": float(max(obs[k].yhi for k in idx)),
                    "n_cells": len(idx),
                    "C": C,
                    "E2": E2,
                    "n_eff": neff,
                    "below_threshold": bool(neff < neff_min),
                }
            )
    n_sc = len(cells)
    C_arr = np.array([c["C"] for c in cells], dtype=float)
    E2_arr = np.array([c["E2"] for c in cells], dtype=float)
    neff_arr = np.array([c["n_eff"] for c in cells], dtype=float)
    ncell_arr = np.array([c["n_cells"] for c in cells], dtype=float)
    return {
        "band_nx": NX,
        "neff_min": neff_min,
        "n_supercells": n_sc,
        "n_bands": len(by_band),
        "n_band_groups": len(groups),
        "band_groups": [
            {
                "bands": [int(b) for b in g[0]],
                "C": g[1],
                "E2": g[2],
                "n_eff": (g[1] * g[1] / g[2] if g[2] > 0.0 else 0.0),
            }
            for g in groups
        ],
        "n_below_threshold": int(np.sum(neff_arr < neff_min)),
        "n_pos_obs": len(pos_idx),
        "n_empty_obs": int(np.sum(empty_mask)),
        "w_glob": w_glob,
        "n_eff_total": sum_c * sum_c / sum_e2,
        "neff_sc_min": float(neff_arr.min()) if n_sc else None,
        "neff_sc_median": float(np.median(neff_arr)) if n_sc else None,
        "neff_sc_max": float(neff_arr.max()) if n_sc else None,
        "cells_per_sc_min": int(ncell_arr.min()) if n_sc else None,
        "cells_per_sc_median": float(np.median(ncell_arr)) if n_sc else None,
        "cells_per_sc_max": int(ncell_arr.max()) if n_sc else None,
        "sc_index": sc_index,
        "C": C_arr,
        "E2": E2_arr,
        "empty_mask": empty_mask,
        "supercells": cells,
    }


def supercell_gof(sc, mu_hat, thr, npar):
    """Pearson χ² over the pre-built super-cells at the fitted μ.

    χ²_sc = Σ_sc (C − M)² / (M·w̄ + (thr·C)²),  w̄ = Σe²/Σc,  M = Σμ̂;
    plus one term for the zero-content observations: M_e²/(M_e·w_glob).
    Guard: a super-cell with M ≤ 0 (or non-finite variance) falls back to the
    observed variance Σe² + (thr·C)².  ndf = n_terms − npar;
    p = TMath::Prob(χ², ndf) (0 if ndf ≤ 0).
    """
    mu = np.asarray(mu_hat, dtype=float)
    n_sc = int(sc["n_supercells"])
    sci = sc["sc_index"]
    mask = sci >= 0
    M = np.bincount(sci[mask], weights=mu[mask], minlength=n_sc)
    C = sc["C"]
    E2 = sc["E2"]
    thr_f = max(float(thr), 0.0)
    wbar = E2 / C
    acc2 = (thr_f * C) ** 2
    var = M * wbar + acc2
    good = np.isfinite(M) & (M > 0.0) & np.isfinite(var) & (var > 0.0)
    n_guard = int(np.sum(~good))
    var = np.where(good, var, E2 + acc2)
    pull = (C - M) / np.sqrt(var)
    terms = pull * pull
    chi2 = float(np.sum(terms))
    n_terms = n_sc
    empty = None
    em = sc["empty_mask"]
    if np.any(em):
        M_e = float(np.sum(mu[em]))
        var_e = M_e * float(sc["w_glob"])
        if math.isfinite(M_e) and var_e > 0.0:
            chi2_e = M_e * M_e / var_e
        else:
            chi2_e = 0.0
        chi2 += chi2_e
        n_terms += 1
        empty = {
            "n_obs": int(np.sum(em)),
            "C": 0.0,
            "M": M_e,
            "w_glob": float(sc["w_glob"]),
            "var": var_e,
            "chi2": chi2_e,
        }
    ndf = int(n_terms - int(npar))
    if ndf > 0 and math.isfinite(chi2) and chi2 < 1e20:
        p = float(ROOT.TMath.Prob(chi2, ndf))
    else:
        p = 0.0
    cells = []
    for i, c in enumerate(sc["supercells"]):
        d = dict(c)
        d["M"] = float(M[i])
        d["var"] = float(var[i])
        d["pull"] = float(pull[i])
        d["chi2"] = float(terms[i])
        cells.append(d)
    return {
        "stat": "chi2_supercell",
        "chi2": chi2,
        "ndf": ndf,
        "chi2ndf": chi2 / ndf if ndf > 0 else float("inf"),
        "p": p,
        "n_supercells": n_sc,
        "n_terms": n_terms,
        "npar": int(npar),
        "thr": thr_f,
        "neff_min": float(sc["neff_min"]),
        "band_nx": int(sc["band_nx"]),
        "n_bands": int(sc["n_bands"]),
        "n_band_groups": int(sc["n_band_groups"]),
        "n_below_threshold": int(sc["n_below_threshold"]),
        "n_guarded": n_guard,
        "empty_term": empty,
        "supercells": cells,
    }


def _objective_of(fit):
    """−2lnL / χ² value at the minimum (``objective``; legacy: ``chi2``)."""
    return fit.get("objective", fit.get("chi2"))


# --------------------------------------------------------------------------- warm start


def _warm_inits(model, prev_fit):
    """Start vector for ``model`` from a previous fit: coefficients embedded
    by parameter name (new terms → 0), N from the previous fit."""
    if not prev_fit:
        return None
    names = prev_fit.get("param_names") or []
    pars = prev_fit.get("parameters") or []
    if not names or len(names) != len(pars):
        return None
    prev = dict(zip(names, [float(v) for v in pars]))
    if not all(math.isfinite(v) for v in prev.values()):
        return None
    inits = []
    for i, name in enumerate(model.param_names):
        if i == 0:
            inits.append(prev.get("N", float(model.initial_params[0])))
        else:
            inits.append(prev.get(name, 0.0))
    return inits


def _finite_or_inf(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("inf")
    return v if math.isfinite(v) else float("inf")


def run_fit(model, obs, warm_from=None, warm_start=False, **kw):
    """One (degree, thr) trial: cold fit, or warm fit with cold fallback.

    With ``warm_start`` and a usable ``warm_from`` fit, the warm fit is kept
    if it converges with a finite objective; otherwise the cold fit is also
    run and the lower objective wins.  Records ``warm_start``, ``start_kind``
    and both objectives (None when not run).
    """
    w_inits = _warm_inits(model, warm_from) if warm_start else None
    if w_inits is None:
        fit = fit_chi2_norm_cells(model, obs, **kw)
        fit.pop("_keepalive", None)
        fit["warm_start"] = bool(warm_start)
        fit["start_kind"] = "cold"
        fit["objective_cold"] = _objective_of(fit)
        fit["objective_warm"] = None
        return fit
    fit_w = fit_chi2_norm_cells(model, obs, inits_override=w_inits, **kw)
    fit_w.pop("_keepalive", None)
    obj_w = _objective_of(fit_w)
    if bool(fit_w.get("converged")) and _finite_or_inf(obj_w) < 1e29:
        fit_w["warm_start"] = True
        fit_w["start_kind"] = "warm"
        fit_w["objective_warm"] = obj_w
        fit_w["objective_cold"] = None
        return fit_w
    fit_c = fit_chi2_norm_cells(model, obs, **kw)
    fit_c.pop("_keepalive", None)
    obj_c = _objective_of(fit_c)
    if _finite_or_inf(obj_c) < _finite_or_inf(obj_w):
        fit, kind = fit_c, "cold"
    else:
        fit, kind = fit_w, "warm"
    fit["warm_start"] = True
    fit["start_kind"] = kind
    fit["objective_warm"] = obj_w
    fit["objective_cold"] = obj_c
    return fit


def fit_chi2_norm_cells(
    model,
    obs,
    max_calls=50000,
    strategy=2,
    extract_covariance=True,
    stat="chi2_neyman",
    thr=0.0,
    tiling=None,
    inits_override=None,
    gof_sc=None,
):
    """Minuit2 fit: data vs continuous N×pdf integrated over each cell.

    ``stat``: ``chi2_neyman`` (legacy Neyman χ², σ = cell error incl. thr —
    byte-identical to the published behaviour) or ``poisson_eff`` (Bohm–Zech
    scaled Poisson −2lnL: w' = w_eff + thr²·c, n = c/w', λ = μ/w',
    NLL2 = 2Σ[λ − n + n ln(n/λ)]; needs ``annotate_weights``).  For
    poisson_eff the acceptance GoF is the super-cell Pearson χ²
    (``gof_sc`` from ``build_gof_supercells``; reported in ``chi2``/``ndf``/
    ``p_value``, the −2lnL minimum in ``objective``); without ``gof_sc`` the
    ``ValueError`` is raised.  ``gof_deviance`` always
    carries the calibrated-deviance diagnostic.
    ``tiling``: None → window-GL normalisation (legacy) or
    ``{"pack", "fine_to_obs"}`` → fine-bin tiling normalisation (∑μ == N).
    ``inits_override``: full start vector incl. N (warm start; no N reseed).
    """
    npar = model.npar
    key = model.key
    if len(obs) <= npar:
        return {
            "key": key,
            "converged": False,
            "error": "ndata <= npar",
            "ndata": len(obs),
            "npar": npar,
            "chi2": float("inf"),
            "ndf": 0,
            "chi2ndf": float("inf"),
            "p_value": 0.0,
            "parameters": [],
            "errors": [],
            "covariance": None,
            "hesse_ok": False,
        }
    if stat not in STATS:
        raise ValueError("unknown stat %r (choose from %s)" % (stat, STATS))
    poisson = stat == "poisson_eff"

    if tiling is not None:
        pack = tiling["pack"]
        fine_to_obs = tiling["fine_to_obs"]
        n_obs = len(obs)
        gl_pack = None

        def mu_of(par):
            return model.expected_cells_batch_tiling(
                par, pack, fine_to_obs, n_obs=n_obs
            )

    else:
        # Precompute GL geometry once (vectorized χ²)
        gl_pack = model.prepare_cell_gl(obs)
        pack = None

        def mu_of(par):
            return model.expected_cells_batch(par, gl_pack)

    y_data = np.array([b.content for b in obs], dtype=float)
    e_data = np.array([b.error for b in obs], dtype=float)
    inv_e2 = 1.0 / (e_data * e_data)
    w_prime = None
    if poisson:
        if any(b.w_eff is None for b in obs):
            raise ValueError("poisson_eff requires annotate_weights() first")
        w_eff = np.array([b.w_eff for b in obs], dtype=float)
        thr_f = max(float(thr), 0.0)
        w_prime = w_eff + thr_f * thr_f * y_data
        n_scaled = y_data / w_prime
        n_log_n = np.zeros_like(n_scaled)
        pos_n = n_scaled > 0.0
        n_log_n[pos_n] = n_scaled[pos_n] * np.log(n_scaled[pos_n])

    if inits_override is not None:
        inits = [float(v) for v in inits_override]
        if len(inits) != npar:
            raise ValueError("inits_override length %d != npar %d" % (len(inits), npar))
    else:
        inits = [float(v) for v in model.initial_params]
        inits[0] = max(float(y_data.sum()), 1.0)
        if not poisson and tiling is None:
            inits = _reseed_N(model, obs, inits, gl_pack=gl_pack)
        else:
            inits = _reseed_N_general(
                inits, mu_of, y_data, inv_e2=inv_e2, w_prime=w_prime
            )

    if poisson:

        def chi2(par):
            try:
                mu = mu_of(par)
            except Exception:
                return 1e30
            if not np.all(np.isfinite(mu)) or np.any(mu <= 0) or np.any(mu > 1e29):
                return 1e30
            lam = mu / w_prime
            terms = lam - n_scaled + n_log_n - n_scaled * np.log(lam)
            return float(2.0 * np.sum(terms))

    else:

        def chi2(par):
            try:
                mu = mu_of(par)
            except Exception:
                return 1e30
            if not np.all(np.isfinite(mu)) or np.any(mu < 0) or np.any(mu > 1e29):
                return 1e30
            d = y_data - mu
            return float(np.sum(d * d * inv_e2))

    mini = ROOT.Math.Factory.CreateMinimizer("Minuit2", "Migrad")
    mini.SetMaxFunctionCalls(int(max_calls))
    mini.SetMaxIterations(int(max_calls))
    mini.SetTolerance(1e-3)
    mini.SetStrategy(int(strategy))
    mini.SetPrintLevel(0)
    mini.SetErrorDef(1.0)
    functor = ROOT.Math.Functor(chi2, npar)
    mini.SetFunction(functor)
    for i, (name, v0) in enumerate(zip(model.param_names, inits)):
        step = max(abs(v0) * 0.05, 0.05 if i == 0 else 0.01)
        mini.SetVariable(i, name, float(v0), step)
    try:
        mini.SetVariableLimits(0, 0.0, max(inits[0] * 50.0, 1e6))
    except Exception:
        pass

    t0 = ROOT.TStopwatch()
    t0.Start()
    migrad_ok = bool(mini.Minimize())
    if not migrad_ok or mini.Status() not in (0, 1):
        mini.SetStrategy(2)
        migrad_ok = bool(mini.Minimize()) or migrad_ok

    cov = None
    hesse_ok = False
    if extract_covariance:
        for strat in (max(int(strategy), 2), 2, 1):
            try:
                mini.SetStrategy(int(strat))
                xs0 = mini.X()
                for i in range(npar):
                    v = float(xs0[i])
                    mini.SetVariableStepSize(i, max(abs(v) * 0.05, 0.01))
                mini.Hesse()
                cov = _read_cov(mini, npar)
                if cov is not None:
                    hesse_ok = True
                    break
            except Exception:
                cov = None
                hesse_ok = False

    wall = t0.RealTime()
    xs = mini.X()
    params = [float(xs[i]) for i in range(npar)]
    errors = (
        [float(mini.Errors()[i]) for i in range(npar)] if hesse_ok else [0.0] * npar
    )

    chi2val = float(chi2(params))
    status = mini.Status()
    ndata = len(obs)
    ndf = ndata - npar
    chi2ndf = chi2val / ndf if ndf > 0 else float("inf")
    # Predicted cell yields at the minimum (same integration as the objective)
    try:
        mu_hat = mu_of(params)
        n_expected_cells = float(np.sum(mu_hat))
    except Exception:
        mu_hat = None
        n_expected_cells = None
    gof = None
    gof_dev = None
    # Reported GoF triple (χ², ndf, p).  chi2_neyman: the objective itself.
    chi2_rep, ndf_rep, chi2ndf_rep = chi2val, ndf, chi2ndf
    if poisson:
        # Calibrated deviance (diagnostic): D = NLL2 at the minimum vs its
        # Poisson expectation/variance (npar-corrected); one-sided p.
        if mu_hat is not None and math.isfinite(chi2val) and chi2val < 1e20:
            try:
                gof_dev = deviance_gof(chi2val, mu_hat, w_prime, npar, ndf)
            except Exception as exc:
                gof_dev = {"stat": "deviance_calib", "error": str(exc), "p": 0.0}
        else:
            gof_dev = {"stat": "deviance_calib", "D": chi2val, "p": 0.0}
        # Acceptance GoF: Pearson χ² on the pre-built super-cells.
        if gof_sc is not None:
            if mu_hat is not None and math.isfinite(chi2val) and chi2val < 1e20:
                gof = supercell_gof(gof_sc, mu_hat, thr, npar)
            else:
                n_terms = int(gof_sc["n_supercells"]) + (
                    1 if np.any(gof_sc["empty_mask"]) else 0
                )
                gof = {
                    "stat": "chi2_supercell",
                    "chi2": float("inf"),
                    "ndf": n_terms - npar,
                    "chi2ndf": float("inf"),
                    "p": 0.0,
                    "n_supercells": int(gof_sc["n_supercells"]),
                    "error": "objective not finite",
                }
            chi2_rep = float(gof["chi2"])
            ndf_rep = int(gof["ndf"])
            chi2ndf_rep = chi2_rep / ndf_rep if ndf_rep > 0 else float("inf")
            pval = float(gof.get("p") or 0.0)
        else:
            raise ValueError(
                "poisson_eff needs the super-cell GoF partition (gof_sc); the "
                "calibrated deviance is a diagnostic only"
            )
    else:
        pval = (
            float(ROOT.TMath.Prob(chi2val, ndf)) if ndf > 0 and chi2val < 1e20 else 0.0
        )
    # Convergence heuristic on the objective per dof (χ²/ndf, or D/(n_obs−npar))
    soft_ok = (
        status in (0, 1)
        or (status in (2, 3, 4, 5) and chi2ndf < 5.0)
        or (math.isfinite(chi2ndf) and chi2ndf < 1.5 and math.isfinite(chi2val))
    )
    # Yield closure + exponent-clip diagnostics
    sum_data = float(y_data.sum())
    sum_mu = n_expected_cells
    yield_closure = (
        sum_mu / sum_data if (sum_mu is not None and sum_data > 0.0) else None
    )
    max_exp_arg = None
    try:
        sp_hat = params[1:]
        if tiling is not None:
            max_exp_arg = model.max_exp_arg(sp_hat, pack)
        else:
            a1 = model.exp_arg_at_arr(gl_pack["xs"], gl_pack["ys"], sp_hat)
            a2 = model.exp_arg_at_arr(gl_pack["wxs"], gl_pack["wys"], sp_hat)
            if a1 is not None and a2 is not None:
                max_exp_arg = float(max(np.max(a1), np.max(a2)))
    except Exception:
        max_exp_arg = None
    clip_hit = bool(max_exp_arg > 50.0) if max_exp_arg is not None else None
    _keepalive = (functor, mini, gl_pack)

    out = {
        "key": key,
        "label": model.label,
        "npar": npar,
        "param_names": list(model.param_names),
        "parameters": params,
        "errors": errors,
        "covariance": cov,
        "hesse_ok": hesse_ok,
        "chi2": chi2_rep,
        "ndf": ndf_rep,
        "chi2ndf": chi2ndf_rep,
        "p_value": pval,
        "converged": bool(soft_ok and math.isfinite(chi2val)),
        "minuit_status": status,
        "migrad_ok": migrad_ok,
        "edm": float(mini.Edm()),
        "ncalls": int(mini.NCalls()),
        "wall_time": float(wall),
        "ndata": ndata,
        "n_expected_cells": n_expected_cells,
        "fit_mode": "poisson_eff" if poisson else "chi2_norm",
        "stat": stat,
        "norm": "tiling" if tiling is not None else "window_gl",
        "objective": chi2val,
        "deviance_over_nobs": chi2ndf if poisson else None,
        "sum_mu": sum_mu,
        "sum_data": sum_data,
        "yield_closure": yield_closure,
        "max_exp_arg": max_exp_arg,
        "clip_hit": clip_hit,
        "_keepalive": _keepalive,
    }
    if poisson:
        out["gof"] = gof
        out["gof_deviance"] = gof_dev
    return out


def json_safe(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, (np.floating, np.integer)):
            return json_safe(float(obj))
    except ImportError:
        pass
    return obj


def fit_quality_ok(result):
    if not result.get("converged"):
        return False, "fit not converged (minuit/status)"
    if not result.get("covariance"):
        return False, "covariance not defined (Hesse failed or null matrix)"
    p = result.get("p_value")
    if p is None or not math.isfinite(float(p)):
        return False, "p-value not defined"
    if float(p) < P_MIN:
        return False, "p-value %.4g < %.2f" % (float(p), P_MIN)
    return True, "ok"


def _thr_grid(thr_max, thr_step=0.01, thr_start=0.0):
    """Ascending thr values in thr_step steps, always including thr_max."""
    thr_max = max(float(thr_max), 0.0)
    thr_step = max(float(thr_step), 1e-6)
    t0 = max(0.0, float(thr_start))
    n_max = int(round(thr_max / thr_step))
    n0 = int(round(t0 / thr_step))
    grid = [min(n * thr_step, thr_max) for n in range(n0, n_max + 1)]
    if not grid or grid[-1] < thr_max - 0.5 * thr_step:
        grid.append(thr_max)
    return sorted(set(round(x / thr_step) * thr_step for x in grid))


def _trial_summary(d, thr, fit):
    return {
        "degree": int(d),
        "thr": float(thr),
        "stat": fit.get("stat"),
        "objective": _objective_of(fit),
        "chi2": fit.get("chi2"),
        "ndf": fit.get("ndf"),
        "chi2ndf": fit.get("chi2ndf"),
        "p_value": fit.get("p_value"),
        "gof_stat": (fit.get("gof") or {}).get("stat"),
        "n_supercells": (fit.get("gof") or {}).get("n_supercells"),
        "deviance_over_nobs": fit.get("deviance_over_nobs"),
        "yield_closure": fit.get("yield_closure"),
        "converged": fit.get("converged"),
        "hesse_ok": fit.get("hesse_ok"),
        "start_kind": fit.get("start_kind"),
        "objective_warm": fit.get("objective_warm"),
        "objective_cold": fit.get("objective_cold"),
        "quality_ok": fit.get("quality_ok"),
        "ncalls": fit.get("ncalls"),
        "wall_time": fit.get("wall_time"),
    }


def _log_trial(d, thr, fit):
    """One line per (degree, thr) trial."""

    def _f(v, fmt):
        try:
            return fmt % float(v)
        except (TypeError, ValueError):
            return str(v)

    gof_label = "chi2_sc" if fit.get("stat") == "poisson_eff" else "chi2"
    print(
        "[fit_one]   d=%d thr=%.4g  stat=%s  obj=%s  %s/ndf=%s/%s=%s  p=%s  "
        "closure=%s  conv=%s  start=%s  %s"
        % (
            d,
            thr,
            fit.get("stat"),
            _f(_objective_of(fit), "%.6g"),
            gof_label,
            _f(fit.get("chi2"), "%.6g"),
            _f(fit.get("ndf"), "%d"),
            _f(fit.get("chi2ndf"), "%.4f"),
            _f(fit.get("p_value"), "%.4g"),
            _f(fit.get("yield_closure"), "%.4f"),
            fit.get("converged"),
            fit.get("start_kind"),
            "PASS" if fit.get("quality_ok") else "FAIL",
        ),
        flush=True,
    )


def find_min_thr(
    model,
    obs,
    thr_max=0.1,
    thr_step=0.01,
    thr_start=0.0,
    max_calls=50000,
    strategy=2,
    extract_covariance=True,
    stat="chi2_neyman",
    tiling=None,
    warm_start=False,
    warm_from=None,
    trials=None,
    full_fits=None,
    gof_sc=None,
):
    """Smallest thr on the 1% grid with p≥P_MIN + quality.

    Returns (thr_min, fit_dict).  If none pass, thr_max fit (quality FAIL).
    With ``warm_start`` each thr step starts from the previous step's
    converged solution (the first step from ``warm_from``, e.g. the previous
    degree's best fit).  Every trial summary is appended to ``trials`` and
    every full fit dict to ``full_fits`` (if given).
    """
    thr_grid = _thr_grid(thr_max, thr_step=thr_step, thr_start=thr_start)
    last_fit = None
    prev = warm_from
    for i, thr in enumerate(thr_grid):
        apply_acceptable_thr(obs, thr)
        fit = run_fit(
            model,
            obs,
            warm_from=prev,
            warm_start=warm_start,
            max_calls=max_calls,
            strategy=strategy,
            extract_covariance=extract_covariance,
            stat=stat,
            thr=thr,
            tiling=tiling,
            gof_sc=gof_sc,
        )
        fit["acceptable_unc_thr"] = float(thr)
        ok, reason = fit_quality_ok(fit)
        fit["quality_ok"] = ok
        fit["quality_reason"] = reason
        last_fit = fit
        if trials is not None:
            trials.append(_trial_summary(model.degree, thr, fit))
            _log_trial(model.degree, thr, fit)
        if full_fits is not None:
            full_fits.append(fit)
        if fit.get("converged") and fit.get("parameters"):
            prev = fit
        if ok:
            fit["thr_min"] = float(thr)
            fit["thr_search"] = "step_%.4g_n=%d" % (thr_step, i + 1)
            apply_acceptable_thr(obs, thr)
            return float(thr), fit
    last_fit["thr_min"] = float(thr_grid[-1])
    last_fit["thr_search"] = "step_%.4g_fail_n=%d" % (thr_step, len(thr_grid))
    apply_acceptable_thr(obs, thr_grid[-1])
    return float(thr_grid[-1]), last_fit


def _best_trial_fit(fits):
    """Warm-start source among a degree's trials.

    The converged fit at the SMALLEST thr (the next degree's scan starts at
    thr=0, i.e. the same statistical weights); fallback: None.
    """
    conv = [f for f in fits if f.get("converged") and f.get("parameters")]
    if not conv:
        return None
    return min(conv, key=lambda f: _finite_or_inf(f.get("acceptable_unc_thr", 0.0)))


def select_degree_and_thr(
    xmin,
    xmax,
    ymin,
    ymax,
    obs,
    *,
    degree=None,
    max_degree=8,
    thr=0.1,
    thr_step=0.01,
    auto_thr=False,
    n_quad=4,
    max_calls=50000,
    strategy=2,
    stat="chi2_neyman",
    tiling=None,
    warm_start=False,
    y_eps=1e-3,
    trials=None,
    gof_sc=None,
):
    """Pick ExpPolyLogY2D degree and thr.

    * ``degree is None`` (auto): for d=1..max_degree, scan thr ∈ [0, thr]
      in ``thr_step`` steps; return first PASS (minimal d, then minimal thr).
    * ``degree`` fixed, ``auto_thr=False``: fit once at that thr.
    * ``degree`` fixed, ``auto_thr=True``: thr scan at that degree only.

    ``warm_start``: a new degree starts from the previous degree's best
    converged fit (coefficients embedded by term name, new terms 0, same N);
    a new thr step starts from the previous thr step (see ``find_min_thr``).
    """
    integral = max(sum(b.content for b in obs), 1.0)
    if degree is not None:
        degrees = [int(degree)]
        auto_degree = False
    else:
        degrees = list(range(1, int(max_degree) + 1))
        auto_degree = True
        auto_thr = True  # degree auto always scans thr up to thr max

    last = None
    prev_degree_best = None
    for d in degrees:
        model = ExpPolyLogY2D(
            xmin, xmax, ymin, ymax, degree=d, n_quad=n_quad, y_eps=y_eps
        )
        model.initial_params[0] = integral
        print(
            "[fit_one] try degree=%d  npar=%d  thr_mode=%s thr_max=%g  "
            "stat=%s norm=%s warm=%s"
            % (
                d,
                model.npar,
                "scan" if auto_thr else "fixed",
                thr,
                stat,
                "tiling" if tiling is not None else "window_gl",
                bool(warm_start and prev_degree_best is not None),
            ),
            flush=True,
        )
        degree_trials = []
        degree_fits = []
        if auto_thr:
            thr_min, fit = find_min_thr(
                model,
                obs,
                thr_max=thr,
                thr_step=thr_step,
                thr_start=0.0,
                max_calls=max_calls,
                strategy=strategy,
                stat=stat,
                tiling=tiling,
                warm_start=warm_start,
                warm_from=prev_degree_best,
                trials=degree_trials,
                full_fits=degree_fits,
                gof_sc=gof_sc,
            )
        else:
            apply_acceptable_thr(obs, thr)
            fit = run_fit(
                model,
                obs,
                warm_from=prev_degree_best,
                warm_start=warm_start,
                max_calls=max_calls,
                strategy=strategy,
                extract_covariance=True,
                stat=stat,
                thr=thr,
                tiling=tiling,
                gof_sc=gof_sc,
            )
            thr_min = float(thr)
            fit["acceptable_unc_thr"] = thr_min
            fit["thr_min"] = thr_min
            fit["thr_search"] = "fixed"
            ok, reason = fit_quality_ok(fit)
            fit["quality_ok"] = ok
            fit["quality_reason"] = reason
            degree_trials.append(_trial_summary(d, thr, fit))
            degree_fits.append(fit)
            _log_trial(d, thr, fit)

        fit["degree"] = d
        fit["auto_degree"] = auto_degree
        last = (model, fit, thr_min)
        if trials is not None:
            trials.extend(degree_trials)
        # Warm-start source for the next degree (converged fit at smallest thr)
        prev_degree_best = _best_trial_fit(degree_fits) or prev_degree_best
        print(
            "[fit_one]   d=%d thr=%.4g  chi2/ndf=%s  p=%s  %s"
            % (
                d,
                thr_min,
                fit.get("chi2ndf"),
                fit.get("p_value"),
                "PASS" if fit.get("quality_ok") else "FAIL",
            ),
            flush=True,
        )
        if fit.get("quality_ok"):
            return model, fit, thr_min

    model, fit, thr_min = last
    return model, fit, thr_min


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--grid", required=True, help="Grid JSON from grid_merge.py")
    p.add_argument("--output", required=True, help="Output fit JSON path")
    p.add_argument(
        "--degree",
        type=int,
        default=None,
        help="Poly degree (omit → auto-scan degree 1..--max-degree)",
    )
    p.add_argument(
        "--max-degree",
        type=int,
        default=8,
        help="Upper degree for auto scan (default 8)",
    )
    p.add_argument(
        "--thr",
        "--acceptable-unc-thr",
        dest="thr",
        type=float,
        default=0.1,
        help=(
            "Accepted relative unc. (default 0.1).  With auto degree: max thr "
            "scanned in 1%% steps from 0.  With fixed --degree: fixed thr "
            "unless --auto-thr."
        ),
    )
    p.add_argument(
        "--auto-thr",
        action="store_true",
        help="With fixed --degree, still scan thr from 0 to --thr in 1%% steps",
    )
    p.add_argument(
        "--thr-step",
        type=float,
        default=0.01,
        help="thr grid step (default 0.01 = 1%%)",
    )
    p.add_argument(
        "--n-quad",
        type=int,
        default=4,
        help="Gauss–Legendre order per axis (default 4)",
    )
    p.add_argument("--dnn-min", type=float, default=None)
    p.add_argument("--dnn-max", type=float, default=None)
    p.add_argument("--hme-min", type=float, default=None)
    p.add_argument("--hme-max", type=float, default=None)
    p.add_argument("--max-calls", type=int, default=50000)
    p.add_argument("--strategy", type=int, default=2, help="Minuit2 strategy")
    p.add_argument("--mx", type=int, default=None, help="Mass label override")
    p.add_argument(
        "--no-empty-meta",
        action="store_true",
        help="One χ² term per empty cell (legacy)",
    )
    # ---- v2 options (all opt-in; the defaults reproduce the published fits)
    p.add_argument(
        "--preset",
        choices=PRESETS,
        default="v1",
        help=(
            "v1 (default): the August-2026 published behaviour (window_gl, chi2_neyman, cold). "
            "v2: --norm tiling --stat poisson_eff --warm-start unless those "
            "flags are given explicitly."
        ),
    )
    p.add_argument(
        "--norm",
        choices=NORMS,
        default=None,
        help=(
            "Density normalisation: window_gl (one n_quad×n_quad GL rule over "
            "the window; legacy) or tiling (sum of per-fine-bin GL integrals; "
            "sum(mu) == N)."
        ),
    )
    p.add_argument(
        "--stat",
        choices=STATS,
        default=None,
        help=(
            "Objective: chi2_neyman (legacy, sigma = cell error) or poisson_eff "
            "(Bohm-Zech scaled Poisson with effective MC weights; super-cell "
            "Pearson chi2 GoF, calibrated deviance as diagnostic)."
        ),
    )
    p.add_argument(
        "--warm-start",
        dest="warm_start",
        action="store_true",
        default=None,
        help="Warm-start each (degree, thr) trial from the previous solution",
    )
    p.add_argument(
        "--no-warm-start",
        dest="warm_start",
        action="store_false",
        default=None,
        help="Cold start at every trial (legacy; overrides the v2 preset)",
    )
    p.add_argument(
        "--y-log-eps",
        type=float,
        default=1e-3,
        help="HME log-map offset eps [GeV] in ExpPolyLogY2D (default 1e-3)",
    )
    p.add_argument(
        "--n-quad-fine",
        type=int,
        default=2,
        help="GL order per axis per fine bin for --norm tiling (default 2)",
    )
    p.add_argument(
        "--weight-map",
        choices=("global", "local"),
        default="local",
        help="poisson_eff effective-weight map for low-n_eff cells (default local)",
    )
    p.add_argument(
        "--weight-neff-min",
        type=float,
        default=10.0,
        help="Min n_eff = c^2/e^2 to trust a cell's (or super-cell's) own weight",
    )
    p.add_argument(
        "--super-cell",
        type=int,
        nargs=2,
        default=(10, 10),
        metavar=("NX", "NY"),
        help="Fine bins per super-cell for the local weight map (default 10 10)",
    )
    p.add_argument(
        "--gof-band",
        type=int,
        default=10,
        help=(
            "poisson_eff GoF: fine DNN bins per band for the super-cell "
            "partition (default 10 = 1.0 DNN unit, index-aligned from ix=0)"
        ),
    )
    p.add_argument(
        "--gof-neff-min",
        type=float,
        default=10.0,
        help="poisson_eff GoF: min n_eff=(sum c)^2/sum e^2 per super-cell (10)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.grid):
        sys.exit("[ERROR] grid not found: %s" % args.grid)

    grid = load_grid_json(args.grid)
    cells = grid.get("cells") or []
    if not cells:
        sys.exit("[ERROR] grid has no cells: %s" % args.grid)

    n_neg = sum(1 for c in cells if float(c.get("content", 0)) < 0)
    if n_neg:
        sys.exit(
            "[ERROR] grid has %d negative cells — re-run grid_merge (strict)" % n_neg
        )

    auto_xmin, auto_xmax, auto_ymin, auto_ymax = fit_range_from_positive_cells(cells)
    dnn_min = auto_xmin if args.dnn_min is None else float(args.dnn_min)
    dnn_max = auto_xmax if args.dnn_max is None else float(args.dnn_max)
    hme_min = auto_ymin if args.hme_min is None else float(args.hme_min)
    hme_max = auto_ymax if args.hme_max is None else float(args.hme_max)
    empty_meta = not args.no_empty_meta
    thr_max = float(args.thr)
    auto_degree = args.degree is None

    # ---- resolve preset → norm / stat / warm start (explicit flags win)
    preset = args.preset
    v2 = preset == "v2"
    norm = args.norm or ("tiling" if v2 else "window_gl")
    stat = args.stat or ("poisson_eff" if v2 else "chi2_neyman")
    warm_start = bool(args.warm_start) if args.warm_start is not None else v2
    y_eps = float(args.y_log_eps)
    if not (y_eps > 0.0 and math.isfinite(y_eps)):
        sys.exit("[ERROR] --y-log-eps must be > 0")
    n_quad_fine = int(args.n_quad_fine)
    if n_quad_fine < 1:
        sys.exit("[ERROR] --n-quad-fine must be >= 1")
    # Machine-readable caveats travel with the result (and into the templates'
    # meta_json): the legacy statistic is known to be blind to the yield, and the
    # v2 acceptance test is not yet calibrated on the real MC weight mixture.
    caveats = []
    if stat == "chi2_neyman":
        caveats.append(
            "chi2_neyman: Neyman chi2 with sigma = observed cell error is blind to "
            "the yield on single-MC-event cells (published fits: sum(mu)/sum(MC) = "
            "0.29-0.97 at p >= 0.05); use --stat poisson_eff / --preset v2"
        )
    if norm == "window_gl":
        caveats.append(
            "window_gl: single Gauss-Legendre rule over the window; the template "
            "yield depends on --n-quad; use --norm tiling / --preset v2"
        )
    if stat == "poisson_eff":
        caveats.append(
            "chi2_supercell: acceptance GoF not yet calibrated on the real MC "
            "weight mixture (2026-09 toy study); the PASS/FAIL verdict is provisional"
        )
    if preset == "v1":
        print(
            "[fit_one] WARNING: preset v1 reproduces the August-2026 fits, whose "
            "Neyman chi2 is blind to the yield (closure 0.29-0.97) and whose "
            "window_gl templates depend on --n-quad; use --preset v2 for the "
            "corrected statistic (see README, Status)",
            flush=True,
        )

    print(
        "[fit_one] grid=%s  cells=%d  window DNN=[%.4g,%.4g] HME=[%.4g,%.4g]"
        % (args.grid, len(cells), dnn_min, dnn_max, hme_min, hme_max),
        flush=True,
    )
    print(
        "[fit_one] ExpPolyLogY2D  degree=%s  thr_max=%g  thr_step=%g  "
        "empty_meta=%s  n_quad=%d"
        % (
            "auto(1..%d)" % args.max_degree if auto_degree else args.degree,
            thr_max,
            args.thr_step,
            empty_meta,
            args.n_quad,
        ),
        flush=True,
    )
    print(
        "[fit_one] preset=%s  norm=%s  stat=%s  warm_start=%s  y_eps=%g  "
        "n_quad_fine=%d  weight_map=%s neff_min=%g super_cell=%dx%d  "
        "gof_band=%d gof_neff_min=%g"
        % (
            preset,
            norm,
            stat,
            warm_start,
            y_eps,
            n_quad_fine,
            args.weight_map,
            args.weight_neff_min,
            int(args.super_cell[0]),
            int(args.super_cell[1]),
            int(args.gof_band),
            float(args.gof_neff_min),
        ),
        flush=True,
    )

    try:
        obs, obs_info = cells_to_observations(
            grid,
            dnn_min,
            dnn_max,
            hme_min,
            hme_max,
            empty_meta=empty_meta,
        )
    except ValueError as exc:
        sys.exit("[ERROR] %s" % exc)

    if len(obs) <= 5:
        sys.exit("[ERROR] too few cells in fit window (%d)" % len(obs))

    print(
        "[fit_one] n_obs=%d  (pos=%d empty_cells=%d empty_rects=%s empty_obs=%d)  "
        "integral=%.6g"
        % (
            obs_info["n_obs"],
            obs_info["n_pos_obs"],
            obs_info["n_empty_cells"],
            obs_info.get("n_empty_rects"),
            obs_info["n_empty_obs"],
            sum(b.content for b in obs),
        ),
        flush=True,
    )

    x_edges_fine = grid.get("x_edges_fine") or grid.get("x_edges")
    y_edges_fine = grid.get("y_edges_fine") or grid.get("y_edges")
    weight_map_stats = None
    if stat == "poisson_eff":
        if x_edges_fine is None or y_edges_fine is None:
            sys.exit("[ERROR] poisson_eff needs x_edges_fine/y_edges_fine in the grid")
        try:
            weight_map_stats = annotate_weights(
                obs,
                x_edges_fine,
                y_edges_fine,
                weight_map=args.weight_map,
                neff_min=float(args.weight_neff_min),
                super_cell=(int(args.super_cell[0]), int(args.super_cell[1])),
            )
        except ValueError as exc:
            sys.exit("[ERROR] %s" % exc)
        print(
            "[fit_one] weights: w_glob=%.4g  n_eff_total=%.1f  classes cell/local/"
            "global = %d/%d/%d  (yield frac %.3f/%.3f/%.3f)  w_sc[min/med/max]="
            "%s/%s/%s over %d/%d super-cells"
            % (
                weight_map_stats["w_glob"],
                weight_map_stats["n_eff_total"],
                weight_map_stats["n_cell"],
                weight_map_stats["n_local"],
                weight_map_stats["n_global"],
                weight_map_stats["yield_frac_cell"],
                weight_map_stats["yield_frac_local"],
                weight_map_stats["yield_frac_global"],
                weight_map_stats["w_sc_min"],
                weight_map_stats["w_sc_median"],
                weight_map_stats["w_sc_max"],
                weight_map_stats["n_supercells_ok"],
                weight_map_stats["n_supercells"],
            ),
            flush=True,
        )

    gof_sc = None
    if stat == "poisson_eff":
        try:
            gof_sc = build_gof_supercells(
                obs,
                x_edges_fine,
                y_edges_fine,
                band_nx=int(args.gof_band),
                neff_min=float(args.gof_neff_min),
            )
        except ValueError as exc:
            sys.exit("[ERROR] %s" % exc)
        print(
            "[fit_one] gof super-cells: n_sc=%d (+%d empty term)  bands=%d → "
            "groups=%d  neff_min=%g  n_eff_total=%.1f  n_eff/sc[min/med/max]="
            "%.1f/%.1f/%.1f  cells/sc[min/med/max]=%s/%s/%s  below_thr=%d"
            % (
                gof_sc["n_supercells"],
                1 if gof_sc["n_empty_obs"] else 0,
                gof_sc["n_bands"],
                gof_sc["n_band_groups"],
                gof_sc["neff_min"],
                gof_sc["n_eff_total"],
                gof_sc["neff_sc_min"],
                gof_sc["neff_sc_median"],
                gof_sc["neff_sc_max"],
                gof_sc["cells_per_sc_min"],
                gof_sc["cells_per_sc_median"],
                gof_sc["cells_per_sc_max"],
                gof_sc["n_below_threshold"],
            ),
            flush=True,
        )

    tiling = None
    tiling_info = None
    if norm == "tiling":
        if x_edges_fine is None or y_edges_fine is None:
            sys.exit(
                "[ERROR] --norm tiling needs x_edges_fine/y_edges_fine in the grid"
            )
        # The tiled window is the set of fine bins whose centre lies inside it,
        # so an explicit window edge that is not a fine-bin edge would make the
        # continuous window (fit_range, templates on other binnings) and the
        # tiled window (I_W) differ silently.
        try:
            xe_f = np.asarray(x_edges_fine, dtype=float)
            ye_f = np.asarray(y_edges_fine, dtype=float)
            for v in (dnn_min, dnn_max):
                _edge_index(xe_f, v, "window DNN")
            for v in (hme_min, hme_max):
                _edge_index(ye_f, v, "window HME")
        except ValueError as exc:
            sys.exit(
                "[ERROR] tiling: %s (with --norm tiling the fit window must lie on "
                "fine-bin edges; use the automatic window or aligned "
                "--dnn-min/--dnn-max/--hme-min/--hme-max)" % exc
            )
        pack = ContinuousDensity2D(
            dnn_min, dnn_max, hme_min, hme_max
        ).prepare_fine_tiling(x_edges_fine, y_edges_fine, n_quad_fine=n_quad_fine)
        try:
            fine_to_obs, tiling_info = build_fine_to_obs(obs, pack)
        except ValueError as exc:
            sys.exit("[ERROR] tiling: %s" % exc)
        if tiling_info["n_unmapped"]:
            # Fine bins inside the window owned by no observation (a positive cell
            # cut by an explicit window with its centre outside) would be excluded
            # from I_W but integrated by templates on other binnings.
            sys.exit(
                "[ERROR] tiling: %d in-window fine bins are not covered by any "
                "observation (a cell cut by the window with its centre outside); use "
                "the automatic window or align --dnn-min/--dnn-max/--hme-min/--hme-max "
                "with the cell edges" % tiling_info["n_unmapped"]
            )
        if tiling_info["n_multi_assigned"]:
            print(
                "[fit_one] WARNING: %d fine bins were assigned to more than one "
                "observation (last assignment kept)" % tiling_info["n_multi_assigned"],
                flush=True,
            )
        tiling = {"pack": pack, "fine_to_obs": fine_to_obs}
        print(
            "[fit_one] tiling: n_fine_in=%d  nodes=%d (n_quad_fine=%d)  mapped=%d  "
            "unmapped=%d"
            % (
                tiling_info["n_fine_in"],
                tiling_info["n_nodes"],
                tiling_info["n_quad_fine"],
                tiling_info["n_mapped"],
                tiling_info["n_unmapped"],
            ),
            flush=True,
        )

    trials = []
    model, fit, thr_min = select_degree_and_thr(
        dnn_min,
        dnn_max,
        hme_min,
        hme_max,
        obs,
        degree=args.degree,
        max_degree=args.max_degree,
        thr=thr_max,
        thr_step=float(args.thr_step),
        auto_thr=bool(args.auto_thr) or auto_degree,
        n_quad=args.n_quad,
        max_calls=args.max_calls,
        strategy=args.strategy,
        stat=stat,
        tiling=tiling,
        warm_start=warm_start,
        y_eps=y_eps,
        trials=trials,
        gof_sc=gof_sc,
    )
    degree = int(fit.get("degree", model.degree))

    mx = args.mx if args.mx is not None else grid.get("mx")
    fit_mode = "chi2_norm" if stat == "chi2_neyman" else "poisson_eff"
    func_spec = model.to_spec()
    func_spec["fit_mode"] = fit_mode
    if norm == "tiling":
        func_spec["norm"] = "tiling"
        func_spec["n_quad_fine"] = n_quad_fine

    result = {
        "key": model.key,
        "function_name": model.key,
        "model": "exppoly_logy2d",
        "fit_mode": fit_mode,
        "function": func_spec,
        "param_names": fit.get("param_names") or list(model.param_names),
        "parameters": fit.get("parameters") or [],
        "errors": fit.get("errors") or [],
        "covariance": fit.get("covariance"),
        "hesse_ok": fit.get("hesse_ok"),
        "npar": model.npar,
        "chi2": fit.get("chi2"),
        "ndf": fit.get("ndf"),
        "chi2ndf": fit.get("chi2ndf"),
        "p_value": fit.get("p_value"),
        "converged": fit.get("converged"),
        "minuit_status": fit.get("minuit_status"),
        "edm": fit.get("edm"),
        "ncalls": fit.get("ncalls"),
        "wall_time": fit.get("wall_time"),
        "ndata": fit.get("ndata"),
        "degree": degree,
        "n_quad": args.n_quad,
        "acceptable_unc_thr": float(fit.get("acceptable_unc_thr", thr_min)),
        "thr_min": float(fit.get("thr_min", thr_min)),
        "thr_max": thr_max,
        "thr_step": float(args.thr_step),
        "thr_search": fit.get("thr_search"),
        "auto_degree": auto_degree,
        "auto_thr": bool(args.auto_thr) or auto_degree,
        "empty_meta": obs_info["empty_meta"],
        "n_pos_obs": obs_info["n_pos_obs"],
        "n_empty_cells": obs_info["n_empty_cells"],
        "n_empty_obs": obs_info["n_empty_obs"],
        "mx": mx,
        "category": grid.get("category"),
        "process": grid.get("process"),
        "grid_file": os.path.abspath(args.grid),
        "source_file": grid.get("source_file"),
        "hist": grid.get("hist"),
        "fit_range": {
            "dnn": [dnn_min, dnn_max],
            "hme": [hme_min, hme_max],
        },
        "fit_range_auto": {
            "dnn": [auto_xmin, auto_xmax],
            "hme": [auto_ymin, auto_ymax],
            "from": "bbox of cells with content > 0",
        },
        "n_grid_cells": len(cells),
        "n_obs_cells": len(obs),
        "integral_window": float(sum(b.content for b in obs)),
        "n_expected_cells": fit.get("n_expected_cells"),
        "grid_meta": {
            k: grid.get(k)
            for k in (
                "algorithm",
                "n_neg_before",
                "n_neg_after",
                "n_merged",
                "n_singleton",
                "nx_fine",
                "ny_fine",
            )
            if k in grid
        },
        # ---- v2 additions (present for both presets)
        "preset": preset,
        "stat": stat,
        "norm": norm,
        "warm_start": warm_start,
        "start_kind": fit.get("start_kind"),
        "objective": fit.get("objective", fit.get("chi2")),
        "objective_warm": fit.get("objective_warm"),
        "objective_cold": fit.get("objective_cold"),
        "y_eps": y_eps,
        "n_quad_fine": n_quad_fine if norm == "tiling" else None,
        "sum_mu": fit.get("sum_mu"),
        "sum_data": fit.get("sum_data"),
        "yield_closure": fit.get("yield_closure"),
        "max_exp_arg": fit.get("max_exp_arg"),
        "clip_hit": fit.get("clip_hit"),
        "trials": trials,
        "caveats": caveats,
    }
    if stat == "poisson_eff":
        result["weight_map_stats"] = weight_map_stats
        result["gof"] = fit.get("gof")
        result["gof_deviance"] = fit.get("gof_deviance")
        result["deviance_over_nobs"] = fit.get("deviance_over_nobs")
        result["gof_supercells_build"] = {
            k: v
            for k, v in (gof_sc or {}).items()
            if k not in ("sc_index", "C", "E2", "empty_mask", "supercells")
        }
    if norm == "tiling":
        fine_binning = {
            "x_edges": [float(v) for v in x_edges_fine],
            "y_edges": [float(v) for v in y_edges_fine],
        }
        if tiling_info and tiling_info.get("unmapped"):
            fine_binning["unmapped"] = tiling_info["unmapped"]
        result["fine_binning"] = fine_binning
        result["tiling"] = {
            k: v for k, v in (tiling_info or {}).items() if k != "unmapped"
        }
    result = json_safe(result)

    out_path = args.output
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print("[fit_one] wrote %s" % out_path, flush=True)

    ok, reason = fit_quality_ok(result)
    print(
        "[fit_one] %s  degree=%s  thr_min=%s  chi2/ndf=%s  p=%s  "
        "npar=%s  hesse_ok=%s  quality=%s  wall=%.1fs"
        % (
            result.get("key"),
            result.get("degree"),
            result.get("thr_min"),
            result.get("chi2ndf"),
            result.get("p_value"),
            result.get("npar"),
            result.get("hesse_ok"),
            "PASS" if ok else ("FAIL: %s" % reason),
            result.get("wall_time") or 0.0,
        ),
        flush=True,
    )
    print(
        "[fit_one] stat=%s norm=%s start=%s  objective=%s  gof=%s  sum_mu=%s  "
        "sum_data=%s  yield_closure=%s  max_exp_arg=%s clip_hit=%s"
        % (
            result.get("stat"),
            result.get("norm"),
            result.get("start_kind"),
            result.get("objective"),
            (result.get("gof") or {}).get("stat", "chi2"),
            result.get("sum_mu"),
            result.get("sum_data"),
            result.get("yield_closure"),
            result.get("max_exp_arg"),
            result.get("clip_hit"),
        ),
        flush=True,
    )
    if not ok:
        print("[fit_one] ERROR: %s" % reason, flush=True)
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
