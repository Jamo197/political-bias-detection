"""cluster_common.py

Shared account-level (cluster) bootstrap machinery for the chapter 5.3-5.5
re-analyses. Same resampling design as ``cluster_bootstrap.py`` (chapters
5.1/5.2):

* unit = author account (``speaker`` in the logs = ``social_media_handle`` of
  the dataset), 72 accounts, accounts sorted alphabetically;
* 2000 replicates, seed 42, ``default_rng(42).integers(0, A, size=(2000, A))``:
  every replicate draws A accounts with replacement and takes all their posts;
* the SAME draw matrix is used for every model, condition and subset, so paired
  comparisons stay paired. Statistics on a subset of posts (one party, the
  R_top posts, ...) are domain estimators on the full draws: an account that is
  not in the subset contributes nothing to that replicate.

Sensitivity runs drop the largest account (Karl_Lauterbach, 193 posts) and
redraw over the remaining 71 accounts with the same seed, as in
``cluster_bootstrap_followup.py``.
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_metrics as em  # noqa: E402

B = em.DEFAULT_N_BOOT
SEED = em.DEFAULT_SEED
ALPHA_SIG = 0.05


@lru_cache(maxsize=1)
def account_of_post() -> Dict[str, str]:
    """text_index -> account handle, from the main matrix of chapters 5.1/5.2."""
    mm = em.load_main_matrix()
    m = mm.drop_duplicates("text_index").set_index("text_index")["speaker"]
    if mm.groupby("text_index")["speaker"].nunique().max() != 1:
        raise SystemExit("A post maps to more than one account")
    return {str(k): str(v) for k, v in m.items()}


def largest_account() -> Tuple[str, int]:
    s = pd.Series(account_of_post()).value_counts()
    return str(s.index[0]), int(s.iloc[0])


class Design:
    """Account index and shared draws. ``drop`` removes accounts (sensitivity)."""

    def __init__(self, drop: Sequence[str] = ()):
        self.post_acct = {t: a for t, a in account_of_post().items() if a not in set(drop)}
        self.accounts = sorted(set(self.post_acct.values()))
        self.A = len(self.accounts)
        self.code = {a: i for i, a in enumerate(self.accounts)}
        self.idx = np.random.default_rng(SEED).integers(0, self.A, size=(B, self.A))
        cnt = np.zeros((B, self.A))
        np.add.at(cnt, (np.repeat(np.arange(B), self.A), self.idx.ravel()), 1)
        self.counts = cnt  # (B, A) multiplicity of each account in each draw
        self.dropped = tuple(drop)
        self.label = "full (72 accounts)" if not drop else f"without {', '.join(drop)} ({self.A} accounts)"

    # -- mapping ---------------------------------------------------------------
    def keep(self, df: pd.DataFrame, col: str = "text_index") -> pd.DataFrame:
        return df[df[col].astype(str).isin(self.post_acct)].copy()

    def codes(self, text_index: Iterable) -> np.ndarray:
        return np.array([self.code[self.post_acct[str(t)]] for t in text_index], dtype=int)

    def n_accounts(self, text_index: Iterable) -> int:
        return len({self.post_acct[str(t)] for t in text_index})

    # -- resampling ------------------------------------------------------------
    def acct_sum(self, codes: np.ndarray, values: np.ndarray) -> np.ndarray:
        return np.bincount(codes, weights=np.asarray(values, float), minlength=self.A)

    def boot_sum(self, codes: np.ndarray, values: np.ndarray) -> np.ndarray:
        """(B,) resampled totals of ``values``."""
        return self.counts @ self.acct_sum(codes, values)

    def boot_mean(self, codes: np.ndarray, values: np.ndarray) -> np.ndarray:
        """(B,) resampled means (ratio estimator, NaN values ignored)."""
        v = np.asarray(values, float)
        ok = ~np.isnan(v)
        num = self.boot_sum(codes[ok], v[ok])
        den = self.boot_sum(codes[ok], np.ones(ok.sum()))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def weights(self, codes: np.ndarray) -> np.ndarray:
        """(B, n) row multiplicities for statistics that need row-level data."""
        return self.counts[:, codes]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def pct(samples) -> Tuple[float, float]:
    return em._percentile_ci(np.asarray(samples, float))


def p_boot(samples) -> float:
    return em._boot_p_two_sided(np.asarray(samples, float))


def holm(p: Sequence[float]) -> List[float]:
    return em._holm(list(p))


def post_boot(values) -> np.ndarray:
    """(B,) post-level bootstrap means (iid resampling of posts, fresh seed-42
    stream per comparison as in cluster_bootstrap.py). Used only to separate the
    effect of the test (Wilcoxon -> bootstrap of the mean) from that of clustering."""
    v = np.asarray(values, float)
    ix = np.random.default_rng(SEED).integers(0, len(v), size=(B, len(v)))
    return v[ix].mean(1)


def cr1_t(codes: np.ndarray, x: np.ndarray) -> Tuple[float, float, float, float]:
    """Mean of x with a cluster-robust (CR1) SE and a t(G-1) 95 % CI.
    Returns (lo, hi, p, G)."""
    from scipy import stats
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    x, codes = x[ok], codes[ok]
    n, m = len(x), x.mean()
    u = np.bincount(codes, weights=x - m)
    G = int((np.bincount(codes) > 0).sum())
    if G < 2:
        return np.nan, np.nan, np.nan, G
    se = np.sqrt(G / (G - 1) * (u ** 2).sum()) / n
    t = stats.t.ppf(0.975, G - 1)
    p = float(2 * stats.t.sf(abs(m / se), G - 1)) if se > 0 else np.nan
    return float(m - t * se), float(m + t * se), p, G


def excl0(lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def direction(lo: float, hi: float, neg: str = "A better", pos: str = "B better") -> str:
    return neg if hi < 0 else pos if lo > 0 else "n.s."


def r4(x) -> float:
    return em._round(x)


def short(m: str) -> str:
    meta = em._meta_for(m)
    fam = {"Mistral": "Ministral"}.get(meta["family"], meta["family"])
    return f"{fam} {int(meta['params_b'])}B"


def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            lc = c.lower()
            is_p = lc == "p" or lc.startswith(("p_", "p (", "p ")) or lc.endswith((" p", "_p")) or "p_holm" in lc
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else
                                ("<0.001" if is_p and v < 0.001 else f"{v:.{nd}f}"))
        elif pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].map({True: "yes", False: "no"})
    return em._md_table(out)


def weighted_slope(T: np.ndarray, P: np.ndarray, W: np.ndarray) -> np.ndarray:
    """OLS slope of P on T per row of the weight matrix W (B, n)."""
    sw = W.sum(1, keepdims=True)
    mt = (W * T).sum(1, keepdims=True) / sw
    mp = (W * P).sum(1, keepdims=True) / sw
    return (W * (T - mt) * (P - mp)).sum(1) / (W * (T - mt) ** 2).sum(1)


def weighted_sd(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    sw = W.sum(1)
    m = (W * P).sum(1) / sw
    return np.sqrt((W * (P - m[:, None]) ** 2).sum(1) / (sw - 1))


def wmean(x: np.ndarray, W: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is not None:
        W = W * mask
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W * x).sum(1) / W.sum(1)
