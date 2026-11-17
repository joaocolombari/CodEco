"""Correlacao de indices (baseline x codec), scatter plots e curvas taxa-distorcao."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import config as C


def ccc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 2 * np.cov(a, b, ddof=0)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-20)


def _codec_curve_label(row) -> str:
    if row["codec"] == "ae":
        # Separa ae_wave, ae_wave_lite e ae_stft em curvas proprias
        variant = row["setting"].split(",")[0].replace("variant=", "")
        return f"ae_{variant}"
    return row["codec"]


def main():
    res = pd.read_csv(C.OUT_DIR / "results.csv")
    idx = [c for c in res.columns if c.startswith("idx_")]
    base = res[res.codec == "baseline"][["dataset", "clip_id"] + idx]
    lossy = res[res.codec != "baseline"].copy()
    m = lossy.merge(base, on=["dataset", "clip_id"], suffixes=("", "_base"))

    rows = []
    for (ds, codec, setting), g in m.groupby(["dataset", "codec", "setting"]):
        for i in idx:
            a, b = g[i + "_base"].values, g[i].values
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 3 and a[ok].std() > 0 and b[ok].std() > 0:
                rows.append(dict(
                    dataset=ds, codec=codec, setting=setting, index=i,
                    pearson=pearsonr(a[ok], b[ok])[0],
                    spearman=spearmanr(a[ok], b[ok])[0],
                    ccc=ccc(a[ok], b[ok]),
                ))
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "index_correlations.csv", index=False)

    for ds, gds in m.groupby("dataset"):
        for i in [c for c in C.PLOT_INDICES if c in idx]:
            groups = list(gds.groupby(["codec", "setting"]))
            n = len(groups)
            cols = min(4, n)
            rws = int(np.ceil(n / cols))
            fig, axs = plt.subplots(rws, cols, figsize=(3.2 * cols, 3.2 * rws), squeeze=False)
            for ax, ((c, s), g) in zip(axs.ravel(), groups):
                ax.scatter(g[i + "_base"], g[i], s=6, alpha=.5)
                lo, hi = np.nanmin(g[[i, i + "_base"]].values), np.nanmax(g[[i, i + "_base"]].values)
                ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
                ax.set_title(f"{c}\n{s[:28]}", fontsize=7)
                ax.set_xlabel("baseline")
                ax.set_ylabel("codec")
            fig.suptitle(f"{ds}\n{i}")
            fig.tight_layout()
            fig.savefig(C.OUT_DIR / f"scatter_{ds}_{i}.png", dpi=150)
            plt.close(fig)

    rd = lossy.groupby(["dataset", "codec", "setting"])[["bitrate_kbps", "mel_ssim", "snr_db", "mel_rmse_db"]].mean().reset_index()
    rd["curve_label"] = rd.apply(_codec_curve_label, axis=1)
    rd.to_csv(C.OUT_DIR / "rate_distortion.csv", index=False)

    for ds, g in rd.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for label, gc in g.groupby("curve_label"):
            gc = gc.sort_values("bitrate_kbps")
            ax.plot(gc.bitrate_kbps, gc.mel_ssim, "o-", label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Bitrate (kbps)")
        ax.set_ylabel("Mel-Spectrogram SSIM")
        ax.legend(fontsize=8)
        ax.set_title(ds)
        fig.savefig(C.OUT_DIR / f"rate_distortion_{ds}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()