"""Generate every table and every number quoted in the manuscript from stored results.

Usage: python scripts/manuscript_tables.py <out_dir>

Writes <out_dir>/experiment_numbers.tex (one macro per quoted number) and one
tab_*.tex per table. Inputs: results_v9 (benchmark, Covertype, wide tables),
results_v7_unchanged (plain greedy loops, core stability, radius sweep) and
results_v11 (FNC, FHARA, Core-init-P; see experiments/external_report.py).
No number is typed by hand.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from manuscript_numbers import (  # noqa: E402
    DS_NAME, DS_ORDER, METHODS, cost_fit, friedman, holm, mark_best, pivot, wilcoxon, wtl,
)

ROOT = HERE.parents[1]
V9 = ROOT / "results_v9"
V7 = ROOT / "results_v7_unchanged" if (ROOT / "results_v7_unchanged").exists() else ROOT / "results_v7"
V11 = ROOT / "results_v11"
HEAD = ["CCAR-E", "CCAR-H", "Core-init", "QR", "NIP-NRS", "GBNRS", "WNRS"]
GROUPS = [("Small-scale", ["iris", "wine", "sonar", "glass", "heart", "ecoli", "ionosphere", "dermatology"]),
          ("Medium-scale", ["vote", "wdbc", "australian", "diabetes", "vehicle", "vowel", "credit"]),
          ("Large-scale", ["yeast", "segment", "chess"])]
NUM: dict[str, object] = {}


def wide_tabular(rows: list[str], label: str, caption: str, spec: str, head: str, size: str = "footnotesize") -> str:
    return "\n".join([
        "\\begin{table*}[!t]", "  \\centering", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", f"  \\{size}",
        "  \\setlength{\\tabcolsep}{2.5pt}", "  \\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%",
        f"  \\begin{{tabular}}{{{spec}}}", "    \\toprule", f"    {head} \\\\", "    \\midrule",
        *[f"    {r}" for r in rows], "    \\bottomrule", "  \\end{tabular}}", "\\end{table*}", ""])


def column_table(rows: list[str], label: str, caption: str, spec: str, head: list[str], size: str = "footnotesize", sep: str = "3pt") -> str:
    return "\n".join([
        "\\begin{table}[!t]", "  \\centering", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", f"  \\{size}",
        f"  \\setlength{{\\tabcolsep}}{{{sep}}}", f"  \\begin{{tabular*}}{{\\columnwidth}}{{@{{\\extracolsep{{\\fill}}}}{spec}@{{}}}}",
        "    \\toprule", *[f"    {h}" for h in head], "    \\midrule",
        *[f"    {r}" for r in rows], "    \\bottomrule", "  \\end{tabular*}", "\\end{table}", ""])


def acc_full() -> pd.DataFrame:
    """Accuracy on all attributes and class counts; computed once, then read from results_v9."""
    path = V9 / "acc_full.csv"
    if not path.exists():
        from experiments.datasets import BENCHMARK_DATASETS
        from experiments.generate_tables import acc_full_scores
        rows = []
        for d in DS_ORDER:
            _, y = next(i for i in BENCHMARK_DATASETS if i.key == d).loader()
            rows.append({"dataset": d, "knn": acc_full_scores(d, "knn")[0], "svm": acc_full_scores(d, "svm")[0],
                         "n_classes": len(np.unique(y))})
        pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path).set_index("dataset")


def fmt_p(p: float) -> str:
    return f"{p:.3f}"


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(V9 / "benchmark_20datasets.csv")
    df = raw[raw.algorithm.isin(METHODS)].copy()
    assert len(df) == 126 and not df[["n_reduct_mean", "runtime_mean", "knn_mean", "svm_mean"]].isna().any().any()
    df["valid_reduct"] = df.valid_reduct.astype(str).str.lower() == "true"
    abl = raw[raw.algorithm == "_ablation"].set_index("dataset").loc[DS_ORDER]
    meta = df[df.algorithm == "ccar_exact"].set_index("dataset")
    full = acc_full()
    pr, prt = pivot(df, "n_reduct_mean"), pivot(df, "runtime_mean")
    pk, ps = pivot(df, "knn_mean"), pivot(df, "svm_mean")
    valid = df.pivot(index="dataset", columns="algorithm", values="valid_reduct").loc[DS_ORDER]
    nvalid = df.pivot(index="dataset", columns="algorithm", values="n_folds_valid").loc[DS_ORDER].astype(int)
    nmin = df.pivot(index="dataset", columns="algorithm", values="n_folds_minimal").loc[DS_ORDER].astype(int)
    all_valid = [m for m in METHODS if valid[m].all()]
    consistent = [d for d in DS_ORDER if meta.gamma_full[d] >= 1 - 1e-12]
    inconsistent = [d for d in DS_ORDER if d not in consistent]
    empty = [d for d in DS_ORDER if abl.n_core[d] == 0]
    nonempty = [d for d in DS_ORDER if abl.n_core[d] > 0]
    assert len(consistent) == 8 and len(empty) == 4
    # literal counts and ranges in the setup paragraph
    assert meta.n_samples.min() == 150 and meta.n_samples.max() == 3196 and meta.n_features.min() == 4 and meta.n_features.max() == 60
    assert full.n_classes.min() == 2 and full.n_classes.max() == 11 and meta.n_features["dermatology"] == 34

    # ---- datasets ------------------------------------------------------------------------
    rows = []
    for g, ds in GROUPS:
        for i, d in enumerate(ds):
            first = f"\\multirow{{{len(ds)}}}{{*}}{{{g}}}" if i == 0 else ""
            rows.append(f"{first} & {DS_NAME[d]} & {int(meta.n_samples[d])} & {int(meta.n_features[d])} & {int(full.n_classes[d])} & {meta.gamma_full[d]:.3f} \\\\")
        if g != GROUPS[-1][0]:
            rows.append("\\midrule")
    (out / "tab_datasets.tex").write_text(column_table(
        rows, "tab:datasets", "UCI datasets. $\\gamma_\\delta(\\C,\\D)$: dependency of all attributes at $\\delta=0.1$.",
        "llrrrr", ["Group & Dataset & $|\\U|$ & $|\\C|$ & $|\\D|$ & $\\gamma_\\delta(\\C,\\D)$ \\\\"], sep="2.8pt"))

    # ---- validity -------------------------------------------------------------------------
    rows = []
    for m, h in zip(METHODS, HEAD):
        bad = [d for d in DS_ORDER if not valid.loc[d, m]]
        if m == "gbnrs":
            by: dict[int, list[str]] = {}
            for d in bad:
                by.setdefault(int(nvalid.loc[d, m]), []).append(DS_NAME[d])
            ok = [DS_NAME[d] for d in DS_ORDER if valid.loc[d, m]]
            txt = f"All except {' and '.join(ok)}: " + "; ".join(f"{', '.join(v)} ({k})" for k, v in sorted(by.items()))
        else:
            txt = ", ".join(f"{DS_NAME[d]} ({int(nvalid.loc[d, m])})" for d in bad) or "---"
        name = "QuickReduct" if h == "QR" else h
        rows.append(f"{name} & {18 - len(bad)}/18 & {int(nmin[m].sum())} & {txt} \\\\")
        NUM[f"Valid{h.replace('-', '')}"] = 18 - len(bad)
        NUM[f"Min{h.replace('-', '')}"] = int(nmin[m].sum())
    (out / "tab_validity.tex").write_text(column_table(
        rows, "tab:reduct-validity",
        "Validity and minimality of the reducts. Valid: datasets on which all five training folds pass. Minimal: folds, of 90, with a valid reduct from which no attribute can be removed. Numbers in brackets: valid folds, of five.",
        "lcc>{\\raggedright\\arraybackslash}p{0.47\\columnwidth}", ["Method & Valid & Minimal & Invalid datasets \\\\"]))

    # ---- reduct size, accuracy, runtime --------------------------------------------------
    rows = []
    for d in DS_ORDER:
        c = mark_best(pr.loc[d].to_dict(), 1, True, {a for a in METHODS if valid.loc[d, a]})
        rows.append(f"{DS_NAME[d]} & {int(meta.n_features[d])} & " + " & ".join(c[a] for a in METHODS) + " \\\\")
    avg = mark_best(pr.mean().to_dict(), 1, True, set(all_valid))
    rows += ["\\midrule", "\\emph{Average} & --- & " + " & ".join(avg[a] for a in METHODS) + " \\\\",
             "\\emph{W/T/L} & --- & --- & " + " & ".join(wtl(pr.ccar_exact.values, pr[a].values, 1, False) for a in METHODS[1:]) + " \\\\"]
    (out / "tab_size.tex").write_text(wide_tabular(
        rows, "tab:reduct-size", "Mean reduct size over five folds. Bold and underline: best and second best among valid cells. W/T/L: datasets on which CCAR-E is smaller than, equal to or larger than the method. QR: QuickReduct.",
        "l r *{7}{c}", "Dataset & $|\\C|$ & " + " & ".join(HEAD)))

    for metric, key, label, title in [("knn", "knn", "tab:main-knn", "$k$NN ($k=3$)"), ("svm", "svm", "tab:main-svm", "SVM-RBF")]:
        pm, sd = pivot(df, f"{metric}_mean"), pivot(df, f"{metric}_std")
        rows = []
        for d in DS_ORDER:
            mk = mark_best(pm.loc[d].to_dict(), 2, False, set(METHODS))
            cells = []
            for a in METHODS:
                cell = f"{pm.loc[d, a]:.2f}\\,$\\pm$\\,{sd.loc[d, a]:.2f}"
                if mk[a].startswith("\\textbf"):
                    cell = f"\\textbf{{{cell}}}"
                elif mk[a].startswith("\\underline"):
                    cell = f"\\underline{{{cell}}}"
                cells.append(cell)
            rows.append(f"{DS_NAME[d]} & {full.loc[d, key]:.2f} & " + " & ".join(cells) + " \\\\")
        avg = mark_best(pm.mean().to_dict(), 2, False, set(METHODS))
        rows += ["\\midrule", f"\\emph{{Average}} & {full[key].mean():.2f} & " + " & ".join(avg[a] for a in METHODS) + " \\\\",
                 "\\emph{W/T/L} & --- & --- & " + " & ".join(wtl(pm.ccar_exact.values, pm[a].values, None, True) for a in METHODS[1:]) + " \\\\"]
        (out / f"tab_{metric}.tex").write_text(wide_tabular(
            rows, label, f"{title} accuracy on the test folds (mean $\\pm$ std). All: accuracy with all attributes. Bold and underline: best and second best as printed. W/T/L: datasets on which CCAR-E is more accurate than, equal to or less accurate than the method. QR: QuickReduct.",
            "l r *{7}{c}", "Dataset & All & " + " & ".join(HEAD)))

    rows = []
    for d in DS_ORDER:
        c = mark_best(prt.loc[d].to_dict(), 3, True, set(METHODS))
        rows.append(f"{DS_NAME[d]} & {int(meta.n_samples[d])} & " + " & ".join(c[a] for a in METHODS) + " \\\\")
    avg = mark_best(prt.mean().to_dict(), 3, True, set(METHODS))
    rows += ["\\midrule", "\\emph{Average} & --- & " + " & ".join(avg[a] for a in METHODS) + " \\\\",
             "\\emph{W/T/L} & --- & --- & " + " & ".join(wtl(prt.ccar_exact.values, prt[a].values, None, False) for a in METHODS[1:]) + " \\\\"]
    (out / "tab_runtime.tex").write_text(wide_tabular(
        rows, "tab:runtime", "Reduction time in seconds (mean over five folds). Bold and underline: fastest and second fastest. W/T/L: datasets on which CCAR-E is faster than, equal to or slower than the method. QR: QuickReduct.",
        "l r *{7}{c}", "Dataset & $|\\U|$ & " + " & ".join(HEAD), size="small"))

    # ---- statistics -----------------------------------------------------------------------
    fr = {m: friedman(p, h) for m, p, h in [("size", pr, False), ("time", prt, False), ("knn", pk, True), ("svm", ps, True)]}
    rows = []
    for name, key in [("$|R|$", "size"), ("Time", "time"), ("$k$NN accuracy", "knn"), ("SVM accuracy", "svm")]:
        rows.append(f"{name} & " + " & ".join(f"{fr[key][3][a]:.2f}" for a in METHODS) + " \\\\")
    (out / "tab_ranks.tex").write_text(wide_tabular(
        rows, "tab:avg-ranks", "Average Friedman ranks over the 18 datasets (lower is better). QR: QuickReduct.",
        "lccccccc", "Metric & " + " & ".join(HEAD), size="small"))

    def tests(p):
        res = [wilcoxon(p.ccar_exact.values, p[b].values) for b in METHODS[1:]]
        return dict(zip(METHODS[1:], zip(holm([r[0] for r in res]), [r[1] for r in res])))

    ts, tt = tests(pr), tests(prt)
    NUM.update(CD=f"{fr['size'][2]:.3f}", CDshort=f"{fr['size'][2]:.2f}",
               MeanE=f"{pr.ccar_exact.mean():.2f}", MeanQR=f"{pr.quick_reduct.mean():.2f}", MeanCI=f"{pr.core_init_greedy.mean():.2f}",
               MeanNIP=f"{pr.nicp.mean():.2f}", MeanWNRS=f"{pr.wnrs.mean():.2f}", MeanEshort=f"{pr.ccar_exact.mean():.1f}", MeanQRshort=f"{pr.quick_reduct.mean():.1f}",
               SizeChi=f"{fr['size'][0]:.3f}", SizeP=fmt_p(fr['size'][1]), SizeRankE=f"{fr['size'][3].ccar_exact:.2f}",
               SizeGapQR=f"{fr['size'][3].quick_reduct - fr['size'][3].ccar_exact:.2f}",
               SizeHolmQR=fmt_p(ts['quick_reduct'][0]), SizeRQR=f"{ts['quick_reduct'][1]:.2f}", SizeHolmCI=fmt_p(ts['core_init_greedy'][0]),
               KnnChi=f"{fr['knn'][0]:.3f}", KnnP=fmt_p(fr['knn'][1]), SvmChi=f"{fr['svm'][0]:.3f}", SvmP=fmt_p(fr['svm'][1]),
               TimeChi=f"{fr['time'][0]:.3f}", TimeRankCI=f"{fr['time'][3].core_init_greedy:.2f}", TimeRankE=f"{fr['time'][3].ccar_exact:.2f}",
               TimeRankQR=f"{fr['time'][3].quick_reduct:.2f}", TimeRankH=f"{fr['time'][3].ccar_h:.2f}", TimeRankNIP=f"{fr['time'][3].nicp:.2f}",
               TimeRankWNRS=f"{fr['time'][3].wnrs:.2f}", TimeRankGB=f"{fr['time'][3].gbnrs:.2f}",
               TimeHolmGB=fmt_p(tt['gbnrs'][0]), TimeRGB=f"{tt['gbnrs'][1]:.2f}", TimeHolmWNRS=fmt_p(tt['wnrs'][0]), TimeRWNRS=f"{tt['wnrs'][1]:.2f}",
               TimeHolmNIP=fmt_p(tt['nicp'][0]), TimeRNIP=f"{tt['nicp'][1]:.2f}", TimeHolmOther=fmt_p(tt['quick_reduct'][0]),
               TimeMeanE=f"{prt.ccar_exact.mean():.2f}", TimeMeanCI=f"{prt.core_init_greedy.mean():.2f}", TimeMeanQR=f"{prt.quick_reduct.mean():.2f}",
               TimeMeanGB=f"{prt.gbnrs.mean():.1f}", TimeMeanNIP=f"{prt.nicp.mean():.2f}",
               ChessGapQR=f"{prt.quick_reduct['chess'] / prt.ccar_exact['chess']:.1f}",
               SonarEms=f"{1000 * prt.ccar_exact['sonar']:,.0f}".replace(",", "{,}"), SonarHms=f"{1000 * prt.ccar_h['sonar']:.0f}",
               DermEms=f"{1000 * prt.ccar_exact['dermatology']:.0f}", DermHms=f"{1000 * prt.ccar_h['dermatology']:.0f}")
    assert all(abs(tt[m][0] - tt['quick_reduct'][0]) < 1e-9 for m in ["ccar_h", "core_init_greedy"])
    # counts quoted in words in the text
    assert (prt.quick_reduct > prt.ccar_exact).sum() == 12 and (prt.core_init_greedy < prt.ccar_exact).sum() == 13
    assert (prt.ccar_exact <= prt.ccar_h).sum() == 11 and (prt.gbnrs == prt.max(axis=1)).all()
    assert fr['size'][3].ccar_exact == fr['size'][3].min()

    q5 = friedman(pivot(df, "n_reduct_mean", methods=all_valid), False)
    NUM.update(FiveChi=f"{q5[0]:.3f}", FiveCD=f"{q5[2]:.3f}", FiveRankE=f"{q5[3].ccar_exact:.2f}", FiveRankH=f"{q5[3].ccar_h:.2f}",
               FiveRankCI=f"{q5[3].core_init_greedy:.2f}", FiveRankNIP=f"{q5[3].nicp:.2f}", FiveRankQR=f"{q5[3].quick_reduct:.2f}")
    assert q5[1] < 1e-4 and fr['time'][1] < 1e-4
    sc, si = friedman(pivot(df, "n_reduct_mean", datasets=consistent), False), friedman(pivot(df, "n_reduct_mean", datasets=inconsistent), False)
    tc, ti = friedman(pivot(df, "runtime_mean", datasets=consistent), False), friedman(pivot(df, "runtime_mean", datasets=inconsistent), False)
    spread = pivot(df, "n_reduct_mean", methods=all_valid, datasets=inconsistent).mean()
    NUM.update(ConsP=fmt_p(sc[1]), ConsRankE=f"{sc[3].ccar_exact:.2f}", InconsP=fmt_p(si[1]), InconsSpread=f"{spread.max() - spread.min():.2f}",
               InconsRankGB=f"{si[3].gbnrs:.2f}", ConsTimeQR=f"{tc[3].quick_reduct:.2f}", ConsTimeCI=f"{tc[3].core_init_greedy:.2f}", ConsTimeE=f"{tc[3].ccar_exact:.2f}",
               InconsTimeCI=f"{ti[3].core_init_greedy:.2f}", InconsTimeE=f"{ti[3].ccar_exact:.2f}",
               GBInvalidCons=int((~valid.gbnrs[consistent]).sum()), GBInvalidIncons=int((~valid.gbnrs[inconsistent]).sum()))
    assert sc[1] < 0.05 and si[1] < 0.05 and tc[1] < 1e-3 and ti[1] < 1e-3
    assert (~valid.wnrs[inconsistent]).sum() == 0 and si[3].gbnrs == si[3].min()

    # accuracy cases named in the text
    ds_cells = {"DermE": pk.ccar_exact["dermatology"], "DermFull": full.knn["dermatology"], "DermCI": pk.core_init_greedy["dermatology"],
                "DermSvmE": ps.ccar_exact["dermatology"], "DermSvmFull": full.svm["dermatology"], "SonarKnnE": pk.ccar_exact["sonar"],
                "SonarSvmE": ps.ccar_exact["sonar"], "SonarSvmFull": full.svm["sonar"]}
    NUM.update({k: f"{v:.2f}" for k, v in ds_cells.items()})
    others = ps.loc["sonar", ["ccar_h", "core_init_greedy", "quick_reduct", "nicp"]]
    NUM.update(SonarSvmLo=f"{others.min():.2f}", SonarSvmHi=f"{others.max():.2f}", DermRE=f"{pr.ccar_exact['dermatology']:.1f}", DermRCI=f"{pr.core_init_greedy['dermatology']:.1f}")

    # CCAR-E against all attributes
    gk, gs = full.knn.reindex(pk.index), full.svm.reindex(ps.index)
    NUM.update(AllGapKnn=f"{(gk - pk.ccar_exact).mean() * 100:.1f}", AllGapSvm=f"{(gs - ps.ccar_exact).mean() * 100:.1f}",
               AllPKnn=f"{stats.wilcoxon(pk.ccar_exact, gk).pvalue:.3f}", AllPSvm=f"{stats.wilcoxon(ps.ccar_exact, gs).pvalue:.3f}")
    # empty-core tables are exactly the ones where NIP-NRS beats CCAR-E; pair-free inconsistent tables
    assert set(ratio_empty := empty) == {"wine", "ionosphere", "sonar", "dermatology"}
    assert sum(abl.n_pairs[d] == 0 for d in inconsistent) == 7

    # head-to-head with NIP-NRS
    h2h = []
    for p in (pr, prt, pk, ps):
        h2h.append(wilcoxon(p.ccar_exact.values, p.nicp.values))
    adj = holm([x[0] for x in h2h])
    ratio = prt.nicp / prt.ccar_exact
    assert (ratio > 1).sum() == 14 and set(ratio[ratio < 1].index) == {"wine", "ionosphere", "sonar", "dermatology"}
    NUM.update(NipHolmSize=fmt_p(adj[0]), NipHolmTime=fmt_p(adj[1]), NipRTime=f"{h2h[1][1]:.2f}", NipSpeed=f"{ratio.median():.1f}")
    assert min(adj[2:]) > 0.05

    # leave-one-out
    loo = {}
    for key, p in (("Time", prt), ("Size", pr)):
        vals = []
        for d in p.index:
            r = pd.Series(np.apply_along_axis(stats.rankdata, 1, p.drop(index=d).values).mean(axis=0), index=p.columns)
            vals.append(r.ccar_exact)
            assert (r.ccar_exact > r.min()) if key == "Time" else (r.ccar_exact == r.min())
        loo[key] = vals
    NUM.update(LooTimeLo=f"{min(loo['Time']):.2f}", LooTimeHi=f"{max(loo['Time']):.2f}", LooSizeLo=f"{min(loo['Size']):.2f}", LooSizeHi=f"{max(loo['Size']):.2f}")

    # plain loops
    v7 = pd.read_csv(V7 / "benchmark_20datasets.csv")
    a7, a9 = v7[v7.algorithm.isin(METHODS)].groupby("algorithm").runtime_mean.sum(), df.groupby("algorithm").runtime_mean.sum()
    slow = (a7 / a9)[["core_init_greedy", "quick_reduct", "nicp", "wnrs"]]
    NUM.update(PlainLo=f"{slow.min():.0f}", PlainHi=f"{slow.max():.0f}", PlainCells=len(df) * 5)
    d7 = v7[v7.algorithm.isin(METHODS)]
    p7 = pivot(d7, "runtime_mean")
    t7 = tests(p7)
    r7 = friedman(p7, False)[3]
    assert r7.ccar_exact == r7.min() and t7["nicp"][0] < 0.05 and t7["quick_reduct"][0] < 0.05

    # ---- ablation -------------------------------------------------------------------------
    def rho_fmt(v):
        return "0" if v == 0 else (f"{v:.4f}" if v >= 1e-3 else f"{v:.2e}")
    rows = [f"{DS_NAME[d]} & {int(abl.n_core[d])} & {rho_fmt(abl.pair_ratio[d])} & {abl.speedup_pair_vs_full[d]:.1f}$\\times$ \\\\" for d in DS_ORDER]
    (out / "tab_ablation.tex").write_text(column_table(
        rows, "tab:ablation-pairs", "Conflicting pairs on the full data ($\\delta=0.1$, single run). $\\rho=|P_\\delta|/\\binom{|\\U|}{2}$. Speedup: time to build $\\Mset$ over the time of the core-first path.",
        "lrrr", ["Dataset & $|\\CORE_\\delta|$ & $\\rho$ & Speedup \\\\"], size="small", sep="4pt"))
    sp = abl.speedup_pair_vs_full
    frac = abl.n_core / meta.n_features[DS_ORDER]
    r_c, r_f, r_u, r_w = (stats.spearmanr(sp, meta.n_features[DS_ORDER]), stats.spearmanr(sp, frac),
                          stats.spearmanr(sp, meta.n_samples[DS_ORDER]), stats.spearmanr(frac, meta.n_features[DS_ORDER]))
    low = abl.pair_ratio[nonempty]
    n_low = int((low < 2.2e-3).sum())
    assert r_w.pvalue < 1e-3 and n_low == 13
    narrow = meta.n_features[DS_ORDER] < 20
    NUM.update(NonEmpty=len(nonempty), SpeedMed=f"{sp[nonempty].median():.1f}", SpeedLo=f"{sp[nonempty].min():.1f}", SpeedHi=f"{sp[nonempty].max():.1f}",
               EmptyLo=f"{sp[empty].min():.2f}", EmptyHi=f"{sp[empty].max():.2f}", RhoWidth=f"{r_c.statistic:.2f}", RhoWidthP=f"{r_c.pvalue:.2f}",
               RhoFrac=f"{r_f.statistic:.2f}", RhoFracP=f"{r_f.pvalue:.2f}", RhoSize=f"{r_u.statistic:.2f}", RhoSizeP=f"{r_u.pvalue:.3f}",
               RhoCoreWidth=f"{r_w.statistic:.3f}".replace("-", "-"), FracNarrow=f"{frac[narrow].median():.2f}", FracWide=f"{frac[~narrow].median():.2f}",
               DensityMax=f"{abl.pair_ratio.max():.2f}", DensityCount=n_low)
    assert sp[nonempty].idxmin() == "diabetes" and sp[nonempty].idxmax() == "chess"

    # ---- cost fits ------------------------------------------------------------------------
    fits = {a: cost_fit(df, a, "n_samples") for a in METHODS}
    for a, k in [("ccar_exact", "E"), ("ccar_h", "H"), ("core_init_greedy", "CI"), ("quick_reduct", "QR"), ("nicp", "NIP"), ("wnrs", "WNRS")]:
        NUM[f"Fitb{k}"] = f"{fits[a]['b']:.2f}"
        NUM[f"Fitc{k}"] = f"{fits[a]['c']:.2f}"
    rows = [f"{h} & {fits[a]['b']:.2f} & {fits[a]['c']:.2f} & {fits[a]['r2']:.2f} \\\\" for a, h in
            [("ccar_exact", "CCAR-E"), ("ccar_h", "CCAR-H"), ("core_init_greedy", "Core-init"), ("quick_reduct", "QuickReduct"), ("nicp", "NIP-NRS"), ("wnrs", "WNRS")]]
    (out / "tab_fits.tex").write_text(column_table(
        rows, "tab:cost-fits", "Fitted exponents of $\\log t=a+b\\log|\\U|+c\\log|\\C|$ over the 18 datasets.",
        "lrrr", ["Method & $b$ & $c$ & $R^2$ \\\\"], sep="4pt"))
    e = fits["ccar_exact"]

    def ratio_at(a, pp):
        f = fits[a]
        return np.exp(f["a"] - e["a"] + (f["b"] - e["b"]) * np.log(1000) + (f["c"] - e["c"]) * np.log(pp))

    def cross(a):
        f = fits[a]
        return np.exp(-(f["a"] - e["a"] + (f["b"] - e["b"]) * np.log(1000)) / (f["c"] - e["c"]))
    NUM.update(FitRsq=f"{e['r2']:.2f}", RatioQR=f"{ratio_at('quick_reduct', 10):.1f}", RatioNIP=f"{ratio_at('nicp', 10):.1f}",
               CrossQR=f"{cross('quick_reduct'):.0f}", CrossNIP=f"{cross('nicp'):.0f}", CrossCI=f"{cross('core_init_greedy'):.0f}")

    # ---- Covertype ------------------------------------------------------------------------
    c = pd.read_csv(V9 / "scalability_v9.csv")
    assert c.valid.all()
    w = c.pivot(index="n", columns="method", values="runtime")
    z = c.pivot(index="n", columns="method", values="n_reduct")
    rows = []
    for n in w.index:
        s = c[c.n == n].iloc[0]
        nn = f"{n:,}".replace(",", "\\,")
        rows.append(f"{nn} & {s.n_core} & {s.gamma_C:.3f} & {z.loc[n, 'CCAR-H']} & {z.loc[n, 'QuickReduct']} & {z.loc[n, 'Core-init']} & "
                    f"{w.loc[n, 'CCAR-H']:.1f} & {w.loc[n, 'QuickReduct']:.1f} & {w.loc[n, 'Core-init']:.1f} \\\\")
    (out / "tab_covertype.tex").write_text(column_table(
        rows, "tab:scalability", "Reduct size and time in seconds on one stratified Covertype subsample per $n$ ($|\\C|=54$). H: CCAR-H. QR: QuickReduct. CI: Core-init.",
        "rrrcccrrr", ["& & & \\multicolumn{3}{c}{$|R|$} & \\multicolumn{3}{c}{Time (s)} \\\\", "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}",
                      "$n$ & $|\\CORE_\\delta|$ & $\\gamma_\\delta(\\C,\\D)$ & H & QR & CI & H & QR & CI \\\\"]))
    gci, gqr = w["Core-init"] / w["CCAR-H"], w["QuickReduct"] / w["CCAR-H"]
    assert gci.is_monotonic_increasing
    assert {54 - k for k in c.n_core} == {44, 45}
    NUM.update(CovCILo=f"{gci.iloc[0]:.1f}", CovCIHi=f"{gci.iloc[-1]:.1f}", CovQRLo=f"{gqr.min():.1f}", CovQRHi=f"{gqr.max():.1f}",
               CovH=f"{w.loc[10000, 'CCAR-H']:.0f}", CovCI=f"{w.loc[10000, 'Core-init']:.0f}", CovQR=f"{w.loc[10000, 'QuickReduct']:.0f}")

    # ---- wide tables ----------------------------------------------------------------------
    t = pd.read_csv(V9 / "wide_table_stress_v9.csv")
    wt = t.pivot(index="p", columns="algorithm", values="runtime")
    rows = []
    for p in wt.index:
        te = "---" if np.isnan(wt.ccar_e[p]) else f"{wt.ccar_e[p]:.2f}"
        rows.append(f"{p} & {int(t[t.p == p].n_core.iloc[0])} & {te} & {wt.ccar_h[p]:.2f} & {wt.nicp[p]:.2f} & {wt.wang_knn[p]:.2f} & {wt.nicp[p] / wt.ccar_h[p]:.2f} & {wt.wang_knn[p] / wt.ccar_h[p]:.2f} \\\\")
    (out / "tab_wide.tex").write_text(column_table(
        rows, "tab:wide-table-stress", "Time in seconds on synthetic wide tables ($|\\U|=400$, single run). E, H, NIP, Wang: CCAR-E, CCAR-H, NIP-NRS and the $k$NN-NRS greedy method ($k=3$). CCAR-E is run only up to $|\\C|=40$.",
        "rrrrrrrr", ["$|\\C|$ & $|\\CORE_\\delta|$ & $t_{\\mathrm{E}}$ & $t_{\\mathrm{H}}$ & $t_{\\mathrm{NIP}}$ & $t_{\\mathrm{Wang}}$ & $t_{\\mathrm{NIP}}/t_{\\mathrm{H}}$ & $t_{\\mathrm{Wang}}/t_{\\mathrm{H}}$ \\\\"], sep="2.2pt"))
    rn = wt.nicp / wt.ccar_h
    assert (t.n_core == 0).all() and (rn < 1).all() and (t[t.algorithm == "wang_knn"].n_reduct == 1).all() and not t[t.algorithm == "wang_knn"].valid_delta.any()
    NUM.update(WideRatioLo=f"{rn.min():.2f}", WideRatioHi=f"{rn.max():.2f}", WideE=f"{wt.ccar_e[40]:.1f}", WideH=f"{wt.ccar_h[40]:.2f}", WideNIP=f"{wt.nicp[40]:.2f}")

    # ---- core stability -------------------------------------------------------------------
    js = json.loads((V7 / "core_stability.json").read_text())
    st = {x["dataset"]: x for x in js["summaries"]}
    rows = []
    for d in ("wdbc", "heart", "credit"):
        x = st[d]
        rows.append(f"{DS_NAME[d]} & {x['n_features']} & {x['core_size_mean']:.1f}$\\pm${x['core_size_std']:.1f} & {x['jaccard_mean']:.2f} & {len(x['stable_core_indices'])} \\\\")
    allm, allj = np.mean([x["core_size_mean"] for x in st.values()]), np.mean([x["jaccard_mean"] for x in st.values()])
    ne = [x for x in st.values() if x["core_size_mean"] > 0]
    nem, nej = np.mean([x["core_size_mean"] for x in ne]), np.mean([x["jaccard_mean"] for x in ne])
    rows += ["\\midrule", f"All 18 (mean) & --- & {allm:.1f} & {allj:.2f} & --- \\\\", f"Non-empty core ({len(ne)}) & --- & {nem:.1f} & {nej:.2f} & --- \\\\"]
    (out / "tab_stability.tex").write_text(column_table(
        rows, "tab:core-stability", "Stability of the core across folds. Jaccard: mean pairwise similarity. Stable: attributes in at least four of five folds.",
        "lrrrr", ["Dataset & $|\\C|$ & $\\overline{|\\CORE_\\delta|}$ & Jaccard & Stable \\\\"]))
    assert len(st) == 18 and len(ne) == 14
    for d, k in (("wdbc", "WDBC"), ("credit", "Credit"), ("heart", "Heart")):
        NUM.update({f"Core{k}": f"{st[d]['core_size_mean']:.1f}", f"CoreSd{k}": f"{st[d]['core_size_std']:.1f}", f"Jac{k}": f"{st[d]['jaccard_mean']:.2f}"})
    NUM.update(JacAll=f"{allj:.2f}", JacNonEmpty=f"{nej:.2f}")

    # ---- radius sweep ---------------------------------------------------------------------
    sw = pd.read_csv(V7 / "delta_sweep_v7.csv")
    size = sw.pivot(index="dataset", columns="delta", values="n_reduct_mean")
    gam = sw.pivot(index="dataset", columns="delta", values="gamma_full")
    kn = sw.pivot(index="dataset", columns="delta", values="knn_mean")
    assert (gam.diff(axis=1).iloc[:, 1:] <= 1e-12).all().all()
    spr = (kn.max(axis=1) - kn.min(axis=1)) * 100
    near = (kn.max(axis=1) - kn[0.1]) * 100
    assert (near < 2).sum() == 7 and near.idxmax() == "sonar"
    inc = [d for d in size.index if gam.loc[d, 0.1] < 1]
    assert len(inc) == 5
    NUM["SweepIncons"] = f"{(size.loc[inc, 0.3] / size.loc[inc, 0.05]).max():.1f}"
    NUM.update(SweepHeart=f"{size.loc['heart', 0.3] / size.loc['heart', 0.05]:.1f}", SweepSonar=f"{size.loc['sonar', 0.3] / size.loc['sonar', 0.05] + 1e-9:.1f}",
               SweepDiabLo=f"{gam.loc['diabetes', 0.05]:.3f}", SweepDiabHi=f"{gam.loc['diabetes', 0.3]:.3f}",
               SweepStay=f"{gam.loc[['wine', 'sonar', 'heart'], 0.3].min():.2f}", SweepSpread=f"{spr.median():.1f}",
               SweepSonarLo=f"{size.loc['sonar', 0.1]:.1f}", SweepSonarHi=f"{size.loc['sonar', 0.3]:.1f}", SweepSonarAcc=f"{(kn.loc['sonar', 0.3] - kn.loc['sonar', 0.1]) * 100:.1f}")

    # ---- one backward pass on the greedy baselines -----------------------------------------
    pf = pd.read_csv(V11 / "pruned_folds.csv")
    assert pf.valid.all()
    pu = pf[~pf.table.str.startswith("covertype")]
    assert len(pu) == 180
    pm_ = pu.groupby(["table", "method"]).n_after.mean().unstack()
    pp = pr.copy()
    pp["quick_reduct"] = pm_["QuickReduct"].reindex(pp.index)
    h_ = holm([wilcoxon(pp.ccar_exact.values, pp[b].values)[0] for b in METHODS[1:]])
    assert min(h_) > 0.05
    NUM.update(PruneMeanQR=f"{pm_['QuickReduct'].mean():.2f}", PruneHolmQR=fmt_p(h_[METHODS[1:].index("quick_reduct")]),
               PruneSec=f"{pu[pu.method == 'QuickReduct'].prune_time.mean():.2f}", PruneFixed=int((~pu[pu.method == "QuickReduct"].minimal_before).sum()))

    # ---- pair filter and FHARA, from the session of results_v11 ------------------------------
    ef = pd.read_csv(V11 / "external_folds.csv")
    et = ef.groupby(["table", "method"]).runtime.mean().unstack()
    es = pd.read_csv(V11 / "external_scalability.csv")
    ten = es[es.n == 10000].groupby("method").runtime.median()
    NUM.update(FilterUCI=f"{np.exp(np.log(et['Core-init'] / et['Core-init-P']).mean()):.2f}", FilterCov=f"{ten['Core-init'] / ten['Core-init-P']:.1f}",
               FharaSlower=int((et.FHARA > et.QuickReduct).sum()),
               MemPairs=f"{es[(es.n == 10000) & (es.fold == 42) & (es.method == 'FNC')].n_pairs.iloc[0] / 1e6:.0f}")
    nef = frac[nonempty]
    rne = stats.spearmanr(sp[nonempty], nef)
    NUM.update(RhoFracNE=f"{rne.statistic:.2f}".replace("-", "-"), RhoFracNEP=f"{rne.pvalue:.3f}")

    # ---- output ---------------------------------------------------------------------------
    lines = "".join(f"\\newcommand{{\\num{k.replace('-', '')}}}{{{v}}}\n" for k, v in sorted(NUM.items()))
    (out / "experiment_numbers.tex").write_text(lines)
    shutil.copy(V11 / "external_macros.tex", out / "external_macros.tex")
    shutil.copy(V11 / "external_table.tex", out / "tab_external.tex")
    print(f"{len(NUM)} macros, tables in {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
