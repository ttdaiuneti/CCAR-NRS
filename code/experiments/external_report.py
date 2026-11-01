"""Tables and macros for the comparison with FNC, FHARA and Core-init-P, from results_v11 only."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results_v11"
M = ["CCAR-E", "CCAR-H", "Core-init-P", "QuickReduct", "FNC", "FHARA"]
NAMES = {"iris": "Iris", "wine": "Wine", "sonar": "Sonar", "glass": "Glass", "heart": "Heart", "ecoli": "Ecoli",
         "ionosphere": "Ionosphere", "dermatology": "Dermatology", "vote": "Congressional", "wdbc": "WDBC",
         "australian": "Australian", "diabetes": "Diabetes", "vehicle": "Vehicle", "vowel": "Vowel",
         "credit": "German Credit", "yeast": "Yeast", "segment": "Segment", "chess": "Chess"}
d = pd.read_csv(RES / "external_folds.csv")
s = pd.read_csv(RES / "external_scalability.csv")
w = pd.read_csv(RES / "external_wide.csv")
mem = pd.read_csv(RES / "external_memory.csv").set_index("method")
assert len(d) == 630 and d.valid.all() and s.valid.all() and w.valid.all()
t = d.groupby(["table", "method"]).runtime.mean().unstack()[M]
p = d.pivot_table(index=["table", "fold"], columns="method", values="reduct_set", aggfunc="first")
ps = s.pivot_table(index=["n", "fold"], columns="method", values="reduct_set", aggfunc="first")
same = int(((p.FNC == p.QuickReduct) & (p.FHARA == p.QuickReduct)).sum())
same_ci = int((p["Core-init-P"] == p["Core-init"]).sum())
assert same == 90 and same_ci == 90 and (ps["Core-init-P"] == ps["Core-init"]).all() and (ps.FNC == ps.QuickReduct).all()
pub = d[d.method == "FNC"]
lost = pub[pub.valid_published_stop == False]
ranks = t.rank(axis=1).mean()
chi = friedmanchisquare(*[t[m] for m in M])
fam = [(a, b) for a in ("CCAR-E", "CCAR-H") for b in ("FNC", "FHARA", "Core-init-P")]
raw = [wilcoxon(t[a], t[b]).pvalue for a, b in fam]
order = sorted(range(6), key=lambda i: raw[i])
holm, run = [0.0] * 6, 0.0
for k, i in enumerate(order):
    run = max(run, min(1.0, (6 - k) * raw[i]))
    holm[i] = run
H = dict(zip(fam, holm))
gm = lambda a, b: float(np.exp(np.log(t[a] / t[b]).mean()))
q = s.groupby(["n", "method"]).runtime.agg(["median", "min", "max", "count"])
sz = s.groupby(["n", "method"]).n_reduct.median()
pairs = s.groupby(["n", "method"]).n_pairs.median()
wd = w.pivot_table(index="p", columns="method", values="runtime")
gb = lambda m: (mem.loc[m, "rss_peak_bytes"] - mem.loc[m, "rss_before_bytes"]) / 1e9
v9 = pd.concat(pd.read_csv(f) for f in glob.glob(str(ROOT / "results_v9" / "*_perfold.csv")))
guar = int((~v9[v9.algorithm == "ccar_exact"].used_surrogate.astype(bool)).sum())
sec = lambda n, m: f"{q.loc[(n, m), 'median']:.0f}" if q.loc[(n, m), "median"] >= 10 else f"{q.loc[(n, m), 'median']:.2f}"
mac = {
    "Same": same, "SameCI": same_ci, "LostFolds": len(lost), "Chi": f"{chi.statistic:.2f}", "RankFNC": f"{ranks.FNC:.2f}",
    "RankP": f"{ranks['Core-init-P']:.2f}", "RankE": f"{ranks['CCAR-E']:.2f}", "RankH": f"{ranks['CCAR-H']:.2f}",
    "RankFH": f"{ranks.FHARA:.2f}", "HolmEF": f"{H['CCAR-E', 'FNC']:.3f}", "HolmHF": f"{H['CCAR-H', 'FNC']:.3f}",
    "HolmEFH": f"{H['CCAR-E', 'FHARA']:.3f}", "HolmHFH": f"{H['CCAR-H', 'FHARA']:.3f}",
    "HolmEP": f"{H['CCAR-E', 'Core-init-P']:.3f}", "HolmHP": f"{H['CCAR-H', 'Core-init-P']:.3f}",
    "WinsEF": int((t["CCAR-E"] < t.FNC).sum()), "GmEF": f"{gm('CCAR-E', 'FNC'):.1f}", "GmHP": f"{gm('CCAR-H', 'Core-init-P'):.1f}",
    "SegE": f"{t.loc['segment', 'CCAR-E']:.2f}", "SegF": f"{t.loc['segment', 'FNC']:.2f}",
    "ChessE": f"{t.loc['chess', 'CCAR-E']:.2f}", "ChessF": f"{t.loc['chess', 'FNC']:.2f}",
    "TenH": sec(10000, "CCAR-H"), "TenP": sec(10000, "Core-init-P"), "TenF": sec(10000, "FNC"),
    "TenFmin": f"{q.loc[(10000, 'FNC'), 'min']:.0f}", "TenFmax": f"{q.loc[(10000, 'FNC'), 'max']:.0f}",
    "TenCI": sec(10000, "Core-init"), "TenQR": sec(10000, "QuickReduct"),
    "OneH": sec(1000, "CCAR-H"), "OneF": sec(1000, "FNC"),
    "RTenH": int(sz.loc[(10000, "CCAR-H")]), "RTenF": int(sz.loc[(10000, "FNC")]),
    "PairsF": f"{pairs.loc[(10000, 'FNC')] / 1e6:.0f}", "PairsP": f"{pairs.loc[(10000, 'Core-init-P')] / 1e3:.0f}",
    "MemF": f"{gb('FNC'):.1f}", "MemH": f"{gb('CCAR-H'):.1f}", "MemP": f"{gb('Core-init-P'):.1f}", "MemQR": f"{gb('QuickReduct'):.1f}",
    "WideH": f"{wd.loc[200, 'CCAR-H']:.2f}", "WideF": f"{wd.loc[200, 'FNC']:.2f}", "GuarFolds": guar,
}
(RES / "external_macros.tex").write_text("".join(f"\\newcommand{{\\ext{k}}}{{{v}}}\n" for k, v in mac.items()))


def cells(row):
    txt = [f"{row[m]:.3f}" for m in M]
    best = min(float(x) for x in txt)
    return " & ".join(f"\\textbf{{{x}}}" if float(x) == best else x for x in txt)


meta = d.groupby("table").agg(n=("n", "first"), core=("n_core", "mean"))
a = ["\\begin{table}[!t]", "  \\caption{Reduction runtime in seconds on the UCI tables, timed in one session (means over five folds). "
     "$|\\U|$: training-fold size. CI-P: Core-init-P. QR: QuickReduct. Bold: fastest as printed.}", "  \\label{tab:external}", "  \\footnotesize",
     "  \\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrrrrrr@{}}", "    \\toprule",
     "    Table & $|\\U|$ & $|\\CORE_\\delta|$ & CCAR-E & CCAR-H & CI-P & QR & FNC & FHARA \\\\", "    \\midrule"]
for k in meta.sort_values("n").index:
    a.append(f"    {NAMES[k]} & {meta.n[k]:,} & {meta.core[k]:.1f} & {cells(t.loc[k])} \\\\".replace(",", "{,}"))
a += ["    \\bottomrule", "  \\end{tabular*}", "\\end{table}"]
C = ["CCAR-H", "Core-init-P", "QuickReduct", "FNC", "FHARA"]
b = ["\\begin{table}[!t]", "  \\caption{Median reduction time in seconds, with the range in brackets, and median $|R|$ on Covertype subsamples. "
     "Five subsamples per $n$, three at $n=10{,}000$, where FHARA has one run. CI-P: Core-init-P. QR: QuickReduct.}", "  \\label{tab:external-cov}", "  \\footnotesize",
     "  \\setlength{\\tabcolsep}{2pt}", "  \\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}rlllllrr@{}}", "    \\toprule",
     "    $n$ & CCAR-H & CI-P & QR & FNC & FHARA & $|R|_{\\mathrm{H}}$ & $|R|_{\\mathrm{FNC}}$ \\\\", "    \\midrule"]
f = lambda v: f"{v:.1f}" if v >= 10 else f"{v:.2f}"
for n in sorted(s.n.unique()):
    med = " & ".join(f(q.loc[(n, m), "median"]) for m in C)
    rng = " & ".join("" if q.loc[(n, m), "count"] == 1 else f"({f(q.loc[(n, m), 'min'])}--{f(q.loc[(n, m), 'max'])})" for m in C)
    b.append(f"    {n:,} & ".replace(",", "{,}") + med + f" & {sz.loc[(n, 'CCAR-H')]:.0f} & {sz.loc[(n, 'FNC')]:.0f} \\\\")
    b.append(f"     & {rng} & & \\\\")
b += ["    \\bottomrule", "  \\end{tabular*}", "\\end{table}"]
(RES / "external_table.tex").write_text("\n".join(a + [""] + b) + "\n")
print(open(RES / "external_macros.tex").read())
print(dict(zip(fam, [round(x, 5) for x in raw])), t.index[t["CCAR-E"] < t.FNC].tolist())
