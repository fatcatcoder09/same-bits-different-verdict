"""Every figure and table in the paper, from the score files, with no hand-entered numbers.

Run: python make_figures.py   (writes fig1_det.pdf, fig2_forest.pdf, fig3_rate.pdf, table1.tex, table1.json)

Generating the tables and the figures from one code path means a number can only be wrong in one place.
`table1.json` and `table2.json` carry every value the paper prints.
"""
import json, os, sys, collections, numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.environ.get("RTC_PAPER_DATA", "/data/audio-deepfake/RTC-SDD-PAPER")
OUT = os.path.dirname(os.path.abspath(__file__))
print(f"reading scores from {R}", file=sys.stderr)
ARMS = ("osce_off", "lace", "nolace")
PRETTY = {"osce_off": "OSCE off (cx 4)", "lace": "LACE (cx 6)", "nolace": "NoLACE (cx 7)"}
DET = {"t1": "T1", "t2": "T2", "p1": "P1", "p2": "P2"}

plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "ps.fonttype": 42})

RT = {l.split()[0]: l.split()[1] for l in open(os.environ.get("RTCFAKE_DEV_LABEL",
                                 "/home/shigang/RTC-SDD-SIDE/data/rtcfake/dev_label.txt"))
      if l.startswith("offline/")}
AV = {l.split()[0]: l.split()[-1] for l in open(f"{R}/asvspoof/label.txt")}
CORPUS = {
    "RTCFake": (RT, {"t1": "scores_t1_full.jsonl", "t2": "scores_t2_full.jsonl",
                     "p1": "scores_p1_full.jsonl", "p2": "scores_p2_full_osce.jsonl"}),
    "ASVspoof": (AV, {"t1": "scores_t1_asv.jsonl", "t2": "scores_t2_asv.jsonl",
                      "p1": "scores_p1_asv.jsonl", "p2": "scores_p2_asv_osce.jsonl"}),
    "FLEURS": (None, {"t1": "scores_t1_fleurs.jsonl", "t2": "scores_t2_fleurs.jsonl",
                      "p1": "scores_p1_fleurs.jsonl", "p2": "scores_p2_fleurs_osce.jsonl"}),
}


def load(fname):
    S = collections.defaultdict(dict)
    try:
        for line in open(os.path.join(R, fname)):
            r = json.loads(line)
            if r["arm"] in ARMS:
                S[r["item"]][r["arm"]] = r["score"]
    except FileNotFoundError:
        print(f"  no {fname}: this corpus/detector is absent from the figures and tables below",
              file=sys.stderr)
        return None
    return {i: v for i, v in S.items() if len(v) == 3}


def det_curve(bona, spoof):
    """False-alarm and miss rates over every threshold. Higher score = more bona fide."""
    b, s = np.sort(np.asarray(bona, float)), np.sort(np.asarray(spoof, float))
    ts = np.unique(np.concatenate([b, s]))
    fa = np.searchsorted(b, ts, "left") / len(b)            # genuine scored below t
    miss = (len(s) - np.searchsorted(s, ts, "left")) / len(s)  # spoof scored at or above t
    return ts, fa, miss


def eer(bona, spoof):
    ts, fa, miss = det_curve(bona, spoof)
    k = int(np.argmin(np.abs(fa - miss)))
    return float((fa[k] + miss[k]) / 2)


def at_target(S, bona, spoof, target=0.01):
    """Threshold fixed on the osce_off arm at `target` false alarms, then applied to all three arms."""
    t = float(np.quantile([S[i]["osce_off"] for i in bona], target))
    out = {}
    for a in ARMS:
        out[a] = (float(np.mean([S[i][a] < t for i in bona])),
                  float(np.mean([S[i][a] >= t for i in spoof])))
    return t, out


# Cells whose absolute level is withheld. T1 and T2 are this author's entries in an ongoing challenge and
# RTCFake dev-offline is that challenge's development set, so printing their level would disclose system
# quality the challenge's own leaderboard has not. Every claim in the paper is a change, so the change is
# printed instead and nothing is lost. T1/T2 on ASVspoof are out of domain and carry no such information,
# and P1/P2 are public checkpoints anyone can reproduce; both keep their levels.
WITHHELD = {("RTCFake", "T1"), ("RTCFake", "T2")}


def sci(p):
    """1.9e-06 is programmer notation; a table prints $1.9\\times10^{-6}$."""
    m, e = f"{p:.1e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def signed(v):
    """A text-mode hyphen is not a minus sign."""
    return f"${v:+.2f}$"


def lvl(r, key_off, key_on, scale=100.0):
    """(off, on) as levels, or ('--', signed change) where the level is withheld."""
    a, b = r[key_off] * scale, r[key_on] * scale
    if (r["corpus"], r["detector"]) in WITHHELD:
        return "--", signed(b - a)
    return f"{a:.2f}", f"{b:.2f}"


# ---------------------------------------------------------------- Table 1 and its artefact
rows = []
for corpus, (lab, files) in CORPUS.items():
    if lab is None:
        continue                                    # FLEURS is genuine-only: no EER, no miss rate
    for d, f in files.items():
        S = load(f)
        if not S:
            continue
        bo = [i for i in S if lab.get(i) == "bonafide"]
        sp = [i for i in S if lab.get(i) == "spoof"]
        t, tg = at_target(S, bo, sp, 0.01)
        # A ratio of two small counts says little; the paired decision flips are the testable quantity.
        new_fa = sum(1 for i in bo if S[i]["nolace"] < t and S[i]["osce_off"] >= t)
        fixed_fa = sum(1 for i in bo if S[i]["osce_off"] < t and S[i]["nolace"] >= t)
        p_fa = binomtest(new_fa, new_fa + fixed_fa, 0.5).pvalue if new_fa + fixed_fa else 1.0
        rows.append({"corpus": corpus, "detector": DET[d], "n_bona": len(bo), "n_spoof": len(sp),
                     "fa_items_off": int(round(tg["osce_off"][0] * len(bo))),
                     "fa_items_nolace": int(round(tg["nolace"][0] * len(bo))),
                     "miss_items_off": int(round(tg["osce_off"][1] * len(sp))),
                     "miss_items_nolace": int(round(tg["nolace"][1] * len(sp))),
                     "fa_newly": new_fa, "fa_fixed": fixed_fa, "fa_mcnemar_p": float(p_fa),
                     "eer_off": eer([S[i]["osce_off"] for i in bo], [S[i]["osce_off"] for i in sp]),
                     "eer_nolace": eer([S[i]["nolace"] for i in bo], [S[i]["nolace"] for i in sp]),
                     "fa_off": tg["osce_off"][0], "fa_nolace": tg["nolace"][0],
                     "miss_off": tg["osce_off"][1], "miss_nolace": tg["nolace"][1]})
json.dump(rows, open(f"{OUT}/table1.json", "w"), indent=1)

with open(f"{OUT}/table1.tex", "w") as fh:
    fh.write("\\begin{tabular}{llccccccc}\n\\toprule\n")
    fh.write("& & \\multicolumn{2}{c}{EER (\\%)} & \\multicolumn{2}{c}{FA (\\%)} "
             "& \\multicolumn{2}{c}{miss (\\%)} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n")
    fh.write("corpus & det. & off & NoLACE & off & NoLACE & off & NoLACE & $p$ \\\\\n\\midrule\n")
    last = None
    for r in rows:
        c = r["corpus"] if r["corpus"] != last else ""
        last = r["corpus"]
        thin = "$^{\\dagger}$" if min(r["miss_items_off"], r["miss_items_nolace"]) < 10 else ""
        e0, e1 = lvl(r, "eer_off", "eer_nolace")
        m0, m1 = lvl(r, "miss_off", "miss_nolace")
        fh.write(f"{c} & {r['detector']} & {e0} & {e1} & "
                 f"{r['fa_off']*100:.2f} & \\textbf{{{r['fa_nolace']*100:.2f}}} & "
                 f"{m0} & {m1}{thin} & {sci(r['fa_mcnemar_p'])} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- Figure 1: schematic + DET curves
# Panels must name cells outside WITHHELD: a detection error trade-off curve discloses the miss level at the
# marker and the equal-error rate at its crossing of the diagonal, so a withheld cell cannot be plotted. The
# ASVspoof P1/P2 cells are excluded for a separate reason -- a single miss item at this operating point means
# their curve below 1% carries no information.
panels = [("RTCFake", "p1"), ("ASVspoof", "t2")]
assert not ({(c, DET[d]) for c, d in panels} & WITHHELD), "a panel would disclose a withheld cell"
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05),
                         gridspec_kw={"width_ratios": [1.25, 1, 1], "wspace": 0.42})

# (a) schematic: one encode, three decodes, identical bytes on the wire.
# Boxes are drawn by the text itself through `bbox`, so a frame can never be narrower than the label inside
# it, which a hand-sized rectangle cannot guarantee.
ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
FS = 5.6
BUS_IN, BUS_OUT, BOX_CX = 2.75, 7.9, 5.35
YS = (8.4, 5.0, 1.6)
BOXSTYLE = dict(boxstyle="square,pad=0.42", fc="w")

ax.text(1.05, 5.0, "Opus\nencoder", ha="center", va="center", fontsize=FS,
        bbox=dict(boxstyle="square,pad=0.42", fc="0.93", ec="0.35", lw=0.7))
ax.text(1.05, 2.3, "one bitstream,\nidentical bytes", ha="center", va="center",
        fontsize=5.0, color="0.4", style="italic")
ax.plot([BUS_IN, BUS_IN], [YS[2], YS[0]], lw=0.8, color="0.6", solid_capstyle="round")
ax.plot([BUS_OUT, BUS_OUT], [YS[2], YS[0]], lw=0.8, color="0.6", solid_capstyle="round")
ax.annotate("", xy=(BUS_IN, 5.0), xytext=(2.05, 5.0),
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.4",
                            shrinkA=0, shrinkB=0, mutation_scale=7))
for (lbl, col), y in zip((("cx 4: OSCE off", "0.25"), ("cx 6: LACE", "0.55"), ("cx 7: NoLACE", "C3")), YS):
    ax.plot([BUS_IN, BOX_CX], [y, y], lw=0.7, color="0.6", zorder=1)
    ax.plot([BOX_CX, BUS_OUT], [y, y], lw=0.7, color="0.6", zorder=1)
    ax.text(BOX_CX, y, lbl, ha="center", va="center", fontsize=FS, color=col, zorder=3,
            bbox=dict(ec=col, lw=0.9, **BOXSTYLE))
ax.annotate("", xy=(8.45, 5.0), xytext=(BUS_OUT, 5.0),
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.6",
                            shrinkA=0, shrinkB=0, mutation_scale=7))
ax.text(9.3, 5.0, "detector", ha="center", va="center", fontsize=FS,
        bbox=dict(boxstyle="square,pad=0.42", fc="0.93", ec="0.35", lw=0.7))
ax.set_title("(a) one encode, three decodes", fontsize=7)

for ax, (corpus, d) in zip(axes[1:], panels):
    lab, files = CORPUS[corpus]
    S = load(files[d])
    bo = [i for i in S if lab.get(i) == "bonafide"]
    sp = [i for i in S if lab.get(i) == "spoof"]
    _, tg = at_target(S, bo, sp, 0.01)
    # Two of the three operating points nearly coincide in each panel -- LACE with NoLACE on P1/RTCFake,
    # LACE with OSCE-off on T2/ASVspoof -- which is a fact about the data, not a plotting accident. Distinct
    # shapes, with LACE drawn as a larger open square, keep an overlapping pair readable instead of one
    # marker hiding the other.
    MARK = {"osce_off": dict(marker="o", ms=4.4, zorder=5),
            "lace":     dict(marker="s", ms=7.4, mfc="none", mew=1.1, zorder=4),
            "nolace":   dict(marker="D", ms=3.4, zorder=6)}
    for a, style in zip(ARMS, ["-", "--", "-"]):
        col = {"osce_off": "0.25", "lace": "0.55", "nolace": "C3"}[a]
        _, fa, miss = det_curve([S[i][a] for i in bo], [S[i][a] for i in sp])
        ax.plot(fa * 100, miss * 100, style, label=PRETTY[a], color=col)
        # The marker sits on its OWN rendering's curve: same numeric threshold, different point.
        m = dict(MARK[a])
        ax.plot(tg[a][0] * 100, tg[a][1] * 100, color=col, mec=col if a == "lace" else "w",
                mew=m.pop("mew", 0.6), linestyle="none", **m)
    # No coordinate labels: the empty region of a detection error trade-off plot differs per panel, since a
    # weak detector's curve hugs the top edge, so no fixed placement is clear of the curves in every panel.
    # The markers carry the point and Table I carries the rates.
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("false alarms on genuine (%)")
    ax.set_ylabel("misses on spoof (%)")
    ax.set_title(f"({'b' if corpus == 'RTCFake' else 'c'}) {DET[d]} on {corpus}", fontsize=7)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
axes[1].legend(loc="lower left", fontsize=5.6)
fig.savefig(f"{OUT}/fig1_det.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: the 40-cell forest plot
cells = []
for corpus, (lab, files) in CORPUS.items():
    for d, f in files.items():
        S = load(f)
        if not S:
            continue
        groups = {"bona": list(S)} if lab is None else {
            "bona": [i for i in S if lab.get(i) == "bonafide"],
            "spoof": [i for i in S if lab.get(i) == "spoof"]}
        for cls, items in groups.items():
            if not items:
                continue
            for a in ("lace", "nolace"):
                v = np.array([S[i][a] - S[i]["osce_off"] for i in items])
                cells.append({"corpus": corpus, "det": DET[d], "cls": cls, "arm": a,
                              "mean": float(v.mean()),
                              "se": float(v.std(ddof=1) / np.sqrt(len(v))), "n": len(v)})
json.dump(cells, open(f"{OUT}/fig2_cells.json", "w"), indent=1)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, arm in zip(axes, ("lace", "nolace")):
    sub = [c for c in cells if c["arm"] == arm]
    sub.sort(key=lambda c: (c["corpus"], c["det"], c["cls"]))
    y = np.arange(len(sub))
    m = np.array([c["mean"] for c in sub]); e = 1.96 * np.array([c["se"] for c in sub])
    col = ["C0" if c["cls"] == "bona" else "C1" for c in sub]
    ax.errorbar(m, y, xerr=e, fmt="none", ecolor="0.5", elinewidth=0.8, capsize=1.5)
    ax.scatter(m, y, s=9, c=col, zorder=3)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c['corpus'][:4]} {c['det']} {c['cls']}" for c in sub], fontsize=5.5)
    ax.set_xlabel("mean score shift vs. OSCE off (logits)")
    ax.set_title(PRETTY[arm], fontsize=7)
    ax.grid(True, axis="x", lw=0.3, alpha=0.4)
axes[0].invert_yaxis()
fig.savefig(f"{OUT}/fig2_forest.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: realised payload
src = f"{R}/primary_rate_24k_rand600.json"
if os.path.exists(src):
    d = json.load(open(src))
    losses = sorted(int(k[4:]) for k in d[0] if k.startswith("loss"))
    mean = [np.mean([r[f"loss{L}"]["frame_kbps"] for r in d]) for L in losses]
    sd = [np.std([r[f"loss{L}"]["frame_kbps"] for r in d], ddof=1) for L in losses]
    fig, ax = plt.subplots(figsize=(2.6, 1.9))
    ax.errorbar(losses, mean, yerr=sd, fmt="o-", ms=3, color="C3", capsize=2)
    ax.axhline(24.0, ls=":", color="0.4", lw=0.8)
    ax.text(0.4, 24.2, "nominal 24 kb/s", fontsize=5.5, color="0.4")
    ax.set_xlabel("encoder's declared packet loss (%)")
    ax.set_ylabel("realised SILK payload (kb/s)")
    ax.set_ylim(10, 25.5)
    ax.grid(True, lw=0.3, alpha=0.4)
    json.dump({"losses": losses, "mean_kbps": mean, "sd_kbps": sd, "n_items": len(d)},
              open(f"{OUT}/fig3_rate.json", "w"), indent=1)
    fig.savefig(f"{OUT}/fig3_rate.pdf")
    plt.close(fig)
    print(f"fig3: {len(d)} items, {[f'{L}%:{m:.2f}' for L, m in zip(losses, mean)]}")
else:
    print(f"fig3 SKIPPED: {src} not present yet")

print(f"table1: {len(rows)} rows -> table1.tex, table1.json")
print(f"fig2: {len(cells)} cells, "
      f"{sum(1 for c in cells if c['arm']=='nolace' and c['mean']<0)}/"
      f"{sum(1 for c in cells if c['arm']=='nolace')} NoLACE negative, "
      f"{sum(1 for c in cells if c['arm']=='lace' and c['mean']<0)}/"
      f"{sum(1 for c in cells if c['arm']=='lace')} LACE negative")


# ---------------------------------------------------------------- Table 2: held-out threshold transfer
# Three columns, not two. The comparator a deployer actually starts from is the unenhanced rendering judged
# with its own held-out threshold ("matched"), not a nominal 1% that even that control does not hit: a 1%
# quantile of ~500-700 calibration items has a five-to-seven-item resolution. Against the matched column the
# direction is uniform; against the nominal target it looked checkpoint-dependent, which was an artefact.
rec_path = f"{OUT}/recalibration.json"
if os.path.exists(rec_path):
    rec = json.load(open(rec_path))

    def cell(corpus, det, audio, thr):
        r = [x for x in rec if x["corpus"] == corpus and x["detector"] == det
             and x["audio"] == audio and x["threshold_from"] == thr]
        return r[0] if r else None

    rows2 = []
    for corpus in ("RTCFake", "ASVspoof"):
        for det in ("T1", "T2", "P1", "P2"):
            m, s_, r_ = (cell(corpus, det, "osce_off", "osce_off"),
                         cell(corpus, det, "nolace", "osce_off"),
                         cell(corpus, det, "nolace", "nolace"))
            if not (m and s_ and r_):
                continue
            rows2.append({"corpus": corpus, "detector": det,
                          "fa_matched": m["fa"], "fa_stale": s_["fa"], "fa_refit": r_["fa"],
                          "miss_matched": m["miss"], "miss_stale": s_["miss"], "miss_refit": r_["miss"],
                          "fa_items_matched": m["fa_items"], "fa_items_stale": s_["fa_items"],
                          "fa_ratio_vs_matched": (s_["fa"] / m["fa"]) if m["fa"] else None,
                          "miss_delta_refit_minus_matched": (r_["miss"] - m["miss"]) * 100})
    json.dump(rows2, open(f"{OUT}/table2.json", "w"), indent=1)

    with open(f"{OUT}/table2.tex", "w") as fh:
        fh.write("\\begin{tabular}{llcccccc}\n\\toprule\n")
        fh.write("& & \\multicolumn{3}{c}{FA (\\%)} & \\multicolumn{3}{c}{miss (\\%)} \\\\\n"
                 "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n")
        fh.write("corpus & det. & matched & stale & refit & matched & stale & refit \\\\\n\\midrule\n")
        last = None
        for r in rows2:
            c = r["corpus"] if r["corpus"] != last else ""
            last = r["corpus"]
            if (r["corpus"], r["detector"]) in WITHHELD:   # changes against the matched cell, not levels
                mm = ("--", signed((r['miss_stale']-r['miss_matched'])*100),
                      signed((r['miss_refit']-r['miss_matched'])*100))
            else:
                mm = (f"{r['miss_matched']*100:.2f}", f"{r['miss_stale']*100:.2f}",
                      f"{r['miss_refit']*100:.2f}")
            fh.write(f"{c} & {r['detector']} & {r['fa_matched']*100:.2f} & "
                     f"\\textbf{{{r['fa_stale']*100:.2f}}} & {r['fa_refit']*100:.2f} & "
                     f"{mm[0]} & {mm[1]} & {mm[2]} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    up = sum(1 for r in rows2 if r["fa_stale"] > r["fa_matched"])
    print(f"table2: {len(rows2)} rows; FA rises vs matched in {up}/{len(rows2)}")
    print("  miss delta refit-minus-matched:",
          [f"{r['corpus'][:4]}/{r['detector']} {r['miss_delta_refit_minus_matched']:+.2f}" for r in rows2])
else:
    print("table2 SKIPPED: run recalibration.py first")
