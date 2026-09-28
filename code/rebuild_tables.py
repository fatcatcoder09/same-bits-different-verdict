"""Regenerate both LaTeX tables from the released JSON alone -- no corpus, no score files, no GPU.

`make_figures.py` needs the per-item score files, which are derived from a gated corpus and are not released,
so it cannot be the "reproduce without the data" path even though an earlier README said it was. This script
is that path: it reads `results/table1.json` and `results/table2.json`, which are aggregate statistics, and
writes exactly the `.tex` the paper includes. A reader can diff its output against the paper's tables.

Run from the release root:  python code/rebuild_tables.py
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(os.path.dirname(HERE), "tables")
WITHHELD = {("RTCFake", "T1"), ("RTCFake", "T2")}


CELLS = [("RTCFake", d) for d in ("T1", "T2", "P1", "P2")] + \
        [("ASVspoof", d) for d in ("T1", "T2", "P1", "P2")]
KEYS = {"table1": ("fa_off", "fa_nolace", "fa_mcnemar_p"),
        "table2": ("fa_matched", "fa_stale", "fa_refit")}
LEVELS = {"table1": ("eer_off", "eer_nolace", "miss_off", "miss_nolace"),
          "table2": ("miss_matched", "miss_stale", "miss_refit")}
DELTAS = {"table1": ("delta_eer_pp", "delta_miss_pp"),
          "table2": ("delta_miss_stale_pp", "delta_miss_refit_pp")}
# rate <-> item-count pairs the files already carry; a rate that disagrees with its own numerator is the
# one error a reader cannot see, because both numbers look plausible on their own
PAIRS = {"table1": (("fa_off", "fa_items_off", "n_bona"), ("fa_nolace", "fa_items_nolace", "n_bona"),
                    ("miss_off", "miss_items_off", "n_spoof"), ("miss_nolace", "miss_items_nolace", "n_spoof")),
         "table2": ()}
# where no denominator ships, the two rates in a row still share one, so their order must match their counts'
ORDER = {"table1": (("fa_off", "fa_items_off", "fa_nolace", "fa_items_nolace"),),
         "table2": (("fa_matched", "fa_items_matched", "fa_stale", "fa_items_stale"),)}


def sci(p):
    """1.9e-06 is programmer notation; a table prints $1.9\\times10^{-6}$."""
    m, e = f"{p:.1e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def signed(v):
    """A text-mode hyphen is not a minus sign."""
    return f"${v:+.2f}$"


def check(name, rows):
    """Refuse a file that would still render. A table that is merely wrong is worse than no table:
    the reader diffs it against the paper, sees a difference, and cannot tell which side is at fault."""
    def bad(msg):
        raise SystemExit(f"{name}.json does not describe this paper's experiment: {msg}")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        bad("expected a list of row objects")
    seen = [(r.get("corpus"), r.get("detector")) for r in rows]
    if len(seen) != len(set(seen)):
        bad(f"a cell appears more than once: {sorted(c for c in set(seen) if seen.count(c) > 1)}")
    if sorted(seen) != sorted(CELLS):
        bad(f"cells are {sorted(seen)}, expected the 8 of {sorted(CELLS)}")
    for r in rows:
        cell = (r["corpus"], r["detector"])
        need = KEYS[name] + (DELTAS[name] if cell in WITHHELD else LEVELS[name])
        for k in need:
            if k not in r:
                bad(f"{cell[0]}/{cell[1]} has no {k}")
            v = r[k]
            if not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
                bad(f"{cell[0]}/{cell[1]} has a non-finite {k}: {v!r}")
        for k in KEYS[name] + (() if cell in WITHHELD else LEVELS[name]):
            if not 0.0 <= r[k] <= 1.0:
                bad(f"{cell[0]}/{cell[1]} has {k} = {r[k]}, which is not a rate in [0, 1]")
        # a change is in points, so it cannot exceed 100; this is the only class of number in the file
        # with no level and no count beside it, which is exactly why it needs its own bound
        for k in DELTAS[name]:
            if k in r and not -100.0 <= r[k] <= 100.0:
                bad(f"{cell[0]}/{cell[1]} has {k} = {r[k]}, which is not a change in points")
        for rate, items, denom in PAIRS[name]:
            if rate in r and items in r and denom in r and r[denom]:
                got = r[items] / r[denom]
                if abs(got - r[rate]) > 1e-9:
                    bad(f"{cell[0]}/{cell[1]}: {rate} = {r[rate]} but {items} = {r[items]}, "
                        f"which over {r[denom]} is {got}")
        for ra, ia, rb, ib in ORDER[name]:
            if all(k in r for k in (ra, ia, rb, ib)):
                if (r[ra] - r[rb] > 0) != (r[ia] - r[ib] > 0) or (r[ra] == r[rb]) != (r[ia] == r[ib]):
                    bad(f"{cell[0]}/{cell[1]}: {ra}/{rb} = {r[ra]}/{r[rb]} but {ia}/{ib} = "
                        f"{r[ia]}/{r[ib]}; over one denominator these cannot both be right")
    return sorted(rows, key=lambda r: (r["corpus"] != "RTCFake", r["corpus"], CELLS.index(
        (r["corpus"], r["detector"])) % 4))


def lvl(r, a, b):
    """Levels, or a signed change where the level is withheld (see the paper's Table I caption)."""
    if (r["corpus"], r["detector"]) in WITHHELD:
        key = f"delta_{a.split('_')[0]}_pp"
        if key not in r:
            raise SystemExit(f"{r['corpus']}/{r['detector']} is a withheld cell but carries no {key}. "
                             f"The published results/ files carry it; a file produced by make_figures.py "
                             f"does not, and its absolute levels belong to whoever computed them.")
        return "--", signed(r[key])
    return f"{r[a]*100:.2f}", f"{r[b]*100:.2f}"


def table1(rows):
    L = ["\\begin{tabular}{llccccccc}", "\\toprule",
         "& & \\multicolumn{2}{c}{EER (\\%)} & \\multicolumn{2}{c}{FA (\\%)} "
         "& \\multicolumn{2}{c}{miss (\\%)} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
         "corpus & det. & off & NoLACE & off & NoLACE & off & NoLACE & $p$ \\\\", "\\midrule"]
    last = None
    for r in rows:
        c = r["corpus"] if r["corpus"] != last else ""
        last = r["corpus"]
        e0, e1 = lvl(r, "eer_off", "eer_nolace")
        m0, m1 = lvl(r, "miss_off", "miss_nolace")
        # a withheld row ships no item counts, so it can carry no dagger; say so rather than defaulting
        counts = [r[k] for k in ("miss_items_off", "miss_items_nolace") if k in r]
        thin = "$^{\\dagger}$" if counts and min(counts) < 10 else ""
        L.append(f"{c} & {r['detector']} & {e0} & {e1} & {r['fa_off']*100:.2f} & "
                 f"\\textbf{{{r['fa_nolace']*100:.2f}}} & {m0} & {m1}{thin} & {sci(r['fa_mcnemar_p'])} \\\\")
    return "\n".join(L + ["\\bottomrule", "\\end{tabular}", ""])


def table2(rows):
    L = ["\\begin{tabular}{llcccccc}", "\\toprule",
         "& & \\multicolumn{3}{c}{FA (\\%)} & \\multicolumn{3}{c}{miss (\\%)} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
         "corpus & det. & matched & stale & refit & matched & stale & refit \\\\", "\\midrule"]
    last = None
    for r in rows:
        c = r["corpus"] if r["corpus"] != last else ""
        last = r["corpus"]
        if (r["corpus"], r["detector"]) in WITHHELD:
            mm = ("--", signed(r['delta_miss_stale_pp']), signed(r['delta_miss_refit_pp']))
        else:
            mm = tuple(f"{r[k]*100:.2f}" for k in ("miss_matched", "miss_stale", "miss_refit"))
        L.append(f"{c} & {r['detector']} & {r['fa_matched']*100:.2f} & "
                 f"\\textbf{{{r['fa_stale']*100:.2f}}} & {r['fa_refit']*100:.2f} & "
                 f"{mm[0]} & {mm[1]} & {mm[2]} \\\\")
    return "\n".join(L + ["\\bottomrule", "\\end{tabular}", ""])


if __name__ == "__main__":
    missing = [f for f in ("table1.json", "table2.json") if not os.path.exists(os.path.join(RES, f))]
    if missing:
        sys.exit(f"missing {missing} in {RES}")
    built = {}
    for name, fn in (("table1", table1), ("table2", table2)):
        rows = check(name, json.load(open(os.path.join(RES, f"{name}.json"))))
        built[name] = (fn(rows), len(rows))
    expect = {}
    ref = os.path.join(RES, "tables.sha256")
    if os.path.exists(ref):
        for line in open(ref):
            h, f = line.split()
            expect[f] = h
    os.makedirs(OUT, exist_ok=True)
    for name, (tex, n) in built.items():
        with open(os.path.join(OUT, f"{name}.tex"), "w") as fh:
            fh.write(tex)
        got = hashlib.sha256(tex.encode()).hexdigest()
        want = expect.get(f"{name}.tex")
        verdict = ("" if want is None else
                   "  matches the published table" if got == want else
                   "  DIFFERS from the published table")
        print(f"wrote tables/{name}.tex from results/{name}.json ({n} rows){verdict}")
    if expect and any(hashlib.sha256(t.encode()).hexdigest() != expect.get(f"{k}.tex")
                      for k, (t, _) in built.items()):
        sys.exit("at least one table differs from the one this paper printed; results/ has been altered")
