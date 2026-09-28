"""The 2x2 threshold-transfer experiment: can configuration-specific recalibration undo the effect?

The measured shift is only interesting if it cannot be calibrated away for free. Split each corpus into a
calibration half and a disjoint test half; set a threshold at a 1% false-alarm target on each rendering's
calibration half; then evaluate both thresholds on both renderings of the test half.

  matched    = threshold from the same rendering it is applied to  -> what a recalibrated deployment gets
  mismatched = threshold from the other rendering                  -> what an unchanged deployment gets

Two questions this answers:
  1. does the mismatched cell miss its 1% target, and by how much, on held-out data;
  2. does the matched cell restore the target, and what does it cost in missed spoofs.

Split is by a hash of the item id, so it is deterministic, disjoint and independent of any score.
"""
import json, os, sys, hashlib, collections, numpy as np

R = os.environ.get("RTC_PAPER_DATA", "/data/audio-deepfake/RTC-SDD-PAPER")
OUT = os.path.dirname(os.path.abspath(__file__))
print(f"reading scores from {R}", file=sys.stderr)
TARGET = 0.01
RT = {l.split()[0]: l.split()[1] for l in open(os.environ.get("RTCFAKE_DEV_LABEL",
                                 "/home/shigang/RTC-SDD-SIDE/data/rtcfake/dev_label.txt"))
      if l.startswith("offline/")}
AV = {l.split()[0]: l.split()[-1] for l in open(f"{R}/asvspoof/label.txt")}
CORPUS = {
    "RTCFake": (RT, {"T1": "scores_t1_full.jsonl", "T2": "scores_t2_full.jsonl",
                     "P1": "scores_p1_full.jsonl", "P2": "scores_p2_full_osce.jsonl"}),
    "ASVspoof": (AV, {"T1": "scores_t1_asv.jsonl", "T2": "scores_t2_asv.jsonl",
                      "P1": "scores_p1_asv.jsonl", "P2": "scores_p2_asv_osce.jsonl"}),
}


def load(fname):
    S = collections.defaultdict(dict)
    for line in open(os.path.join(R, fname)):
        r = json.loads(line)
        if r["arm"] in ("osce_off", "nolace"):
            S[r["item"]][r["arm"]] = r["score"]
    return {i: v for i, v in S.items() if len(v) == 2}


def is_cal(item):
    """Deterministic half-split on the item id, independent of any score."""
    return hashlib.sha256(item.encode()).digest()[0] < 128


N_BOOT, BOOT_SEED = 4000, 0


def boot_rate(flags, n=N_BOOT, seed=BOOT_SEED):
    """Percentile interval for a rate, resampling items."""
    a = np.asarray(flags, float)
    r = np.random.default_rng(seed)
    bs = a[r.integers(0, len(a), (n, len(a)))].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


rows = []
for corpus, (lab, files) in CORPUS.items():
    for det, f in files.items():
        S = load(f)
        cal_b = [i for i in S if lab.get(i) == "bonafide" and is_cal(i)]
        tst_b = [i for i in S if lab.get(i) == "bonafide" and not is_cal(i)]
        tst_s = [i for i in S if lab.get(i) == "spoof" and not is_cal(i)]
        thr = {a: float(np.quantile([S[i][a] for i in cal_b], TARGET)) for a in ("osce_off", "nolace")}
        for applied_to in ("osce_off", "nolace"):
            for thr_from in ("osce_off", "nolace"):
                t = thr[thr_from]
                fa = [S[i][applied_to] < t for i in tst_b]
                ms = [S[i][applied_to] >= t for i in tst_s]
                lo, hi = boot_rate(fa)
                rows.append({"corpus": corpus, "detector": det, "audio": applied_to,
                             "threshold_from": thr_from,
                             "matched": applied_to == thr_from,
                             "n_cal_bona": len(cal_b), "n_test_bona": len(tst_b), "n_test_spoof": len(tst_s),
                             "fa": float(np.mean(fa)), "fa_ci": [lo, hi],
                             "fa_ci_n_boot": N_BOOT, "fa_ci_seed": BOOT_SEED,
                             "fa_items": int(np.sum(fa)),
                             "miss": float(np.mean(ms)), "miss_items": int(np.sum(ms))})
json.dump(rows, open(f"{OUT}/recalibration.json", "w"), indent=1)

print(f"Held-out 2x2, target FA = {TARGET*100:.0f}%, threshold fitted on a disjoint calibration half.\n")
print(f"{'corpus':9s}{'det':4s}{'audio':10s}{'thr from':10s}{'FA %':>7s}{'95% CI':>16s}{'items':>7s}{'miss %':>8s}")
for r in rows:
    tag = "  <- matched" if r["matched"] else ""
    print(f"{r['corpus']:9s}{r['detector']:4s}{r['audio']:10s}{r['threshold_from']:10s}"
          f"{r['fa']*100:7.2f}{'[%.2f, %.2f]' % (r['fa_ci'][0]*100, r['fa_ci'][1]*100):>16s}"
          f"{r['fa_items']:7d}{r['miss']*100:8.2f}{tag}")

print("\nThe cell the paper is about: NoLACE audio judged with the OSCE-off threshold (mismatched),")
print("against the same audio judged with its own threshold (matched).")
print(f"{'corpus':9s}{'det':4s}{'FA mismatched':>15s}{'FA matched':>12s}{'miss mismatched':>17s}{'miss matched':>14s}")
for corpus, (_, files) in CORPUS.items():
    for det in files:
        mm = next(r for r in rows if r["corpus"] == corpus and r["detector"] == det
                  and r["audio"] == "nolace" and r["threshold_from"] == "osce_off")
        mt = next(r for r in rows if r["corpus"] == corpus and r["detector"] == det
                  and r["audio"] == "nolace" and r["threshold_from"] == "nolace")
        print(f"{corpus:9s}{det:4s}{mm['fa']*100:15.2f}{mt['fa']*100:12.2f}"
              f"{mm['miss']*100:17.2f}{mt['miss']*100:14.2f}")
