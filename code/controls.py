"""The two control results the paper states in Section II, recomputed from the renders and the scores and
written to `controls.json` so that both numbers have an artefact behind them.

  bit_identity     complexity 4 and complexity 5 decode identically wherever no packet is lost, which is what
                   makes the OSCE contrast single-variable: the deep-PLC gate at >= 5 is inert here.
  encoder_loss     the one encoder-side variable that could confound the contrast, measured with rate control and
                   realised payload matched: SILK VBR at 14,800 bps, no DRED, `-loss 0` against `-loss 20`.
"""
import json, os, sys, collections, numpy as np

R = os.environ.get("RTC_PAPER_DATA", "/data/audio-deepfake/RTC-SDD-PAPER")
OUT = os.path.dirname(os.path.abspath(__file__))
print(f"reading scores from {R}", file=sys.stderr)
RT = {l.split()[0]: l.split()[1] for l in open(os.environ.get("RTCFAKE_DEV_LABEL",
                                 "/home/shigang/RTC-SDD-SIDE/data/rtcfake/dev_label.txt"))
      if l.startswith("offline/")}


def eer(bona, spoof):
    b, s = np.sort(np.asarray(bona, float)), np.sort(np.asarray(spoof, float))
    ts = np.unique(np.concatenate([b, s]))
    fa = np.searchsorted(b, ts, "left") / len(b)
    miss = (len(s) - np.searchsorted(s, ts, "left")) / len(s)
    k = int(np.argmin(np.abs(fa - miss)))
    return float((fa[k] + miss[k]) / 2)


# ---------------------------------------------------------------- bit identity, complexity 4 vs 5
identity = {}
for name, manifest in (("RTCFake", "full.jsonl"), ("ASVspoof", "asv.jsonl"), ("FLEURS", "fleurs.jsonl")):
    crc = {}
    for line in open(f"{R}/renders/{manifest}"):
        r = json.loads(line)
        for x in r.get("rows", [r]):
            if x.get("arm") in ("classic", "deepplc") and "crc" in x:
                crc[(x["arm"], x.get("loss_pct"), x["item"], x.get("lost"))] = x["crc"]
    # every mask instance in which nothing was lost: the two complexities must agree there
    loss_free = [(p, i) for (a, p, i, lost) in crc if a == "classic" and lost == 0]
    same = sum(1 for p, i in loss_free
               if crc.get(("classic", p, i, 0)) == crc.get(("deepplc", p, i, 0)))
    identity[name] = {"loss_free_instances": len(loss_free), "bit_identical": same}
identity["total"] = {"loss_free_instances": sum(v["loss_free_instances"] for v in identity.values()),
                     "bit_identical": sum(v["bit_identical"] for v in identity.values())}

# ---------------------------------------------------------------- encoder declared loss at matched rate
S = collections.defaultdict(dict)
for line in open(f"{R}/scores_t1_review_vbr.jsonl"):
    r = json.loads(line)
    S[r["item"]][r["arm"]] = r["score"]
S = {i: v for i, v in S.items() if len(v) == 2 and i in RT}
bo = [i for i in S if RT[i] == "bonafide"]
sp = [i for i in S if RT[i] == "spoof"]
e0 = eer([S[i]["vbr_l0"] for i in bo], [S[i]["vbr_l0"] for i in sp])
e20 = eer([S[i]["vbr_l20"] for i in bo], [S[i]["vbr_l20"] for i in sp])
rng = np.random.default_rng(0)
d = []
for _ in range(2000):
    bb, ss = rng.choice(bo, len(bo), True), rng.choice(sp, len(sp), True)
    d.append(eer([S[i]["vbr_l0"] for i in bb], [S[i]["vbr_l0"] for i in ss])
             - eer([S[i]["vbr_l20"] for i in bb], [S[i]["vbr_l20"] for i in ss]))
d = np.array(d) * 100
enc = {"detector": "T1", "corpus": "RTCFake", "n_items": len(S), "n_bona": len(bo), "n_spoof": len(sp),
       "condition": "SILK VBR, 14800 bps nominal, no DRED, zero mask, decoder complexity 4",
       "eer_loss0": e0, "eer_loss20": e20, "delta_eer_pp": 100 * (e0 - e20),
       "delta_ci95_pp": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
       "shift_bona": float(np.mean([S[i]["vbr_l0"] - S[i]["vbr_l20"] for i in bo])),
       "shift_spoof": float(np.mean([S[i]["vbr_l0"] - S[i]["vbr_l20"] for i in sp]))}

json.dump({"bit_identity": identity, "encoder_declared_loss": enc}, open(f"{OUT}/controls.json", "w"), indent=1)

print("bit identity, complexity 4 vs 5 on loss-free mask instances:")
for k, v in identity.items():
    print(f"  {k:9s} {v['bit_identical']}/{v['loss_free_instances']}")
print(f"\nencoder declared loss at matched rate and rate-control mode, T1, n={enc['n_items']}:")
print(f"  EER {enc['eer_loss0']*100:.2f}% (-loss 0) vs {enc['eer_loss20']*100:.2f}% (-loss 20)")
print(f"  delta {enc['delta_eer_pp']:+.2f} pp  95% [{enc['delta_ci95_pp'][0]:+.2f}, {enc['delta_ci95_pp'][1]:+.2f}]")
