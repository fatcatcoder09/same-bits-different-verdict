"""Declaration A': render every dev-offline item through the concealment ladder and the OSCE contrast.

Per item and per loss rate P in {2,5,10,20}%: ONE encode (VoIP, SILK WB, 24 kb/s CBR, 20 ms, expected loss = P,
DRED 1 s), then four decodes of that one bitstream - noloss reference, classic PLC (complexity 4), Deep PLC (5),
DRED (5, DRED used when it covers the gap) - under a Gilbert-Elliott mask seeded from the item id.
Plus an OSCE contrast: one encode with no loss and no DRED, decoded at complexity 4 / 6 (LACE) / 7 (NoLACE).
Outputs are FLAC; a JSONL manifest records the mask, the DRED coverage and a checksum of every rendering.
"""
import os, sys, json, hashlib, subprocess, argparse, zlib, numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_arms import read_bits, decode, LIB  # noqa: F401
ROOT = os.environ.get("SIDE_ROOT", "/home/shigang/RTC-SDD-SIDE")
DEMO = os.environ.get("OPUS_DEMO", f"{ROOT}/side_exp/opus-build/opus-1.6.1/opus_demo")
WAV = f"{ROOT}/data/rtcfake/wav/dev"; RATES = (2, 5, 10, 20); FR = 320
# The encoder only spends bits on DRED when it is told to expect loss: opus_encoder.c:700-704 switches
# dred_frac at packetLossPercentage > 5 and opus_encoder.c:723-724 zeroes the DRED bitrate when fewer than two
# chunks fit, so at -loss 2 and 5 almost no packet carries redundancy and the `dred` arm is the Deep PLC arm.
# ENC_LOSS_R decouples the encoder's expected loss from the mask's actual loss, which turns "is redundancy
# present" into a controlled factor crossed with the four mask rates instead of a confound riding on them.
ENC_LOSS_R = 20
# In-band FEC suppresses DRED, and how much depends on the rate-control mode. Measured on this build over 25
# items at 20% declared loss, share of packets carrying DRED, FEC off -> on:
#   CBR  24 kb/s 79.0% -> 1.1%   32 kb/s 99.6% -> 1.3%   40 kb/s 99.7% -> 93.4%   48 kb/s 99.7% -> 99.7%
#   VBR  24 kb/s 99.4% -> 0.0%   32 kb/s 99.7% -> 85.8%  40 kb/s 99.7% -> 99.7%   48 kb/s 99.7% -> 99.7%
# opus_encoder.c:696-698 raises bitrate_offset from 12000 to 20000 when useInBandFEC is set. The target_chunks
# clamp at opus_encoder.c:2610 is guarded by `if (st->use_vbr)`, so the "fewer than two chunks" gate suppresses
# DRED in VBR only; in CBR, DRED is emitted opportunistically and what crowds it out is LBRR taking the packet
# room, which is gradual and content-dependent. Suppression is therefore near-total rather than total below the
# crossover, and the crossover sits between 24 and 32 kb/s in VBR, higher in CBR.
# These renders are CBR at 24 kb/s, where DRED is effectively absent once FEC is on, so the FEC question has to
# be asked twice: once at the declared 24 kb/s, where deployment means LBRR without DRED, and once at 48 kb/s,
# the lowest rate where both coexist. The 48 kb/s block carries its own FEC-off control so rate is not a confound.
BITRATE_HI = 48000
# Rate-matched control for the declared-loss contrast. At nominal 24 kb/s CBR the realised primary layer is
# 22.48 kb/s under `-loss 0/2/5` and 13.19 kb/s under `-loss 20`, the rest being DRED padding
# (measured separately over 40 items; that script is not part of the published kit). 14,500 bps with `-loss 0` realises 13.20 kb/s of primary, so this arm
# holds the primary rate of the `-loss 20` arm while the encoder never expects loss - which separates "the
# declared loss rate starves the primary layer" from "the declared loss rate changes the coding itself".
# Overridable so the match can be moved. Corpus-wide the `-loss 20` arm realises 13.94 kb/s on a seeded
# random 600-item sample; the 300-item slice that gave 13.53 was the first 300 offline entries, which are
# all English bona fide, so any bracket built on it does not hold.
BITRATE_C1 = int(os.environ.get("BITRATE_C1", 14500))
ONLY = "all"

def ge_mask(n, rate, seed, mean_burst=2.0):
    """Gilbert-Elliott: mean burst length `mean_burst`, average loss `rate`."""
    rng = np.random.default_rng(seed); p_bg = (rate / 100) / (1 - rate / 100) / mean_burst; p_gb = 1 / mean_burst
    m, bad = np.zeros(n, np.int8), False
    for i in range(n):
        bad = (rng.random() > p_gb) if bad else (rng.random() < p_bg)
        m[i] = int(bad)
    return m

def one(item, outdir):
    seed = int(hashlib.sha256(item.encode()).hexdigest()[:8], 16)
    x, sr = sf.read(f"{WAV}/{item}", dtype="int16")
    if sr != 16000 or x.ndim != 1: return {"item": item, "error": f"sr={sr} ndim={x.ndim}"}
    x = x[: (len(x) // FR) * FR]
    if len(x) < 10 * FR: return {"item": item, "error": "too short"}
    base = os.path.join(outdir, item.replace("/", "__")[:-4]); os.makedirs(os.path.dirname(base), exist_ok=True)
    raw = base + ".raw"; x.tofile(raw); rows = []
    enc = [DEMO, "-e", "voip", "16000", "1", "24000", "-cbr", "-bandwidth", "WB", "-framesize", "20"]
    try:
        if ONLY == "fec48":
            # 2x2 at 48 kb/s: {in-band FEC off, on} x {DRED off, on}, the only regime where both fit.
            for fec in (False, True):
                for dred in (False, True):
                    tag = f"{'f' if fec else 'n'}48_{'dred' if dred else 'plc'}"
                    bitx = f"{base}.{tag}.bit"
                    cmd = [DEMO, "-e", "voip", "16000", "1", str(BITRATE_HI), "-cbr", "-bandwidth", "WB",
                           "-framesize", "20", "-loss", str(ENC_LOSS_R)]
                    if dred: cmd += ["-dred", "100"]
                    if fec: cmd += ["-inbandfec"]
                    subprocess.run(cmd + [raw, bitx], check=True, capture_output=True)
                    pkx = read_bits(bitx)
                    arm = ("fec_dred" if dred else "fec_plc") if fec else ("dred" if dred else "deepplc")
                    for P in RATES:
                        m = ge_mask(len(pkx), P, seed + P)
                        y, cov, lbrr = decode(pkx, m, arm, complexity=5)
                        f = f"{base}.p{P}.{tag}.flac"; sf.write(f, y, 16000)
                        rows.append({"item": item, "arm": tag, "loss_pct": P, "lost": int(m.sum()),
                                     "packets": len(pkx), "bitrate": BITRATE_HI, "inbandfec": fec,
                                     "dred_enabled": dred, "dred_covered": cov, "lbrr_used": lbrr,
                                     "crc": zlib.crc32(y.tobytes()), "file": f})
                    os.remove(bitx)
            return {"item": item, "rows": rows}
        if ONLY == "c1":
            # Zero mask and complexity 4, exactly as the `noloss` arms are decoded, so the only thing that
            # differs from `noloss` at 24 kb/s is the nominal rate and the declared loss it was reached by.
            bitc = f"{base}.c1.bit"
            subprocess.run([DEMO, "-e", "voip", "16000", "1", str(BITRATE_C1), "-cbr", "-bandwidth", "WB",
                            "-framesize", "20", "-loss", "0", "-dred", "100", raw, bitc],
                           check=True, capture_output=True)
            pkc = read_bits(bitc)
            y, cov, _ = decode(pkc, np.zeros(len(pkc), np.int8), "classic", complexity=4)
            f = f"{base}.c1.flac"; sf.write(f, y, 16000)
            rows.append({"item": item, "arm": "c1_ratematched", "loss_pct": 0, "lost": 0, "packets": len(pkc),
                         "bitrate": BITRATE_C1, "enc_loss": 0, "dred_covered": cov,
                         "crc": zlib.crc32(y.tobytes()), "file": f})
            os.remove(bitc)
            return {"item": item, "rows": rows}
        if ONLY == "fec":
            # One bitstream with in-band FEC on as well as DRED, which is what a deployed RTC decoder at this
            # expected loss would send. opus_encoder.c:696-698 switches the DRED allocation to a different
            # formula when useInBandFEC is set, so this is a third encoder condition and not a decoder switch.
            bitf = f"{base}.f.bit"
            subprocess.run(enc + ["-loss", str(ENC_LOSS_R), "-dred", "100", "-inbandfec", raw, bitf],
                           check=True, capture_output=True)
            pkf = read_bits(bitf)
            for P in RATES:
                m = ge_mask(len(pkf), P, seed + P)           # the same mask as every other arm at this rate
                for arm in ("fec_plc", "fec_dred"):
                    y, cov, lbrr = decode(pkf, m, arm, complexity=5)
                    f = f"{base}.p{P}.{arm}.flac"; sf.write(f, y, 16000)
                    rows.append({"item": item, "arm": arm, "loss_pct": P, "lost": int(m.sum()),
                                 "packets": len(pkf), "enc_loss": ENC_LOSS_R, "inbandfec": True,
                                 "dred_covered": cov, "lbrr_used": lbrr,
                                 "crc": zlib.crc32(y.tobytes()), "file": f})
            os.remove(bitf)
            return {"item": item, "rows": rows}
        if ONLY == "redundancy":
            # One bitstream that really carries DRED, decoded under each mask twice: the only difference
            # between the two arms is whether the decoder uses the redundancy that is present in both.
            bitr = f"{base}.r.bit"
            subprocess.run(enc + ["-loss", str(ENC_LOSS_R), "-dred", "100", raw, bitr], check=True, capture_output=True)
            pkr = read_bits(bitr)
            for P in RATES:
                m = ge_mask(len(pkr), P, seed + P)               # same mask as the matching p{P} arms
                for arm, name in (("deepplc", "r_deepplc"), ("dred", "r_dred")):
                    y, cov, _ = decode(pkr, m, arm, complexity=5)
                    f = f"{base}.p{P}.{name}.flac"; sf.write(f, y, 16000)
                    rows.append({"item": item, "arm": name, "loss_pct": P, "lost": int(m.sum()),
                                 "packets": len(pkr), "enc_loss": ENC_LOSS_R, "dred_covered": cov,
                                 "crc": zlib.crc32(y.tobytes()), "file": f})
            os.remove(bitr)
            return {"item": item, "rows": rows}
        if ONLY != "osce":
            for P in RATES:
                bit = f"{base}.p{P}.bit"
                subprocess.run(enc + ["-loss", str(P), "-dred", "100", raw, bit], check=True, capture_output=True)
                pk = read_bits(bit); mask = ge_mask(len(pk), P, seed + P)
                for arm, cx, m in (("noloss", 4, np.zeros(len(pk), np.int8)), ("classic", 4, mask), ("deepplc", 5, mask), ("dred", 5, mask)):
                    y, cov, _ = decode(pk, m, arm, complexity=cx)
                    f = f"{base}.p{P}.{arm}.flac"; sf.write(f, y, 16000)
                    rows.append({"item": item, "arm": arm, "loss_pct": P, "lost": int(m.sum()), "packets": len(pk),
                                 "dred_covered": cov, "crc": zlib.crc32(y.tobytes()), "file": f})
                os.remove(bit)
        bit0 = f"{base}.osce.bit"
        subprocess.run(enc + ["-loss", "0", raw, bit0], check=True, capture_output=True)
        pk0 = read_bits(bit0); z = np.zeros(len(pk0), np.int8)
        for arm, cx in (("osce_off", 4), ("lace", 6), ("nolace", 7)):
            y, _, _ = decode(pk0, z, "classic", complexity=cx)
            f = f"{base}.{arm}.flac"; sf.write(f, y, 16000)
            rows.append({"item": item, "arm": arm, "loss_pct": 0, "lost": 0, "packets": len(pk0), "dred_covered": 0,
                         "crc": zlib.crc32(y.tobytes()), "file": f})
        os.remove(bit0)
    finally:
        if os.path.exists(raw): os.remove(raw)
    return {"item": item, "rows": rows}

_OUT = None
def _init(out, only=None, wav=None):
    """Worker setup. A forked worker inherits ONLY and WAV; one started by forkserver or spawn does not,
    and would silently fall back to the defaults, so they are passed explicitly."""
    global _OUT, ONLY, WAV
    _OUT = out
    if only is not None:
        ONLY = only
    if wav is not None:
        WAV = wav
def _work(item):
    return one(item, _OUT)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=14); ap.add_argument("--manifest", required=True)
    ap.add_argument("--list", default=""); ap.add_argument("--wav_root", default="")   # second corpus (e.g. FLEURS)
    ap.add_argument("--only", default="all", choices=["all", "osce", "redundancy", "fec", "fec48", "c1"])
    a = ap.parse_args()
    globals()["ONLY"] = a.only
    if a.wav_root:
        globals()["WAV"] = a.wav_root
    items = ([l.strip() for l in open(a.list) if l.strip()] if a.list else
             [l.split()[0] for l in open(os.environ.get("RTCFAKE_DEV_LABEL", f"{ROOT}/data/rtcfake/dev_label.txt")) if l.startswith("offline/")])
    if a.limit: items = items[: a.limit]
    done = set()
    if os.path.exists(a.manifest):
        done = {json.loads(l)["item"] for l in open(a.manifest) if l.strip()}
    items = [i for i in items if i not in done]
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(items)} items to render, {len(done)} already done, {a.jobs} workers", flush=True)
    from multiprocessing import Pool
    errs = 0
    with Pool(a.jobs, initializer=_init, initargs=(a.out, ONLY, WAV)) as pool, open(a.manifest, "a") as mf:
        for k, r in enumerate(pool.imap_unordered(_work, items, chunksize=4), 1):
            mf.write(json.dumps(r) + "\n"); mf.flush(); errs += "error" in r
            if k % 200 == 0: print(f"  {k}/{len(items)} errors={errs}", flush=True)
    print(f"done, errors={errs}", flush=True)
