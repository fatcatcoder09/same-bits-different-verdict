"""Decode one Opus bitstream (opus_demo -e format) under a loss mask with a chosen concealment arm.

Arms: classic (decoder complexity 4, libopus PLC), deepplc (complexity 5, PLC only - DRED never parsed),
dred (complexity 5, DRED decoded whenever the packet carries any, which is the gate opus_demo.c:1148 uses),
fec_plc and fec_dred (complexity 5, in-band FEC on, without and with DRED).
opus_demo cannot provide a Deep-PLC-only arm on a DRED bitstream: it parses DRED itself, independent of
OPUS_SET_IGNORE_EXTENSIONS.

Two deviations from the reference loop, both deliberate:
  - `max_dred_samples` is `fs` rather than opus_demo.c:1137's `IMIN(sampling_rate, lost_count*output_samples)`.
    Capping it at the gap length made a single-packet loss unreachable, and single losses are half of all gaps
    under a mean-burst-2 mask; asking for the whole window is an upper bound on what the library will give.
  - The `fec_plc` and `fec_dred` arms add the reference in-band-FEC path: when the packet that ends a gap
    carries an LBRR copy, the frame immediately before it is decoded from that copy, in preference to DRED,
    exactly as opus_demo.c:1143-1146 does. The other arms have no LBRR to use because their encodes do not
    pass `-inbandfec`.

`covered` is descriptive, not an endpoint. Measured against ground truth (decoder state cloned at each
concealed frame, both paths run from it, 2,042 frames): precision 0.990, recall 0.951. Every miss is the
5 ms overlap-add bleed at `offset - frame - avail == 80`, which `offset - frame - 80 <= avail` would recover
at no cost to precision. That change is deliberately NOT applied: the renders on disk were produced with the
predicate below and the counter must keep matching them. The exact indicator of applied DRED is
`crc(dred) != crc(deepplc)`, which is free because both arms are rendered.
"""
import ctypes, struct, glob, os, numpy as np


def _libopus():
    """OSCE is a non-default build option, so this must be a libopus built with --enable-osce, not the
    system one. LIBOPUS names it; otherwise fall back to the in-tree build."""
    if os.environ.get("LIBOPUS"):
        return os.environ["LIBOPUS"]
    here = os.path.dirname(os.path.abspath(__file__))
    hits = sorted(glob.glob(os.path.join(here, "../../../side_exp/opus-build/opus-1.6.1/.libs/libopus.so.*")))
    if hits:
        return hits[-1]
    raise SystemExit("no libopus found: set LIBOPUS to a libopus.so built with --enable-osce "
                     "--enable-deep-plc --enable-dred. The system libopus will not do -- a stock build "
                     "has no OSCE, so the contrast this code measures does not exist in it.")


LIB = ctypes.CDLL(_libopus())
for f in ("opus_decoder_create", "opus_dred_decoder_create", "opus_dred_alloc"): getattr(LIB, f).restype = ctypes.c_void_p
LIB.opus_packet_has_lbrr.restype = ctypes.c_int
OPUS_SET_COMPLEXITY = 4010

def read_bits(path):
    b, pos, pkts = open(path, "rb").read(), 0, []
    while pos + 8 <= len(b):
        n = struct.unpack(">I", b[pos:pos + 4])[0]; pkts.append(b[pos + 8:pos + 8 + n]); pos += 8 + n
    return pkts

def decode(pkts, lost, arm, fs=16000, frame=320, complexity=None):
    err = ctypes.c_int(0)
    dec = ctypes.c_void_p(LIB.opus_decoder_create(fs, 1, ctypes.byref(err))); assert err.value == 0
    use_lbrr = arm.startswith("fec")          # in-band FEC: recover the frame adjacent to the next received
    use_dred = arm in ("dred", "fec_dred")     # packet from its LBRR copy, as opus_demo.c:1143 does
    cx = complexity if complexity is not None else (4 if arm == "classic" else 5)
    assert LIB.opus_decoder_ctl(dec, OPUS_SET_COMPLEXITY, ctypes.c_int(cx)) == 0
    dd = ctypes.c_void_p(LIB.opus_dred_decoder_create(ctypes.byref(err))); assert err.value == 0, err.value
    dr = ctypes.c_void_p(LIB.opus_dred_alloc(ctypes.byref(err))); assert err.value == 0, err.value
    out, pcm, pending, covered, lbrr_used = [], (ctypes.c_int16 * frame)(), 0, 0, 0
    try:
        for i, p in enumerate(pkts):
            if lost[i]: pending += 1; continue
            buf = ctypes.create_string_buffer(p, len(p))
            avail, end = 0, ctypes.c_int(0)
            if pending and use_dred:
                # Ask for the whole window the library is willing to give, not just the gap: capping
                # max_dred_samples at the gap length made a single-packet loss unreachable.
                avail = LIB.opus_dred_parse(dd, dr, buf, len(p), fs, fs, ctypes.byref(end), 0)
                if avail < 0: raise RuntimeError(f"opus_dred_parse returned {avail}")
            has_lbrr = bool(LIB.opus_packet_has_lbrr(buf, len(p))) if (pending and use_lbrr) else False
            for k in range(pending):                             # conceal the gap, oldest frame first
                offset = (pending - k) * frame
                if has_lbrr and k == pending - 1:
                    # Only the frame immediately before this packet has an LBRR copy in it, and the reference
                    # loop prefers that copy over DRED for exactly that frame (opus_demo.c:1143-1146).
                    n = LIB.opus_decode(dec, buf, len(p), pcm, frame, 1)
                    lbrr_used += 1
                elif use_dred and avail > 0:                     # the gate opus_demo itself uses
                    n = LIB.opus_decoder_dred_decode(dec, dr, offset, pcm, frame)
                    # opus_dred_parse returns the far edge of the DRED window and writes the near edge into
                    # `end`; the redundancy spans [end, avail) samples back. Count this frame only when that
                    # span actually reaches it, so the counter is applied coverage and not attempts.
                    covered += int(end.value < offset and offset - frame < avail)
                else:
                    n = LIB.opus_decode(dec, None, 0, pcm, frame, 0)
                if n < 0: raise RuntimeError(f"conceal decode returned {n}")
                out.append(np.frombuffer(pcm, np.int16)[:n].copy())
            pending = 0
            n = LIB.opus_decode(dec, buf, len(p), pcm, frame, 0)
            if n < 0: raise RuntimeError(f"decode returned {n}")
            out.append(np.frombuffer(pcm, np.int16)[:n].copy())
        for _ in range(pending):
            n = LIB.opus_decode(dec, None, 0, pcm, frame, 0)
            if n < 0: raise RuntimeError(f"tail conceal returned {n}")
            out.append(np.frombuffer(pcm, np.int16)[:n].copy())
    finally:
        LIB.opus_dred_free(dr); LIB.opus_dred_decoder_destroy(dd); LIB.opus_decoder_destroy(dec)
    return np.concatenate(out), covered, lbrr_used
