"""Score every A' rendering with one detector and write per-item logit scores.

Usage: score_renders.py --detector {t1,t2,p1,p2} --manifest full.jsonl --out scores_<det>.jsonl [--limit N]
Score convention throughout the paper line: s = logit(bona fide) - logit(spoof); higher = more bona fide.
Full utterances, never truncated or tiled (declaration v3, common protocol).
"""
import argparse, json, os, sys, numpy as np, torch, soundfile as sf

def load_p(tag):
    from huggingface_hub import PyTorchModelHubMixin
    from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
    class SSL(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.model = Wav2Vec2Model(Wav2Vec2Config(quantize_targets=True, extractor_mode="layer_norm", layer_norm_first=True,
                final_dim=768, latent_temp=(2.0, 0.1, 0.999995), encoder_layerdrop=0.0, dropout_input=0.0, dropout_features=0.0,
                dropout=0.0, encoder_layers=24, encoder_embed_dim=1024, encoder_ffn_embed_dim=4096, encoder_attention_heads=16, conv_bias=True))
        def extract_feat(s, x): return s.model(x, mask=False, features_only=True)["x"]
    class Det(torch.nn.Module, PyTorchModelHubMixin):
        def __init__(s):
            super().__init__(); s.m_ssl = SSL(); s.adap_pool1d = torch.nn.AdaptiveAvgPool1d(1); s.proj_fc = torch.nn.Linear(1024, 2)
        def forward(s, w): return s.proj_fc(s.adap_pool1d(s.m_ssl.extract_feat(w).transpose(1, 2)).squeeze(-1))
    repo = "nii-yamagishilab/wav2vec-large-anti-deepfake" + ("-nda" if tag == "p1" else "")
    m = Det.from_pretrained(repo)
    def prep(x):                                   # official recipe: layer-norm the waveform
        t = torch.from_numpy(x.astype(np.float32))
        return torch.nn.functional.layer_norm(t, t.shape)[None]
    return m, prep, (lambda o: float(o[0, 1] - o[0, 0]))

def load_t(tag, ckpt):
    raise SystemExit(
        "T1 and T2 are not released: they are trained on an access-gated corpus and are the "
        "author's entries in a running challenge. Use --detector p1 or p2 -- those checkpoints "
        "are public and download themselves.")


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--detector", required=True); a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True); a.add_argument("--ckpt", default=""); a.add_argument("--limit", type=int, default=0)
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = a.parse_args()
    m, prep, sc = (load_p(a.detector) if a.detector in ("p1", "p2") else load_t(a.detector, a.ckpt))
    dev = torch.device(a.device); m = m.to(dev).eval()
    # Two manifest shapes exist: the render manifests nest their rows under "rows", the window-crop manifest
    # is one row per line. Reading only the nested shape silently scored nothing and still reported success.
    rows = []
    for l in open(a.manifest):
        if not l.strip(): continue
        rec = json.loads(l)
        rows.extend(rec["rows"] if "rows" in rec else [rec])
    if a.limit: rows = rows[: a.limit]
    done = set()
    if os.path.exists(a.out): done = {json.loads(l)["file"] for l in open(a.out) if l.strip()}
    rows = [r for r in rows if r["file"] not in done]
    print(f"{len(rows)} renderings to score with {a.detector} on {a.device}", flush=True)
    with torch.no_grad(), open(a.out, "a") as f:
        for i, r in enumerate(rows, 1):
            x, sr = sf.read(r["file"], dtype="int16"); assert sr == 16000
            s = sc(m(prep(x / 32768.0).to(dev)))
            # carry every manifest field through, so a join key the manifest defines (e.g. `window`) survives
            out = {k: v for k, v in r.items() if k != "score"}; out["score"] = s
            f.write(json.dumps(out) + "\n")
            if i % 2000 == 0: f.flush(); print(f"  {i}/{len(rows)}", flush=True)
    n_out = sum(1 for _ in open(a.out)) if os.path.exists(a.out) else 0
    if n_out == 0:
        print(f"FAILED: scored nothing from {a.manifest}"); sys.exit(1)
    print(f"done, {n_out} rows", flush=True)