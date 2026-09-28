# Same Bits, Different Verdict --- reproduction kit

Holding an Opus bitstream byte-identical and changing only the receiver's decoder complexity --- which
selects a neural post-filter the sender cannot observe and no protocol records --- shifts speech deepfake
detector scores toward "synthetic" in 20 of 20 cells, and raises the false-alarm rate at a transferred 1 %
operating point in 8 of 8, by 1.7 to 6.0 times in the seven cells where the calibrated rate is non-zero.
The transmitted file does not determine the decision.

## What this kit does and does not do

**It rebuilds both of the paper's tables from released aggregates, with no corpus and no network.** That
path is the point of the kit and it is tested: standard library only, bare interpreter, empty environment.

**Half of it reproduces from audio; half cannot.** P1 and P2 are public checkpoints that download
themselves, so `score_renders.py` will reproduce their four rows of each table from the audio. T1 and T2
are trained on an access-gated corpus and are entries in a running challenge; their loader is replaced
here by a refusal that says so. The render harness and the analysis scripts are complete either way.

Scores match the paper's bit for bit on a GPU. On CPU they differ in the third decimal place -- device
numerics, not a broken run; the direction and the size of the effect are unchanged.

So: `code/render_arms.py` runs end to end if you hold the audio and an OSCE-enabled libopus.
`code/rebuild_tables.py` runs on nothing but this repository. `make_figures.py`, `recalibration.py` and
`controls.py` read per-utterance score files that are not here; they are the arithmetic, open to inspection,
and they will run against score files of your own in the same layout.

```
code/     rebuild_tables.py                regenerates both LaTeX tables from results/ alone
          render_arms.py, decode_arms.py   the render harness; `--only osce` is the paper's contrast
          score_renders.py                 scores renders with P1 or P2 (public checkpoints)
          make_figures.py                  Figure 1, the 40 direction cells and Table I (needs scores)
          recalibration.py                 the held-out 2x2 threshold-transfer experiment (needs scores)
          controls.py                      bit-identity check and the encoder-side control (needs renders)
results/  table1.json, table2.json         every cell of both tables
          recalibration.json               the held-out grid with bootstrap intervals
          fig2_cells.json                  the 40 class-mean shift cells
          controls.json                    the bit-identity instances and the encoder-loss null
```

`results/` holds the published copies, redacted as the paper is (see below). They are outputs, not inputs:
the analysis scripts write their JSON next to themselves in `code/`, and only `rebuild_tables.py` reads
`results/`.

## Requirements

`rebuild_tables.py` needs Python 3.8+ and the standard library, and nothing else --- that is the point of
it. `recalibration.py` and `controls.py` need `numpy`; `make_figures.py` also needs `matplotlib` and
`scipy`; `render_arms.py` also needs `soundfile` and a libopus built as below. `score_renders.py` needs
`torch`, `soundfile`, `huggingface_hub` and `fairseq` -- `fairseq` only because that is what the
checkpoints' own model card uses to build the wav2vec 2.0 encoder, and it is the least pleasant install
here. Nothing is vendored; these are declared dependencies of that one script. Compiling the generated
tables needs the `booktabs` LaTeX package.

## Rebuilding the tables

```bash
python3 code/rebuild_tables.py     # results/*.json -> tables/table1.tex, tables/table2.tex
```

Those `.tex` files are the ones the paper includes; diff them against the paper to check every number in
Tables I and II. It creates `tables/` itself and works from any working directory.

## Running the render harness

The decoder post-filters are a **non-default build option** --- a stock libopus has no OSCE at all, so the
contrast cannot be reproduced against a distribution package unless it was built with the flags below
(Debian passes these; Arch sets the meson equivalents; Chromium and Firefox build without them).

```bash
./configure --enable-osce --enable-deep-plc --enable-dred   # libopus 1.6.1
make

export LIBOPUS=/path/to/opus-1.6.1/.libs/libopus.so.0   # the shared object built above
export OPUS_DEMO=/path/to/opus-1.6.1/opus_demo          # the binary built above
export SIDE_ROOT=/path/to/your/checkout                 # holds data/rtcfake/wav/dev
export RTCFAKE_DEV_LABEL=/path/to/dev_label.txt
export RTC_PAPER_DATA=/path/to/your/analysis/tree       # read by the three analysis scripts
export BITRATE_C1=14500                                 # rate for the `--only c1` control render

python3 code/render_arms.py --only osce --out RENDERS --manifest renders.jsonl
```

`--only osce` is the paper's contrast and only that: one encode per item and three decodes, at complexity
4, 6 and 7. The default `--only all` additionally renders a four-rate packet-loss ladder belonging to a
different experiment --- five encodes and nineteen files per item --- so pass `--only osce` unless you want
that too.

`LIBOPUS` must point at a libopus built with the flags above; the system one has no OSCE, so the contrast
this code measures does not exist in it. Nothing here checks that for you beyond a symbol lookup.

`RTC_PAPER_DATA` is the root the analysis scripts read, and they expect more under it than this kit can
give you: `asvspoof/label.txt`, `renders/{full,asv,fleurs}.jsonl`, `scores_*.jsonl` per detector and
corpus, `scores_t1_review_vbr.jsonl` for the encoder-side control, and `primary_rate_24k_rand600.json`.
Three corpora appear there --- RTCFake, ASVspoof 2021 LA and FLEURS, the last used for direction only
because it carries one class.

Our renders were byte-identical across two machines on 29,337 of 29,337 files, under a toolchain this kit
does not pin for you: one `soundfile`/`libsndfile` version, since that is what reads the WAV and writes
the FLAC, mask seeds from `hashlib.sha256` rather than Python's `hash()`, which is salted per process, and

```bash
export NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512BW,AVX512DQ,AVX512VL,AVX512IFMA,AVX512VBMI,AVX512VNNI,AVX512VBMI2,AVX512BITALG,AVX512VPOPCNTDQ"
```

## What is not here, and why

**No audio, no renders, no trained checkpoints, no scoring step.** RTCFake is CC-BY-NC-4.0 and
access-gated to registered teams of the ICASSP 2027 RTC-SDD Challenge; ASVspoof 2021 has its own terms.
Redistributing the audio, or renderings of it, would route around an access gate we are not entitled to
open. The two detectors trained for this work are trained on that corpus and are not released either,
which is why the audio-to-scores step is absent rather than merely undocumented.

**No VBR render path.** Every encode in `render_arms.py` is CBR. `controls.py` also computes an
encoder-declared-loss null from `scores_t1_review_vbr.jsonl`, measured on a SILK VBR encode at 14,800 bps
that this harness does not produce. That null is not a claim in the paper --- the contrast decodes one
bitstream, so the declared loss is identical across the three renderings by construction --- and it is kept
in `controls.json` only because it was measured. Do not confuse it with `BITRATE_C1`, which is a different
control at CBR 14,500.

**Two cells report changes for two of the three quantities.** T1 and T2 on RTCFake are the author's
entries in that challenge and RTCFake dev-offline is its development set, so their equal-error and miss
*levels* are withheld from the published files and the changes are given instead. Their false-alarm rates
are published as levels, in the files and in the paper alike, because a false-alarm rate here is set by the
1 % calibration target rather than by how good the detector is, and it is the quantity the paper's
threshold-transfer claim is about. The redaction is applied to
`results/` after the analysis scripts run, so those scripts still compute the levels --- against your data,
not ours --- and the shipped `recalibration.json` no longer carries the `miss` field that
`make_figures.py` would read from a file of its own making.

## Terms

The code is MIT (`LICENSE`), which covers `code/` only. The files under `results/` are aggregate statistics computed from
access-gated corpora and are published so the paper's tables can be checked; they are offered under
CC-BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/), matching the most restrictive upstream
term. Neither covers the corpora themselves.

## The paper

*Same Bits, Different Verdict: The Decoder in Speech Deepfake Detection.* Under review. A link will be
added here when the paper is published; until then `tables/` is the artefact to diff against, and
`results/tables.sha256` says whether your rebuild matches the tables the paper prints.

## Citing the corpora

RTCFake: Xue et al., *Findings of ACL 2026*, pp. 5763--5775, doi 10.18653/v1/2026.findings-acl.285.
ASVspoof 2021: Liu et al., *IEEE/ACM TASLP* 31:2507--2522, 2023.
FLEURS: Conneau et al., *Proc. IEEE SLT*, 2023, pp. 798--805.
