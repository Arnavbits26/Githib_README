# KV-Cache Inference Benchmark

A from-scratch NumPy implementation of a decoder-only transformer, built
to measure and demonstrate the single most important optimization in LLM
inference serving: **KV caching**.

## Why this project exists

Production LLM serving systems (vLLM, TGI, TensorRT-LLM) all build on the
same core idea: don't recompute what you've already computed. Every new
token only needs attention against the tokens *before* it — if you cache
each layer's keys and values as you go, generating token *N+1* only
requires work proportional to the new token, not the whole sequence again.

Rather than call vLLM as a black box, I implemented the mechanism itself
— causal self-attention, multi-head splitting, and the KV cache — from
scratch, to actually understand what these serving systems are doing
under the hood, then built a benchmark to measure the effect for real.

## What's actually being compared

- **`generate_naive`** — at every generation step, re-run the *entire*
  forward pass over the whole sequence so far. This is what a "just call
  the model in a loop" implementation looks like if you never think
  about caching. Cost grows close to quadratically with sequence length.
- **`generate_kv_cached`** — compute the new token's Q/K/V only, reuse
  cached K/V for every token already seen, and append to the cache. Cost
  per step stays close to constant.

Both produce output from the exact same model and the same random seed
— the only difference is how much redundant work each one does.

## Real results (measured on this machine, single CPU core, no GPU)

```
new tokens |  naive (s) | kv-cache (s) |  speedup |  naive tok/s |   kv tok/s |  naive MB |   kv MB
----------------------------------------------------------------------------------------------------
         8 |      0.092 |        0.042 |    2.20x |         87.0 |      191.1 |     9.96 |   9.25
        16 |      0.168 |        0.080 |    2.11x |         95.4 |      201.0 |    10.64 |   9.38
        32 |      0.421 |        0.155 |    2.72x |         75.9 |      206.3 |    12.01 |  10.07
        64 |      1.003 |        0.296 |    3.39x |         63.8 |      216.1 |    14.77 |  11.45
       128 |      3.133 |        0.620 |    5.06x |         40.9 |      206.6 |    20.26 |  14.19
```

**Batching comparison** (8 independent sequences, 32 new tokens each):

```
Sequential (one at a time): 1.074s (238.3 tok/s)
Batched (all together):     0.399s (640.9 tok/s)
Batching speedup:           2.69x
```

Raw numbers (all runs) are saved to `results/results.json`.

## What the numbers actually show

- **The speedup grows with sequence length** (2.2x → 5.1x from 8 to 128
  new tokens), exactly as expected: naive generation's cost scales close
  to quadratically (recomputing more each step), while KV-cached
  generation's throughput stays roughly flat (~190–215 tok/s) regardless
  of how long the sequence gets. This is the actual mechanism behind why
  serving systems care about caching at all.
- **Naive throughput visibly degrades** as sequences get longer (87 →
  41 tok/s) — this is the real-world reason a naive implementation
  becomes unusable for long generations, not just "slower."
- **Batching alone gave a 2.7x throughput gain** even without any
  dynamic scheduling — just processing 8 sequences in one batched matrix
  operation instead of 8 sequential calls. This is the same underlying
  principle (better hardware utilization per call) that vLLM's
  continuous batching builds on, just without the dynamic
  request-admission logic a real serving system adds on top.

## An honest note on the memory numbers

KV-cached generation shows *lower* peak memory here than naive, which is
the opposite of the usual "KV cache costs memory" framing you'll read
about in production LLM serving. That's specific to what's being
compared: naive here means "recompute everything," which involves large,
repeatedly-reallocated attention matrices over a growing sequence — not
"no cache" in the production sense of a model serving zero context. In a
real serving system, the KV cache is compared against *not keeping any
state at all* (impossible for autoregressive generation) — there, the
cache is a pure memory cost paid in exchange for the compute savings
shown above. This is exactly the trade-off `quantization` (also on my
list to explore next) is designed to help with, by shrinking the size of
each cached entry.

## How to run

```bash
bash run.sh
```

That's it — one command, no downloads, no GPU required, no external
model weights, done. If your Python environment blocks system-wide pip
installs (common on newer Debian/Ubuntu), `run.sh` automatically retries
with `--break-system-packages`.

If you'd rather run the pieces manually:

```bash
pip install -r requirements.txt
python3 benchmark.py
```

## Project structure

```
kv-cache-benchmark/
├── src/
│   ├── model.py        # MiniTransformer: attention, layers, forward_full + forward_step
│   └── generate.py      # generate_naive vs generate_kv_cached
├── benchmark.py          # Runs both, measures latency/throughput/memory, saves results
├── results/results.json  # Raw output from the last run
├── requirements.txt
└── run.sh                 # Single-command entry point
```

## Honest limitations, and what I'd build next with more time/hardware

- **No GPU, no PyTorch, no real pretrained weights.** This was built and
  fully run in a CPU-only sandbox with no internet access to a model
  hub, so I built the mechanism from first principles in NumPy instead
  of using a real pretrained model via HuggingFace/vLLM. The model's
  weights are randomly initialized — this measures *inference mechanics*
  (attention cost, caching, batching), not output quality, which is
  exactly what a from-scratch implementation is suited to demonstrate.
- **Not GPU-parallel.** These results are on a single CPU core. On a GPU,
  the absolute numbers would look completely different (much higher
  throughput, memory measured in VRAM not RAM), but the *relative*
  story — why caching matters, why batching matters — holds regardless
  of hardware, since it's about avoiding redundant compute, not about
  the hardware itself.
- **Next steps if I had GPU access:** port `MiniTransformer` to PyTorch
  (the forward-pass logic translates almost directly), then compare
  against a real vLLM deployment of the same architecture to see how
  much of vLLM's advantage comes from KV caching alone (demonstrated
  here) versus its more advanced techniques — PagedAttention, continuous
  batching with dynamic scheduling, and quantized weights.
