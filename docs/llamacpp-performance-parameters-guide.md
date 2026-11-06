# llama.cpp Performance Parameters: The Complete Guide

[English](llamacpp-performance-parameters-guide.md) | [简体中文](llamacpp-performance-parameters-guide.zh-CN.md)

> **About this document**: llama.cpp is a fast-moving project — parameter names and defaults can change almost monthly. This guide was compiled in July 2026 from the official [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) main-branch documentation (the official `llama-server` and `llama-bench` READMEs, etc.) plus a range of community benchmarking write-ups, aiming for accuracy and completeness. That said: **the definitive source for your specific parameters is always the `llama-cli --help` / `llama-server --help` output for the version you're running**; this guide notes, where possible, how old or new a given parameter is.
>
> **Scope**: this guide only covers parameters that affect **inference speed, VRAM/RAM usage, and throughput**. Sampling parameters (`temperature`, `top-p`, `top-k`, `repeat-penalty`, Mirostat, DRY, XTC, grammar/json-schema, etc.) only affect the content and style of the output — **they don't affect tokens/sec** — so they're outside the scope of this guide; feel free to adjust them without worrying about performance.

## Table of Contents

1. [Overview of the Three Core Tools](#1-overview-of-the-three-core-tools)
2. [Quick Diagnosis: Symptom → Parameter to Tune](#2-quick-diagnosis-symptom-parameter-to-tune)
3. [Thread & CPU Scheduling Parameters](#3-thread-cpu-scheduling-parameters)
4. [Context Length & Batch Processing Parameters](#4-context-length-batch-processing-parameters)
5. [GPU Offloading & VRAM Parameters](#5-gpu-offloading-vram-parameters)
6. [Mixture-of-Experts (MoE) Offloading Parameters](#6-mixture-of-experts-moe-offloading-parameters)
7. [Flash Attention & KV Cache Quantization Parameters](#7-flash-attention-kv-cache-quantization-parameters)
8. [Memory Mapping, Locking & Loading Parameters](#8-memory-mapping-locking-loading-parameters)
9. [Multi-GPU & Distributed Inference Parameters](#9-multi-gpu-distributed-inference-parameters)
10. [Automatic VRAM Fitting: the `--fit` Family (Newer Feature)](#10-automatic-vram-fitting-the-fit-family-newer-feature)
11. [RoPE Scaling & Long-Context Extension Parameters](#11-rope-scaling-long-context-extension-parameters)
12. [Speculative Decoding Parameters](#12-speculative-decoding-parameters)
13. [Server Concurrency & Throughput Parameters](#13-server-concurrency-throughput-parameters)
14. [How Model Quantization Format Affects Performance](#14-how-model-quantization-format-affects-performance)
15. [Compile-Time Backend Selection (Brief)](#15-compile-time-backend-selection-brief)
16. [Complete Guide to llama-bench Benchmarking](#16-complete-guide-to-llama-bench-benchmarking)
17. [Curated Real-World Tuning Recipes](#17-curated-real-world-tuning-recipes)
18. [High-Impact Parameter Quick Reference](#18-high-impact-parameter-quick-reference)
19. [Performance Timing & Diagnostic Parameters (--perf)](#19-performance-timing-diagnostic-parameters-perf)
20. [References](#20-references)

---

## 1. Overview of the Three Core Tools

| Tool | Purpose | Typical invocation |
|---|---|---|
| `llama-cli` | Command-line one-shot inference / interactive chat | `llama-cli -m model.gguf -p "Hello" -ngl 99` |
| `llama-server` | Long-running HTTP service, OpenAI-compatible API | `llama-server -m model.gguf -ngl 99 --port 8080` |
| `llama-bench` | Dedicated performance-benchmarking tool; excludes sampling/tokenization time | `llama-bench -m model.gguf -ngl 99` |

All three share the same underlying argument-parsing logic (`common_params`), so **the vast majority of parameters listed in this guide apply to all three tools**. The exceptions: some `llama-bench` parameters additionally support a "comma-separated multi-value / range sweep" batch-testing syntax (see Section 16), and `llama-server` has extra concurrency (slots) parameters of its own (see Section 13).

---

## 2. Quick Diagnosis: Symptom → Parameter to Tune

| Symptom | Parameters to check first |
|---|---|
| Out of VRAM (OOM / CUDA out of memory) | Lower `-ngl`; enable `-ctk q8_0 -ctv q8_0`; reduce `-c`; for MoE models use `--n-cpu-moe` / `-ot`; confirm `-fit on` |
| Slow generation speed (tg, per-token output) | Push `-ngl` up to all layers if possible; turn on `-fa on`; quantize the KV cache; for MoE models lower `--n-cpu-moe`; switch to a smaller quantization format; for CPU-only setups, check the thread count |
| Slow prompt processing (pp, first-token latency) | Increase `-b` / `-ub`; max out `-ngl`; turn on `-fa on`; confirm `--op-offload` hasn't been disabled |
| Stuttering/queuing under multi-user concurrency | Adjust `-np` (parallel slot count); confirm `-cb` (continuous batching) is enabled; `--cache-reuse`, `-sps` |
| Long context (32K+) won't run / exceeds VRAM | Quantize the KV cache (`-ctk`/`-ctv`) before increasing `-c`; use `--rope-scaling yarn` for context extension; `--swa-full` |
| Multi-GPU shows no real speedup, or is even slower | Check whether `-sm` mode suits your interconnect (use `layer` without NVLink); check whether `-ts` ratios match your VRAM; try `GGML_CUDA_P2P=1` |
| Model loads very slowly | Keep the `--mmap` default enabled; SSD users can try `-dio` |
| CPU inference gets slower when all threads are maxed out | Lower `-t` to the physical core count (not the logical thread count), and drop it by another 1-2 if needed |

---

## 3. Thread & CPU Scheduling Parameters

| Parameter | Default | Description |
|---|---|---|
| `-t, --threads N` | `-1` (auto) | Number of CPU threads used during generation |
| `-tb, --threads-batch N` | same as `--threads` | Number of threads for the prompt-processing / batch stage |
| `-C, --cpu-mask M` | `""` | Hexadecimal CPU affinity mask |
| `-Cr, --cpu-range lo-hi` | — | CPU affinity range; use either this or `-C` |
| `--cpu-strict <0\|1>` | `0` | Whether to strictly pin cores |
| `--prio N` | `0` | Process/thread priority: low (-1), normal (0), medium (1), high (2), realtime (3) |
| `--poll <0...100>` | `50` | Polling intensity while waiting for work (higher = more CPU usage but lower latency) |
| `-Cb, --cpu-mask-batch M` | same as `-C` | Mask specific to the batch-processing stage |
| `-Crb, --cpu-range-batch lo-hi` | — | Range specific to the batch-processing stage |
| `--numa TYPE` | off | NUMA optimization: `distribute` (spread evenly across nodes) / `isolate` (use only the launch node) / `numactl` (use numactl's mapping) |

### Tuning tips

- **More threads isn't always better.** The official thread-count benchmark (7B Q4_0, CPU backend) shows:

  | Threads | pp64 (t/s) | tg16 (t/s) |
  |---|---|---|
  | 1 | 6.17 | 4.05 |
  | 2 | 12.31 | 7.80 |
  | 4 | 23.18 | 12.22 |
  | 8 | 32.29 | 16.71 |
  | 16 | 33.52 | **15.32** (already declining) |
  | 32 | 59.00 | **16.41** (prompt processing still climbing, generation essentially flat) |

  The takeaway: **prompt processing (pp) is highly parallel, so more threads generally means faster**; but **per-token generation (tg) is fundamentally serial and bound by memory bandwidth** — beyond the physical core count, gains flatten out quickly and can even go negative (cache contention, hyper-threading/SMT fighting over shared execution units).
- Rule of thumb: set `-t` to the **physical core count** (not the logical thread count — hyper-threading/SMT usually doesn't help here), leaving 1-2 cores for the OS/other processes if needed. `-tb` can be set higher, since prompt processing benefits more from parallelism.
- If the whole model is already offloaded to GPU via `-ngl 99`, thread count typically affects generation speed by **less than 10%** — not worth obsessing over.
- Intel's 12th-gen+ hybrid P-core/E-core architecture and AMD's CCD boundaries can both make "maxing out every core" backfire; benchmark your own hardware with `llama-bench -t 4,8,12,16,...` (see Section 16).
- Dual-socket servers (multiple NUMA nodes): llama.cpp currently doesn't handle cross-node access gracefully, and cross-socket access noticeably slows things down. If the model fits in a single NUMA node's memory, bind threads to that node first (`--numa isolate`, or pin with `numactl`); only try `--numa distribute` if it doesn't fit.

---

## 4. Context Length & Batch Processing Parameters

| Parameter | Default | Description |
|---|---|---|
| `-c, --ctx-size N` | `0` (read from model metadata) | Context window size (in tokens); directly determines KV-cache VRAM usage |
| `-n, --predict, --n-predict N` | `-1` (unlimited) | Maximum number of tokens to generate |
| `-b, --batch-size N` | `2048` | **Logical** maximum batch size (upper limit on tokens submitted for processing at once) |
| `-ub, --ubatch-size N` | `512` | **Physical** maximum batch size (the actual number of tokens fed into the compute graph at once; determines compute-buffer VRAM usage) |
| `--keep N` | `0` (`-1` = all) | Number of tokens retained from the initial prompt once context overflows |
| `--swa-full` | `false` | Use a full-size SWA cache for models using sliding-window attention (SWA) |

### Tuning tips

- `-b` (logical batch size) and `-ub` (physical batch size): `-ub` cannot exceed `-b`. **Only the prompt-processing stage (prefill) is meaningfully affected by batch size** — during per-token generation, batch size is effectively fixed at 1 and barely matters.
- Official batch-size benchmark (7B Q4_0, CUDA, `-p 1024`):

  | `-b` | pp1024 (t/s) |
  |---|---|
  | 128 | 1436.51 |
  | 256 | 1932.43 |
  | 512 | 2254.45 |
  | 1024 | 2498.61 |

  Increasing batch size substantially improves prompt-processing throughput, but linearly increases the VRAM used by the compute buffer.
- **CPU+GPU hybrid inference (especially MoE models partially offloaded to CPU) is unusually sensitive to batch size**: the default `-b 2048 -ub 512` is tuned for pure-GPU inference and tends to be too small when expert weights live on the CPU. Raising `-b`/`-ub` to `2048–4096` (or even matching them) often significantly boosts throughput, because it determines the threshold for triggering a "one-shot copy of CPU weights to the GPU for batched compute" (default threshold is 32 tokens; adjustable via the `GGML_OP_OFFLOAD_MIN_BATCH` environment variable).
- `-c 0` makes the model use the native training context length recorded in its metadata. If you need a context longer than that, you'll need the RoPE/YaRN parameters from Section 11.
- The larger the context, the larger the KV cache (roughly linear in `-c`) — this is the number-one cause of running out of VRAM in long-context scenarios; see the KV-cache quantization discussion in Section 7.

---

## 5. GPU Offloading & VRAM Parameters

| Parameter | Default | Description |
|---|---|---|
| `-ngl, --gpu-layers, --n-gpu-layers N` | `auto` | Number of model layers offloaded to VRAM; setting a large number (e.g. `99` / `999`) is equivalent to "offload everything" |
| `-sm, --split-mode {none,layer,row,tensor}` | `layer` | Multi-GPU split strategy — see table below |
| `-ts, --tensor-split N0,N1,...` | auto-allocated by VRAM | Ratio of the model assigned to each GPU, e.g. `3,1` |
| `-mg, --main-gpu INDEX` | `0` | Main GPU (the only GPU used when `split-mode=none`; in `row` mode, used to hold intermediate results and the KV cache) |
| `-dev, --device <dev1,dev2,..>` | all available devices | List of devices to offload to; `none` = no offloading |
| `--list-devices` | — | Print the list of available devices, then exit |
| `-ot, --override-tensor <regex>=<device>` | — | Use a regular expression to precisely control which device a given tensor goes to — the most flexible option, but requires manual tuning |
| `-kvo, --kv-offload` | enabled | Whether the KV cache is placed in VRAM |
| `--op-offload` | enabled (`true`) | Whether CPU-side tensor operations are temporarily offloaded to the GPU for execution (has a large impact on prompt-processing speed in CPU+GPU hybrid inference) |
| `--repack` | enabled | Whether quantized weights are repacked to match the CPU's SIMD instruction set (AVX/NEON, etc.) to speed up CPU computation |

### The three `-sm` split modes compared

| Mode | How it works | Best suited for |
|---|---|---|
| `none` | Uses a single GPU (specified via `-mg`) | A single GPU is enough, or you want to force use of just one card |
| `layer` (default) | Pipeline parallelism: different layers live on different GPUs, executed in sequence | **The most general-purpose and best-compatibility option**; low inter-GPU bandwidth requirements — the go-to for consumer PCIe multi-GPU setups |
| `row` | Splits the weight matrix of a single layer across GPUs by row, in principle allowing parallel computation | The official docs now describe this as an **older, generally weaker-performing** tensor-parallel path; without NVLink or similar fast interconnects, it's typically slower than `layer` on consumer PCIe platforms |
| `tensor` | A more thorough tensor-level split, with multiple GPUs computing the same layer simultaneously (experimental) | Needs a fast inter-GPU interconnect (like NVLink) to pay off — otherwise communication overhead becomes the bottleneck |

### Tuning tips

- `-ngl 99` (or any larger number) is the standard shorthand for "offload everything to GPU": as long as the value is ≥ the model's total layer count, llama.cpp automatically clamps it to the actual number of layers.
- **Partial offloading shows a pronounced "cliff effect"** — official benchmark (7B Q4_0, 35 layers, CUDA):

  | `-ngl` | pp512 (t/s) | tg128 (t/s) |
  |---|---|---|
  | 10 | 373 | 13.45 |
  | 20 | 473 | 21.36 |
  | 30 | 632 | 40.04 |
  | 33 | 704 | 60.62 |
  | 34 | 881 | 71.76 |
  | 35 (all) | **2400** | **131.66** |

  Moving the last few layers from CPU to GPU gives the biggest speedup, because intermediate results no longer need to shuttle back and forth between CPU and GPU. **As long as VRAM allows, prioritize maxing out `-ngl`**; only fall back to the MoE-specific offloading in Section 6 or a lower quantization level if it genuinely doesn't fit.
- Multi-GPU: when VRAM is uneven across cards, use `-ts` to allocate proportionally to each card's remaining VRAM (e.g., for three cards with 32G/16G/8G, you might set `-ts 32,16,6` to leave headroom on the smallest card). Set `-mg` to whichever card has the most bandwidth/compute.
- On multi-GPU NVIDIA setups, you can try enabling direct P2P transfers (requires driver support; more common on workstation/datacenter cards):
  ```bash
  GGML_CUDA_P2P=1 llama-server -m model.gguf -sm tensor ...
  ```
  On some motherboards/BIOS configurations (especially with IOMMU enabled), this can be unstable — if you run into issues, just unset this environment variable.
- `-ot` is the ultimate tool: it lets you precisely specify "which tensors go on CPU, which go on which GPU," and is commonly used for manual tuning of large MoE models (see Section 6) — though it requires understanding the target model's tensor-naming conventions.

---

## 6. Mixture-of-Experts (MoE) Offloading Parameters

MoE models (Mixture-of-Experts — Qwen3-MoE, DeepSeek-V3, GLM-4.x, Kimi-K2, gpt-oss, etc.) have two categories of parameters:

- **"Always-active" components** (attention, embeddings, shared experts, output layer, etc.): used by every token, and well-suited to living in VRAM.
- **Routed experts**: only a small subset is activated per token, but because routing changes token-by-token, **all experts must stay resident in memory**, and they account for the vast majority of the model's size.

The core idea: **keep the "always-active" components on the GPU, and put the routed-expert weights in system RAM (CPU)**, transferring only small activation vectors between GPU and CPU as needed, instead of moving entire expert weights around.

| Parameter | Default | Description |
|---|---|---|
| `-cmoe, --cpu-moe` | off | Puts **all** MoE expert weights on the CPU — the simplest starting configuration |
| `-ncmoe, --n-cpu-moe N` | `0` | Puts only the MoE expert weights for the **first N layers** on the CPU; the rest still follow `-ngl` into VRAM. Larger N saves more VRAM but is slower |
| `-ot "exps=CPU"` | — | Uses a regex to match all expert tensors and force them onto the CPU — similar effect to `--cpu-moe`, but more readable and more extensible |

### Tuning tips

- The simplest starting point (put attention/dense FFN/shared experts/KV cache all on GPU, and routed experts on CPU):
  ```bash
  llama-server -m moe-model.gguf -ngl 999 -ot "exps=CPU" -fa on
  ```
- When you have some spare VRAM but not enough for all the experts, start `--n-cpu-moe` at 0 and increase it gradually, using `llama-bench` or actual VRAM usage (`nvidia-smi`) to find the smallest value that just avoids OOM:
  ```bash
  # Official example: gpt-oss-20b on an 8GB VRAM machine
  llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 32768 --n-cpu-moe 16 -ub 2048 -b 2048 --jinja
  ```
- With multiple GPUs plus MoE, you can use `-ot` to finely control which layers go to which GPU and which experts go to CPU, for example:
  ```bash
  -ot "blk\.([0-9])\.=CUDA0,blk\.(1[0-9])\.=CUDA1,exps=CPU"
  ```
- **CPU+GPU hybrid inference is very sensitive to `-b`/`-ub`** (see Section 4) — it's recommended to raise both to 2048–4096 at the same time.
- **PCIe bandwidth can become the bottleneck**: if the CPU-side weights are large and the GPU connects over PCIe (rather than NVLink), the amount of data to transfer per batch can reach a point where a single copy takes several seconds — in which case increasing batch size actually pays off (doing more work per transfer amortizes the transfer overhead).
- The dual-socket NUMA issues also apply to MoE + CPU offloading scenarios — see the tip at the end of Section 3.

---

## 7. Flash Attention & KV Cache Quantization Parameters

| Parameter | Default | Description |
|---|---|---|
| `-fa, --flash-attn [on\|off\|auto]` | `auto` | Whether to use the Flash Attention kernel |
| `-ctk, --cache-type-k TYPE` | `f16` | K-cache data type: `f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1` |
| `-ctv, --cache-type-v TYPE` | `f16` | V-cache data type; same options as above |
| `-ctxcp, --ctx-checkpoints N` | `32` | Maximum context checkpoints retained per slot (used for incremental reuse in long conversations) |
| `-cms, --checkpoint-min-step N` | `256` | Minimum token interval between checkpoints |
| `-cram, --cache-ram N` | `8192` (MiB) | Memory cap for the prompt cache (`-1` = unlimited, `0` = disabled) |
| `-kvu, --kv-unified` | enabled by default when slots is auto | Whether all concurrent sequences share a single KV buffer |

### Tuning tips

- **Quantizing the KV cache generally requires Flash Attention to be on** (`-fa on`) — the two are meant to be used together, and there's rarely a reason to enable only one.
- **`q8_0` KV cache is close to a free lunch**: it halves VRAM usage, and multiple independent tests show perplexity loss on the order of 0.1% or less — a good default for long-context scenarios:
  ```bash
  llama-server -m model.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 32768
  ```
- **`q4_0` halves VRAM again, but calls for more caution**:
  - Quality: community testing gives **somewhat inconsistent conclusions** about whether K or V is more sensitive — most sources suggest the V cache (value vectors, which directly participate in the weighted sum over outputs) is more sensitive to low precision, while the K cache (used in computing attention weights) tolerates quantization better, though some sources report the opposite. Treat this as model-dependent: if output quality matters to you, test on your own task first (you can quantize only K and keep V at `f16` as an asymmetric setup) rather than assuming either direction is automatically safer.
  - Speed: multiple sources report that at **long context lengths**, `q4_0`'s dequantization overhead can offset or even outweigh its bandwidth savings — some cases show it running **slower than f16** at a 64K context (because attention computation has to dequantize on the fly, and that overhead grows with context length). If speed goes down rather than up after enabling `q4_0`, fall back to `q8_0` or `f16` and compare.
- A reasonable starting point: **default to `-ctk q8_0 -ctv q8_0`**; only try `q4_0` once VRAM is genuinely tight and you've verified acceptable quality on your specific model/task; keep `f16` (the default) if quality matters most and VRAM is ample.

---

## 8. Memory Mapping, Locking & Loading Parameters

| Parameter | Default | Description |
|---|---|---|
| `--mmap, --no-mmap` | enabled | Whether to load the model file via memory mapping (disabling it slows loading, but combined with no `mlock` it may reduce paging) |
| `--mlock` | off | Force the model to be locked in physical memory, preventing the system from swapping it to disk/compressed memory |
| `-dio, --direct-io` | off | If the backend supports it, load via Direct I/O, bypassing the system page cache (helps with fast NVMe drives, but skips cache reuse) |
| `--no-host` | off | Bypass the host buffer, allowing more buffers to be used (advanced option) |
| `--check-tensors` | off | Validate tensor values during loading (slows down loading; generally only used when debugging) |

### Tuning tips

- The `--mmap` default (enabled) is the right call for the vast majority of scenarios: faster loading, and multiple processes can share the same model's page cache.
- If RAM/VRAM is tight and you're worried the system will swap the model out, add `--mlock` (this pins an equivalent amount of physical RAM, so it's not a good fit for machines that are also short on RAM).
- With some CPU+GPU hybrid offloading setups — especially when repeatedly re-tuning parameters on a large MoE model — you'll see people add `--no-mmap`, mainly to prevent page-cache state from skewing comparisons between test runs. That doesn't mean it's generally faster; for actual deployment, it's best to leave the default as-is.

---

## 9. Multi-GPU & Distributed Inference Parameters

Beyond `-sm` / `-ts` / `-mg` / `-dev` from Section 5, there's also:

| Parameter | Default | Description |
|---|---|---|
| `-rpc, --rpc <rpc_servers>` | none | Registers remote RPC compute nodes (comma-separated), letting you spread the model/computation across multiple machines on a LAN |
| `GGML_CUDA_P2P=1` (environment variable) | off | Allows direct P2P data transfer between NVIDIA GPUs, bypassing system RAM as an intermediary |

### Tuning tips

- If a single GPU has enough VRAM, **don't force multi-GPU just to "use every card you have"** — layer-wise pipeline parallelism (`layer`) can't keep multiple GPUs busy simultaneously during generation, so the actual throughput gain is limited, and scheduling overhead can even make it slower. Multi-GPU is better suited to cases where the model simply doesn't fit on one card.
- For genuine throughput scaling (multiple GPUs computing at once), you need `-sm tensor` or `-sm row` plus a fast interconnect (NVLink); on ordinary consumer PCIe multi-GPU setups the payoff is limited, and `layer` is usually still the more reliable choice.
- `-rpc` suits the "I have several machines, but none of them has enough VRAM alone" scenario, splitting the model across multiple machines — but you'll need to accept the extra latency from LAN bandwidth.

---

## 10. Automatic VRAM Fitting: the --fit Family (Newer Feature)

This is a relatively recent addition to llama.cpp: **when parameters like `-ngl` / `-ot` / `-c` aren't explicitly specified, it automatically works backward from the device's available VRAM to derive a configuration that will actually run** (layer allocation, whether to offload to CPU, shrinking the context if necessary, etc.).

| Parameter | Default | Description |
|---|---|---|
| `-fit, --fit [on\|off]` | `on` | Whether to enable automatic VRAM fitting |
| `-fitt, --fit-target MiB0,MiB1,...` | `1024` | VRAM headroom reserved per device (MiB); can be set individually per GPU |
| `-fitc, --fit-ctx N` | `4096` | Minimum context size `--fit` is allowed to shrink down to |

The companion diagnostic tool `llama-fit-params` can **just print** the parameters the auto-fit algorithm computes, without actually launching inference — handy for copying the result down into a fixed configuration:

```bash
llama-fit-params --model model.gguf -c 65536
# Sample output:
# llama_params_fit_impl: context size reduced from 65536 to 4096 -> ...
# -c 4096 -ngl 41 -ot "blk\.22\.ffn_(gate|down).*=CPU,..."
```

### Tuning tips

- For beginners / quick sanity checks: **keep `-fit on` (the default)** and don't specify `-ngl` — let llama.cpp figure it out. This usually gets you a conservative, "it runs" configuration.
- During fine-grained tuning (e.g., manually assigning MoE layers per Section 6): **it's best to set `-fit off`**, otherwise auto-fit may quietly alter parameters you didn't explicitly set, making it hard to tell which change actually caused an effect.
- Known issue: VRAM for the `mmproj` (vision projector) in multimodal models is currently **not** included in the auto-fit estimate, which can lead to "fitting looked successful but OOM'd while loading mmproj." For these models, it's best to manually increase the `-fitt` headroom, or use `--no-mmproj-offload` to keep the projector on the CPU.

---

## 11. RoPE Scaling & Long-Context Extension Parameters

Used to extend a model beyond its native training context length. These parameters **don't directly affect tokens/sec**, but determine whether you can run at that larger context size at all, plus output quality after extension — and they indirectly affect VRAM usage (longer context = larger KV cache).

| Parameter | Default | Description |
|---|---|---|
| `--rope-scaling {none,linear,yarn}` | whatever the model specifies, usually `linear` | Frequency scaling method |
| `--rope-scale N` | — | Context extension multiplier (an alternate way of writing linear scaling) |
| `--rope-freq-base N` | model-specific value | RoPE base frequency, used for NTK-aware scaling |
| `--rope-freq-scale N` | — | Frequency scaling factor; extends context to `1/N` times |
| `--yarn-orig-ctx N` | `0` (= the model's original training length) | YaRN: original context length |
| `--yarn-ext-factor N` | `-1.00` | YaRN: extrapolation mixing factor (`0.0` = pure interpolation) |
| `--yarn-attn-factor N` | `-1.00` | YaRN: attention magnitude scaling |
| `--yarn-beta-slow / --yarn-beta-fast N` | `-1.00` | YaRN: low/high-frequency correction dimension parameters |

### Tuning tips

- If the model's native context is already long enough (per the training length in its metadata), **you don't need to touch these** — just adjust `-c` directly.
- To extend context beyond the native training length:
  ```bash
  # Simple 4x linear extension
  llama-cli -m model.gguf -c 16384 --rope-scaling linear --rope-freq-scale 0.25

  # YaRN (usually better quality than linear scaling)
  llama-cli -m model.gguf -c 32768 --rope-scaling yarn --yarn-orig-ctx 4096
  ```
- Some models (such as certain Llama 3.1 variants) officially support long context natively via a specific `--rope-freq-base` value — check the model card / official release notes for the exact value, rather than reusing another model's parameters from memory.

---

## 12. Speculative Decoding Parameters

How it works: a smaller, faster "draft model" guesses several tokens in a row, and the target (large) model **verifies all of those guesses in parallel, in one pass**. Correctly guessed tokens are essentially obtained "for free," lowering the average time per token. This works best on code completion and other highly predictable text; the draft model and target model **must have compatible vocabularies**.

| Parameter | Default | Description |
|---|---|---|
| `-md, --model-draft FNAME` | none | Path to the draft model |
| `-ngld, --gpu-layers-draft N` | `auto` | Number of draft-model layers offloaded to GPU |
| `-td, --threads-draft N` | same as `--threads` | Number of threads used by the draft model |
| `--spec-draft-n-max N` | `3` | Maximum tokens drafted per round (the old name `--draft-max` is deprecated) |
| `--spec-draft-n-min N` | `0` | Minimum tokens drafted per round (the old name `--draft-min` is deprecated) |
| `--spec-draft-p-split P` | `0.10` | Probability-split threshold for speculative decoding |
| `--spec-draft-p-min P` | `0.00` | Minimum probability to accept a draft (greedy mode) |
| `--spec-draft-backend-sampling` | enabled | Whether draft sampling is delegated to the backend (GPU) |
| `--spec-type` | `none` | Speculation strategy: `draft-simple` / `draft-eagle3` / `draft-mtp` (requires a specially trained draft head) / `ngram-simple` / `ngram-map-k` / `ngram-map-k4v` / `ngram-mod` / `ngram-cache` (needs no extra model — mines n-grams from the context itself to use as drafts) |
| `--spec-draft-cache-type-k/v` | `f16` | KV-cache quantization type for the draft model's own cache |

### Tuning tips

- **When a suitable small model is available** (a smaller version from the same family, with the same vocabulary), use the classic `-md` draft-model setup:
  ```bash
  llama-server -m target.gguf -md draft.gguf -ngl 99 -ngld 99 --spec-draft-n-max 16
  ```
- **Without a suitable draft model** (e.g., VRAM only fits one model), try the `--spec-type ngram-*` family — no extra model needed; it mines repeated fragments straight from the current context/generation history to use as drafts. This works well for tasks where the output heavily repeats the input (like code editing or long-document summarization), costs almost nothing, and is worth trying by default.
- Whether speculative decoding pays off depends heavily on how predictable the task is: free-form creative text has a low hit rate and can actually slow things down, while structured output, code, and tool-calling — with lots of repeating patterns — see clear gains. Verify with `llama-server`'s built-in `tools/server/bench/speed-bench`, or by directly comparing `predicted_per_second` with the feature on vs. off.

---

## 13. Server Concurrency & Throughput Parameters

(`llama-server`-only)

| Parameter | Default | Description |
|---|---|---|
| `-np, --parallel N` | `-1` (auto) | Number of concurrent processing slots, determining how many requests can be served at once |
| `-cb, --cont-batching` | enabled | Continuous (dynamic) batching — packs the decode steps of multiple concurrent requests into the same batch |
| `--cache-prompt` | enabled | Whether to reuse the common prefix already processed in a previous request (KV-cache reuse) |
| `--cache-reuse N` | `0` | Minimum fragment length for reusing cache via KV shifting; notably effective for long conversation histories |
| `-sps, --slot-prompt-similarity SIM` | `0.10` | Similarity threshold a new request's prompt must meet against a slot's existing prompt to reuse that slot |
| `--threads-http N` | `-1` | Number of threads handling the HTTP requests themselves (separate from inference threads) |
| `--metrics` | off | Enables a Prometheus-format metrics endpoint, useful for monitoring throughput/latency |

### Tuning tips

- **`-cb` (continuous batching) should almost always stay enabled**: benchmarks show throughput improvements of **over 2x** at 4-way concurrency compared to handling requests serially. It's the core mechanism that lets llama-server serve multiple users at once.
- A larger `-np` (concurrent slot count) can theoretically serve more users at once, but **each slot consumes its own separate share of KV-cache VRAM** — slot count × per-slot VRAM must stay within your total VRAM budget. Setting the slot count too high shrinks the context each slot gets, or causes outright OOM.
- For long conversations or agent-style workloads that resend the full history repeatedly, `--cache-reuse` together with the default-enabled `--cache-prompt` can substantially cut down redundant prompt computation — it's worth keeping these on and raising `--cache-reuse` a bit (e.g., 256–512).
- To check whether your concurrency configuration is reasonable, the most informative approach is to set your load-testing tool's concurrency to match `-np` (the `tools/server/bench/speed-bench` tool itself recommends aligning `--concurrency` with `-np`).

---

## 14. How Model Quantization Format Affects Performance

Quantization format isn't a runtime parameter — it's **fixed when the model is downloaded/converted** (it's baked into the GGUF file itself) — but it's the single biggest factor in whether a model fits in VRAM and what its generation-speed ceiling is, so it's included here.

### Quantization families at a glance

| Family | Representative formats | Characteristics |
|---|---|---|
| Legacy block quantization | `Q4_0` `Q4_1` `Q5_0` `Q5_1` `Q8_0` | Uniform precision within each block; simple to implement and fast. `Q8_0` remains a common choice for "near-lossless" 8-bit quantization |
| K-quants | `Q2_K` `Q3_K_S/M/L` `Q4_K_S/M` `Q5_K_S/M` `Q6_K` | Mixed precision within the same weight matrix; usually better quality than legacy block quantization at the same size. `_M` (medium) uses higher precision on critical layers than `_S` (small), and is usually worth the small size increase |
| I-quants (importance matrix) | `IQ1_S` `IQ2_XXS` `IQ3_XXS` `IQ4_NL` `IQ4_XS`, etc. | Use calibration data (an imatrix) to preserve precision as much as possible at extremely low bit-widths; smallest size, but computation can be slower on some hardware/backends. Quality drops noticeably at ultra-low bit-widths (below 2-bit), so these are generally only worth considering when VRAM is extremely tight |

### A quantization comparison for a 7B–8B-class model (figures from third-party community testing — for reference only; results vary by model/hardware)

| Format | Relative file size | Perplexity change vs. FP16 | Relative generation speed | Best for |
|---|---|---|---|---|
| `Q8_0` | Largest (~7GB range) | +0.03% | Baseline (slower) | Nearly lossless; the top choice when VRAM/bandwidth is ample |
| `Q6_K` | Slightly smaller | +0.13% | Slightly faster | One of the best quality/size trade-offs |
| `Q5_K_M` | Medium | +0.39% | Faster | A balanced choice |
| `Q4_K_M` | Smaller | +1.68% | Faster | **The most common default recommendation** — the sweet spot for most use cases |
| `Q4_K_S` | Even smaller | +2.62% | Faster | Smaller and faster than `Q4_K_M`, with a slight quality drop |
| `Q3_K_M` | Small | +6.07% | Fast | Only recommended when you absolutely must save VRAM and the model itself is large |
| `Q2_K` | Smallest | +15.3% | Fastest | Noticeable quality loss; generally not recommended |

### Tuning tips

- **Start from `Q4_K_M` by default**: for most scenarios, it's the best balance of size, speed, and quality, and it's also the format most commonly offered by inference frameworks and model publishers.
- If VRAM/bandwidth is ample (e.g., running a 7B–14B model on a 24GB+ GPU), feel free to go straight to `Q6_K` or `Q8_0` — the quality loss is barely perceptible, and the speed cost is acceptable.
- If VRAM is tight and the model is fairly large (30B+), prefer dropping to a lower quantization level over cutting context length or GPU layers — **quantization is the only way to save VRAM without sacrificing speed** (whereas cutting `-ngl` pushes some layers onto CPU, which usually costs more speed than dropping one quantization level).
- LLM generation is fundamentally **VRAM-bandwidth-bound**: smaller quantization means fewer bytes need to be read from VRAM per generated token, which generally means faster generation — this is also why quantization saves VRAM *and* boosts speed.
- Once a model has been quantized, you **cannot** convert it back to a higher-precision level without requantizing from the original FP16 weights — decide on the format you need before downloading.

---

## 15. Compile-Time Backend Selection (Brief)

An even more fundamental performance factor than any runtime parameter is **the compute backend chosen at compile time**, which determines whether llama.cpp can actually use your hardware's acceleration units:

| Backend | Suitable hardware | Build flag (example) |
|---|---|---|
| CUDA | NVIDIA GPUs | `-DGGML_CUDA=ON` |
| HIP/ROCm | AMD GPUs (including the Instinct series) | `-DGGML_HIP=ON` |
| Vulkan | Almost any modern GPU (NVIDIA/AMD/Intel); the best cross-vendor compatibility | `-DGGML_VULKAN=ON` |
| SYCL | Intel GPUs | `-DGGML_SYCL=ON` |
| Metal | Apple Silicon | Enabled automatically by default on macOS |
| CPU (AVX2/AVX-512/AMX/NEON) | No discrete GPU, or CPU-only deployment | Instruction set auto-detected by default |

If you're using an official prebuilt distribution, simply picking the right one — CUDA build / Vulkan build / ROCm build — is itself the biggest performance decision you'll make. On the same hardware, Vulkan has the best general compatibility but typically doesn't reach the peak performance of a vendor-specific backend (CUDA/ROCm). This guide doesn't go into build details; for a complete Windows + Vulkan walkthrough, see [Building llama.cpp Locally (Windows + Vulkan)](build-llamacpp-windows-vulkan.md).

---

## 16. Complete Guide to llama-bench Benchmarking

`llama-bench` is the official dedicated performance-testing tool. It **excludes tokenization and sampling time**, measuring only the throughput of the model's forward computation itself — it's the standard tool for verifying whether the tuning advice in this guide actually helps.

### Syntax

```text
llama-bench [options]

General options:
  --numa <distribute|isolate|numactl>       NUMA mode
  -r, --repetitions <n>                     Repetitions per test (default 5, averaged)
  --prio <-1|0|1|2|3>                       Process/thread priority
  --delay <0...N>                           Seconds of delay between tests
  -o, --output <csv|json|jsonl|md|sql>      Output format (default md)
  --list-devices                            List available devices
  -v, --verbose                             Verbose output
  --no-warmup                               Skip warmup
  -fitt, --fit-target <MiB>                 Auto-fit by VRAM (see Section 10)
  -rpc, --rpc <rpc_servers>                 Register remote RPC devices

Test parameters (nearly all support comma-separated multi-values / "start-end[+step|*multiplier]" range syntax):
  -m, --model <filename>       Path to the model file
  -p, --n-prompt <n>           Number of tokens for the prompt-processing test (default 512)
  -n, --n-gen <n>              Number of tokens for the generation test (default 128)
  -pg <pp,tg>                  Mixed test: process pp prompt tokens, then generate tg tokens
  -d, --n-depth <n>            Pre-fill the KV cache with n tokens before testing (simulates speed mid-way through a long conversation)
  -b, --batch-size <n>         Same as Section 4 (default 2048)
  -ub, --ubatch-size <n>       Same as Section 4 (default 512)
  -ctk / -ctv <t>              KV cache type (same as Section 7, default f16)
  -t, --threads <n>            Thread count
  -ngl, --n-gpu-layers <n>     Number of GPU layers
  -ncmoe, --n-cpu-moe <n>      MoE offload layer count (same as Section 6)
  -sm, --split-mode <...>      Multi-GPU split mode
  -fa, --flash-attn <on|off|auto>
  -mmp, --mmap <0|1>
  -ts, --tensor-split <...>
```

### Reading the output

```text
| model            | size     | params | backend | ngl | test   | t/s            |
| ---------------- | -------- | ------ | ------- | --- | ------ | -------------- |
| qwen2 1.5B Q4_0   | 885.97 MiB | 1.54 B | CUDA    | 99  | pp512  | 5765.41 ± 20.55 |
| qwen2 1.5B Q4_0   | 885.97 MiB | 1.54 B | CUDA    | 99  | tg128  | 197.71 ± 0.81   |
```

- `pp512` = prompt-processing test, processing 512 tokens; higher is better — it corresponds to lower first-token latency for long documents/long system prompts.
- `tg128` = per-token text-generation test, generating 128 tokens; this number tracks more closely with the felt "typing speed" of everyday interaction.
- `t/s` = tokens per second, with the standard deviation (variation across repeated runs) shown after `±`.
- The two **cannot be compared directly**: `pp` is typically far higher than `tg`, because the prefill stage can be massively parallelized while the decode stage is inherently serial. When comparing different configurations, make sure you're comparing the same test type (`pp` to `pp`, or `tg` to `tg`).

### Common test combination examples

```bash
# Basic test: pp/tg speed for a fully offloaded model
llama-bench -m model.gguf -ngl 99

# Sweep across GPU layer counts (find the sweet spot for partial offloading)
llama-bench -m model.gguf -ngl 10,20,30,31,32,33,34,35

# Sweep across thread counts (find the optimal thread count for pure-CPU scenarios)
llama-bench -m model.gguf -ngl 0 -t 1,2,4,8,16,32

# Sweep across batch sizes (focus on long-prompt processing speed)
llama-bench -m model.gguf -n 0 -p 1024 -b 128,256,512,1024,2048

# Compare Flash Attention on vs. off
llama-bench -m model.gguf -ngl 99 -fa 0
llama-bench -m model.gguf -ngl 99 -fa 1

# Compare KV cache quantization
llama-bench -m model.gguf -ngl 99 -fa 1 -ctk f16,q8_0,q4_0 -ctv f16,q8_0,q4_0

# Simulate mid-conversation (pre-fill a 4096-token context, then measure the speed drop-off)
llama-bench -m model.gguf -ngl 99 -d 0,4096,16384

# Compare the same model across several quantization levels
llama-bench -m model-Q4_K_M.gguf -m model-Q5_K_M.gguf -m model-Q8_0.gguf -ngl 99
```

---

## 17. Curated Real-World Tuning Recipes

The commands below are example templates — tune the actual values against your own `llama-bench` results and hardware.

### 17.1 Pure CPU inference (no discrete GPU)

```bash
llama-cli -m model-Q4_K_M.gguf \
  -t $(nproc --all) -tb $(nproc --all) \
  -ngl 0 -c 4096 -b 512
```
Prefer `Q4_K_M` or a smaller level; be cautious with `Q8_0` if RAM is under 16GB.

### 17.2 Single GPU, fully offloaded (standard scenario)

```bash
llama-server -m model-Q4_K_M.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -c 8192 --port 8080
```

### 17.3 Single GPU with insufficient VRAM, needs partial offloading (non-MoE dense model)

```bash
# First, use llama-bench to find the largest -ngl that avoids OOM
llama-bench -m model.gguf -ngl 10,20,30,31,32,33,34,35

llama-server -m model.gguf -ngl 32 -fa on -ctk q8_0 -ctv q8_0 -c 4096
```

### 17.4 Large MoE model (CPU+GPU hybrid, consumer GPU)

```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF \
  --ctx-size 32768 --jinja \
  -ub 2048 -b 2048 \
  --n-cpu-moe 16 \
  -ngl 999 -fa on
```

### 17.5 Multi-GPU (model doesn't fit on a single card)

```bash
llama-cli -m big-model-Q4_K_M.gguf \
  -ngl 999 -sm layer -ts 32,16,6 -mg 0 \
  -c 8192 -fa on
```

### 17.6 Long context + VRAM optimization

```bash
llama-server -m model.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -c 65536
```

### 17.7 High-concurrency, multi-user server

```bash
llama-server -m model-Q4_K_M.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -np 4 -cb -c 32768 \
  --cache-reuse 256 --metrics
```
`-np` (slot count) has to be weighed together with the per-slot context budget: total VRAM ≈ model weights + `np × per-slot KV cache`.

### 17.8 Speculative decoding speedup (when a matching small model is available)

```bash
llama-server -m target-model.gguf -md draft-model.gguf \
  -ngl 99 -ngld 99 \
  --spec-draft-n-max 16 -fa on
```

---

## 18. High-Impact Parameter Quick Reference

Roughly ranked by typical impact on performance (higher up generally means more worth tuning first):

| Rank | Parameter | What it does, in one line |
|---|---|---|
| 1 | `-ngl` | Determines how much computation happens on GPU (usually the single biggest speed lever) |
| 2 | Quantization level (chosen when downloading the model) | Sets the floor for VRAM usage and bandwidth needs |
| 3 | `--n-cpu-moe` / `-ot` | The VRAM/speed balance valve for MoE models |
| 4 | `-fa` | Paves the way for KV quantization once enabled, and usually has a performance benefit on its own |
| 5 | `-ctk` / `-ctv` | KV-cache quantization — a VRAM lifesaver for long-context scenarios |
| 6 | `-b` / `-ub` | The main lever for prompt-processing throughput (especially critical for CPU+GPU hybrid setups) |
| 7 | `-c` | The larger the context, the larger the KV-cache VRAM cost |
| 8 | `-sm` / `-ts` | Whether the multi-GPU split strategy matches your hardware's interconnect |
| 9 | `-t` / `-tb` | Thread configuration for CPU inference or partial offloading |
| 10 | `-np` / `-cb` | Multi-user concurrent throughput on the server |
| 11 | `--spec-*` speculative decoding | Extra speedup for specific task types |
| 12 | `--mmap` / `--mlock` / `-dio` | Mainly affects load time; little impact on steady-state inference speed |

---

## 19. Performance Timing & Diagnostic Parameters (--perf)

Besides using `llama-bench` for dedicated comparison testing (Section 16), llama.cpp can also print detailed timing statistics directly during **any real run**, without needing to run a separate benchmark.

| Parameter | Default | Description |
|---|---|---|
| `--perf, --no-perf` | `false` (off) | Whether to print libllama's internal performance timing stats (env: `LLAMA_ARG_PERF`) |

### `llama-cli` and other command-line tools

This is **off** by default; you need to explicitly add `--perf` to get a timing breakdown like the following printed at the end of the run:

```text
llama_perf_context_print:        load time =   579.15 ms
llama_perf_context_print:      sample time =     0.72 ms /    28 runs   (    0.03 ms per token, 38888.89 tokens per second)
llama_perf_context_print: prompt eval time =   655.63 ms /    10 tokens (   65.56 ms per token,    15.25 tokens per second)
llama_perf_context_print:        eval time =  2180.97 ms /    27 runs   (   80.78 ms per token,    12.38 tokens per second)
llama_perf_context_print:       total time =  2891.13 ms
```

(In different versions, the function-name prefix on these log lines may be `llama_print_timings` or `llama_perf_context_print` — check your build — but the fields mean the same thing either way.)

- `load time`: the one-time cost of loading the model; not counted as part of inference speed.
- `sample time`: sampling overhead (the cost of picking the next token from the logits); usually tiny and rarely the bottleneck.
- `prompt eval time`: prompt-processing (prefill) time; the tokens-per-second figure derived from this is what the rest of this guide calls **pp speed**.
- `eval time`: per-token generation (decode) time; the derived speed corresponds to **tg speed**.
- `total time`: total end-to-end time for the run.

Usage example:

```bash
llama-cli -m model.gguf -ngl 99 -p "Write a short passage about autumn" -n 128 --perf
```

This helps you quickly pin down whether a slow run is slow because of loading, because of processing a long prompt, or because of the generation stage — which tells you which of Sections 4, 5, 6, or 7 to go tune.

### `llama-server`: per-request timing

The server doesn't have an "print on process exit" mechanism — instead, timing comes through fields in the API response:

- Every `/completion` (or compatible chat/completions) response includes a `timings` object by default, with aggregate speed metrics like `predicted_per_second` — no extra flag needed.
- Adding `"timings_per_token": true` to the request body (default `false`) attaches timing/speed information to **every token in the streamed output**, which is useful for building a real-time monitoring dashboard, or for diagnosing whether speed is dropping off as the conversation gets longer.
- To monitor throughput/latency trends over the long term (rather than a one-off diagnosis), use `--metrics` to enable the Prometheus-format metrics endpoint (Section 13) and feed it into a tool like Grafana.

### `--perf` / `timings` vs. `llama-bench` — which to use

| Scenario | Recommended tool |
|---|---|
| Want to know how long this one real request actually took, and whether the bottleneck is prefill or decode | `--perf` (CLI) or the `timings` field in the response (server) |
| Want to compare whether a given parameter is faster on or off, with repeated runs averaged and variables controlled | `llama-bench` (Section 16) |
| Want to monitor production throughput/latency trends over the long term | `llama-server --metrics` + Prometheus/Grafana |

---

## 20. References

- llama.cpp official repository: <https://github.com/ggml-org/llama.cpp>
- Complete server parameter documentation: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Complete llama-bench documentation: <https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md>
- Official multi-GPU documentation: <https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md>
- `--fit` / `llama-fit-params` tool documentation: <https://github.com/ggml-org/llama.cpp/tree/master/tools/fit-params>
- Community MoE CPU+GPU hybrid inference tuning guide: <https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide>

> One more reminder: llama.cpp moves fast, and parameter names or defaults may have already changed by the time you're reading this. Whenever in doubt, your first move should always be running `llama-cli --help` / `llama-server --help` to check the actual parameter list for your local version.
