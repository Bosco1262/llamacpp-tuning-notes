# Why `llama-fit-params` Returns `-c 0`

[English](why-fit-params-returns-c-0.md) | [简体中文](why-fit-params-returns-c-0.zh-CN.md)

Seeing `llama-fit-params` print `-c 0 -ngl -1` is confusing the first time: with only 8 GB of VRAM, how can everything possibly be offloaded? And what does `-c 0` even mean? This note untangles both questions.

---

## First, a common misconception

The meaning of `-ngl` is easy to read backwards:

- `-ngl` stands for `n-gpu-layers` — the **number of model layers offloaded to the GPU**
- `-ngl -1` means **offload every model layer to VRAM**, i.e. maximum GPU acceleration — it does **not** mean "no GPU offloading"
- The flag that truly disables GPU offloading and runs everything on CPU is `-ngl 0`

---

## Why 8 GB of VRAM is enough for full offloading (`-ngl -1`)

`llama-fit-params` adds up the total VRAM footprint of model weights + KV cache + runtime overhead. As long as everything fits into your VRAM (minus the reserved headroom), it recommends offloading all layers. In the Gemma 4 E4B Q4_K_M case on an 8 GB card, the total is far below 8 GB. Here's the breakdown.

### 1. The model weights themselves are tiny

**Gemma 4 E4B Q4_K_M** is a small-parameter model at a medium quantization level:

- Parameters: 4B (4 billion)
- Quantization: Q4_K_M (~4.5 bits per parameter on average)
- Actual VRAM footprint: only **2.4–2.7 GB**

So the weights alone take less than a third of the 8 GB, leaving plenty of room.

### 2. The KV cache is far smaller than the remaining VRAM

The KV cache stores past attention data during inference, and its size scales with context length. Enabling an 8-bit quantized KV cache (`q8_0`) shrinks it further.

For Gemma 4 E4B's architecture (34 layers, hidden size 2560), an 8-bit quantized KV cache costs roughly:

| Context length | Estimated KV cache |
|------------|----------------|
| 32768 (32K) | ~0.34 GB |
| 131072 (128K) | ~1.35 GB |

### 3. Total VRAM footprint

At full 128K context:

model weights (~2.6 GB) + KV cache (~1.35 GB) + CUDA runtime / intermediate tensors (~0.5 GB) ≈ **4.45 GB**

Subtract the 512 MB of headroom reserved by `--fit-target 512` (for the OS and driver), and about 7.5 GB is still available — far more than needed. That's why the tool concludes all layers can live on the GPU, with no layers left to run on CPU.

---

## Bonus: what `-c 0` means in the output

`-c` is the context-length parameter, and `-c 0` means **use the model's built-in default context window** — no manual limit needed. Because the requested 32K/128K comfortably fit in VRAM, the tool has no reason to shrink the context, so it prints `0` and lets you use the model's native window as-is.

If you later switch to a bigger model (9B, 13B), a higher-precision quantization, or a larger context, and the total exceeds available VRAM, the tool will automatically lower `-ngl`, pushing some layers back to the CPU — that's "partial GPU offloading".
