# `llama-fit-params` Explained

[English](llama-fit-params.md) | [简体中文](llama-fit-params.zh-CN.md)

`llama-fit-params` calculates and optimizes the VRAM allocation parameters for running large models locally. It only prints the computed parameters and never actually starts inference, which makes it a cheap way to check your VRAM budget before launching anything. At runtime you can enable `--fit on` together with `--fit-ctx N` (minimum context size) and `--fit-target N` (VRAM headroom reserved per GPU) to let the same logic adjust resources dynamically.

## Model categories

- **Choice 1–4**: MoE models (smallest to largest parameter count)
- **Choice 5–13**: dense models (smallest to largest parameter count)

## Using it with vision models

Vision models need an extra mmproj file (the CLIP projector), which occupies additional VRAM.

Steps:

1. First check the actual size of the mmproj file with `dir <mmproj path>`.
2. Then run `llama-fit-params`, computing `--fit-target` as follows:

```
--fit-target = mmproj file size (MB) + 512
```

Here `512` is the base reservation for the model weights plus the target context, and `mmproj file size (MB)` is the value obtained in step 1. Each Vision choice below already lists a `--fit-target` based on an estimated mmproj size. If the real mmproj file differs substantially from the estimate, adjust it yourself using the formula above.

## Estimated `--fit-target` per model

| Model | Estimated mmproj | `--fit-target` |
| ---------------------------- | ----------- | -------------- |
| Gemma4-26B-A4B Vision (MoE)  | ~2200 MB    | 2700           |
| Qwen3.6-35B-A3B Vision (MoE) | ~900 MB     | 1400           |
| Gemma4-E4B Vision            | ~1900 MB    | 2400           |
| Qwen3.5-9B Vision            | ~900 MB     | 1400           |
| Gemma4-12B Vision            | ~200 MB     | 700            |
| Qwen3.6-27B Vision           | ~900 MB     | 1400           |

---

## Examples: MoE models

### Choice 1 — Gemma-4-26B-A4B Text MoE (32K)

```bat
llama-fit-params -m ./models/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --fit-ctx 32768 --fit-target 512
```

Output:

```text
-c 32768 -ngl 31 -ot "blk\.9\.ffn_down.*=CPU,blk\.10\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.11\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.12\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.13\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.14\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.15\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.16\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.17\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.18\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.19\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.20\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.21\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.22\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.23\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.24\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.25\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.26\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.27\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.28\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.29\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.30\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
```

---

### Choice 2 — Gemma-4-26B-A4B Vision MoE (32K)

Estimated mmproj ~2200 MB; by the formula at the top, `--fit-target` = `2200 + 512 = 2712`, rounded down to 2700 here.

```bat
llama-fit-params -m ./models/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --fit-ctx 32768 --fit-target 2700
```

Output:

```text
-c 32768 -ngl 31 -ot "blk\.4\.ffn_(gate|gate_up|down).*=CPU,blk\.5\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.6\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.7\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.8\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.9\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.10\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.11\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.12\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.13\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.14\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.15\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.16\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.17\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.18\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.19\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.20\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.21\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.22\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.23\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.24\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.25\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.26\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.27\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.28\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.29\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.30\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
```

---

### Choice 3 — Qwen3.6-35B-A3B Text MoE (32K)

```bat
llama-fit-params -m ./models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf --fit-ctx 32768 --fit-target 512 --cache-type-k q4_0 --cache-type-v q4_0
```

Output:

```text
-c 32768 -ngl 41 -ot "blk\.10\.ffn_(gate|gate_up|down).*=CPU,blk\.11\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.12\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.13\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.14\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.15\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.16\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.17\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.18\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.19\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.20\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.21\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.22\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.23\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.24\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.25\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.26\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.27\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.28\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.29\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.30\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.31\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.32\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.33\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.34\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.35\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.36\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.37\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.38\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.39\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.40\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
```

---

### Choice 4 — Qwen3.6-35B-A3B Vision MoE (32K)

Estimated mmproj ~900 MB; by the formula at the top, `--fit-target` = `900 + 512 = 1412`, rounded down to 1400 here.

```bat
llama-fit-params -m ./models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf --fit-ctx 32768 --fit-target 1400 --cache-type-k q4_0 --cache-type-v q4_0
```

Output:

```text
-c 32768 -ngl 41 -ot "blk\.8\.ffn_(gate|gate_up|down).*=CPU,blk\.9\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.10\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.11\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.12\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.13\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.14\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.15\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.16\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.17\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.18\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.19\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.20\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.21\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.22\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.23\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.24\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.25\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.26\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.27\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.28\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.29\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.30\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.31\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.32\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.33\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.34\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.35\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.36\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.37\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.38\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.39\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.40\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
```

---

## Examples: dense models

Dense models should ideally run entirely on the GPU: as soon as CPU + GPU hybrid inference kicks in, generation speed drops sharply. So try to fit every layer into VRAM first, and only step down `-ngl` or move to a lower quantization level when it genuinely doesn't fit.

### Choice 5 — Gemma-4-E4B-it Text (128K)

```bat
llama-fit-params -m ./models/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf --fit-ctx 131072 --fit-target 512
```

Output:

```text
-c 0 -ngl -1
```

---

### Choice 6 — Gemma-4-E4B-it Vision (128K)

Estimated mmproj ~1900 MB; by the formula at the top, `--fit-target` = `1900 + 512 = 2412`, rounded down to 2400 here.

```bat
llama-fit-params -m ./models/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf --fit-ctx 131072 --fit-target 2400
```

Output:

```text
-c 0 -ngl 40
```

---

### Choice 7 — DeepSeek-R1-0528-Qwen3-8B Text (32K)

```bat
llama-fit-params -m ./models/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf --fit-ctx 32768 --fit-target 512 --cache-type-k q8_0 --cache-type-v q8_0
```

Output:

---

### Choice 8 — Qwen3.5-9B Text (64K)

```bat
llama-fit-params -m ./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf --fit-ctx 85760 --fit-target 512 --cache-type-k q8_0 --cache-type-v q8_0
```

Output:

---

### Choice 9 — Qwen3.5-9B Vision (32K)

Estimated mmproj ~900 MB; by the formula at the top, `--fit-target` = `900 + 512 = 1412`, rounded down to 1400 here.

```bat
llama-fit-params -m ./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf --fit-ctx 32768 --fit-target 1400 --cache-type-k q8_0 --cache-type-v q8_0
```

Output:

---

### Choice 10 — Gemma-4-12B Text (32K)

```bat
llama-fit-params -m ./models/gemma-4-12B-it-QAT-GGUF/gemma-4-12b-it-qat-q4_0.gguf --fit-ctx 32768 --fit-target 512
```

Output:

---

### Choice 11 — Gemma-4-12B Vision (32K)

Estimated mmproj ~200 MB; by the formula at the top, `--fit-target` = `200 + 512 = 712`, rounded down to 700 here.

```bat
llama-fit-params -m ./models/gemma-4-12B-it-QAT-GGUF/gemma-4-12b-it-qat-q4_0.gguf --fit-ctx 32768 --fit-target 700
```

Output:

---

### Choice 12 — Qwen3.6-27B Text (8K)

```bat
llama-fit-params -m ./models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf --fit-ctx 8192 --fit-target 512 --cache-type-k q8_0 --cache-type-v q8_0
```

Output:

---

### Choice 13 — Qwen3.6-27B Vision (8K)

Estimated mmproj ~900 MB; by the formula at the top, `--fit-target` = `900 + 512 = 1412`, rounded down to 1400 here.

```bat
llama-fit-params -m ./models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf --fit-ctx 8192 --fit-target 1400 --cache-type-k q8_0 --cache-type-v q8_0
```

Output:

---

## Using the output

Each command prints something like `-c 0 -ngl 33`, or `-c 0 -ngl 33 -ot "..."` for MoE models. **Append these values directly to your `llama-server` launch command**; leave the remaining sampling parameters untouched.
