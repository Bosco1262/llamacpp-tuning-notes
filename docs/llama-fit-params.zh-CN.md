# llama-fit-params 使用讲解

[English](llama-fit-params.md) | [简体中文](llama-fit-params.zh-CN.md)

`llama-fit-params` 用于自动计算和优化大模型在本地运行时的显存分配参数。它只打印计算结果、不真正启动推理，因此适合在正式启动前先确认显存预算；实际运行时可以开启 `--fit on`，搭配 `--fit-ctx N`（最小上下文大小）和 `--fit-target N`（为每个 GPU 预留的空闲内存量）来动态调整资源分配。

## 模型分类

- **Choice 1~4**：MoE 模型（参数量从小到大）
- **Choice 5~13**：稠密模型（参数量从小到大）

## 视觉模型（Vision）使用说明

视觉模型需要额外加载 mmproj 文件（CLIP 投影），这部分会占用额外显存。

步骤：

1. 先用 `dir <mmproj路径>` 查看 mmproj 文件实际大小
2. 然后运行 `llama-fit-params`，`--fit-target` 按以下公式计算：

```
--fit-target = mmproj文件大小(MB) + 512
```

其中 `512` 是模型权重 + 目标上下文的基础预留值，`mmproj文件大小(MB)` 为步骤 1 查到的值。各 Vision Choice 中已给出基于预估 mmproj 大小的 `--fit-target` 值。如果实际 mmproj 文件大小与表中预估值偏差较大，请按上述公式自行调整。

## 各模型 `--fit-target` 预估对照

| 模型                         | 预估 mmproj | `--fit-target` |
| ---------------------------- | ----------- | -------------- |
| Gemma4-26B-A4B Vision (MoE)  | ~2200 MB    | 2700           |
| Qwen3.6-35B-A3B Vision (MoE) | ~900 MB     | 1400           |
| Gemma4-E4B Vision            | ~1900 MB    | 2400           |
| Qwen3.5-9B Vision            | ~900 MB     | 1400           |
| Gemma4-12B Vision            | ~200 MB     | 700            |
| Qwen3.6-27B Vision           | ~900 MB     | 1400           |

---

## 使用示例：MoE 模型

### Choice 1 — Gemma-4-26B-A4B Text MoE（32K）

```bat
llama-fit-params -m ./models/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --fit-ctx 32768 --fit-target 512
```

输出：

```text
-c 32768 -ngl 31 -ot "blk\.9\.ffn_down.*=CPU,blk\.10\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.11\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.12\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.13\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.14\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.15\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.16\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.17\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.18\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.19\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.20\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.21\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.22\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.23\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.24\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.25\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.26\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.27\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.28\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.29\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.30\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
```

---

### Choice 2 — Gemma-4-26B-A4B Vision MoE（32K）

mmproj 预估 ~2200 MB，`--fit-target` 按顶部公式为 `2200 + 512 = 2712`，此处使用 2700。

```bat
llama-fit-params -m ./models/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --fit-ctx 32768 --fit-target 2700
```

输出：

```text
-c 32768 -ngl 31 -ot "blk\.4\.ffn_(gate|gate_up|down).*=CPU,blk\.5\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.6\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.7\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.8\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.9\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.10\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.11\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.12\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.13\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.14\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.15\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.16\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.17\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.18\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.19\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.20\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.21\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.22\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.23\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.24\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.25\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.26\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.27\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.28\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.29\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.30\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
```

---

### Choice 3 — Qwen3.6-35B-A3B Text MoE（32K）

```bat
llama-fit-params -m ./models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf --fit-ctx 32768 --fit-target 512 --cache-type-k q4_0 --cache-type-v q4_0
```

输出：

```text
-c 32768 -ngl 41 -ot "blk\.10\.ffn_(gate|gate_up|down).*=CPU,blk\.11\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.12\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.13\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.14\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.15\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.16\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.17\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.18\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.19\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.20\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.21\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.22\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.23\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.24\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.25\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.26\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.27\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.28\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.29\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.30\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.31\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.32\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.33\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.34\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.35\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.36\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.37\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.38\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.39\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.40\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
```

---

### Choice 4 — Qwen3.6-35B-A3B Vision MoE（32K）

mmproj 预估 ~900 MB，`--fit-target` 按顶部公式为 `900 + 512 = 1412`，此处使用 1400。

```bat
llama-fit-params -m ./models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf --fit-ctx 32768 --fit-target 1400 --cache-type-k q4_0 --cache-type-v q4_0
```

输出：

```text
-c 32768 -ngl 41 -ot "blk\.8\.ffn_(gate|gate_up|down).*=CPU,blk\.9\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.10\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.11\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.12\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.13\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.14\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.15\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.16\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.17\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.18\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.19\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.20\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.21\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.22\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.23\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.24\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.25\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.26\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.27\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.28\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.29\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.30\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.31\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.32\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.33\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.34\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.35\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.36\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.37\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.38\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.39\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.40\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
```

---

## 使用示例：稠密模型

稠密模型建议尽量全部跑在 GPU 上：一旦出现 CPU + GPU 混合推理，生成速度会明显下降。因此优先尝试把全部层放进显存，只有在装不下时才逐步下调 `-ngl` 或改用更低的量化档位。

### Choice 5 — Gemma-4-E4B-it Text（128K）

```bat
llama-fit-params -m ./models/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf --fit-ctx 131072 --fit-target 512
```

输出：

```text
-c 0 -ngl -1
```

---

### Choice 6 — Gemma-4-E4B-it Vision（128K）

mmproj 预估 ~1900 MB，`--fit-target` 按顶部公式为 `1900 + 512 = 2412`，此处使用 2400。

```bat
llama-fit-params -m ./models/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf --fit-ctx 131072 --fit-target 2400
```

输出：

```text
-c 0 -ngl 40
```

---

### Choice 7 — DeepSeek-R1-0528-Qwen3-8B Text（32K）

```bat
llama-fit-params -m ./models/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf --fit-ctx 32768 --fit-target 512 --cache-type-k q8_0 --cache-type-v q8_0
```

输出：

---

### Choice 8 — Qwen3.5-9B Text（64K）

```bat
llama-fit-params -m ./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf --fit-ctx 85760 --fit-target 512 --cache-type-k q8_0 --cache-type-v q8_0
```

输出：

---

### Choice 9 — Qwen3.5-9B Vision（32K）

mmproj 预估 ~900 MB，`--fit-target` 按顶部公式为 `900 + 512 = 1412`，此处使用 1400。

```bat
llama-fit-params -m ./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf --fit-ctx 32768 --fit-target 1400 --cache-type-k q8_0 --cache-type-v q8_0
```

输出：

---

### Choice 10 — Gemma-4-12B Text（32K）

```bat
llama-fit-params -m ./models/gemma-4-12B-it-QAT-GGUF/gemma-4-12b-it-qat-q4_0.gguf --fit-ctx 32768 --fit-target 512
```

输出：

---

### Choice 11 — Gemma-4-12B Vision（32K）

mmproj 预估 ~200 MB，`--fit-target` 按顶部公式为 `200 + 512 = 712`，此处使用 700。

```bat
llama-fit-params -m ./models/gemma-4-12B-it-QAT-GGUF/gemma-4-12b-it-qat-q4_0.gguf --fit-ctx 32768 --fit-target 700
```

输出：

---

### Choice 12 — Qwen3.6-27B Text（8K）

```bat
llama-fit-params -m ./models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf --fit-ctx 8192 --fit-target 512 --cache-type-k q8_0 --cache-type-v q8_0
```

输出：

---

### Choice 13 — Qwen3.6-27B Vision（8K）

mmproj 预估 ~900 MB，`--fit-target` 按顶部公式为 `900 + 512 = 1412`，此处使用 1400。

```bat
llama-fit-params -m ./models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf --fit-ctx 8192 --fit-target 1400 --cache-type-k q8_0 --cache-type-v q8_0
```

输出：

---

## 使用输出结果

每个命令运行后会输出类似 `-c 0 -ngl 33` 或 `-c 0 -ngl 33 -ot "..."`（MoE 模型）。**把这些值直接拼进 `llama-server` 的启动命令**，其余采样参数保持原样即可。
