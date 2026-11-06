# llama.cpp 性能参数完整指南

[English](llamacpp-performance-parameters-guide.md) | [简体中文](llamacpp-performance-parameters-guide.zh-CN.md)

> **关于本文档**：llama.cpp 是一个更新极快的项目，参数名称、默认值几乎每个月都有变化。本文基于 2026 年 7 月对官方仓库 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 主分支文档（`llama-server`、`llama-bench` 官方 README 等）及多份社区实测资料整理，力求准确、完整。但请注意：**具体参数以你所用版本的 `llama-cli --help` / `llama-server --help` 输出为准**，本文尽量标注了参数的新旧程度。
>
> **范围说明**：本文只收录**影响推理速度、显存/内存占用、吞吐量**的参数。采样参数（`temperature`、`top-p`、`top-k`、`repeat-penalty`、Mirostat、DRY、XTC、grammar/json-schema 等)只影响输出内容和风格，**不影响 token/s**，因此不在本文范围内，可放心调整而无需担心性能。

## 目录

1. [三大核心工具速览](#1-三大核心工具速览)
2. [快速诊断：症状 → 该调的参数](#2-快速诊断症状--该调的参数)
3. [线程与 CPU 调度参数](#3-线程与-cpu-调度参数)
4. [上下文长度与批处理参数](#4-上下文长度与批处理参数)
5. [GPU 卸载与显存参数](#5-gpu-卸载与显存参数)
6. [混合专家模型（MoE）专用卸载参数](#6-混合专家模型moe专用卸载参数)
7. [Flash Attention 与 KV Cache 量化参数](#7-flash-attention-与-kv-cache-量化参数)
8. [内存映射、锁定与加载参数](#8-内存映射锁定与加载参数)
9. [多 GPU 与分布式推理参数](#9-多-gpu-与分布式推理参数)
10. [自动显存适配 --fit 系列（较新特性）](#10-自动显存适配---fit-系列较新特性)
11. [RoPE 缩放与长上下文扩展参数](#11-rope-缩放与长上下文扩展参数)
12. [推测解码（Speculative Decoding）参数](#12-推测解码speculative-decoding参数)
13. [服务器并发与吞吐参数](#13-服务器并发与吞吐参数)
14. [模型量化格式对性能的影响](#14-模型量化格式对性能的影响)
15. [编译后端选择（简述）](#15-编译后端选择简述)
16. [llama-bench 基准测试完整指南](#16-llama-bench-基准测试完整指南)
17. [实战调优方案精选](#17-实战调优方案精选)
18. [高影响参数速查表](#18-高影响参数速查表)
19. [性能计时与诊断参数（--perf）](#19-性能计时与诊断参数--perf)
20. [参考资料](#20-参考资料)

---

## 1. 三大核心工具速览

| 工具 | 用途 | 典型调用 |
|---|---|---|
| `llama-cli` | 命令行单次推理 / 交互对话 | `llama-cli -m model.gguf -p "你好" -ngl 99` |
| `llama-server` | 常驻 HTTP 服务，OpenAI 兼容 API | `llama-server -m model.gguf -ngl 99 --port 8080` |
| `llama-bench` | 专用性能基准测试工具，不含采样/分词耗时 | `llama-bench -m model.gguf -ngl 99` |

三者共享同一套底层参数解析逻辑（`common_params`），所以**本文列出的绝大多数参数在三个工具里通用**，仅 `llama-bench` 的部分参数支持"逗号分隔多值 / 区间扫描"的批量测试语法（见第 16 节），`llama-server` 额外有并发（slots）相关参数（见第 13 节）。

---

## 2. 快速诊断：症状 → 该调的参数

| 症状 | 优先检查的参数 |
|---|---|
| 显存不足（OOM / CUDA out of memory） | 降低 `-ngl`；开启 `-ctk q8_0 -ctv q8_0`；减小 `-c`；对 MoE 模型用 `--n-cpu-moe` / `-ot`；确认 `-fit on` |
| 生成速度（tg，逐 token 输出）慢 | `-ngl` 尽量拉满到全部层；开 `-fa on`；量化 KV 缓存；MoE 模型调低 `--n-cpu-moe`；换更小的量化格式；纯 CPU 时检查线程数 |
| 提示词处理（pp，首 token 延迟）慢 | 增大 `-b` / `-ub`；`-ngl` 拉满；开 `-fa on`；确认 `--op-offload` 未被关闭 |
| 多用户并发卡顿、排队 | 调整 `-np`（并行槽数）；确认 `-cb`（连续批处理）开启；`--cache-reuse`、`-sps` |
| 长上下文（32K+）跑不动/超显存 | 增大 `-c` 前先量化 KV（`-ctk/-ctv`）；`--rope-scaling yarn` 做上下文扩展；`--swa-full` |
| 多卡没有明显加速甚至更慢 | 检查 `-sm` 模式是否适合你的互联方式（无 NVLink 用 `layer`）；`-ts` 配比是否匹配显存；尝试 `GGML_CUDA_P2P=1` |
| 模型加载很慢 | 保持 `--mmap` 默认开启；SSD 用户可试 `-dio` |
| CPU 推理吃满所有线程反而更慢 | 把 `-t` 降到物理核心数（而非逻辑线程数），必要时再降 1-2 |

---

## 3. 线程与 CPU 调度参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-t, --threads N` | `-1`（自动） | 生成阶段使用的 CPU 线程数 |
| `-tb, --threads-batch N` | 同 `--threads` | 提示词处理 / 批处理阶段线程数 |
| `-C, --cpu-mask M` | `""` | 十六进制 CPU 亲和性掩码 |
| `-Cr, --cpu-range lo-hi` | — | CPU 亲和性区间，与 `-C` 二选一 |
| `--cpu-strict <0\|1>` | `0` | 是否严格绑核 |
| `--prio N` | `0` | 进程/线程优先级：低(-1)、正常(0)、中(1)、高(2)、实时(3) |
| `--poll <0...100>` | `50` | 等待任务时的轮询强度（越高越占 CPU，但延迟更低） |
| `-Cb, --cpu-mask-batch M` | 同 `-C` | 批处理阶段专用掩码 |
| `-Crb, --cpu-range-batch lo-hi` | — | 批处理阶段专用区间 |
| `--numa TYPE` | 关闭 | NUMA 优化：`distribute`（均分到各节点）/ `isolate`（只用启动节点）/ `numactl`（使用 numactl 的映射） |

### 调优建议

- **线程数不是越多越好**。官方线程数基准测试（7B Q4_0，CPU 后端）显示：

  | 线程数 | pp64 (t/s) | tg16 (t/s) |
  |---|---|---|
  | 1 | 6.17 | 4.05 |
  | 2 | 12.31 | 7.80 |
  | 4 | 23.18 | 12.22 |
  | 8 | 32.29 | 16.71 |
  | 16 | 33.52 | **15.32**（已下降） |
  | 32 | 59.00 | **16.41**（提示词处理仍在涨，生成几乎不再涨） |

  可以看到：**提示词处理（pp）是高度并行的任务，线程数越多通常越快**；但**逐 token 生成（tg）本质上是串行的，受内存带宽限制**，超过物理核心数后收益迅速归零甚至变负（缓存争用、超线程/SMT 争抢共享执行单元）。
- 经验法则：`-t` 设为**物理核心数**（不是逻辑线程数，超线程/SMT 通常不带来提升），必要时留 1-2 个核心给操作系统/其他进程；`-tb` 可以设得更高一些，因为提示词处理更吃并行度。
- 若整个模型已经 `-ngl 99` 全部卸载到 GPU，线程数对生成速度的影响通常 **< 10%**，不必过度纠结。
- Intel 12代+ 的 P-core/E-core 混合架构、AMD 的 CCD 边界，都可能让"全部核心拉满"反而更慢，建议用 `llama-bench -t 4,8,12,16,...` 实测你自己的硬件（见第 16 节）。
- 双路服务器（多 NUMA 节点）：llama.cpp 目前对跨节点访问处理得并不优雅，跨 socket 访问会明显拖慢速度。若模型能放进单个 NUMA 节点的内存里，优先把线程绑定在该节点（`--numa isolate` 或用 `numactl` 绑定）；放不下时再尝试 `--numa distribute`。

---

## 4. 上下文长度与批处理参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-c, --ctx-size N` | `0`（从模型元数据读取） | 上下文窗口大小（token 数），直接决定 KV 缓存显存占用 |
| `-n, --predict, --n-predict N` | `-1`（无限） | 最多生成的 token 数 |
| `-b, --batch-size N` | `2048` | **逻辑**最大批大小（一次能提交处理的 token 数上限） |
| `-ub, --ubatch-size N` | `512` | **物理**最大批大小（真正一次送入计算图的 token 数，决定计算缓冲区显存） |
| `--keep N` | `0`（`-1` = 全部） | 上下文超限时，从初始提示词中保留的 token 数 |
| `--swa-full` | `false` | 对使用滑动窗口注意力（SWA）的模型使用全尺寸 SWA 缓存 |

### 调优建议

- `-b`（逻辑批大小）与 `-ub`（物理批大小）：`-ub` 不能超过 `-b`。**只有提示词处理阶段（prefill）明显受批大小影响**，逐 token 生成阶段批大小通常固定为 1，几乎不受影响。
- 官方批大小基准（7B Q4_0，CUDA，`-p 1024`）：

  | `-b` | pp1024 (t/s) |
  |---|---|
  | 128 | 1436.51 |
  | 256 | 1932.43 |
  | 512 | 2254.45 |
  | 1024 | 2498.61 |

  增大批大小能显著提升提示词处理吞吐量，但会线性增加计算缓冲区的显存占用。
- **CPU+GPU 混合推理（尤其是 MoE 模型部分卸载到 CPU）对批大小格外敏感**：默认的 `-b 2048 -ub 512` 是为纯 GPU 推理调的，对于把专家权重放在 CPU 上的场景通常偏小；把 `-b`/`-ub` 提到 `2048~4096`（甚至相等）往往能显著提升吞吐，因为这决定了触发"一次性拷贝 CPU 权重到 GPU 批量计算"的阈值（默认阈值 32 token，可用环境变量 `GGML_OP_OFFLOAD_MIN_BATCH` 调整）。
- `-c 0` 让模型使用其元数据里记录的原生训练上下文长度；若你要的上下文超过原生长度，需要配合第 11 节的 RoPE/YaRN 参数。
- 上下文越大，KV 缓存越大（近似与 `-c` 线性相关），是长上下文场景下显存不足的头号原因，参见第 7 节的 KV 缓存量化。

---

## 5. GPU 卸载与显存参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-ngl, --gpu-layers, --n-gpu-layers N` | `auto` | 卸载到显存的模型层数；填一个很大的数（如 `99` / `999`）等价于"全部卸载" |
| `-sm, --split-mode {none,layer,row,tensor}` | `layer` | 多卡拆分方式，见下表 |
| `-ts, --tensor-split N0,N1,...` | 按显存自动分配 | 各 GPU 分配模型的比例，如 `3,1` |
| `-mg, --main-gpu INDEX` | `0` | 主 GPU（`split-mode=none` 时的唯一使用卡；`row` 模式下用于存放中间结果与 KV） |
| `-dev, --device <dev1,dev2,..>` | 全部可用设备 | 指定参与卸载的设备列表；`none` = 不卸载 |
| `--list-devices` | — | 打印可用设备列表后退出 |
| `-ot, --override-tensor <正则>=<设备>` | — | 用正则表达式精细指定"某些张量放哪个设备"，最灵活但需手工调试 |
| `-kvo, --kv-offload` | 启用 | KV 缓存是否放入显存 |
| `--op-offload` | 启用(`true`) | 是否把 CPU 侧张量运算临时卸载到 GPU 执行（对 CPU+GPU 混合推理的提示词处理速度影响很大） |
| `--repack` | 启用 | 是否对量化权重按 CPU SIMD 指令集（AVX/NEON 等）重新打包排布以加速 CPU 计算 |

### `-sm` 三种拆分模式对比

| 模式 | 原理 | 适用场景 |
|---|---|---|
| `none` | 只用一张卡（`-mg` 指定） | 单卡够用，或想强制只用某一张卡 |
| `layer`（默认） | 流水线并行：不同层放不同卡，按顺序执行 | **最通用、兼容性最好**，对卡间带宽要求低，PCIe 消费级多卡首选 |
| `row` | 把同一层的权重矩阵按行拆到多卡，理论上可并行计算 | 官方文档已将其描述为**较旧、性能普遍偏弱**的张量并行路径；没有 NVLink 等高速互联时，在消费级 PCIe 平台上通常比 `layer` 更慢 |
| `tensor` | 更彻底的张量级拆分，多卡同时算同一层（实验性） | 需要较快的卡间互联（如 NVLink）才能体现优势，否则通信开销会成为瓶颈 |

### 调优建议

- `-ngl 99`（或更大的数）是"全部卸载到 GPU"的通用写法：只要该值 ≥ 模型总层数，llama.cpp 会自动裁剪到实际层数。
- **部分卸载存在明显的"悬崖效应"**——官方基准（7B Q4_0，35 层，CUDA）：

  | `-ngl` | pp512 (t/s) | tg128 (t/s) |
  |---|---|---|
  | 10 | 373 | 13.45 |
  | 20 | 473 | 21.36 |
  | 30 | 632 | 40.04 |
  | 33 | 704 | 60.62 |
  | 34 | 881 | 71.76 |
  | 35（全部） | **2400** | **131.66** |

  最后几层从 CPU 挪到 GPU 时提速最明显，这是因为不再需要在 CPU/GPU 之间来回搬运中间结果。**只要显存允许，优先把 `-ngl` 拉满**，实在放不下再考虑第 6 节的 MoE 专用卸载或降量化档位。
- 多卡：显存不均衡时用 `-ts` 按各卡剩余显存比例分配（如三张卡 32G/16G/8G，可设 `-ts 32,16,6`，给小卡留出余量），`-mg` 建议设为带宽/算力最强的那张卡。
- NVIDIA 多卡可尝试启用 P2P 直连（需驱动支持，工作站/数据中心卡更常见）：
  ```bash
  GGML_CUDA_P2P=1 llama-server -m model.gguf -sm tensor ...
  ```
  部分主板/BIOS（尤其开启 IOMMU 时）启用后可能不稳定，出问题就撤掉这个环境变量。
- `-ot` 是终极武器：可以精确指定"哪些张量放 CPU、哪些放哪张 GPU"，常见于 MoE 大模型的手工调优（见第 6 节），但需要了解目标模型的张量命名规则。

---

## 6. 混合专家模型（MoE）专用卸载参数

MoE（Mixture-of-Experts，如 Qwen3-MoE、DeepSeek-V3、GLM-4.x、Kimi-K2、gpt-oss 等）模型的参数分两类：

- **"常驻激活"部分**（attention、embedding、共享专家、输出层等）：每个 token 都要用到，适合放显存。
- **路由专家（routed experts）**：每个 token 只激活其中一小部分，但因为路由结果逐 token 变化，**全部专家都要常驻内存**，占了模型绝大部分体积。

核心思路：**把"常驻激活"部分留在 GPU，把路由专家权重放到系统内存（CPU）**，只在需要时把很小的激活向量在 GPU/CPU 间传递，而不是搬运整个专家权重。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-cmoe, --cpu-moe` | 关闭 | 把**全部** MoE 专家权重放到 CPU，最简单的起手配置 |
| `-ncmoe, --n-cpu-moe N` | `0` | 只把**前 N 层**的 MoE 专家权重放到 CPU，其余仍按 `-ngl` 走显存；N 越大，省显存越多但也越慢 |
| `-ot "exps=CPU"` | — | 用正则匹配所有专家张量强制放 CPU，效果类似 `--cpu-moe` 但可读性更好、可扩展性更强 |

### 调优建议

- 最简单的起点（把 attention/dense FFN/共享专家/KV 缓存都放 GPU，把路由专家放 CPU）：
  ```bash
  llama-server -m moe-model.gguf -ngl 999 -ot "exps=CPU" -fa on
  ```
- 显存有富余但装不下全部专家时，用 `--n-cpu-moe` 从 0 开始逐步增加，配合 `llama-bench` 或实际显存占用（`nvidia-smi`）找到刚好不 OOM 的最小值：
  ```bash
  # gpt-oss-20b，8GB 显存机器上的官方示例
  llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 32768 --n-cpu-moe 16 -ub 2048 -b 2048 --jinja
  ```
- 多卡 + MoE 时可用 `-ot` 精细分配各层去哪张卡、专家去 CPU，例如：
  ```bash
  -ot "blk\.([0-9])\.=CUDA0,blk\.(1[0-9])\.=CUDA1,exps=CPU"
  ```
- **CPU+GPU 混合推理对 `-b`/`-ub` 十分敏感**（详见第 4 节），建议同时调大到 2048～4096。
- **PCIe 带宽会成为瓶颈**：如果 CPU 侧权重很大而显卡走 PCIe（非 NVLink），单批次要传输的数据量可能达到"需要数秒才能完成一次拷贝"的量级，此时增大批大小反而更划算（一次多做点事，分摊传输开销）。
- 双路服务器 NUMA 问题在 MoE + CPU 卸载场景同样适用，见第 3 节末尾建议。

---

## 7. Flash Attention 与 KV Cache 量化参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-fa, --flash-attn [on\|off\|auto]` | `auto` | 是否使用 Flash Attention 内核 |
| `-ctk, --cache-type-k TYPE` | `f16` | K 缓存数据类型：`f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1` |
| `-ctv, --cache-type-v TYPE` | `f16` | V 缓存数据类型，可选值同上 |
| `-ctxcp, --ctx-checkpoints N` | `32` | 每个 slot 最多保留的上下文检查点数（用于长对话增量复用） |
| `-cms, --checkpoint-min-step N` | `256` | 检查点之间的最小 token 间隔 |
| `-cram, --cache-ram N` | `8192`（MiB） | 提示词缓存占用内存上限（`-1` 无限制，`0` 关闭） |
| `-kvu, --kv-unified` | slots 为自动时默认启用 | 是否让所有并发序列共用一块 KV 缓冲区 |

### 调优建议

- **量化 KV 缓存通常要求开启 Flash Attention**（`-fa on`），二者建议一起用，几乎没有理由只开一个。
- **`q8_0` KV 缓存是"近乎免费的午餐"**：显存减半，多份实测显示困惑度（perplexity）损失在 0.1% 量级以内，建议作为长上下文场景的默认选择：
  ```bash
  llama-server -m model.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 32768
  ```
- **`q4_0` 再减半显存，但要更谨慎**：
  - 质量方面：社区实测在"K/V 谁更敏感"上存在**不完全一致的结论**——多数资料认为 V 缓存（值向量，直接参与输出内容加权求和）对低精度更敏感，K 缓存（参与注意力权重计算）相对耐量化；但也有资料给出相反结论。建议把它当作"因模型而异"，如果对输出质量有要求，先用你自己的任务实测（可以只量化 K、保留 V 为 `f16` 做非对称配置），而不要直接假设某一方向必然更安全。
  - 速度方面：多个来源报告在**长上下文**下 `q4_0` 的反量化开销可能抵消甚至反超省下的带宽收益，个别场景在 64K 上下文下实测**比 f16 慢**（因为注意力计算时要临时反量化，开销随上下文长度增长）。如果启用 `q4_0` 后速度不升反降，先退回 `q8_0` 或 `f16` 试试。
- 经验起点：**默认用 `-ctk q8_0 -ctv q8_0`**；显存实在紧张、且已验证过特定模型/任务的质量可接受时再尝试 `q4_0`；对质量要求高、显存又充足时可以保持 `f16`（默认值）。

---

## 8. 内存映射、锁定与加载参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mmap, --no-mmap` | 启用 | 是否用内存映射方式加载模型文件（关闭后加载变慢，但配合无 `mlock` 时可能减少换页） |
| `--mlock` | 关闭 | 强制把模型锁在物理内存中，防止被系统换出到交换分区/压缩内存 |
| `-dio, --direct-io` | 关闭 | 若后端支持，使用 Direct I/O 加载，绕过系统页缓存（对高速 NVMe 有帮助，但会跳过缓存复用） |
| `--no-host` | 关闭 | 绕过 host buffer，允许使用更多缓冲区（进阶选项） |
| `--check-tensors` | 关闭 | 加载时校验张量数值是否合法（会拖慢加载速度，一般只在排查问题时用） |

### 调优建议

- 默认的 `--mmap` 开启对绝大多数场景都是对的：加载更快，多进程可共享同一份模型的页缓存。
- 内存/显存紧张、担心系统把模型换出时，加 `--mlock`（会占用等量的常驻物理内存，不适合内存也紧张的机器）。
- 某些 CPU+GPU 混合卸载（尤其大 MoE 模型反复调参测试时）会看到有人加 `--no-mmap`，主要是为了避免测试过程中的页缓存状态影响本次结果的可比性，不代表它总是更快，正式部署一般保持默认。

---

## 9. 多 GPU 与分布式推理参数

除第 5 节的 `-sm` / `-ts` / `-mg` / `-dev` 外，还有：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-rpc, --rpc <rpc_servers>` | 无 | 注册远程 RPC 计算节点（逗号分隔），可以把模型/计算摊到局域网内的多台机器上 |
| `GGML_CUDA_P2P=1`（环境变量） | 关闭 | 允许 NVIDIA 显卡间直接 P2P 传输数据，跳过系统内存中转 |

### 调优建议

- 单卡显存够用时，**不要为了"用满所有显卡"而强行多卡**——层间流水线并行（`layer`）在生成阶段无法让多张卡同时忙碌，实际吞吐提升有限，甚至可能因为调度开销更慢。多卡更适合"单卡装不下模型"的场景。
- 追求真正的吞吐扩展（多卡同时算），需要 `-sm tensor` 或 `-sm row` + 快速互联（NVLink），普通 PCIe 消费级多卡收益有限，`layer` 通常仍是更稳的选择。
- `-rpc` 适合"我有好几台机器但每台显存都不够"的场景，把模型拆到多机上跑，但要接受局域网带宽带来的额外延迟。

---

## 10. 自动显存适配 `--fit` 系列（较新特性）

这是 llama.cpp 相对较新加入的功能：**在没有显式指定 `-ngl` / `-ot` / `-c` 等参数时，自动根据当前设备可用显存反推一套能跑起来的配置**（层数分配、是否卸载到 CPU、必要时收缩上下文等）。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-fit, --fit [on\|off]` | `on` | 是否启用自动显存适配 |
| `-fitt, --fit-target MiB0,MiB1,...` | `1024` | 每个设备预留的显存余量（MiB），可对每张卡单独设置 |
| `-fitc, --fit-ctx N` | `4096` | `--fit` 收缩上下文时允许的最小上下文大小 |

配套的诊断工具 `llama-fit-params` 可以**只打印**自动适配算出来的参数，而不真正启动推理，便于你把结果抄下来做成固定配置：

```bash
llama-fit-params --model model.gguf -c 65536
# 输出形如：
# llama_params_fit_impl: context size reduced from 65536 to 4096 -> ...
# -c 4096 -ngl 41 -ot "blk\.22\.ffn_(gate|down).*=CPU,..."
```

### 调优建议

- 新手/快速验证阶段：**保持 `-fit on`（默认）**，不指定 `-ngl`，让 llama.cpp 自己算，通常能得到一个"能跑起来"的保守配置。
- 精细调优阶段（例如按第 6 节手工分配 MoE 层）：**建议 `-fit off`**，否则自动适配可能悄悄改动你没显式指定的参数，让你难以判断某次调整到底是谁生效了。
- 已知问题：多模态模型的 `mmproj`（视觉投影层）显存目前**不在**自动适配的估算范围内，可能出现"看起来适配成功但加载 mmproj 时 OOM"的情况——这类模型建议手动加大 `-fitt` 余量，或用 `--no-mmproj-offload` 把投影层留在 CPU。

---

## 11. RoPE 缩放与长上下文扩展参数

用于把模型扩展到超出原生训练长度的上下文窗口。这类参数**不直接影响 token/s**，但决定了"能不能开到那么大的上下文"以及扩展后的输出质量，间接关系到显存占用（更长上下文 = 更大 KV 缓存）。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--rope-scaling {none,linear,yarn}` | 按模型自带配置，一般为 `linear` | 频率缩放方式 |
| `--rope-scale N` | — | 上下文扩展倍数（等价于线性缩放的另一种写法） |
| `--rope-freq-base N` | 模型自带值 | RoPE 基频，用于 NTK-aware 缩放 |
| `--rope-freq-scale N` | — | 频率缩放因子，扩展至 `1/N` 倍上下文 |
| `--yarn-orig-ctx N` | `0`（= 模型原始训练长度） | YaRN：原始上下文长度 |
| `--yarn-ext-factor N` | `-1.00` | YaRN：外推混合因子（`0.0` = 完全插值） |
| `--yarn-attn-factor N` | `-1.00` | YaRN：注意力幅度缩放 |
| `--yarn-beta-slow / --yarn-beta-fast N` | `-1.00` | YaRN：低/高频修正维度参数 |

### 调优建议

- 若模型本身原生支持的上下文已经够用（元数据里的训练长度），**不需要碰这些参数**，直接调 `-c` 即可。
- 需要把上下文扩展到超出原生训练长度时：
  ```bash
  # 简单线性扩展 4 倍
  llama-cli -m model.gguf -c 16384 --rope-scaling linear --rope-freq-scale 0.25

  # YaRN（通常比线性缩放质量更好）
  llama-cli -m model.gguf -c 32768 --rope-scaling yarn --yarn-orig-ctx 4096
  ```
- 一些模型（如部分 Llama 3.1 系列）官方就是通过特定的 `--rope-freq-base` 数值原生支持长上下文，具体数值请以模型卡片/官方发布说明为准，不要凭经验照搬其他模型的参数。

---

## 12. 推测解码（Speculative Decoding）参数

原理：用一个更小、更快的"草稿模型（draft model）"连续猜测多个 token，再由目标大模型**一次性并行验证**这些猜测，猜对的部分可以"免费"获得，从而降低平均每 token 的耗时。在代码补全、格式高度可预测的文本上效果最明显；草稿模型与目标模型的**词表需要兼容**。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-md, --model-draft FNAME` | 无 | 草稿模型路径 |
| `-ngld, --gpu-layers-draft N` | `auto` | 草稿模型卸载到 GPU 的层数 |
| `-td, --threads-draft N` | 同 `--threads` | 草稿模型使用的线程数 |
| `--spec-draft-n-max N` | `3` | 单轮最多起草的 token 数（旧版名称 `--draft-max` 已废弃） |
| `--spec-draft-n-min N` | `0` | 单轮最少起草 token 数（旧版名称 `--draft-min` 已废弃） |
| `--spec-draft-p-split P` | `0.10` | 推测解码分裂概率阈值 |
| `--spec-draft-p-min P` | `0.00` | 接受草稿的最小概率（贪婪模式） |
| `--spec-draft-backend-sampling` | 启用 | 是否把草稿采样下放到后端（GPU）执行 |
| `--spec-type` | `none` | 推测策略：`draft-simple` / `draft-eagle3` / `draft-mtp`（需要专门训练的草稿头）/ `ngram-simple` / `ngram-map-k` / `ngram-map-k4v` / `ngram-mod` / `ngram-cache`（无需额外模型，从上下文里挖 n-gram 做草稿） |
| `--spec-draft-cache-type-k/v` | `f16` | 草稿模型自己的 KV 缓存量化类型 |

### 调优建议

- **有现成小模型可配对**（同系列、同词表的小尺寸版本）时用经典的 `-md` 草稿模型模式：
  ```bash
  llama-server -m target.gguf -md draft.gguf -ngl 99 -ngld 99 --spec-draft-n-max 16
  ```
- **没有合适草稿模型**（比如显存只够放一个模型）时，可以尝试 `--spec-type ngram-*` 系列——不需要额外模型，直接从当前上下文/生成历史里挖重复片段做草稿，对"输出会大量重复输入内容"的任务（如代码编辑、长文摘要）比较有效，代价几乎为零，值得默认打开试一下。
- 推测解码是否划算高度依赖任务的可预测性：自由创作类文本命中率低、可能反而拖慢；结构化输出、代码、工具调用等重复模式多的任务收益明显。建议用 `llama-server` 自带的 `tools/server/bench/speed-bench` 或直接对比开关前后的 `predicted_per_second` 来验证。

---

## 13. 服务器并发与吞吐参数

（仅 `llama-server` 特有）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-np, --parallel N` | `-1`（自动） | 并发处理槽（slot）数量，决定同时能服务多少个请求 |
| `-cb, --cont-batching` | 启用 | 连续批处理（动态批处理），把多个并发请求的解码步骤打包到同一批里执行 |
| `--cache-prompt` | 启用 | 是否复用上一次请求中已处理过的公共前缀（KV 缓存复用） |
| `--cache-reuse N` | `0` | 通过 KV 位移复用缓存的最小片段长度，配合长对话历史效果明显 |
| `-sps, --slot-prompt-similarity SIM` | `0.10` | 新请求与某个 slot 已有提示词的相似度需达到多少才复用该 slot |
| `--threads-http N` | `-1` | 处理 HTTP 请求本身的线程数（与推理线程分开） |
| `--metrics` | 关闭 | 开启 Prometheus 格式的性能指标端点，便于监控吞吐/延迟 |

### 调优建议

- **`-cb`（连续批处理）几乎总该保持开启**：有实测显示在 4 并发场景下，相比串行处理请求，吞吐量提升可达 **2 倍以上**。它是 llama-server 能同时服务多用户的核心机制。
- `-np`（并发槽数）越大，理论上能同时服务的用户越多，但**每个槽都要占用独立的一份 KV 缓存显存**——槽数 × 单槽显存 必须控制在总显存预算内，槽数设太高会导致每个槽能分到的上下文变短，或者直接 OOM。
- 长对话、Agent 类重复发送完整历史的场景，`--cache-reuse` 和默认开启的 `--cache-prompt` 能显著减少重复计算的提示词部分，建议保持开启并适当调高 `--cache-reuse`（如 256~512）。
- 想验证并发配置是否合理，把压测工具的并发数设成和 `-np` 相同的值最有参考意义（`tools/server/bench/speed-bench` 中即建议 `--concurrency` 对齐 `-np`）。

---

## 14. 模型量化格式对性能的影响

量化格式不是运行时参数，而是**下载/转换模型时就固定好的**（GGUF 文件本身的格式），但它是决定"模型能否装进显存"以及"生成速度上限"的第一位因素，因此收录于此。

### 量化家族速览

| 家族 | 代表格式 | 特点 |
|---|---|---|
| 传统块量化（legacy） | `Q4_0` `Q4_1` `Q5_0` `Q5_1` `Q8_0` | 每个块内统一精度，实现简单、速度快，`Q8_0` 至今仍是"近乎无损"8 位量化的常用选择 |
| K-quants | `Q2_K` `Q3_K_S/M/L` `Q4_K_S/M` `Q5_K_S/M` `Q6_K` | 同一权重矩阵内混合精度分配，相同体积下质量通常优于传统块量化；`_M`（medium）比 `_S`（small）在关键层用更高精度，通常值得多付出的一点体积 |
| I-quants（重要性矩阵） | `IQ1_S` `IQ2_XXS` `IQ3_XXS` `IQ4_NL` `IQ4_XS` 等 | 借助校准数据（imatrix）在极低比特下尽量保精度，体积最小但在部分硬件/后端上计算速度较慢，超低比特（2 bit 以下）质量下降明显，一般只在显存极度紧张时才考虑 |

### 一个 7B~8B 级模型的量化对比示例（数值为社区第三方实测，仅供参考，不同模型/硬件会有差异）

| 量化格式 | 相对文件大小 | 困惑度相对 FP16 变化 | 相对生成速度 | 定位 |
|---|---|---|---|---|
| `Q8_0` | 最大（约 7GB 档） | +0.03% | 基准（较慢） | 几乎无损，显存/带宽充裕时的首选 |
| `Q6_K` | 略小 | +0.13% | 略快 | 质量/体积性价比最好的档位之一 |
| `Q5_K_M` | 中等 | +0.39% | 更快 | 均衡选择 |
| `Q4_K_M` | 较小 | +1.68% | 更快 | **最常见的默认推荐**，多数场景的甜点位 |
| `Q4_K_S` | 更小 | +2.62% | 更快 | 比 `Q4_K_M` 更小更快，质量略降 |
| `Q3_K_M` | 小 | +6.07% | 快 | 仅建议在必须极限省显存、模型本身较大时使用 |
| `Q2_K` | 最小 | +15.3% | 最快 | 质量损失明显，一般不推荐 |

### 调优建议

- **默认从 `Q4_K_M` 开始**：多数场景下是体积、速度、质量的最佳平衡点，也是各推理框架和模型发布方最常提供的档位。
- 显存/带宽充裕（如 24GB+ 显卡跑 7B~14B 模型）：可以直接上 `Q6_K` 或 `Q8_0`，质量损失几乎不可感知，速度损失也在可接受范围。
- 显存紧张、模型又比较大（30B+）：优先考虑降量化档位而不是砍上下文长度或砍 GPU 层数——**量化是唯一不需要牺牲运行速度就能省显存的方法**（相反，砍 `-ngl` 会让部分层跑在 CPU 上，速度损失往往比降一档量化更大）。
- LLM 的生成阶段本质是**显存带宽受限**：更小的量化 = 每生成一个 token 需要从显存读取的字节数更少 = 生成速度通常更快，这也是"量化不仅省显存、还能提速"的原因。
- 一旦模型已经量化完成，**无法**在不重新量化（需要原始 FP16 权重）的情况下转换到更高精度档位，下载前想清楚需要的档位。

---

## 15. 编译后端选择（简述）

比任何运行时参数都更根本的性能因素是**编译时选择的计算后端**，这决定了 llama.cpp 能否用上你的硬件加速单元：

| 后端 | 适用硬件 | 编译开关（示例） |
|---|---|---|
| CUDA | NVIDIA GPU | `-DGGML_CUDA=ON` |
| HIP/ROCm | AMD GPU（含 Instinct 系列） | `-DGGML_HIP=ON` |
| Vulkan | 几乎所有现代 GPU（NVIDIA/AMD/Intel），跨厂商通用性最好 | `-DGGML_VULKAN=ON` |
| SYCL | Intel GPU | `-DGGML_SYCL=ON` |
| Metal | Apple Silicon | macOS 上默认自动启用 |
| CPU（AVX2/AVX-512/AMX/NEON） | 无独立显卡或纯 CPU 部署 | 默认自动检测指令集 |

若用官方预编译发行版，选对"CUDA 版 / Vulkan 版 / ROCm 版"这一步本身就是最大的性能决策；同硬件下，Vulkan 通用性最好但通常不如厂商专用后端（CUDA/ROCm）极限性能高。本文不展开编译细节，Windows + Vulkan 的完整编译流程见[《本地编译 llama.cpp（Windows + Vulkan）》](build-llamacpp-windows-vulkan.zh-CN.md)。

---

## 16. llama-bench 基准测试完整指南

`llama-bench` 是官方自带的专用性能测试工具，**不包含分词和采样耗时**，只测量模型前向计算本身的吞吐量，是验证本文各项调优是否真的有效的标准工具。

### 语法

```text
llama-bench [options]

通用选项：
  --numa <distribute|isolate|numactl>       NUMA 模式
  -r, --repetitions <n>                     每项测试重复次数（默认 5，取平均）
  --prio <-1|0|1|2|3>                       进程/线程优先级
  --delay <0...N>                           每项测试之间的间隔秒数
  -o, --output <csv|json|jsonl|md|sql>      标准输出格式（默认 md）
  --list-devices                            列出可用设备
  -v, --verbose                             详细输出
  --no-warmup                               跳过预热
  -fitt, --fit-target <MiB>                 按显存自动适配（同第 10 节）
  -rpc, --rpc <rpc_servers>                 注册远程 RPC 设备

测试参数（几乎都支持逗号分隔多值 / "起-止[+步长|*倍数]" 区间语法）：
  -m, --model <filename>       模型文件路径
  -p, --n-prompt <n>           提示词处理测试的 token 数（默认 512）
  -n, --n-gen <n>              生成测试的 token 数（默认 128）
  -pg <pp,tg>                  混合测试：先处理 pp 个提示词 token 再生成 tg 个
  -d, --n-depth <n>            测试前先用 n 个 token 预填充 KV 缓存（模拟长对话中段的速度）
  -b, --batch-size <n>         同第 4 节（默认 2048）
  -ub, --ubatch-size <n>       同第 4 节（默认 512）
  -ctk / -ctv <t>              KV 缓存类型（同第 7 节，默认 f16）
  -t, --threads <n>            线程数
  -ngl, --n-gpu-layers <n>     GPU 层数
  -ncmoe, --n-cpu-moe <n>      MoE 卸载层数（同第 6 节）
  -sm, --split-mode <...>      多卡拆分方式
  -fa, --flash-attn <on|off|auto>
  -mmp, --mmap <0|1>
  -ts, --tensor-split <...>
```

### 输出解读

```text
| model            | size     | params | backend | ngl | test   | t/s            |
| ---------------- | -------- | ------ | ------- | --- | ------ | -------------- |
| qwen2 1.5B Q4_0   | 885.97 MiB | 1.54 B | CUDA    | 99  | pp512  | 5765.41 ± 20.55 |
| qwen2 1.5B Q4_0   | 885.97 MiB | 1.54 B | CUDA    | 99  | tg128  | 197.71 ± 0.81   |
```

- `pp512` = 提示词处理（prompt processing）测试，处理 512 个 token；数值越高，长文档/长系统提示的首 token 延迟越低。
- `tg128` = 逐 token 生成（text generation）测试，生成 128 个 token；数值越接近日常交互中"打字速度"的直观感受。
- `t/s` = 每秒 token 数，`±` 后为标准差（重复测试的波动）。
- 两者**不能直接比较**：`pp` 通常远高于 `tg`，因为 prefill 阶段可以大批量并行，而 decode 阶段本质串行。对比不同配置时，务必确保对比的是同一类测试（都看 `pp` 或都看 `tg`）。

### 常用测试组合示例

```bash
# 基础测试：某模型全卸载后的 pp/tg 速度
llama-bench -m model.gguf -ngl 99

# 扫描不同 GPU 层数（找到部分卸载的性价比拐点）
llama-bench -m model.gguf -ngl 10,20,30,31,32,33,34,35

# 扫描不同线程数（纯 CPU 场景找最优线程数）
llama-bench -m model.gguf -ngl 0 -t 1,2,4,8,16,32

# 扫描不同批大小（关注长提示词处理速度）
llama-bench -m model.gguf -n 0 -p 1024 -b 128,256,512,1024,2048

# 对比开关 Flash Attention
llama-bench -m model.gguf -ngl 99 -fa 0
llama-bench -m model.gguf -ngl 99 -fa 1

# 对比 KV 缓存量化
llama-bench -m model.gguf -ngl 99 -fa 1 -ctk f16,q8_0,q4_0 -ctv f16,q8_0,q4_0

# 模拟长对话中段（先填充 4096 token 上下文再测速度衰减）
llama-bench -m model.gguf -ngl 99 -d 0,4096,16384

# 对比多个不同量化档位的同一模型
llama-bench -m model-Q4_K_M.gguf -m model-Q5_K_M.gguf -m model-Q8_0.gguf -ngl 99
```

---

## 17. 实战调优方案精选

以下命令均为示例框架，实际数值请结合 `llama-bench` 实测结果和你的硬件调整。

### 17.1 纯 CPU 推理（无独立显卡）

```bash
llama-cli -m model-Q4_K_M.gguf \
  -t $(nproc --all) -tb $(nproc --all) \
  -ngl 0 -c 4096 -b 512
```
优先用 `Q4_K_M` 或更小档位；若 RAM < 16GB 慎用 `Q8_0`。

### 17.2 单 GPU 全部卸载（标准场景）

```bash
llama-server -m model-Q4_K_M.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -c 8192 --port 8080
```

### 17.3 单 GPU 显存不足，需部分卸载（非 MoE 稠密模型）

```bash
# 先用 llama-bench 找到不 OOM 的最大 -ngl
llama-bench -m model.gguf -ngl 10,20,30,31,32,33,34,35

llama-server -m model.gguf -ngl 32 -fa on -ctk q8_0 -ctv q8_0 -c 4096
```

### 17.4 大型 MoE 模型（CPU+GPU 混合，消费级显卡）

```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF \
  --ctx-size 32768 --jinja \
  -ub 2048 -b 2048 \
  --n-cpu-moe 16 \
  -ngl 999 -fa on
```

### 17.5 多 GPU（模型单卡装不下）

```bash
llama-cli -m big-model-Q4_K_M.gguf \
  -ngl 999 -sm layer -ts 32,16,6 -mg 0 \
  -c 8192 -fa on
```

### 17.6 长上下文 + 显存优化

```bash
llama-server -m model.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -c 65536
```

### 17.7 高并发多用户服务器

```bash
llama-server -m model-Q4_K_M.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -np 4 -cb -c 32768 \
  --cache-reuse 256 --metrics
```
`-np`（槽数）与单槽上下文预算需要一起权衡：总显存 ≈ 模型权重 + `np × 单槽KV缓存`。

### 17.8 推测解码加速（有配套小模型时）

```bash
llama-server -m target-model.gguf -md draft-model.gguf \
  -ngl 99 -ngld 99 \
  --spec-draft-n-max 16 -fa on
```

---

## 18. 高影响参数速查表

按对性能的典型影响大小粗略排序（越靠前通常越值得优先调整）：

| 排名 | 参数 | 一句话作用 |
|---|---|---|
| 1 | `-ngl` | 决定多少计算发生在 GPU（通常是单一最大速度杠杆） |
| 2 | 量化档位（下载模型时选择） | 决定显存占用与带宽需求的下限 |
| 3 | `--n-cpu-moe` / `-ot` | MoE 模型的显存/速度平衡阀门 |
| 4 | `-fa` | 打开后为 KV 量化铺路，且本身通常有性能收益 |
| 5 | `-ctk` / `-ctv` | KV 缓存量化，长上下文场景的显存救星 |
| 6 | `-b` / `-ub` | 提示词处理吞吐量的主要杠杆（对 CPU+GPU 混合尤其关键） |
| 7 | `-c` | 上下文越大，KV 缓存显存开销越大 |
| 8 | `-sm` / `-ts` | 多卡场景下拆分策略是否匹配硬件互联 |
| 9 | `-t` / `-tb` | CPU 推理或部分卸载时的线程配置 |
| 10 | `-np` / `-cb` | 服务器多用户并发吞吐 |
| 11 | `--spec-*` 推测解码 | 特定任务类型下的额外加速 |
| 12 | `--mmap` / `--mlock` / `-dio` | 主要影响加载速度，对稳态推理速度影响较小 |

---

## 19. 性能计时与诊断参数（--perf）

除了用 `llama-bench` 做专项对比测试（第 16 节），llama.cpp 自身也能在**每一次真实运行**时直接打印详细的耗时统计，不需要额外跑一遍基准测试。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--perf, --no-perf` | `false`（关闭） | 是否输出 libllama 内部的性能计时统计（env: `LLAMA_ARG_PERF`） |

### `llama-cli` 等命令行工具

默认是**关闭**的，需要显式加上 `--perf` 才会在运行结束后打印类似下面的耗时明细：

```text
llama_perf_context_print:        load time =   579.15 ms
llama_perf_context_print:      sample time =     0.72 ms /    28 runs   (    0.03 ms per token, 38888.89 tokens per second)
llama_perf_context_print: prompt eval time =   655.63 ms /    10 tokens (   65.56 ms per token,    15.25 tokens per second)
llama_perf_context_print:        eval time =  2180.97 ms /    27 runs   (   80.78 ms per token,    12.38 tokens per second)
llama_perf_context_print:       total time =  2891.13 ms
```

（不同版本里这几行日志的函数名前缀可能是 `llama_print_timings` 或 `llama_perf_context_print`，具体以你的构建为准，但字段含义是一致的。）

- `load time`：加载模型的一次性耗时，不计入推理速度。
- `sample time`：采样耗时（从 logits 里选出下一个 token 的开销），通常很小，一般不是瓶颈。
- `prompt eval time`：提示词处理（prefill）耗时，换算出的 tokens per second 就是本文其它章节说的 **pp 速度**。
- `eval time`：逐 token 生成（decode）耗时，换算出的速度对应 **tg 速度**。
- `total time`：本次运行端到端总耗时。

用法示例：

```bash
llama-cli -m model.gguf -ngl 99 -p "写一段关于秋天的描述" -n 128 --perf
```

这能帮你快速判断"这次慢，到底是慢在加载、慢在处理长提示词，还是慢在生成阶段"，从而决定该去调第 4、5、6、7 节里的哪个参数。

### `llama-server`：单次请求级别的计时

服务器场景没有"进程退出时打印"这种机制，而是通过 API 响应里的字段获取：

- 每次 `/completion`（或兼容的 chat/completions）响应默认都带有一个 `timings` 对象，包含 `predicted_per_second` 等聚合速度指标，无需额外开关。
- 请求体加上 `"timings_per_token": true`（默认 `false`），可以让**流式输出的每一个 token** 都附带当时的耗时/速度信息，适合做实时监控面板，或诊断"速度是不是随着对话变长而掉下来了"。
- 想长期、持续地监控吞吐/延迟趋势（而不是单次诊断），用 `--metrics` 开启 Prometheus 格式的指标端点（第 13 节），接入 Grafana 等工具。

### `--perf` / `timings` 与 `llama-bench` 该用哪个

| 场景 | 推荐工具 |
|---|---|
| 想知道"这一次真实请求到底花了多久、瓶颈在 prefill 还是 decode" | `--perf`（CLI）或响应里的 `timings` 字段（server） |
| 想对比"开/关某个参数到底谁快"，需要多次重复取平均、控制变量 | `llama-bench`（第 16 节） |
| 想长期监控生产环境的吞吐/延迟趋势 | `llama-server --metrics` + Prometheus/Grafana |

---

## 20. 参考资料

- llama.cpp 官方仓库：<https://github.com/ggml-org/llama.cpp>
- Server 完整参数文档：<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- llama-bench 完整文档：<https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md>
- 多 GPU 官方文档：<https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md>
- `--fit` / `llama-fit-params` 工具说明：<https://github.com/ggml-org/llama.cpp/tree/master/tools/fit-params>
- MoE CPU+GPU 混合推理调优指南（社区）：<https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide>

> 再次提醒：llama.cpp 迭代很快，参数名、默认值可能在你阅读本文时已经变化。任何时候有疑问，第一优先级是运行 `llama-cli --help` / `llama-server --help` 核对你本地版本的真实参数列表。
