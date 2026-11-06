# llama.cpp 调优笔记

[English](README.md) | [简体中文](README.zh-CN.md)

**实用的 llama.cpp 调优笔记：性能参数、显存适配、内存映射，以及 `llama-server` 的 API 密钥配置。**

写给所有在本地跑 GGUF 模型的人：到底哪个参数真的影响 token/s、怎么不再靠猜来决定 `-ngl`、以及如何避免把推理服务变成谁都能访问的开放接口。

## 文档目录

| # | 文档 | 内容简介 |
| --- | --- | --- |
| 1 | [本地编译 llama.cpp（Windows + Vulkan）](docs/build-llamacpp-windows-vulkan.zh-CN.md) | VS 2022 + CMake + Vulkan SDK 的环境准备、真实存在的编译参数（以及两个并不存在的参数）、编译验证、DLL 部署与常见问题排查。 |
| 2 | [llama.cpp 性能参数完整指南](docs/llamacpp-performance-parameters-guide.zh-CN.md) | 所有影响速度、显存/内存占用与吞吐量的参数——快速诊断表、线程、批大小、GPU 卸载、MoE 卸载、KV 缓存量化、`--fit`、RoPE、推测解码、并发、量化格式、后端选择，以及完整的 `llama-bench` 用法、实战调优方案与高影响参数速查表。 |
| 3 | [llama-fit-params 使用讲解](docs/llama-fit-params.zh-CN.md) | 自动显存适配的诊断工具怎么用、视觉模型的 `--fit-target` 计算公式，以及 13 组可直接复制的配置（MoE 与稠密、文本与视觉）。 |
| 4 | [`--no-mmap` 和 `--mlock` 详解](docs/no-mmap-and-mlock.zh-CN.md) | 两个参数分别对页缓存做了什么、各自的代价，以及按场景给出的选择表。 |
| 5 | [为什么 llama-fit-params 会返回 `-c 0`](docs/why-fit-params-returns-c-0.zh-CN.md) | 8 GB 显存为什么也能 `-ngl -1`、显存预算到底怎么算，以及 `-c 0` 的含义。 |
| 6 | [为 llama-server 添加 API 密钥](docs/llama-server-api-key.zh-CN.md) | `--api-key` 与 `--api-key-file` 的区别，附可直接运行的 cURL 与 Python（OpenAI SDK）客户端示例。 |

## 目录结构

```text
.
├── README.md                 # 英文索引
├── README.zh-CN.md           # 中文索引（本文件）
└── docs/
    ├── *.md                  # 英文文档
    └── *.zh-CN.md            # 中文文档
```

## 约定

- **每篇文档都有中英两个版本。** `English | 简体中文` 切换位于每篇文档 `H1` 标题正下方，始终指向另一语言的同一篇文档，切换后不会丢失阅读位置。
- 英文文件命名为 `<name>.md`，中文文件命名为 `<name>.zh-CN.md`，均放在 `docs/` 目录下。
- 命令块中，`bat` 表示 Windows `cmd`/PowerShell 环境，`bash` 表示 POSIX shell 环境。

## 关于准确性

`llama.cpp` 迭代非常快，参数名和默认值可能逐月变化。本仓库内容基于 2026 年 7 月官方仓库主分支整理。如有疑问，请以你本地版本实际运行的 `llama-cli --help` / `llama-server --help` 输出为准。

## 上游资源

- llama.cpp：<https://github.com/ggml-org/llama.cpp>
- Server 参数文档：<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>

## License

本项目采用 [MIT License](LICENSE)。
