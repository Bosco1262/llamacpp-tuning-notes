# llama.cpp Tuning Notes

[English](README.md) | [简体中文](README.zh-CN.md)

**Practical llama.cpp tuning notes — performance parameters, VRAM fitting, memory mapping, and API-key setup for `llama-server`.**

A field-tested companion for anyone running GGUF models locally: which flag actually moves tokens/sec, how to stop guessing at `-ngl`, and how to keep the server from being an open endpoint.

## Contents

| # | Document | What's inside |
| --- | --- | --- |
| 1 | [Building llama.cpp Locally (Windows + Vulkan)](docs/build-llamacpp-windows-vulkan.md) | VS 2022 + CMake + Vulkan SDK setup, the build flags that actually exist (and the two that don't), build verification, DLL deployment and troubleshooting. |
| 2 | [llama.cpp Performance Parameters: The Complete Guide](docs/llamacpp-performance-parameters-guide.md) | Every parameter that affects speed, VRAM/RAM and throughput — quick-diagnosis table, threads, batch sizes, GPU offloading, MoE offloading, KV-cache quantization, `--fit`, RoPE, speculative decoding, concurrency, quantization formats, backend selection, a full `llama-bench` walkthrough, tuning recipes and a high-impact cheat sheet. |
| 3 | [`llama-fit-params` Explained](docs/llama-fit-params.md) | What the auto-fit diagnostic does, the `--fit-target` formula for vision models, and 13 copy-paste configurations (MoE and dense, text and vision). |
| 4 | [`--no-mmap` vs `--mlock`](docs/no-mmap-and-mlock.md) | What each flag really does to the page cache, what each one costs, and a decision table by scenario. |
| 5 | [Why `llama-fit-params` Returns `-c 0`](docs/why-fit-params-returns-c-0.md) | Why an 8 GB card still gets `-ngl -1`, how the VRAM budget is actually calculated, and what `-c 0` means. |
| 6 | [Adding an API Key to `llama-server`](docs/llama-server-api-key.md) | `--api-key` vs `--api-key-file`, plus working cURL and Python (OpenAI SDK) client examples. |

## Repository layout

```text
.
├── README.md                 # English index (this file)
├── README.zh-CN.md           # Chinese index
└── docs/
    ├── *.md                  # English documents
    └── *.zh-CN.md            # Chinese documents
```

## Conventions

- **Every document exists in both languages.** The `English | 简体中文` switch sits directly under each document's `H1` and always points to the same document in the other language, so you never lose your place.
- English files use `<name>.md`, Chinese files use `<name>.zh-CN.md`, both in the same `docs/` folder.
- Commands are written for `llama.cpp` on any platform; a `bat` block means a Windows `cmd`/PowerShell session, a `bash` block means a POSIX shell.

## A note on accuracy

`llama.cpp` moves fast — argument names and defaults can change month to month. The material here was compiled against the official repository's main branch as of July 2026. Whenever something looks off, trust `llama-cli --help` / `llama-server --help` for the build you're actually running.

## Upstream

- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- Server documentation: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>

## License

This project is licensed under the [MIT License](LICENSE).
