# `--no-mmap` and `--mlock` Explained

[English](no-mmap-and-mlock.md) | [简体中文](no-mmap-and-mlock.zh-CN.md)

Both flags come up constantly in llama.cpp discussions. Drawing on the official llama.cpp documentation and community practice, they are both about **memory management** — but they act on entirely different things, through entirely different mechanisms.

---

## Core definitions

| Flag | Official description | In plain terms |
| ---------------- | --------------------------------------------------------------- | ---------------------------------------------------------------- |
| **`--no-mmap`** | Disables memory mapping; forces the entire model to be loaded into RAM | **"Load it directly"**: the model file is read into memory in one go, with no reliance on a file mapping |
| **`--mlock`** | Forces the system to lock the model in RAM, preventing the OS from swapping or compressing it | **"Lock it in memory"**: tells the OS "don't touch this memory", so it never gets swapped to disk |

By default, llama.cpp enables `--mmap` (memory mapping) and does not enable `--mlock`.

---

## `--no-mmap` in detail

### What is mmap?

Memory mapping is an operating-system mechanism: **the model file is not loaded into memory; instead a "mapping relationship" is established**. When the program needs a block of model weights, the OS reads it from disk on demand, and it behaves just like memory access. This has two benefits:

- **Extremely fast startup**: the whole model doesn't need to be read into memory at launch
- **Little RAM pressure**: the OS page cache manages it automatically — pages that have been accessed stay resident (like a hot cache), untouched pages are freed

### When to use `--no-mmap`

**Scenario 1: your RAM is slightly smaller than the model, and you see frequent pageouts**

Say you have 16 GB of RAM and a 14 GB model — add the KV cache and other overhead, and the system may repeatedly page parts of the model weights out to swap, causing frequent stutter during generation. `--no-mmap` forces all weights to stay resident in RAM, avoiding that thrashing.

**Scenario 2: avoiding warm-up cost**

With mmap, weights are loaded on demand. On the very first inference, every weight's first access triggers a page fault, adding latency. For production environments that need to respond immediately, `--no-mmap` makes sure all weights are loaded during initialization.

**Scenario 3: certain unusual hardware or VM environments**

In some virtualized environments or on certain CPU architectures, mmap behaviour may be unstable or slow — another case where `--no-mmap` helps.

### The cost of `--no-mmap`

- **Loading time increases significantly** (the whole model file has to be read)
- **RAM usage multiplies** (the model occupies both RAM and VRAM)
- **It can actually waste VRAM during GPU inference** (e.g. a Gemma-4B with `--no-mmap` ended up under more pressure than a 9B Qwen)

> **Recommendation**: for pure GPU inference (`-ngl -1`, or most layers on GPU), **you usually don't need `--no-mmap`**. It's really only worth considering for CPU+GPU hybrid inference.

---

## `--mlock` in detail

### What is mlock?

`mlock` is a POSIX system call (on Windows it's implemented via `VirtualLock`). It tells the OS: **this memory region may not be swapped out, and may not be compressed**.

### When to use `--mlock`

**Scenario 1: plenty of RAM, but the OS still likes to swap**

Even with 64 GB of RAM, Linux may still swap out some pages that "look inactive" to free space for the file cache. When llama.cpp touches those pages again, a swap-in occurs and you get a latency spike. `--mlock` prevents that.

**Scenario 2: low-latency, real-time applications**

Chat bots, voice assistants and similar workloads, where any hiccup — even a few hundred milliseconds of swap-in — hurts the user experience.

**Scenario 3: CPU inference (no GPU)**

With CPU inference the weights live directly in RAM; `--mlock` guarantees they are never swapped out, keeping generation speed stable.

### The cost of `--mlock`

- Requires enough free RAM (locked memory can't be used by other programs)
- On Linux you may need to raise the `memlock` ulimit (the default is usually very low, e.g. 64 KB)

---

## Side-by-side comparison

| Dimension | `--no-mmap` | `--mlock` |
| ---------------- | --------------------------------- | -------------------------------------- |
| **Mechanism** | No memory mapping; everything is read into RAM | Memory mapping stays, but mapped pages are locked and never swapped |
| **Load speed** | Slow (full read) | Fast (on-demand reads) |
| **RAM usage** | High (whole model + copy) | Normal (pages managed by mmap) |
| **Runtime stability** | Guarantees everything is in RAM | Guarantees no swap, but first access still incurs a page fault |
| **Best for** | RAM barely sufficient; need to avoid page faults | Ample RAM; need to prevent swap-induced stutter |
| **GPU inference** | Usually unnecessary | Usually unnecessary |

---

## How to choose

For a GPU-inference workload with a wide variety of models:

| Situation | Recommendation |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| **Pure GPU inference (`-ngl -1`)** | **Neither flag.** Weights are in VRAM; RAM usage is small anyway |
| **CPU+GPU hybrid (some layers in RAM)** | Use `--mlock` rather than `--no-mmap`. Fast loading, stable runtime, no wasted RAM |
| **RAM just barely enough (model + KV cache ≈ 80–90% of RAM)** | Consider `--no-mmap` to avoid frequent pageouts |
| **Gemma models (small parameter count)** | You can **drop `--no-mmap`** — the model is small, and mmap is more than adequate for GPU inference |
| **Large MoE models (35B/26B, IQ2_M/Q4_0)** | If most layers are on CPU (RAM), prefer `--mlock`; if only a few expert layers are offloaded to CPU (`-ot exps=CPU`), you can skip it |

**Rules of thumb**:

- The model fits entirely in GPU → **you need neither flag**
- The model mostly runs on CPU → **add `--mlock`**
- The model size is close to your RAM limit → **add `--no-mmap`**
