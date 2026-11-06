# Building llama.cpp Locally (Windows + Vulkan)

[English](build-llamacpp-windows-vulkan.md) | [简体中文](build-llamacpp-windows-vulkan.zh-CN.md)

The official prebuilt releases cover most use cases, but if you want the Vulkan backend, control over compile-time options, or just to know what's actually inside the binary, building from source is the direct route. This guide walks through Windows + MSVC + Vulkan end to end, and flags the traps people fall into.

## Requirements

| Component | Requirement | Notes |
| --- | --- | --- |
| Visual Studio 2022 / Build Tools 2022 | Select the "Desktop development with C++" workload | That workload already includes the MSVC v143 compiler and the Windows SDK — you do **not** need C++/CLI support |
| CMake | ≥ 3.14, 3.20+ recommended | Ticking "C++ CMake tools for Windows" in the VS installer is enough |
| Git | any recent version | To clone the source |
| Vulkan SDK | 1.3.x or newer ([vulkan.lunarg.com](https://vulkan.lunarg.com/)) | Keep the installer defaults; the critical parts are that `Bin\glslc.exe` exists and that `VULKAN_SDK` gets set |
| GPU driver | Vulkan 1.3 capable | Recent NVIDIA / AMD / Intel drivers all qualify; check with `vulkaninfo` |
| Node.js | optional | Neither building nor running llama.cpp needs it; it's only required if you modify the `llama-server` Web UI front-end sources |

> After installing the Vulkan SDK, **open a new terminal** — `VULKAN_SDK` won't be present in an already-running session.

## Building llama.cpp

Start by cloning the source:

```bat
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

### Option 1: Visual Studio generator (cmd)

Run this from "x64 Native Tools Command Prompt for VS 2022" or "Developer PowerShell for VS 2022". A normal terminal usually works too, since CMake locates VS through the registry.

```bat
:: 1. Clean the old build (optional but recommended: avoids weird link errors after changing flags or pulling new source)
if exist build rmdir /s /q build

:: 2. Configure: Vulkan backend + host CPU optimizations
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
  -DGGML_VULKAN=ON ^
  -DGGML_NATIVE=ON ^
  -DGGML_OPENMP=ON

:: 3. Build (--config Release selects the build type)
cmake --build build --config Release --parallel

:: 4. Verify
build\bin\Release\llama-server.exe --version
```

### Option 2: Visual Studio generator (PowerShell)

```powershell
# 1. Clean the old build
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue

# 2. Configure (PowerShell line continuation uses backticks)
cmake -B build -G "Visual Studio 17 2022" -A x64 `
  -DGGML_VULKAN=ON `
  -DGGML_NATIVE=ON `
  -DGGML_OPENMP=ON

# 3. Build
cmake --build build --config Release --parallel $env:NUMBER_OF_PROCESSORS

# 4. Verify
.\build\bin\Release\llama-server.exe --version
```

> `--parallel` (a synonym for `-j`) only *asks* the build tool to parallelize; with the VS generator it's MSBuild that does the actual work. If you notice the build still using a single core, bypass CMake and pass the flag straight to MSBuild: `cmake --build build --config Release -- /m:%NUMBER_OF_PROCESSORS%`.

### Option 3: Ninja (optional, faster builds)

Ninja is a single-config generator, so the build type comes from `CMAKE_BUILD_TYPE`. Incremental builds are usually noticeably faster than MSBuild:

```bat
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON
cmake --build build
```

This requires `ninja.exe` on `PATH` (VS's CMake toolchain ships one).

### What the flags do

| Flag | Effect | Notes |
| --- | --- | --- |
| `-DGGML_VULKAN=ON` | Builds the Vulkan backend | Configuration hard-requires `glslc` via `find_package(Vulkan COMPONENTS glslc REQUIRED)` and fails immediately if it's missing |
| `-DGGML_NATIVE=ON` | Tunes the build for the host CPU | Already ON by default. Under GCC/Clang it means `-march=native`; under MSVC it runs a configure-time probe of the host and then decides `/arch:AVX2` etc. |
| `-DGGML_OPENMP=ON` | Enables OpenMP (CPU-side multithreading) | Already ON by default; listed only to be explicit |
| `-DCMAKE_BUILD_TYPE=Release` | Build type for single-config generators (Ninja / Makefiles) | **The VS generator is a multi-config generator and ignores this** — use `--config Release` instead |
| `-DBUILD_SHARED_LIBS=OFF` | Static linking; the output no longer needs a pile of DLLs | Optional — see "Deployment" below |

A common misconception: the widely-copied `-DGGML_VULKAN_DEVICE=0` and `-DGGML_VULKAN_MEMORY_ALLOCATOR=1` **do not exist in llama.cpp**. The Vulkan backend's CMake options are exactly `GGML_VULKAN`, `GGML_VULKAN_CHECK_RESULTS`, `GGML_VULKAN_DEBUG`, `GGML_VULKAN_MEMORY_DEBUG`, `GGML_VULKAN_SHADER_DEBUG_INFO`, `GGML_VULKAN_VALIDATE` and `GGML_VULKAN_RUN_TESTS`. Passing an unknown variable only produces `Manually-specified variables were not used by the project`, then configuration proceeds normally — which makes it very easy to believe the flag did something.

Choosing a GPU is a **runtime** decision, not a compile-time one:

```bat
:: List the available devices first
llama-server.exe --list-devices

:: Use the first Vulkan device
llama-server.exe -m model.gguf --device Vulkan0 -ngl 99
```

With multiple GPUs, combine that with `--main-gpu`, `--split-mode` and `--tensor-split` (see [Section 5 of the performance guide](llamacpp-performance-parameters-guide.md), "GPU Offloading & VRAM Parameters"). You can also limit visibility via an environment variable, e.g. expose only device 0: `set GGML_VK_VISIBLE_DEVICES=0`.

## Verifying the build

```bat
build\bin\Release\llama-server.exe --version
```

This prints the version plus build info (git commit, compiler, build options). `--version` works in every tool and doesn't need a model.

Then confirm the backend actually got compiled in:

```bat
build\bin\Release\llama-cli.exe -m model.gguf -ngl 99 -p "hello"
```

The startup log should contain lines like `ggml_vulkan: Found 1 Vulkan devices` and `ggml_vulkan: 0 = NVIDIA GeForce ...`. If they're absent, either the Vulkan backend wasn't built or the driver/device selection is off.

## Deployment

Binaries land in `build\bin\Release\` (the exact path varies by generator; with Ninja it's usually `build\bin\`). Besides `llama-server.exe`, `llama-cli.exe` and `llama-bench.exe`, that folder contains `llama.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `ggml-vulkan.dll` and friends, because `BUILD_SHARED_LIBS` defaults to ON.

So **copying just the .exe won't run** — it'll fail with a missing-DLL error. Two options:

- Copy the whole `bin\Release` directory to wherever you launch from; or
- Configure with `-DBUILD_SHARED_LIBS=OFF` for static linking, giving a more self-contained (and larger) set of binaries.

The Vulkan shaders are generated and compiled into `ggml-vulkan.dll` at build time, so no `.spv` files are needed at runtime.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Could NOT find Vulkan (missing: Vulkan_GLSLC_EXECUTABLE)` during configure | The Vulkan SDK is missing, was installed without the shader toolchain, or `VULKAN_SDK` isn't in the environment. Reinstall the SDK and **open a new terminal**; set `VULKAN_SDK` manually if needed |
| Configure can't find a C++ compiler / can't find VS | Use "Developer PowerShell for VS 2022", or specify the generator and platform explicitly: `-G "Visual Studio 17 2022" -A x64` |
| The binary immediately dies with `0xC000001D` (illegal instruction) | It was compiled for the build machine's instruction set and is now on an older CPU. Rebuild with `-DGGML_NATIVE=OFF`, or move to `GGML_BACKEND_DL` with multiple variants |
| Missing `ggml-vulkan.dll` (or similar) at startup | You copied only the .exe — see "Deployment" |
| The build takes a very long time | The first build generates and compiles every Vulkan shader; that's expected. Use Ninja, add `--parallel`, and keep the CPU free. Incremental builds are far quicker |
| It runs, but `--list-devices` shows no Vulkan device | Outdated or disabled driver; run `vulkaninfo` to see whether devices can be enumerated |

## Optional: switch to the CUDA backend

On NVIDIA cards the CUDA backend is usually faster than Vulkan, at the cost of installing the CUDA Toolkit:

```bat
cmake -B build-cuda -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DGGML_NATIVE=ON
cmake --build build-cuda --config Release --parallel
```

Vulkan's advantage is being vendor-neutral: AMD, Intel and machines without a CUDA setup all follow the same procedure. Both backends can be compiled into one binary and selected at runtime with `--device`.

## References

- Official build documentation: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>
- CMake option definitions (all `option(GGML_...)` live here): <https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt>
- Vulkan backend build script: <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/CMakeLists.txt>
- Vulkan SDK download: <https://vulkan.lunarg.com/>

> Verified against the main branch as of July 2026. Options can change between releases; the most reliable check is to have CMake print every cache variable during configuration: `cmake -B build -DGGML_VULKAN=ON -LH`.
