# 本地编译 llama.cpp（Windows + Vulkan）

[English](build-llamacpp-windows-vulkan.md) | [简体中文](build-llamacpp-windows-vulkan.zh-CN.md)

官方预编译包已经覆盖了大多数场景，但如果你想要 Vulkan 优化、想自己控制编译选项，或者只是想知道二进制里到底装了什么，从源码编译是最直接的方式。本文给出 Windows + MSVC + Vulkan 后端的完整流程，并标注几个容易踩的坑。

## 环境要求

| 组件 | 要求 | 说明 |
| --- | --- | --- |
| Visual Studio 2022 / Build Tools 2022 | 勾选「使用 C++ 的桌面开发」 | 该工作负载已包含 MSVC v143 编译器与 Windows SDK，**不需要**额外勾选 C++/CLI 支持 |
| CMake | ≥ 3.14，建议 3.20 以上 | 安装 VS 时勾选「C++ CMake tools for Windows」即可一并装好 |
| Git | 任意较新版本 | 拉取源码 |
| Vulkan SDK | 1.3.x 或更新（[vulkan.lunarg.com](https://vulkan.lunarg.com/)） | 保持安装器默认选项即可，关键是 `Bin\glslc.exe` 必须存在，且安装后 `VULKAN_SDK` 环境变量已被写入 |
| 显卡驱动 | 支持 Vulkan 1.3 | NVIDIA / AMD / Intel 近几年的驱动都满足；可用 `vulkaninfo` 自检 |
| Node.js | 可选 | llama.cpp 的编译与运行都不需要它，只有当你自己修改 `llama-server` 的 Web UI 前端源码时才需要 |

> 安装完 Vulkan SDK 后请**新开一个终端**，否则 `VULKAN_SDK` 不会出现在当前会话的环境变量里。

## 编译 llama.cpp

先拉取源码：

```bat
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

### 方式一：Visual Studio 生成器（cmd）

建议从「x64 Native Tools Command Prompt for VS 2022」或「Developer PowerShell for VS 2022」中执行；普通终端通常也能工作，因为 CMake 会通过注册表找到 VS。

```bat
:: 1. 清理旧构建（可选，但推荐：换了参数或升级源码后能避免诡异的链接错误）
if exist build rmdir /s /q build

:: 2. 配置：Vulkan 后端 + 本机指令集优化
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
  -DGGML_VULKAN=ON ^
  -DGGML_NATIVE=ON ^
  -DGGML_OPENMP=ON

:: 3. 编译（--config Release 决定构建类型）
cmake --build build --config Release --parallel

:: 4. 验证
build\bin\Release\llama-server.exe --version
```

### 方式二：Visual Studio 生成器（PowerShell）

```powershell
# 1. 清理旧构建
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue

# 2. 配置（PowerShell 用反引号续行）
cmake -B build -G "Visual Studio 17 2022" -A x64 `
  -DGGML_VULKAN=ON `
  -DGGML_NATIVE=ON `
  -DGGML_OPENMP=ON

# 3. 编译
cmake --build build --config Release --parallel $env:NUMBER_OF_PROCESSORS

# 4. 验证
.\build\bin\Release\llama-server.exe --version
```

> `--parallel`（`-j` 的同义词）只是向构建工具请求并行；在 VS 生成器下真正并行的是 MSBuild。如果发现编译仍然只跑一个核，可以绕过 CMake 直接传参：`cmake --build build --config Release -- /m:%NUMBER_OF_PROCESSORS%`。

### 方式三：Ninja（可选，构建更快）

Ninja 是单配置生成器，构建类型由 `CMAKE_BUILD_TYPE` 决定，增量编译通常比 MSBuild 快不少：

```bat
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON
cmake --build build
```

需要 `ninja.exe` 在 `PATH` 中（VS 的 CMake 工具链会自带一份）。

### 参数说明

| 参数 | 作用 | 备注 |
| --- | --- | --- |
| `-DGGML_VULKAN=ON` | 编译 Vulkan 后端 | 配置阶段会以 `find_package(Vulkan COMPONENTS glslc REQUIRED)` 强依赖 `glslc`，找不到直接失败 |
| `-DGGML_NATIVE=ON` | 按构建机的 CPU 探测指令集 | 默认已是 ON。GCC/Clang 下等价于 `-march=native`；MSVC 下则是在 configure 阶段探测本机能力，再决定 `/arch:AVX2` 等开关 |
| `-DGGML_OPENMP=ON` | 启用 OpenMP（CPU 侧多线程） | 默认已是 ON，写出来只是为了显式说明 |
| `-DCMAKE_BUILD_TYPE=Release` | 单配置生成器（Ninja / Makefiles）的构建类型 | **VS 生成器是多配置生成器，会忽略该变量**，构建类型由 `--config Release` 决定 |
| `-DBUILD_SHARED_LIBS=OFF` | 静态链接，产物不再依赖一堆 DLL | 可选，见下文「部署」一节 |

一个常见的误区：网上流传的 `-DGGML_VULKAN_DEVICE=0` 和 `-DGGML_VULKAN_MEMORY_ALLOCATOR=1` **在 llama.cpp 中并不存在**。Vulkan 后端相关的 CMake 选项只有 `GGML_VULKAN`、`GGML_VULKAN_CHECK_RESULTS`、`GGML_VULKAN_DEBUG`、`GGML_VULKAN_MEMORY_DEBUG`、`GGML_VULKAN_SHADER_DEBUG_INFO`、`GGML_VULKAN_VALIDATE`、`GGML_VULKAN_RUN_TESTS` 七个。传入不存在的变量时 CMake 只会打印一句 `Manually-specified variables were not used by the project`，然后正常配置，很容易让人误以为参数生效了。

选哪块 GPU 是**运行时**的事，不是编译期的事：

```bat
:: 先列出可用设备
llama-server.exe --list-devices

:: 指定用第一块 Vulkan 设备
llama-server.exe -m model.gguf --device Vulkan0 -ngl 99
```

多卡时再配合 `--main-gpu`、`--split-mode`、`--tensor-split`（详见[性能参数指南](llamacpp-performance-parameters-guide.zh-CN.md)第 5 节「GPU 卸载与显存参数」）。也可以用环境变量限制可见设备，例如只暴露 0 号卡：`set GGML_VK_VISIBLE_DEVICES=0`。

## 验证编译结果

```bat
build\bin\Release\llama-server.exe --version
```

会打印版本号与构建信息（git commit、编译器、编译选项等）。`--version` 在所有工具里都可用，且不需要加载模型。

再看一眼后端是否真的启用：

```bat
build\bin\Release\llama-cli.exe -m model.gguf -ngl 99 -p "hello"
```

启动日志里应出现类似 `ggml_vulkan: Found 1 Vulkan devices` 和 `ggml_vulkan: 0 = NVIDIA GeForce ...` 的行；如果没有，说明 Vulkan 后端没编进去，或者驱动/设备选择有问题。

## 部署

编译产物在 `build\bin\Release\`（具体路径随生成器略有不同，Ninja 下通常是 `build\bin\`）。里面除了 `llama-server.exe`、`llama-cli.exe`、`llama-bench.exe`，还有 `llama.dll`、`ggml.dll`、`ggml-base.dll`、`ggml-cpu.dll`、`ggml-vulkan.dll` 等动态库（默认 `BUILD_SHARED_LIBS=ON`）。

所以**只拷 exe 是跑不起来的**，会提示找不到 DLL。两个选择：

- 把整个 `bin\Release` 目录一起拷到你的启动目录；
- 或者首次配置时加 `-DBUILD_SHARED_LIBS=OFF` 做静态链接，产物更自包含（体积会大一些）。

Vulkan 的着色器已经在编译期生成并编进 `ggml-vulkan.dll`，运行时不需要额外的 `.spv` 文件。

## 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 配置时报 `Could NOT find Vulkan (missing: Vulkan_GLSLC_EXECUTABLE)` | Vulkan SDK 没装、装的时候去掉了着色器工具链，或 `VULKAN_SDK` 未生效。重装 SDK 并**新开终端**；必要时手动设置 `VULKAN_SDK` 环境变量 |
| 配置时报找不到 C++ 编译器 / 找不到 VS | 换成 「Developer PowerShell for VS 2022」；或显式指定生成器与平台：`-G "Visual Studio 17 2022" -A x64` |
| 程序一启动就 `0xC000001D`（非法指令） | 二进制是按构建机的指令集编的，拿到了更老的 CPU 上。用 `-DGGML_NATIVE=OFF` 重新编译，或改用 `GGML_BACKEND_DL` + 多 variant |
| 启动报缺少 `ggml-vulkan.dll` 之类 | 只拷了 exe，见「部署」一节 |
| 编译时间很长 | 首次构建要给所有 Vulkan 着色器生成并编译代码，属正常现象。用 Ninja、加 `--parallel`、并确保 CPU 没被其他任务占满；后续增量编译会快得多 |
| 能启动但 `--list-devices` 里没有 Vulkan 设备 | 驱动过旧或被禁用；运行 `vulkaninfo` 看是否能枚举设备 |

## 可选：改用 CUDA 后端

NVIDIA 显卡上，CUDA 后端通常比 Vulkan 更快，代价是要装 CUDA Toolkit：

```bat
cmake -B build-cuda -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DGGML_NATIVE=ON
cmake --build build-cuda --config Release --parallel
```

Vulkan 的优势是跨厂商通用：AMD、Intel 以及没有 CUDA 环境的机器都能用同一套流程。两套后端也可以同时编进同一个二进制，运行时用 `--device` 选择。

## 参考资料

- 官方构建文档：<https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>
- CMake 选项定义（`option(GGML_...)` 都在这里）：<https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt>
- Vulkan 后端构建脚本：<https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/CMakeLists.txt>
- Vulkan SDK 下载：<https://vulkan.lunarg.com/>

> 本文基于 2026 年 7 月的主分支核对。llama.cpp 的选项可能随版本调整，最稳妥的做法是配置时加 `-LH` 打印全部缓存变量：`cmake -B build -DGGML_VULKAN=ON -LH`。
