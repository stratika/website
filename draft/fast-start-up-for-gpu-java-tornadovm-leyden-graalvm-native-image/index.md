---
title: "Fast Start-up for GPU Java: TornadoVM v8.0.0 with Project Leyden and GraalVM Native Image"
date: ""
description: "TornadoVM v8.0.0 apps can start from a Project Leyden AOT cache or run as a GraalVM Native Image: about 0.3 s instead of 0.6 s for a CUDA program."
authors:
  - "thanos-stratikopoulos"
image: "fast-start-up.jpg"
categories:
  - "TornadoVM"
  - "Performance"
  - "OpenJDK"
related_posts:
  - "how-is-leyden-improving-java-performance-part-1-of-3"
  - "the-tornadovm-programming-model-explained"
  - "getting-started-with-tornadovm-2-0"
canonical: "https://www.tornadovm.org/blogs/fast-start-up-for-gpu-java-tornadovm-leyden-graalvm-native-image"
---

{{< note >}}
This article was first published on the TornadoVM blog on 27 October 2026, for TornadoVM 8.0.0. Everything it describes works in the latest release, TornadoVM 8.2.0: install it with `sdk install tornadovm 8.2.0-jdk22plus-cuda` (or `-opencl`, `-metal`, `-hip`). One thing has improved since: from 8.1.0, TornadoVM creates each backend only when a program first needs it, so the saving that `-Dtornado.enabled.backends` gives below now comes without the flag.
{{< /note >}}

**TornadoVM v8.0.0 applications can start from a Project Leyden AOT cache or ship as a GraalVM Native Image.** Until now, neither was possible. TornadoVM compiled kernels with a vendored copy of the Graal compiler, which needs JVMCI, and JVMCI rules out both a Leyden cache and a Native Image. TornadoVM v8.0.0 replaces Graal with [TC, TornadoVM's own compiler](https://www.tornadovm.org/blogs/tornadovm-8-0-0-tc-tornadovm-compiler), which is plain Java. So a TornadoVM application can now use every start-up feature the JDK offers.

On an RTX 4090, a small CUDA program that compiles and runs one kernel used to take about **0.6 s** from launch to exit. It now takes about **0.3 s** from a Leyden AOT cache or as a Native Image. TC's own compile time for the kernel drops from about 30 ms to 11 ms, and to under 1 ms in a Native Image.

This post shows where TornadoVM's start-up time goes, what each option saves, and the commands to use each one. The complete reference, including troubleshooting, is the [fast start-up guide](https://github.com/beehive-lab/TornadoVM/blob/master/LEYDEN_GRAALVM_NATIVE_IMAGE.md) in the TornadoVM repository.

## 1. Kernels are still compiled at run time

An AOT cache or a Native Image doesn't mean your GPU kernels are compiled ahead of time. In every mode, TC still reads the kernel's class files when the application runs and specialises the kernel on the actual arguments: array lengths, scalar values and object layouts. That's how TornadoVM has always worked, and it's why the generated code is the same whichever mode you choose.

What these features remove is the cost *around* that compile: starting the JVM, loading and linking classes, and warming up TC and the drivers. For a long-running service that cost is paid once and barely matters. For a command-line tool, a serverless function or a test suite that starts hundreds of JVMs, it's most of the run.

### Why TC makes this possible

Both features were design requirements for TC from [the start](https://github.com/beehive-lab/TornadoVM/issues/1227), not something added later:

- **Leyden.** TC doesn't generate bytecode or define classes at run time, loads its classes in a deterministic order, and keeps its static initialisers simple. That lets a Leyden AOT cache record a training run and replay it.
- **Native Image.** TC uses no JVMCI, no reflection outside the metadata it registers, no dynamic proxies, and no bindings to LLVM, MLIR or NVVM. A Native Image build of the compiler produces **byte-identical** kernels to the JVM build.

Removing Graal also removed the JVMCI flags. The argfile behind every TornadoVM launch shrank from about 75 lines of exports, `--patch-module` and `--upgrade-module-path` flags to about a dozen lines, and it's the same on every JDK the SDK supports.

## 2. Where the start-up time goes

All measurements below use the same workload: saxpy on 2²⁰ floats on an RTX 4090. The process compiles one kernel, runs it once and exits. Times are the wall time of the whole process, median of 10 cold runs, on Linux with NVIDIA driver 610.57 and CUDA 13.3, measured on the TornadoVM v8.0.0 development branch just before release. The SDK has both the OpenCL and CUDA backends installed, and the run uses CUDA.

Several costs make up that time, and only some of them belong to TornadoVM:

- **Unused backends.** TornadoVM initialises every installed backend at start-up. On a CUDA run, an OpenCL backend you never use costs about 170 ms: loading the OpenCL library, finding devices and creating a context.
- **The JVM and class loading.** Loading and linking the runtime, the drivers' FFM bindings and TC itself.
- **TC.** About 30 ms to compile saxpy on a fresh JVM.
- **The driver.** Creating a CUDA context alone takes about 50 ms, and tearing contexts down at exit takes another 35–60 ms. No JVM feature can remove these.
- **The `tornado` launcher.** It's a Python script, and it adds about 90 ms per start. Running `java` with an argfile avoids it.

## 3. Improving cold runs with AOT

As in Section 2, every time in the first two tables is the wall time of a whole cold run: a fresh process that starts, compiles the saxpy kernel, runs it once and exits (median of 10 runs). Each row adds one option. The columns are the JDK the same application runs on:

| Mode | JDK 21 | JDK 25 | JDK 26 | JDK 27-ea |
|---|---|---|---|---|
| `java @argfile` | 652 ms | 616 ms | 619 ms | 638 ms |
| + load only CUDA | 495 ms | 465 ms | 467 ms | 483 ms |
| + AppCDS archive | 572 ms | 519 ms | 527 ms | 532 ms |
| + AppCDS + load only CUDA | **405 ms** | 370 ms | 378 ms | 381 ms |
| + Leyden AOT cache | — | 455 ms | 452 ms | 449 ms |
| + AOT cache + load only CUDA | — | **310 ms** | **309 ms** | **301 ms** |

As a GraalVM Native Image (GraalVM CE for JDK 25), the whole application is a single 32 MB executable:

| Native Image | Time |
|---|---|
| `./saxpy` on CUDA, both backends initialised | 471 ms |
| `./saxpy` on OpenCL, the default backend | 334 ms |
| `./saxpy -Dtornado.enabled.backends=cuda` | **324 ms** |

The last table measures something narrower: only the time TC takes to compile the saxpy kernel, still inside a cold run, not the whole process:

| Mode | TC compile time |
|---|---|
| JVM | ~30 ms |
| Leyden AOT cache | ~11 ms |
| Native Image | ~0.9 ms |

Two things stand out. First, the cheapest win needs no build step at all: loading only the backend you use saves about 150 ms on any JDK. Second, the AOT cache and the Native Image end up close to each other, at around 0.3 s, because what's left is mostly the CUDA driver itself.

### Which to use

| Your setup | Use |
|---|---|
| Any JDK, more than one backend installed | `-Dtornado.enabled.backends=<backend>`: no build step, saves about 150 ms |
| JDK 25 or newer | A Leyden AOT cache: one training run, saves about another 150 ms |
| JDK 21 | AppCDS |
| Smallest start-up, no JDK at run time | A GraalVM Native Image |
| CLIs, serverless functions, tests started many times | Combine `enabled.backends` with any of the above |
| Long-running services | These mostly help start-up and change little once the application is warm |

Leyden and Native Image need the `jdk22plus` SDK. The `jdk21` SDK supports AppCDS only, because the FFM API is still a preview feature on JDK 21.

## 4. How to use each one

### Step 0: one argfile instead of the launcher

Every mode below runs `java` directly instead of the Python launcher. Generate the argfile once:

```bash
tornado --generate-argfile                # writes $TORNADOVM_HOME/tornado-argfile
java @$TORNADOVM_HOME/tornado-argfile -cp app.jar com.example.Main
```

For the AOT cache and AppCDS, point the argfile's `--module-path` at a directory holding only the SDK's runtime modules, not `.` and not the bundled test and example jars. Both caches require the module path to match the training run, and the test modules pull in an incubator module that stops the JDK from caching the module graph. That alone is worth about 100 ms (558 ms vs 455 ms). The [guide](https://github.com/beehive-lab/TornadoVM/blob/master/LEYDEN_GRAALVM_NATIVE_IMAGE.md#step-0-one-argfile-instead-of-the-launcher) has a three-line script that sets this up.

### Step 1: load only the backend you use

```bash
java @$TORNADOVM_HOME/tornado-argfile -Dtornado.enabled.backends=cuda -cp app.jar com.example.Main
```

The value is `opencl`, `cuda`, `metal` or a comma-separated list. Backend indices then count only the loaded backends, so with `cuda` alone, CUDA is backend 0. A name that matches no installed backend fails at start-up with `TornadoBackendNotFound`, and the error lists what is installed.

### Step 2: a Project Leyden AOT cache (JDK 25+)

An AOT cache (JEPs [483](https://openjdk.org/jeps/483), [514](https://openjdk.org/jeps/514) and [515](https://openjdk.org/jeps/515)) stores what a training run did: classes already loaded and linked, resolved method handles and lambdas, and method profiles. Later runs start from there. For TornadoVM, it covers the runtime, the drivers' FFM bindings and TC.

The quickest way is the launcher, which trains on the first run and uses the cache afterwards:

```bash
tornado --aotCache app.aot --classpath app.jar com.example.Main    # first run: trains and writes app.aot
tornado --aotCache app.aot --classpath app.jar com.example.Main    # later runs start from it
```

For the full gain, use the argfile from Step 0:

```bash
# training run: give it a workload that compiles and launches your kernels
java @$TORNADOVM_HOME/tornado-argfile -Dtornado.enabled.backends=cuda \
     -XX:AOTCacheOutput=app.aot -cp app.jar com.example.Main

# production runs: same JDK, same flags, same class path
java @$TORNADOVM_HOME/tornado-argfile -Dtornado.enabled.backends=cuda \
     -XX:AOTCache=app.aot -cp app.jar com.example.Main
```

A few rules apply:

- Train and run on the **same JDK build**, with the **same module and class paths**. Your application must be on the class path as jars, not directories of classes.
- **Retrain after changing the application or the SDK.** A cache that doesn't match is silently ignored; add `-XX:AOTMode=on` to make a mismatch an error instead.
- **Train once per backend selection.** A cache trained with `enabled.backends=cuda` holds CUDA's classes, not OpenCL's.
- To check that the cache is used, add `-Xlog:aot` and look for `Opened AOT cache app.aot` and `full module graph: enabled`.

On JDK 21, AppCDS gives a smaller version of the same benefit: `-XX:ArchiveClassesAtExit=app.jsa` on the first run and `-XX:SharedArchiveFile=app.jsa` afterwards. In the table above, that brings JDK 21 from 652 ms to 405 ms.

### Step 3: a GraalVM Native Image (GraalVM for JDK 25+)

`tornado-native-image` builds the whole application, including the TornadoVM runtime, the drivers and TC, into one executable that needs no JVM:

```bash
source setvars.sh                                   # a jdk22plus SDK
export GRAALVM_HOME=/path/to/graalvm-jdk-25         # GraalVM CE or Oracle GraalVM, JDK 25+

tornado-native-image -cp app.jar -o app com.example.Main

./app                                               # the default backend
./app -Dtornado.enabled.backends=cuda               # load only CUDA
./app -Dtornado.tc.verbose=true                     # TornadoVM and TC options work as on the JVM
```

The build runs in two steps. First it runs your application on the GraalVM JVM once per installed backend, with the Native Image agent recording the reflection, resources and FFM downcalls it uses. Then it runs `native-image` with that metadata. For saxpy the whole build takes about 16 s and peaks at 2.5 GB of memory.

The executable loads the device libraries (`libcuda`, `libnvrtc`, the OpenCL ICD or Metal) at run time, just as the JVM does. So the target machine needs the GPU drivers, but not the TornadoVM SDK or a JDK.

Two things to know:

- **Trace every kernel.** Kernels are still compiled at run time, and TC reads their class files from the image. The image only contains what the tracing runs used, so give `tornado-native-image` arguments that compile and launch every kernel and code path your application needs. A kernel that was never traced fails at run time with `ClassNotFoundException`.
- **Not covered yet:** applications that call the JNI-based libraries (cuDNN, cuDF, CUTLASS), and Windows.

### Bonus: the compiler on its own

`tc emit` compiles a single kernel to CUDA C++, OpenCL C or MSL without a GPU or the TornadoVM runtime, which is handy in CI. `build-tc-native` turns it into a Native Image too, cutting a cold `tc emit` from about 0.15 s to 0.01 s with byte-identical output:

```bash
GRAALVM_HOME=/path/to/graalvm-jdk-25 build-tc-native
tc emit --target cuda --cp app.jar com.example.Kernels saxpy
```

## Get it

Leyden and Native Image need the `jdk22plus` SDK of TornadoVM v8.0.0:

```bash
sdk install tornadovm 8.0.0-jdk22plus-cuda   # or -opencl, -metal
tornado --devices
```

Prebuilt SDKs are also on the [downloads page](https://www.tornadovm.org/downloads), or you can build from source with `make jdk22plus BACKEND=opencl,cuda`.

If your application starts often (a CLI, a function, a test suite), try `-Dtornado.enabled.backends` first, then an AOT cache. If you run into problems, the [troubleshooting table](https://github.com/beehive-lab/TornadoVM/blob/master/LEYDEN_GRAALVM_NATIVE_IMAGE.md#troubleshooting) covers the common errors. Please [open an issue](https://github.com/beehive-lab/TornadoVM/issues) for anything it doesn't cover.

[Read about TC, the compiler that made this possible →](https://www.tornadovm.org/blogs/tornadovm-8-0-0-tc-tornadovm-compiler)

