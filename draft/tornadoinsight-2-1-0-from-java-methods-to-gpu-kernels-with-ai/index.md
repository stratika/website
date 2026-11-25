---
title: "TornadoInsight 2.1.0: From Java Methods to GPU Kernels with AI"
date: ""
description: "TornadoInsight 2.1.0 turns a Java method into a TornadoVM kernel with an LLM, checks it on your own GPU, and shows the diff before anything changes."
authors:
  - "thanos-stratikopoulos"
image: "tornadoinsight-2-1-0.jpg"
categories:
  - "TornadoVM"
  - "IntelliJ IDEA"
  - "AI"
  - "Tools"
related_posts:
  - "tornadoinsight-an-intellij-plugin-for-tornavovm"
  - "tornadoinsight-compatibility-with-tornadovm-sdk-2-0-configuration-guide"
  - "build-and-run-tornadovm-with-intellij-idea"
canonical: "https://www.tornadovm.org/blogs/tornadoinsight-2-1-0-from-java-methods-to-gpu-kernels-with-ai"
---

{{< note >}}
This article was first published on the TornadoVM blog on 2 November 2026, for TornadoInsight 2.1.0. Everything it describes is available in the latest release, [TornadoInsight 2.2.0](https://github.com/beehive-lab/tornado-insight/releases/tag/v2.2.0), which also gives Kernel API kernels a default work-group size and adds a Grid / work-group column to **Fastest (measured)**.
{{< /note >}}

**TornadoInsight 2.1.0 can turn a plain Java method into a TornadoVM kernel with the help of an LLM, check that kernel on your own GPU, and show you the change before anything is written.** It is experimental and off by default.

[TornadoInsight](https://plugins.jetbrains.com/plugin/23309-tornadoinsight) is our IntelliJ IDEA plugin for TornadoVM, the plugin to OpenJDK that runs Java on GPUs and multi-core CPUs. Until now it helped you write kernels: static checks flag code that TornadoVM cannot run on a device, and a dynamic inspection runs a kernel and shows the code TornadoVM generates for it. With 2.1.0 it can also help you get to a kernel in the first place. The release also adds quick fixes for the static checks and makes the dynamic inspection more reliable.

## Why we built it

The hardest step with TornadoVM is often the first one: taking a Java method you already have and rewriting it for one of TornadoVM's parallel APIs. You need to know which loops are independent, when to use `@Parallel` or `@Reduce`, how a `KernelContext` kernel maps threads to data, and which Java features a device cannot run.

Our goal is to let an LLM do that rewrite, from inside IntelliJ IDEA, without asking you to trust its answer. TornadoInsight already knows what a valid kernel looks like, and it can run TornadoVM on your machine. So the model proposes, and TornadoInsight checks, runs and compares before you see anything.

## How it works

Right-click a plain Java method and choose **Transform into TornadoVM Kernel (AI)...**, or press `Alt`+`Enter` on its name. A dialog asks which TornadoVM API to target:

- **Loop Parallel API:** the method's own loops, with `@Parallel` on the independent ones and `@Reduce` for reductions. The most readable, and it still runs as plain Java.
- **Kernel API:** a `KernelContext` kernel with explicit threads, work-groups, local memory and barriers, launched with a `GridScheduler`.
- **Tile API:** a `TileContext` kernel that works on whole tiles of elements, for NVIDIA GPUs with TornadoVM 7.0.0 or newer.
- **Fastest (measured):** transforms the method into each API it can, times each verified kernel on your device, and recommends one.

The model's answer has to pass TornadoInsight's checks and, when you ask for it, give the same results as your method on the device; otherwise it goes back to the model, up to three times. A diff then shows the change, and accepting replaces the method or adds the kernel beside it, as one step you can undo. Transformations run one at a time in the background, can be cancelled, and wait their turn in the status bar.

## Every kernel is checked before you see it

An LLM can write a kernel that looks right and computes the wrong thing, so TornadoInsight treats each answer like code it has to prove.

- **It must be a valid kernel.** The answer must parse, its names must resolve, it must keep the method's parameters, be `static void`, and pass the same static checks TornadoInsight runs in the editor. Problems go back to the model, for up to three answers.
- **It must give the same results.** If you tick **Run each kernel on this machine**, TornadoInsight compiles the kernel, runs it with TornadoVM on the device, runs your original method too, and compares the outputs on two synthetic data sets: 4096 values, then 4097 values, an odd length no work-group size divides. A kernel that TornadoVM falls back to running in Java does not pass. A kernel that fails or differs goes back to the model; if the last answer still fails, no kernel is offered.
- **It must be parallel.** TornadoVM does not check that a loop's iterations are independent; a `@Parallel` loop like `y[i] = a * x[i] + y[i - 1]` runs, and computes wrong results. The model is asked to say when a method cannot be parallelised, and why, and TornadoInsight then reports that instead of offering a kernel.

The kernel runs with your user's rights, so the checks also restrict what it may use: only the device side of the TornadoVM API, `Math`, the primitive wrappers and your project's own code, held to the same rules. Your Claude API key is kept out of its environment.

## Example: Black-Scholes, 25x faster on an Apple M4 Pro

With **Fastest (measured)**, a Black-Scholes method became a Loop Parallel kernel that runs **25.4× faster** than the Java method, including the transfers to and from the GPU. Both kernels gave the same outputs as the method on both data sets.

| Kernel | Time per run | vs Java | Device kernel | Transfers |
|---|---|---|---|---|
| Java (the method) | 30.6 ms | 1× | – | – |
| **Loop Parallel API** (recommended) | **1.2 ms** | **25.4× faster** | 0.138 ms | 0.380 ms |
| Kernel API | 1.2 ms | 24.6× faster | 0.330 ms | 0.384 ms |

Median of 10 runs on 1,048,576 elements per array, after 3 warm-up runs, with TornadoVM 8.0.0 and the Metal backend. The Tile API was not compared, since it needs an NVIDIA GPU with the CUDA backend.

The Loop Parallel kernel is the recommendation even though the two times tie: TornadoInsight prefers the more readable kernel unless another is faster by more than a threshold, 20% by default. The Loop Parallel version is the original method with `@Parallel` on its loop and `TornadoMath` in place of `Math`, so it still runs as plain Java when called directly.

Not every method gains. On the same machine, a SAXPY loop (`z[i] = a * x[i] + y[i]`) was about 3× slower on the GPU than in Java: it does too little work per element to pay for the transfers. Measuring is how you find that out before you change your code.

## Your model, your choice

The transformation stays off until you choose a model, because it sends your code to that model: the method, the methods it calls, the fields it reads, and the name and backend of the device the kernel will run on. When a kernel is run, compiler errors and up to 1,500 characters of TornadoVM's output per failure go back too, with your directories replaced by names such as `$TORNADOVM_HOME` and `~`.

You set it up under **Settings | Other Settings | TornadoInsight | AI-Assisted Transformation**, with a provider, a model, an API key and a **Test Connection** button. Two kinds of provider work:

- **A local server with the OpenAI chat API**, such as llama.cpp, vLLM, Ollama or LM Studio. Your code stays on your machine or network.
- **The Claude API.** The key is kept in the IDE password store, or read from `ANTHROPIC_API_KEY`.

## Also in 2.1.0

**Quick fixes for the static checks.** Press `Alt`+`Enter` on a problem the static checks report, and TornadoInsight fixes it:

- remove an `assert` statement
- remove an unused system or native call
- remove a `throw` statement or a `throws` clause
- unwrap a `try`/`catch`, keeping its body
- make a kernel return `void`

**A more reliable dynamic inspection.** It now uses a single TornadoVM toolchain, and explains why a JDK cannot run your TornadoVM SDK, for example a GraalVM JDK with TornadoVM 6.0.0 or newer. The test program is compiled against every jar of the SDK, and the copied kernel code gets clean imports and rewritten class-qualified references. API keys are kept out of the environment of the `tornado` launcher.

The full list is in the [release notes](https://github.com/beehive-lab/tornado-insight/releases/tag/v2.1.0).

## Get started

In IntelliJ IDEA, open **Settings | Plugins | Marketplace**, search for **TornadoInsight**, install it and restart the IDE, or get it from the [plugin page](https://plugins.jetbrains.com/plugin/23309-tornadoinsight). To transform and run kernels you also need:

- a TornadoVM SDK, with `TORNADOVM_HOME` visible to the IDE (`sdk install tornadovm` with [SDKMAN!](https://sdkman.io/), or the [downloads page](https://www.tornadovm.org/downloads))
- JDK 21 or newer as the project JDK, a standard OpenJDK build matching your TornadoVM SDK
- a local server with the OpenAI chat API, or a Claude API key
- for the Tile API, TornadoVM 7.0.0 or newer and an NVIDIA GPU with the CUDA backend

The transformation is experimental, and we want to hear how it does on your code: which methods it gets right, which it does not, and which models work best. Tell us in the [TornadoInsight issues](https://github.com/beehive-lab/tornado-insight/issues).

[Read the TornadoInsight configuration guide →](/today/tornadoinsight-compatibility-with-tornadovm-sdk-2-0-configuration-guide/) for how to point the plugin at your TornadoVM SDK.

