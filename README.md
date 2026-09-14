<h1 align="center">Zhuoran (Jerry) Yin</h1>

<p align="center">
  Compiler engineer at AMD: MLIR, LLVM, and GPU code generation
</p>

<p align="center">
  <sub>Triton · TokenSpeed · IREE · rocMLIR · MIOpen · TensorFlow</sub>
</p>

---

I make machine learning models run fast on AMD GPUs. The work spans the stack that gets them there: LLVM intrinsics and MLIR underneath, Triton and IREE in the middle, inference engines and the ROCm libraries above. Eight years at AMD, nearly all of it in code generation and kernel performance, two of them leading the teams that do it.

When a new opportunity comes up, I pick the one that covers a gap in what I understand rather than the one most likely to last. That makes me a poor judge of longevity, and most of what I've worked on has since been deprioritized or folded into something else. It's also why I stepped out of managing rocMLIR and Triton to work on IREE: we had invented a lot of our own machinery in rocMLIR, and I wanted to understand how the upstream MLIR pieces fit together. The range above is the result.

I start from measurement and from reading the generated assembly. The counters show which shapes are losing, the ISA shows why, and together they decide what to work on. I'd rather fix a problem at the level where it belongs, often a layer below where it surfaced, and land it upstream where it keeps working after I've moved on.

### Where the work lands

| Project | Focus | |
| :--- | :--- | :--- |
| [triton-lang/triton](https://github.com/triton-lang/triton)<br><sub>kernel language</sub> | AMD backend: descriptor-based data movement, address spaces, LDS and bank-conflict heuristics | [pull requests](https://github.com/triton-lang/triton/pulls?q=is%3Apr+author%3Ajerryyin) |
| [lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed)<br><sub>inference engine</sub> | Mixture-of-experts decode kernels for AMD GPUs | [pull requests](https://github.com/lightseekorg/tokenspeed/pulls?q=is%3Apr+author%3Ajerryyin) |
| [iree-org/iree](https://github.com/iree-org/iree)<br><sub>compiler and runtime</sub> | GPU code generation: software pipelining, buffer intrinsics, tiling heuristics, convolution and GEMM performance | [pull requests](https://github.com/iree-org/iree/pulls?q=is%3Apr+author%3Ajerryyin) |
| [llvm/llvm-project](https://github.com/llvm/llvm-project)<br><sub>compiler infrastructure</sub> | Upstream MLIR and AMDGPU changes underpinning the above | [pull requests](https://github.com/llvm/llvm-project/pulls?q=is%3Apr+author%3Ajerryyin) |
| [ROCm/rocMLIR](https://github.com/ROCm/rocMLIR)<br><sub>kernel generator</sub> | Implicit GEMM, int8, tuning, CI, releases | [pull requests](https://github.com/ROCm/rocMLIR/pulls?q=is%3Apr+author%3Ajerryyin) |
| [ROCm/MIOpen](https://github.com/ROCm/MIOpen)<br><sub>DNN library</sub> | Compiler-generated solvers, flexible tensor layouts | [pull requests](https://github.com/ROCm/MIOpen/pulls?q=is%3Apr+author%3Ajerryyin) |
| [ROCm/AMDMIGraphX](https://github.com/ROCm/AMDMIGraphX)<br><sub>graph compiler</sub> | Quantization operators and fusion | [pull requests](https://github.com/ROCm/AMDMIGraphX/pulls?q=is%3Apr+author%3Ajerryyin) |
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)<br><sub>framework</sub> | ROCm enablement, kernels, XLA <sub>(2019–2020)</sub> | [pull requests](https://github.com/tensorflow/tensorflow/pulls?q=is%3Apr+author%3Ajerryyin) |

<sub>Or browse [every pull request I've opened](https://github.com/search?q=is%3Apr+author%3Ajerryyin&type=pullrequests), and the [660+ I've reviewed](https://github.com/search?q=type%3Apr+reviewed-by%3Ajerryyin&type=pullrequests).</sub>

### Selected work

**Memory movement and data layout**

- Added Triton-level `tt.descriptor_gather` and `tt.descriptor_scatter` for gfx1250, with layout-driven warp predication and padded shared-layout support. <sub>[triton#10157](https://github.com/triton-lang/triton/pull/10157)</sub>
- Mapped `tl.const` pointer arguments onto AMD's constant address space, cutting the `readfirstlane` churn that mixture-of-experts kernels provoke. Supported upstream by representing pointer address space as a named enum and carrying the buffer-op base address space in the op format. <sub>[triton#11294](https://github.com/triton-lang/triton/pull/11294)</sub>
- Tuned LDS padding and swizzle heuristics to bank-wrap boundaries to reduce bank conflicts, unifying the heuristic across the async-copy and descriptor-movement paths. <sub>[triton#9741](https://github.com/triton-lang/triton/pull/9741)</sub>
- Established the lowering path in IREE from `vector.transfer_read` to AMD GPU buffer load intrinsics, eliminating explicit bounds checks for roughly a 5× improvement on affected kernels, together with alignment validation, dynamic size checks, and correct linearization for non-packed layouts. <sub>[iree#21269](https://github.com/iree-org/iree/pull/21269)</sub>
- Removed redundant accumulator movement through pack/unpack folding and layout propagation, cutting shared-memory pressure and admitting larger tiles. <sub>[iree#19590](https://github.com/iree-org/iree/pull/19590)</sub>

**Latency hiding and software pipelining**

- Redesigned software pipelining in IREE's GPU backend, replacing a loop-based prefetch model with a stage-based approach that orders pipeline stages by dataflow through backward slicing. Configurable pipeline depth, automatic barrier insertion, and a migration onto upstream `scf` pipelining infrastructure. It generates optimized pipelines across a far wider range of workloads, particularly those with irregular access patterns and complex control flow. <sub>[iree#22605](https://github.com/iree-org/iree/pull/22605)</sub>
- Pipelined Triton's descriptor-based gathers and scatters through the asynchronous data-movement chain, and added multi-buffering and async-copy modes for LDS gathers, down to the MemRef and AMDGPU patterns upstream that make them legal. <sub>[triton#10172](https://github.com/triton-lang/triton/pull/10172) · [iree#23354](https://github.com/iree-org/iree/pull/23354)</sub>
- Wrote a control-flow fission pass isolating transfer operations, so aggressive prefetching remains effective regardless of convolution variant. <sub>[iree#21018](https://github.com/iree-org/iree/pull/21018)</sub>

**Tiling, heuristics, and tuning**

- Replaced IREE's fixed-threshold heuristics with a hardware-adaptive framework that classifies GEMMs by arithmetic intensity and derives tile sizes from compute-unit count and wavefront size, worth 10–20% on untuned convolution and GEMM and consistent across RDNA and CDNA. <sub>[iree#21638](https://github.com/iree-org/iree/pull/21638) · [iree#21546](https://github.com/iree-org/iree/pull/21546)</sub>
- Sized TokenSpeed's mixture-of-experts decode tiles by expert load, and distributed gather and scatter indices across warps. <sub>[tokenspeed#1194](https://github.com/lightseekorg/tokenspeed/pull/1194)</sub>
- Unified rocMLIR's blockwise and matrix-core GEMM paths, trading hard-coded arithmetic for transform-map-driven vectorization and widening the tuning space, so a single interface serves both GEMM and convolution.
- Built separate benchmarking matrices for convolution and GEMM, 380 and 500+ configurations respectively, giving continuous visibility into performance gaps and regressions rather than one-off measurements.

**Operators, precision, and library integration**

- Built the MXFP4 mixture-of-experts decode path in TokenSpeed. <sub>[tokenspeed#1100](https://github.com/lightseekorg/tokenspeed/pull/1100)</sub>
- Brought int8 convolution to rocMLIR through implicit GEMM, from specification to library integration. The result is orders of magnitude faster than the naive BLAS-based path it replaced, and repairing the reduction path along the way lifted general convolution throughput 10–30%. <sub>[rocMLIR#508](https://github.com/ROCm/rocMLIR/pull/508) · [rocMLIR#734](https://github.com/ROCm/rocMLIR/pull/734)</sub>
- Implemented quantization operators, fusion, and broadcast support for the graph compiler, aligning the definition across library APIs. <sub>[rocMLIR#999](https://github.com/ROCm/rocMLIR/pull/999)</sub>
- 52 pull requests into upstream TensorFlow (kernels, XLA, StreamExecutor, and the Python front end) and [83 more](https://github.com/ROCm/tensorflow-upstream/pulls?q=is%3Apr+author%3Ajerryyin) into AMD's ROCm fork, covering 3D convolution, RNN kernel gaps, layer normalization, dropout, and batched GEMM.

<sub>Also [k8s-interactive-pod](https://github.com/jerryyin/k8s-interactive-pod): launches interactive Kubernetes development pods with GPU access.</sub>

### Trajectory

| | |
| :--- | :--- |
| **2026–** | Triton and TokenSpeed: mixture-of-experts kernels, descriptor-based data movement, and gfx1250 enablement |
| **2025–2026** | Back to engineering on IREE's AMD GPU backend: software pipelining, buffer intrinsics, tiling heuristics |
| **2023–2024** | Led the rocMLIR team, then Triton as well: hiring, mentoring, and managing the transitions that moved the team into Triton, IREE, and MIGraphX |
| **2020–2022** | Founding engineer on rocMLIR, second on the project. Grew it from a two-person prototype into a production compiler shipping in ROCm releases, and into a team |
| **2019–2020** | ROCm TensorFlow: features, XLA, upstreaming, releases, CI |
| **2018** | ML deployment tooling; TensorFlow inference performance |

---

<p align="center">
  <sub>
    <a href="https://github.com/jerryyin?tab=repositories">Repositories</a> · <a href="https://gist.github.com/jerryyin">Gists</a> · <a href="https://github.com/search?q=is%3Apr+author%3Ajerryyin&type=pullrequests">Pull requests</a>
  </sub>
</p>
