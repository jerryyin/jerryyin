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

- Triton `tt.descriptor_gather`/`tt.descriptor_scatter` for gfx1250 TDM: warp predication, lowering, pipelining, and follow-on hardening. <sub>[triton#10019](https://github.com/triton-lang/triton/pull/10019) → [triton#10021](https://github.com/triton-lang/triton/pull/10021) → [triton#10157](https://github.com/triton-lang/triton/pull/10157) → [triton#10172](https://github.com/triton-lang/triton/pull/10172) → [triton#10215](https://github.com/triton-lang/triton/pull/10215) → [triton#10230](https://github.com/triton-lang/triton/pull/10230) → [triton#10568](https://github.com/triton-lang/triton/pull/10568) → [triton#10564](https://github.com/triton-lang/triton/pull/10564)</sub>
- Mapped `tl.const` pointer args onto AMD's constant address space, with upstream pointer-address-space enum and buffer-op format support. <sub>[triton#11294](https://github.com/triton-lang/triton/pull/11294) → [triton#11379](https://github.com/triton-lang/triton/pull/11379) → [triton#11385](https://github.com/triton-lang/triton/pull/11385)</sub>
- LDS padding and swizzle heuristics at bank-wrap boundaries, unified across async-copy and TDM paths. <sub>[triton#9741](https://github.com/triton-lang/triton/pull/9741) → [triton#9747](https://github.com/triton-lang/triton/pull/9747) → [triton#9780](https://github.com/triton-lang/triton/pull/9780)</sub>
- Upstream lowering of `vector.transfer_read`/`vector.maskedload` to AMDGPU buffer loads with alignment and size guards. <sub>[llvm#131803](https://github.com/llvm/llvm-project/pull/131803) → [llvm#135014](https://github.com/llvm/llvm-project/pull/135014) → [llvm#135982](https://github.com/llvm/llvm-project/pull/135982) → [llvm#138922](https://github.com/llvm/llvm-project/pull/138922) → [llvm#146705](https://github.com/llvm/llvm-project/pull/146705)</sub>
- Pack/unpack folding and layout propagation across LLVM and IREE. <sub>[llvm#117340](https://github.com/llvm/llvm-project/pull/117340) → [iree#19590](https://github.com/iree-org/iree/pull/19590) → [llvm#138332](https://github.com/llvm/llvm-project/pull/138332) → [llvm#146139](https://github.com/llvm/llvm-project/pull/146139)</sub>

**Latency hiding and software pipelining**

- IREE GPU software pipelining: upstream scf-based prefetching, stage-based prefetcher, then configurable multi-stage pipelines. <sub>[iree#22523](https://github.com/iree-org/iree/pull/22523) → [iree#22605](https://github.com/iree-org/iree/pull/22605)</sub> <sub>[iree#22669](https://github.com/iree-org/iree/pull/22669) → [iree#22673](https://github.com/iree-org/iree/pull/22673) → [iree#22725](https://github.com/iree-org/iree/pull/22725) → [iree#22788](https://github.com/iree-org/iree/pull/22788) → [iree#22818](https://github.com/iree-org/iree/pull/22818) → [iree#22868](https://github.com/iree-org/iree/pull/22868)</sub>
- Multi-buffered, pipelined LDS gathers in IREE on top of upstream MemRef/AMDGPU changes. <sub>[llvm#176941](https://github.com/llvm/llvm-project/pull/176941) → [llvm#182364](https://github.com/llvm/llvm-project/pull/182364) → [iree#23354](https://github.com/iree-org/iree/pull/23354) → [iree#23400](https://github.com/iree-org/iree/pull/23400) → [iree#23648](https://github.com/iree-org/iree/pull/23648) → [iree#24114](https://github.com/iree-org/iree/pull/24114) → [iree#24210](https://github.com/iree-org/iree/pull/24210) → [iree#24116](https://github.com/iree-org/iree/pull/24116)</sub>
- Control-flow fission pass isolating transfer ops so prefetching holds across convolution variants. <sub>[iree#21018](https://github.com/iree-org/iree/pull/21018)</sub>

**Tiling, heuristics, and tuning**

- Hardware-adaptive GEMM tiling heuristics driven by arithmetic intensity. <sub>[iree#21546](https://github.com/iree-org/iree/pull/21546) → [iree#21638](https://github.com/iree-org/iree/pull/21638) → [iree#21691](https://github.com/iree-org/iree/pull/21691) → [iree#21803](https://github.com/iree-org/iree/pull/21803) → [iree#21826](https://github.com/iree-org/iree/pull/21826) → [iree#21834](https://github.com/iree-org/iree/pull/21834)</sub>
- TokenSpeed MXFP4 MoE decode on gfx1250: base path, expert-load tile sizing, warp-distributed gather/scatter indices. <sub>[tokenspeed#1100](https://github.com/lightseekorg/tokenspeed/pull/1100) → [tokenspeed#1194](https://github.com/lightseekorg/tokenspeed/pull/1194) → [tokenspeed#1455](https://github.com/lightseekorg/tokenspeed/pull/1455) → [tokenspeed#1503](https://github.com/lightseekorg/tokenspeed/pull/1503) → [tokenspeed#1578](https://github.com/lightseekorg/tokenspeed/pull/1578) → [tokenspeed#1622](https://github.com/lightseekorg/tokenspeed/pull/1622)</sub>
- Unified rocMLIR's blockwise and matrix-core GEMM paths behind transform-map-driven vectorization, one interface for GEMM and convolution.
- Built convolution and GEMM benchmarking matrices (380 and 500+ configs) for continuous visibility into gaps and regressions.

**Operators, precision, and library integration**

- int8 convolution in rocMLIR via implicit GEMM: lowering, kernels, tuning, CI. <sub>[rocMLIR#485](https://github.com/ROCm/rocMLIR/pull/485) → [rocMLIR#494](https://github.com/ROCm/rocMLIR/pull/494) → [rocMLIR#505](https://github.com/ROCm/rocMLIR/pull/505) → [rocMLIR#508](https://github.com/ROCm/rocMLIR/pull/508) → [rocMLIR#539](https://github.com/ROCm/rocMLIR/pull/539) → [rocMLIR#568](https://github.com/ROCm/rocMLIR/pull/568) → [rocMLIR#601](https://github.com/ROCm/rocMLIR/pull/601) → [rocMLIR#611](https://github.com/ROCm/rocMLIR/pull/611) → [rocMLIR#618](https://github.com/ROCm/rocMLIR/pull/618)</sub>
- Reduction-xdlops tuning-space removal and GEMM perf refactors. <sub>[rocMLIR#689](https://github.com/ROCm/rocMLIR/pull/689) → [rocMLIR#699](https://github.com/ROCm/rocMLIR/pull/699) → [rocMLIR#711](https://github.com/ROCm/rocMLIR/pull/711) → [rocMLIR#724](https://github.com/ROCm/rocMLIR/pull/724) → [rocMLIR#727](https://github.com/ROCm/rocMLIR/pull/727) → [rocMLIR#734](https://github.com/ROCm/rocMLIR/pull/734) → [rocMLIR#754](https://github.com/ROCm/rocMLIR/pull/754)</sub>
- Quantization ops and MIGraphX int8 quantization end-to-end. <sub>[rocMLIR#969](https://github.com/ROCm/rocMLIR/pull/969) → [rocMLIR#990](https://github.com/ROCm/rocMLIR/pull/990) → [rocMLIR#999](https://github.com/ROCm/rocMLIR/pull/999) → [rocMLIR#1005](https://github.com/ROCm/rocMLIR/pull/1005) → [rocMLIR#1024](https://github.com/ROCm/rocMLIR/pull/1024) → [rocMLIR#1031](https://github.com/ROCm/rocMLIR/pull/1031) → [rocMLIR#1034](https://github.com/ROCm/rocMLIR/pull/1034) → [rocMLIR#1078](https://github.com/ROCm/rocMLIR/pull/1078) → [rocMLIR#1089](https://github.com/ROCm/rocMLIR/pull/1089) → [rocMLIR#1105](https://github.com/ROCm/rocMLIR/pull/1105)</sub>
- 52 pull requests into upstream TensorFlow (kernels, XLA, StreamExecutor, and the Python front end) and [83 more](https://github.com/ROCm/tensorflow-upstream/pulls?q=is%3Apr+author%3Ajerryyin) into AMD's ROCm fork, covering 3D convolution, RNN kernel gaps, layer normalization, dropout, and batched GEMM.


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
