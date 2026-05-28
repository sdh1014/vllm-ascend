# brainstorm: 支持上游 vLLM GGUF 量化

## 目标

调研 vLLM Ascend 完整支持上游 vLLM GGUF 量化格式与语义需要做哪些工作，并定义首个 MVP 实现范围。

## 已知信息

* 用户希望支持上游 vLLM 的 GGUF 量化格式和语义。
* 这是一个跨模型加载、量化配置、tensor 元数据和 NPU 执行路径的复杂兼容任务。
* vLLM Ascend 已有自己的量化包：`vllm_ascend/quantization/`。
* 上游 vLLM 本地 checkout 位于 `/Users/songdehao/sdh-lab/code/vllm`，commit 为 `87e31455b`，日期是 2026-05-20。
* 上游 GGUF 当前包含 `GGUFModelLoader`、`GGUFConfig`、Linear、Embedding 和 MoE 方法。
* 上游 GGUF 运行时调用 `torch.ops._C.ggml_*` kernel；vLLM Ascend 目前没有 GGML/GGUF 自定义算子实现。
* `NPUPlatform.supported_quantization` 当前只包含 `ascend` 和 `compressed-tensors`。
* 上游 vLLM 已有 Gemma3 多模态 GGUF 生成测试：`tests/models/multimodal/generation/test_multimodal_gguf.py`。
* 该测试覆盖 Gemma3 GGUF backbone + `mmproj*.gguf`，并用 HF `AutoModelForImageTextToText` 作为 logprobs 比较基线。

## 临时假设

* 目标行为应跟随上游 vLLM 的 GGUF CLI/API 语义，不新增 Ascend 专属用户流程。
* 第一阶段聚焦 dense text 模型。
* 第二阶段明确纳入 GGUF 多模态，优先 Gemma3，因为该路径在单机上可行且上游已有真实生成测试。
* 第三阶段纳入 MoE TP 支持。

## 待确认问题

* 确认阶段边界：第一阶段 dense text GGUF；第二阶段 GGUF 多模态；第三阶段 MoE TP。

## 需求（演进中）

* 识别上游 vLLM GGUF 入口、依赖和运行时假设。
* 识别 vLLM Ascend 可能受影响的文件。
* 输出工作拆分、测试项和验证命令。
* 保持上游用户可见 GGUF 引用方式：本地 `.gguf`、`<repo_id>/<filename>.gguf`、`<repo_id>:<quant_type>`。
* 增加 Ascend GGUF Linear 和 Embedding 执行路径。
* 不把 GGUF 路径接入 ModelSlim `ascend` 量化检测。
* 第二阶段支持 GGUF 多模态，先复用上游 Gemma3 多模态 GGUF 测试方式。
* 第三阶段支持 MoE TP，覆盖 routed experts、expert 权重加载和 TP 切分。
* 第一阶段文本验证模型选定为 Qwen2.5-1.5B-Instruct、Qwen3-0.6B 和 Phi-3.5-mini-instruct。
* 第一阶段测试策略为 3 个 dense text E2E 模型加 GGML quant type 单元测试矩阵。
* 其他上游 GGUF 文本模型纳入扩展测试，不作为第一阶段 PR 必过项。
* NPU packed GGML kernel 优先用 Triton Ascend 做 `dequant` 和 `matvec` 原型。

## 验收标准（演进中）

* [x] 调研记录覆盖上游 GGUF 加载和量化语义。
* [x] 调研记录覆盖 vLLM Ascend 本地缺口和可能修改文件。
* [ ] MVP 范围和明确排除项已确认。
* [ ] Qwen2.5-1.5B-Instruct、Qwen3-0.6B 和 Phi-3.5-mini-instruct 的 GGUF 文本用例可通过与上游 vLLM 相同的引用方式在 NPU 加载。
* [ ] GGML quant type 单元测试矩阵覆盖 standard、K-quants 和 I-matrix 类型的 packed 解包与 dequant 正确性。
* [ ] Linear、embedding、LM head，以及 GGUF 内 F32/F16/BF16 非量化模块行为正确。
* [ ] Triton Ascend `ggml_dequantize` 和 decode `ggml_mul_mat_vec_a8` 原型完成可行性验证，或记录不能作为第一阶段性能路径的证据。
* [ ] 第二阶段 Gemma3 多模态 GGUF 验证方案已记录，包含 Q4_0 + F16 mmproj 和 BF16 + BF16 mmproj 两类上游用例。
* [ ] 第三阶段 MoE TP 支持范围已记录，包含 MoE GGUF 权重加载、expert 映射和 TP 验证。

## 完成定义

* 行为变更有对应测试新增或更新。
* lint 和相关单元测试通过。
* 记录 NPU 运行时验证方案。
* 如果用户可见参数或支持格式变化，更新文档。

## 明确排除

* 本轮初始调研不包含实现。
* 首个 MVP 不要求完成高性能原生 NPU GGML packed-weight kernel。
* 第一阶段不包含 MoE GGUF 和多模态 GGUF 实现。
* 第二阶段包含 GGUF 多模态实现与验证。
* 第三阶段包含 MoE TP 实现与验证。

## 技术记录

* 任务目录：`.trellis/tasks/05-27-gguf-quantization-support`
* 初始本地搜索发现量化模块位于 `vllm_ascend/quantization/`，测试位于 `tests/ut/quantization/`。

## 调研引用

* [`research/upstream-gguf.md`](research/upstream-gguf.md) - 上游 GGUF 用户流程、加载器、量化方法和运行时 kernel 调用。
* [`research/ascend-gap.md`](research/ascend-gap.md) - vLLM Ascend 缺口和推荐 MVP。

## 技术方案

推荐方案：分三阶段推进。第一阶段为 dense text 模型增加 Ascend GGUF 兼容路径，复用上游 GGUF 加载逻辑，并在加载后把 packed GGUF 权重转成 NPU 友好的 dense tensor；同时用 Triton Ascend 验证 packed `dequant` 和 decode `matvec` 原型。第二阶段支持 GGUF 多模态，优先 Gemma3 单机路径。第三阶段支持 MoE TP。

候选方案：

* Dense materialization first - 正确性优先，实现风险最小。
* Native GGML CANN kernels first - 内存和性能更好，但范围明显更大。
* External GGUF plugin dependency first - 适合上游先移除 in-tree GGUF 后再接入。

## 工作拆分

### 第一阶段：Dense Text GGUF

* 将 `gguf` 加入 Ascend 平台量化支持和依赖处理。
* 注册或包装 `GGUFConfig`，增加 Ascend 专用 Linear 和 Embedding 方法。
* 复用上游 `GGUFModelLoader`；只为真实 Ascend 不兼容点增加 patch。
* 实现 packed GGUF 权重加载后的 dense materialization。
* 增加检测、配置选择、synthetic qweight 处理、loader 引用方式的单元测试。
* 增加三组 dense text GGUF NPU smoke test，覆盖 Qwen2.5-1.5B-Instruct、Qwen3-0.6B 和 Phi-3.5-mini-instruct。

第一阶段文本验证模型：

* `Qwen/Qwen2.5-1.5B-Instruct`，GGUF repo 为 `Qwen/Qwen2.5-1.5B-Instruct-GGUF`，文件为 `qwen2.5-1.5b-instruct-q6_k.gguf`。
* `Qwen/Qwen3-0.6B`，GGUF repo 为 `unsloth/Qwen3-0.6B-GGUF`，文件为 `Qwen3-0.6B-BF16.gguf`。
* `microsoft/Phi-3.5-mini-instruct`，GGUF repo 为 `bartowski/Phi-3.5-mini-instruct-GGUF`，文件为 `Phi-3.5-mini-instruct-IQ4_XS.gguf`。

第一阶段测试分层：

* E2E 测试覆盖 loader、架构映射、tokenizer、embedding、linear、LM head 和生成路径。
* GGML quant type 单元测试覆盖 packed 解包、dequant、matvec/matmul 正确性。
* 必测 quant type 分组为 standard、K-quants 和 I-matrix；具体类型参考上游 `tests/kernels/quantization/test_gguf.py`。
* `gpt2-large`、`stablelm-3b-4e1t`、`TinyDolphin-2.8-1.1b`、`gemma-3-270m-it` 作为扩展测试。
* 上游已标 broken 的 `Llama-3.2-1B-Instruct-GGUF` 和 `starcoder2-3b-GGUF` 不进入第一阶段测试矩阵。

第一阶段 Triton Ascend kernel 优先级：

* 优先实现或验证 `ggml_dequantize`，覆盖 Embedding 和 Linear 的 packed weight dequant。
* 优先实现或验证 decode 场景 `ggml_mul_mat_vec_a8`，用于小 batch/token 路径。
* `ggml_mul_mat_a8` 大 batch 路径不作为第一阶段必需性能目标。
* `ggml_moe_a8`、`ggml_moe_a8_vec` 和 MoE TP kernel 放入第三阶段。
* 第一阶段语义验证仍以 dense materialization 为主，Triton Ascend 原型用于判断下一阶段性能路径。

第一阶段预计修改文件：

* `vllm_ascend/utils.py`：新增 `GGUF_QUANTIZATION_METHOD = "gguf"` 常量，避免在平台和量化模块中散写字符串。
* `vllm_ascend/platform.py`：把 `gguf` 加入 `supported_quantization`；在 `pre_register_and_update()` 中把 `gguf` 加入 `--quantization` choices；注册 Ascend GGUF quant config。
* `vllm_ascend/quantization/__init__.py`：增加 `AscendGGUFConfig` 的 lazy export。
* `vllm_ascend/quantization/gguf_config.py`：新增 Ascend GGUF quant config，复用上游 `GGUFConfig` 语义，并返回 Ascend 专用 Linear/Embedding method。
* `vllm_ascend/quantization/methods/gguf.py`：新增 Ascend GGUF Linear/Embedding 执行方法，第一阶段负责 packed GGUF 权重加载后的 dense materialization。
* `vllm_ascend/ops/triton/gguf.py`：新增 Triton Ascend `ggml_dequantize` 和 decode `ggml_mul_mat_vec_a8` 原型。
* `tests/ut/test_platform.py`：覆盖 `gguf` 平台支持和 CLI choices 注册。
* `tests/ut/quantization/test_gguf_config.py`：覆盖 GGUF config 注册、method 选择和非量化模块跳过行为。
* `tests/ut/quantization/methods/test_gguf.py`：覆盖 packed 权重 materialization、Linear/Embedding apply 行为和异常分支。
* `tests/e2e/singlecard/test_gguf_quantization.py`：覆盖三组 dense text GGUF NPU smoke。
* `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gguf.py`：覆盖 Triton Ascend GGUF kernel 原型正确性。
* `docs/source/user_guide/support_matrix/supported_features.md`：第一阶段完成后更新 GGUF 支持状态和限制。

### 第二阶段：GGUF 多模态

* 支持 GGUF backbone + `mmproj*.gguf` 加载。
* 支持 vision config patch 和 Gemma3 architecture patch。
* 支持多模态 processor kwargs，例如 `do_pan_and_scan`。
* 支持图文输入路径下的 `generate_greedy_logprobs` 验证。
* 先覆盖单机 Gemma3 多模态 GGUF，不要求 MoE TP。

### 第三阶段：MoE TP

* 支持 MoE GGUF expert 权重加载。
* 支持 `w13_qweight`、`w2_qweight`、expert id 映射和 TP 切分。
* 处理 Ascend MoE quant method 调用签名与上游 `GGUFMoEMethod` 的差异。
* 增加 MoE TP NPU 验证。

## 第二阶段：Gemma3 多模态 GGUF 验证规划

上游测试文件：`/Users/songdehao/sdh-lab/code/vllm/tests/models/multimodal/generation/test_multimodal_gguf.py`

上游已覆盖两组 Gemma3 用例：

* QAT Q4_0 backbone：`google/gemma-3-4b-it-qat-q4_0-gguf`，backbone 为 `gemma-3-4b-it-q4_0.gguf`，mmproj 为 `mmproj-model-f16-4B.gguf`。
* BF16 backbone：`unsloth/gemma-3-4b-it-GGUF`，backbone 为 `gemma-3-4b-it-BF16.gguf`，mmproj 为 `mmproj-BF16.gguf`，并开启 `do_pan_and_scan`。

第二阶段应复用该验证结构：

* 使用 `tokenizer_name="google/gemma-3-4b-it"`。
* 输入 stop sign 和 cherry blossom 两类图片 prompt。
* 运行 `generate_greedy_logprobs`。
* 以 HF `AutoModelForImageTextToText` 输出作为比较基线。
* 验证 GGUF backbone、`mmproj*.gguf`、vision config patch、processor kwargs 和图文输入路径均可用。
