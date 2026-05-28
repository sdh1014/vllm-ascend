# 上游 vLLM GGUF 调研

## 来源

* 本地上游 checkout：`/Users/songdehao/sdh-lab/code/vllm`，commit `87e31455b`，日期 2026-05-20。
* vLLM 最新 GGUF 文档：https://docs.vllm.ai/en/latest/features/quantization/gguf/
* vLLM 量化硬件支持表：https://docs.vllm.ai/en/latest/features/quantization/
* vLLM GGUF out-of-tree 迁移 RFC：https://github.com/vllm-project/vllm/issues/39583

## 用户可见语义

* 当 `EngineArgs.create_model_config()` 识别到 GGUF 模型引用时，会自动把 `quantization` 和 `load_format` 都设为 `gguf`。
* 支持的模型引用方式：
  * 本地 `*.gguf`；
  * `<repo_id>/<filename>.gguf`；
  * `<repo_id>:<quant_type>`，例如 `unsloth/Qwen3-0.6B-GGUF:Q4_K_M`。
* vLLM 文档建议使用 base HF tokenizer，因为 GGUF tokenizer 转换慢且不稳定。
* 最新文档仍将 GGUF 描述为实验性、未充分优化。
* 最新 vLLM 硬件支持表列出 GGUF 支持 NVIDIA/AMD GPU，不支持 Intel GPU 或 x86 CPU；表中没有 Ascend。

## 上游代码结构

* `vllm/engine/arg_utils.py` 负责检测 GGUF，并设置 `load_format="gguf"` 和 `quantization="gguf"`。
* `vllm/model_executor/model_loader/gguf_loader.py` 负责：
  * 解析本地文件、远端文件和 `repo_id:quant_type` 引用；
  * 将 GGUF tensor 名映射到 HF/vLLM 参数名；
  * 先读取 GGUF tensor type，再读取 tensor data；
  * 根据 F32/F16/BF16 tensor 更新 `unquantized_modules`。
* `vllm/model_executor/layers/quantization/gguf.py` 定义：
  * `GGUFConfig`；
  * `GGUFLinearMethod`；
  * `GGUFEmbeddingMethod`；
  * `GGUFMoEMethod`；
  * 支持的 GGML quant type 分组：standard、K-quants、I-matrix。
* 运行时通过 `vllm._custom_ops` 调用这些上游自定义算子：
  * `ggml_dequantize`；
  * `ggml_mul_mat_vec_a8`；
  * `ggml_mul_mat_a8`；
  * `ggml_moe_a8`；
  * `ggml_moe_a8_vec`；
  * `ggml_moe_get_block_size`。

## 需要保持的关键语义

* GGUF 权重在加载阶段不是普通 dense 权重。量化 tensor 会加载到 `qweight`，标量 GGML 类型会加载到 `qweight_type`。
* merged linear layer 可能先把多个 GGUF shard 放在 `data_container`，加载后再 materialize 成 padded packed tensor。
* GGUF 文件里的 F32/F16/BF16 tensor 被视为非量化模块，绕过 GGUF quant method。
* MoE GGUF 使用特殊的 `w13_qweight`、`w2_qweight` 和类型 tensor。
* 上游在部分模型文件里有模型专属 GGUF 处理，包括 Llama、Gemma3、Exaone、Jais2、Apertus、RNJ1、OpenPangu 和 SigLIP。

## 上游方向

* 上游已有 RFC，计划将 GGUF 迁移为 out-of-tree `vllm-gguf-plugin`。
* 因此 vLLM Ascend 应尽量避免复制大段上游 GGUF loader 代码；优先包装或继承当前上游/plugin API。
