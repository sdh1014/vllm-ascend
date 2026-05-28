# vLLM Ascend GGUF 缺口调研

## 本地发现

* `NPUPlatform.supported_quantization` 当前只包含 `ascend` 和 `compressed-tensors`。
* vLLM Ascend 的 docs、tests、`requirements.txt`、`pyproject.toml` 里没有 `gguf` 条目。
* vLLM Ascend 没有 `csrc/quantization/gguf` 或 GGML 自定义算子实现。
* 上游 GGUF Python 方法可以被 Ascend layer 选中，因为 Ascend linear 和 embedding layer 会调用 `quant_config.get_quant_method(...)`。
* 第一个硬失败点是运行时执行：上游 GGUF 方法会调用 `torch.ops._C.ggml_*`，这些是 CUDA/ROCm 方向的 vLLM 算子，NPU 没有实现。

## 可能工作项

1. 平台和依赖
   * 将 `gguf` 加入 Ascend 支持的量化列表。
   * 如果安装的上游 vLLM 包不再提供 `gguf` 依赖，则在 vLLM Ascend 中补充 Python 依赖。
   * 确保 Ascend 上 `vllm serve <repo>:<quant_type>` 仍能自动选择 `load_format="gguf"`。

2. 量化配置接入
   * 引入 Ascend GGUF config，或注册对 `gguf` 的覆盖实现。
   * 保持上游用户语义：显式 `--quantization gguf`、本地 `.gguf`、远端 `<repo>/<file>.gguf`、`<repo>:<quant_type>`。
   * 不把 GGUF 接入 ModelSlim `ascend` 量化检测。

3. 权重加载兼容
   * 尽量复用上游 `GGUFModelLoader`。
   * 验证 Ascend 自定义 linear、embedding、LM head 和 MoE loader 是否保持上游 GGUF 属性：`is_gguf_weight`、`is_gguf_weight_type`、`qweight`、`qweight_type`、`data_container` 和 shard 元数据。

4. NPU 执行路径
   * 实现 Ascend GGUF `LinearMethod` 和 `EmbeddingMethod`。
   * MVP 路径：加载后把 GGUF packed 权重 dequantize 成 NPU 友好的 dense FP16/BF16 tensor，再复用现有 Ascend 非量化 GEMM/embedding。
   * 完整性能路径：增加 GGML dequantize、matmul-vector、matmul-batch、MoE 的 CANN/custom-op 实现。

5. MoE 支持
   * 上游 `GGUFMoEMethod.apply()` 签名与 `vllm_ascend/ops/fused_moe/fused_moe.py` 不匹配；Ascend MoE 会传入 `router_logits`、`pertoken_scale`、`top_k` 等参数。
   * 除非 MVP 明确要求，MoE 应拆成独立阶段。

6. 测试和验证
   * 增加 GGUF 检测和 NPU platform config 注册单元测试。
   * 用 synthetic GGUF `qweight/qweight_type` tensor 增加 Linear 和 Embedding 单元测试。
   * 增加本地 `.gguf` 和 `repo_id:quant_type` 引用解析的 loader 级测试。
   * 增加小型 dense GGUF 模型 NPU smoke test，并与上游 CPU/GPU 或 HF baseline 做误差比较。

## 推荐 MVP

先支持 dense text 模型：

* 本地 `.gguf`；
* `<repo_id>:<quant_type>`；
* Linear、embedding 和 LM head；
* GGUF 内 F32/F16/BF16 非量化模块；
* tensor parallel 路径仅在 dense materialization 后能复用上游 sharding 时纳入。

暂不包含：

* MoE GGUF；
* 多模态 GGUF/mmproj；
* 原生 NPU GGML packed-weight kernel；
* 与 CUDA/ROCm 的性能一致性。
