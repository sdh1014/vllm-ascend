# kvcache offload native

## 目标

适配并验证 vLLM Ascend 对 `--kv-offloading-backend native` 的支持，使其兼容近期 vLLM upstream main 的 native KV offload API 变化。

## 已知信息

- 需要完善并验证 `native` KV offloading backend。
- 已检查的 vLLM upstream main 最新提交是 `b06813e87207e15b133e903d641e03f237d85b17`。
- vLLM Ascend 当前在 `vllm_ascend/kv_offload/` 下已有 `NPUOffloadingSpec` 和 `CpuNpuOffloadingHandler`。
- 现有 vLLM Ascend 文档和 e2e 测试仍使用显式 `kv_transfer_config`，并配置 `num_cpu_blocks`。
- upstream native CLI 路径现在通过 `--kv-offloading-size` 写入 `cpu_bytes_to_use`。

## 需求

- 支持 native CLI 路径：`--kv-offloading-backend native` + `--kv-offloading-size`。
- 将 `NPUOffloadingSpec` 适配到当前 upstream `OffloadingSpec` API。
- 将 NPU-CPU KV transfer handler 适配到 upstream `CanonicalKVCaches`。
- MVP 只支持新 native CLI 路径，不继续维护旧的显式 `num_cpu_blocks` 配置路径。
- NPU 验证模型使用 dense 中等规模模型 `Qwen/Qwen3-8B`。
- 后续 MoE 完整语义验证使用多卡 TP/EP。
- 默认复用现有 `torch.ops._C_ascend.swap_blocks_batch`；若实现或验证证明必须新增算子，则使用 triton-ascend。
- 为 native backend 的配置路径、spec 构造、handler 构造补充或更新可本地验证的测试。

## 验收标准

- [x] `NPUOffloadingSpec` 使用当前 upstream 模块路径导入依赖。
- [x] native CLI 路径可以创建 Ascend NPU offloading spec，不要求用户继续传旧的 `num_cpu_blocks`。
- [x] handler 注册可以处理 `CanonicalKVCaches`。
- [x] transfer 逻辑可以处理 upstream `GPULoadStoreSpec.group_sizes` 和 `block_indices`。
- [x] 目标单测通过。
- [x] 任务结果中记录 `--kv-offloading-backend native` 的 NPU 验证命令和结果。

## 完成标准

- 已为变更行为新增或更新测试。
- 已运行目标 lint 或单测。
- 已记录 NPU 验证结果；没有 NPU 环境时记录阻塞原因。
- 用户可见配置发生变化时，同步更新文档。

## 研究记录

- [`research/upstream-native-kv-offload.md`](research/upstream-native-kv-offload.md) - upstream native CLI 和 KV offload API 变化。

## 技术方案

推荐 MVP：让 Ascend 的 `NPUOffloadingSpec` 兼容 upstream native API，同时对用户暴露的入口保持为 `--kv-offloading-backend native --kv-offloading-size <GiB>`。

NPU 验证选择 dense 模型 `Qwen/Qwen3-8B`。本任务不选择 MoE 模型作为 MVP 验证对象，避免 expert parallel、路由和 MoE 通信路径干扰 native KV offload 主链路判断。

后续完整语义验证选择 MoE 模型 `Qwen/Qwen3-30B-A3B`。该模型规模适中，能覆盖 MoE 场景，同时不会像超大 MoE 模型那样把验证成本推高。MoE 完整语义验证需要多卡，至少覆盖 TP；涉及 expert 路径时覆盖 EP。

算子策略：优先复用现有 `torch.ops._C_ascend.swap_blocks_batch`。只有当现有算子无法表达 upstream 新语义，或 NPU 验证证明存在功能缺口时，才新增算子；新增算子必须使用 triton-ascend。

## 决策记录

背景：upstream native KV offload 已通过 `--kv-offloading-size` 自动写入 `cpu_bytes_to_use`，旧的 `num_cpu_blocks` 路径与新 CLI 入口不一致。

决策：MVP 只支持新 native CLI 路径：`--kv-offloading-backend native --kv-offloading-size <GiB>`。

影响：实现和测试集中在新路径；旧显式 `num_cpu_blocks` 配置不作为本任务兼容目标。

背景：验证模型需要大小适中，同时尽量减少与 KV offload 无关的变量。

决策：使用 dense 模型 `Qwen/Qwen3-8B` 做 NPU 验证。

影响：MoE 模型不作为本任务验收条件。

背景：完整语义验证需要覆盖 MoE 模型，确认 native KV offload 在更接近生产复杂度的模型结构下仍可用。

决策：后续完整语义验证使用 `Qwen/Qwen3-30B-A3B`，并使用多卡 TP/EP。

影响：该验证作为后续扩展验收，不阻塞 MVP；MVP 仍使用单卡 dense 模型完成主链路验证。

背景：native KV offload 适配主要是接口和语义迁移，当前已有批量 block copy 算子。

决策：默认不写新算子；确需新算子时使用 triton-ascend。

影响：本任务先集中在复用现有 transfer 能力，避免扩大实现范围。

## 技术备注

- 可能涉及文件：`vllm_ascend/kv_offload/npu.py`、`vllm_ascend/kv_offload/cpu_npu.py`、KV offload 文档和相关测试。
- upstream vLLM 源码来自 `/Users/songdehao/sdh-lab/code/vllm`。
- 当前 vLLM Ascend 分支：`codex/kvcache-offload-native`。

## 待确认问题

- 暂无。

## 不在范围内

- 旧显式 `num_cpu_blocks` 配置路径兼容。
- MoE 模型 native KV offload 验证不阻塞 MVP，但后续完整语义验证指定为 `Qwen/Qwen3-30B-A3B` + 多卡 TP/EP。
- LMCache backend 变更。
- Mooncake、AscendStore 或 distributed KV pool connectors 的改造。
- 除 native backend 可用性和可验证性之外的大范围性能调优。
