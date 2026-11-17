# Aegis 交付状态

本分支合并交付第三/第四阶段，基于 PR #2 的 `0d6da6f`；此前没有独立 PR #3。
原 V1/V2 核心、既有测试及32条 `causal_traces_v1.jsonl` 保留。

- CLI：`wish`、`run`、`stats`、`export-traces`，以及 `aegis` / `python -m aegis_core` 双入口。
- `wish` 可反问并续编译同一工单；冻结 manifest 绑定账本和 Git HEAD。
- `run` 在独立 Worktree 双探针验收，记录每次尝试；处理作弊、越界、复读和负向失败。
- 模型调度显式 opt-in，Codex 生成补丁使用 read-only；不自动合并业务代码。
- 12种离线故障模板均验证正负补丁，共24次模板尝试。
- 520条合成介质样本按468/52切分；520条GRPO候选按260组配对，明确为模板指令扩展。
- `training/` 提供可选 QLoRA SFT 和在线物理奖励 GRPO 入口，validate-only 无GPU依赖。
- Windows PowerShell bootstrap 默认只安装核心/pytest并执行完整测试；训练依赖需显式启用。

本地完整验证83项通过。CI配置Linux/Windows × Python3.11/3.12；Windows3.11
通过真实bootstrap脚本运行全部测试。最终CI运行记录见本分支PR，而非把计划当实测。
CUDA训练依赖未在云端安装，未执行RTX5060训练或测量显存可行性。

入口文档：[LOCAL_HANDOFF.md](LOCAL_HANDOFF.md)、[docs/control_tower.md](docs/control_tower.md)。
