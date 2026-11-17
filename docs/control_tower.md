# 统一 CLI 控制塔

安装核心：`python -m pip install -e '.[test]'`。入口 `aegis` 与
`python -m aegis_core` 等价，不导入训练框架。

```bash
# 默认扫描已跟踪文件；多个等分候选会返回结构化反问，exit code 2。
aegis wish "修复某个模块" --repo /path/to/repo --ticket-id task1 --out /tmp/contract.json

# 同一 Wish、同一 ticket 续编译；也可传 --constraints constraints.json。
aegis wish "修复某个模块" --repo /path/to/repo --ticket-id task1 \
  --selected-module engine/iso_renderer.py --out /tmp/contract.json \
  --fail-before '["{python}","-m","pytest","-q"]' \
  --pass-after '["{python}","-m","pytest","-q"]' --patch /tmp/fix.diff

aegis run --contract /tmp/contract.json
aegis stats --repo /path/to/repo
aegis export-traces --repo /path/to/repo --out /tmp/latest.jsonl
```

`--target` 可直接指明已存在源码；`--proposal` 可传精确业务接口存根，避免默认
通用 `apply_change` 存根。`--source-dir` 限定源码目录，默认仓库根目录。
`--patch` 可重复，按顺序尝试。命令 argv 推荐 JSON 数组，`{python}` 在执行时
解析为当前虚拟环境 Python，避免 Windows 路径转义与不同 Python 环境混用。
stdout 为一个 JSON 结果；成功 exit0，缺失约束 exit2，门禁失败/参数错误 exit1。

契约绑定 repo_root、HEAD、编译结果、目标、探针命令和参数摘要；摘要同时写入账本。
重新计算篡改文件的 hash 不能绕过账本绑定；HEAD 或已跟踪工作区改变均拒绝执行。
默认 SQLite 位于 Git common-dir 的 `aegis/ledger.sqlite`，不污染业务源码；
`--db` 可显式指定同一个账本。`run` 不能把外部契约导入另一账本后自动获得授权。

每次尝试创建新 Worktree，运行失败基线再应用补丁，保护 tests/config，拒绝超出
契约目标的修改。工作树结束后清理，成功补丁保存在轨迹的 action_a0.patch 中。
本命令验收并录制，不自动合并上游业务分支。需要接管补丁时先导出轨迹审阅。
任何命令执行均继承宿主权限；测试程序本身不是 OS 安全沙盒。

默认本地补丁模式不调用模型；要显式启用：

```bash
aegis wish "修复目标" --repo /path/to/repo --target app.py \
  --allow-model --models /path/to/aegis-core/config/models.yaml \
  --fail-before '["{python}","-m","pytest","-q"]' \
  --pass-after '["{python}","-m","pytest","-q"]' --out /tmp/model-contract.json
```

Router 共享 deadline、验收预留和配额。Codex CLI 生成补丁时为 read-only 沙盒；
所有实际应用只在独立 Worktree 内发生。API 模型接收明确提供的目标源文件和
冻结测试上下文，配置凭据仅从宿主环境读取。连续两次重发被拒补丁触发隔离；
离线模式没有下一个模型档位，工单以 failed、reason=blocked_needs_human 终止。

`stats` 的 success_rate 分母为 succeeded+failed 终态工单；mean_reward 只统计
repair/repair_attempt 轨迹，忽略编译等零奖励管理动作；cheat_interceptions 和
replay_interceptions 为被拦截的尝试数。导出保持 V1 state_s0/action_a0/state_s1/reward。

## 合成数据与训练边界

原始 `causal_traces_v1.jsonl` 原样保留。新增：

- `mediator_sft_train.jsonl` 468 条，`mediator_sft_eval.jsonl` 52 条；由编译器逐条验证答案。
- `grpo_repair_pairs.jsonl` 520 条，260 个正负对；覆盖12种真实执行过的离线故障模板。
- `manifest_v3.json` 记录文件 SHA-256、模板执行数和来源。

```bash
python -m aegis_core.dataset_builder --output-dir datasets
python -m training.train_sft --validate-only
python -m training.train_grpo --validate-only
```

GRPO 的520条是24次独立模板执行结果的指令扩展，不是520次独立执行或实盘记录；
重复上下文不应被当成泛化评测。SFT 520条为合成参数和问题，不是历史用户对话。
在线 GRPO 奖励会对新生成补丁创建全新靶场、重新执行双探针，不读取样本/模型的
自报分数。训练依赖只在 training/requirements-train.txt；实际 GPU 训练未在云端执行。
Windows三条启动命令及硬件边界见 [LOCAL_HANDOFF.md](../LOCAL_HANDOFF.md)。
