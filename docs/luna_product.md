# Luna 产品入口

Luna 是 Aegis 的产品总控层。用户用自然语言提出需求；Luna 负责环境检查、任务拆解、影响分析、日志、交付建议与总控台，Aegis 负责确定性探针、隔离 Worktree、变异检查、静态审查、Ed25519 和回放。原有 `aegis` 命令保持兼容。

## 命令

```bash
luna run "为订单数量加入边界校验，并为非法输入返回明确结果" --repo .
luna status --repo .
luna dashboard --repo . --port 8766
luna combat-check --repo .
```

普通运行需要有效的 `config/models.yaml` 路由、模型 ID 与服务凭证；运行还要求工作树干净，以及可用的 Ed25519 密钥。缺少验收约束、模型网关或签名材料时，Luna 记录阻断原因并停止，不伪造绿色结果。通过后生成 `luna/delivery/<run_id>` 本地交付分支，不直接改写默认分支。

每次任务在 `artifacts/luna_logs/` 写入机器 JSONL、产品战报、签名清单、签名封套、获胜补丁和交付 ZIP。JSONL 每项包含 UTC 时间、命令、Diff SHA-256、探针耗时、门禁结论、回放句柄、前序哈希和当前事件哈希；`luna` 战报分为需求理解、拆解方案、拦截记录、真实测试证据、交付与合并建议五段。`luna status` 与 dashboard 展示健康度、待验收项和 Quarantine 数量。Dashboard 仅监听 loopback，且只提供只读接口。

## 离线演练

```bash
luna run "将库存数量限制在 0 到 10，非法输入返回空值" --repo . --demo
```

`--demo` 只针对 `examples/luna_combat_sandbox/inventory.py` 内嵌样例，不调用外部模型，也不代表真实商业项目已具备生产凭证。它让 Aegis 冻结真实红灯探针，先拦截一个只处理下界的错误补丁，再验证双边界修复，并生成一个小时有效、仅本次演练使用的临时 Ed25519 密钥签名。临时私钥在签发和验签后删除；演练分支保留在本地，交付包和日志位于 `artifacts/luna_logs/`。

`luna combat-check` 检查 Python/依赖与磁盘、Ed25519、GitHub 写权限、Worktree 隔离、最新默认分支 CI 和模型路由端点。只有这些实战前提都确认可用才输出 `READY`；检查不代替真实任务门禁。
