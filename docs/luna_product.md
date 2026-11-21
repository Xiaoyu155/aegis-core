# Luna 产品入口

Luna 是 Aegis 的产品总控层。用户用自然语言提出需求；Luna 负责环境检查、任务拆解、影响分析、日志、交付建议与总控台，Aegis 负责确定性探针、隔离 Worktree、变异检查、静态审查、Ed25519 和回放。原有 `aegis` 命令保持兼容。

## 命令

```bash
luna init --repo .
luna run "为订单数量加入边界校验，并为非法输入返回明确结果" --repo .
luna status --repo .
luna dashboard --repo . --port 8766
luna combat-check --repo .
luna review-fleet --repo .
```

普通运行需要有效的 `config/luna-models.json` 路由、模型 ID 与服务凭证；运行还要求工作树干净，以及可用的 Ed25519 密钥。缺少验收约束、模型网关或签名材料时，Luna 记录阻断原因并停止，不伪造绿色结果。通过后生成 `luna/delivery/<run_id>` 本地交付分支，不直接改写默认分支。

每次任务在 `artifacts/luna_logs/` 写入机器 JSONL、产品战报、签名清单、签名封套、获胜补丁和交付 ZIP。JSONL 每项包含 UTC 时间、命令、Diff SHA-256、探针耗时、门禁结论、回放句柄、前序哈希和当前事件哈希；`luna` 战报分为需求理解、拆解方案、拦截记录、真实测试证据、交付与合并建议五段。`luna status` 与 dashboard 展示健康度、待验收项、工单时间线、拦截记录及模型配额。Dashboard 仅监听 loopback，且只提供只读接口。

## 本机生产设置

运行 `luna init --repo .` 后，将生成的 `.env.luna.example` 复制为 `.env.luna` 并填写网关参数。进程环境优先于该文件。密钥默认保存在用户目录下的 `.aegis/keys`，也可用 `--keys-dir` 指定仓库外的目录。使用自定义目录时，为后续命令设置 `AEGIS_KEYS_DIR`。重复执行会验证并保留现有密钥。检查状态为 `BLOCKED` 时退出码为 2，输出中的 `remediation` 列出各阻断项的修复步骤。可用 `--json` 获取单个 JSON 对象。

Windows 可运行 `./scripts/bootstrap_local.ps1 -PythonVersion 3.12`。脚本会快进拉取当前跟踪分支，创建虚拟环境，安装依赖并运行测试，随后验证初始化结果并启动总控台。已有本地代码可加 `-NoPull`，仅需验证初始化可加 `-NoDashboard`。

`config/luna-models.json` 按角色定义有序的端点列表。默认 planner 和 reviewer 使用推理端点，coder 使用代码端点，三者均可回落到本地端点。端点额度按 UTC 固定窗口重置，共享账本记录在 `.aegis/luna-quota.sqlite`。每次请求先预留输入估算值和最大输出额度，再按服务返回的用量结算。服务未返回用量或请求超时时保留预留值。可在 `luna status --json` 查看剩余额度和重置时间。旧配置可通过 `--models config/models.yaml` 显式选用。

`luna review-fleet` 通过已认证的 GitHub CLI 读取 origin 的全部 Open PR，默认核对当前 PR HEAD 对应的四项 v16 门禁检查。可用 `--receipt-name` 指定其他项目的门禁检查名。草稿、冲突、缺失收据或非绿色检查会阻止生成合并模板。生成的模板带有 `--match-head-commit`，由审阅人手动执行。

## 离线演练

```bash
luna run "将库存数量限制在 0 到 10，非法输入返回空值" --repo . --demo
```

`--demo` 只针对 `examples/luna_combat_sandbox/inventory.py` 内嵌样例，不调用外部模型，也不代表真实商业项目已具备生产凭证。它让 Aegis 冻结真实红灯探针，先拦截一个只处理下界的错误补丁，再验证双边界修复，并生成一个小时有效、仅本次演练使用的临时 Ed25519 密钥签名。临时私钥在签发和验签后删除；演练分支保留在本地，交付包和日志位于 `artifacts/luna_logs/`。

`luna combat-check` 检查 Python/依赖与磁盘、Ed25519、GitHub 写权限、Worktree 隔离、最新默认分支 CI 和模型路由端点。只有这些实战前提都确认可用才输出 `READY`；检查不代替真实任务门禁。
