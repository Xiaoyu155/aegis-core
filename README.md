# aegis-core V1.2

Python 3.11+ 的确定性控制面与因果轨迹骨架。运行时仅依赖标准库；Git
Worktree 需要系统安装 Git，验收测试需要 pytest。不调用模型、不导入目标业务模块。

## 运行

```bash
python -m venv .venv
# Linux/macOS
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
pytest -v
```

本仓库从空目录独立实现，未复制或修改其他仓库业务代码。首个 Git 提交冻结
[接口存根](contracts/v1.pyi) 和 [行为契约](contracts/README.md)，随后才实现代码。

## 外部仓库与浏览器门禁

- 模型加载顺序为显式 AEGIS_WORLD_MODEL_DIR、目标仓的 artifacts/causal_world_model、安装包内的训练权重。包内模型跨仓默认只记录预测。显式与仓内模型沿用原有路由策略。学到的新权重与报告保存到目标仓的模型目录或 learn --artifact-dir 指定的位置。包内权重不参与写入。
- plan、run、learn 的 JSON 都包含 telemetry。plan 按修复节点计数，run 按补丁候选计数，learn 按账本中的修复尝试计数。below_threshold_count 仅在 learn 中有筛选意义，恰好等于阈值的样本也计入该项。learn 会将同一工单内的真实失败和通过双探针的不同补丁关联，以 verified_recovery 存入 ExperienceBank。重复执行 learn 不会重复写入该案例。
- attach 会识别包含可执行脚本的 HTML 入口，并保留 .aegis/repo_profile.json 中的 tool_source_globs。默认涵盖根目录和子项目 tools/ 下的 Python、JavaScript/TypeScript、HTML 及 CSS，测试和配置仍受保护。用 aegis attach --tool-source-glob 重设工具范围。
- 浏览器修复工单在 DAG 中声明 "probe_kind": "browser_visual"，也可用 aegis wish --no-auto-probe --probe-kind browser_visual 冻结契约。至少要锁定一个探针脚本。直接传入的脚本参数会自动锁定；间接入口或额外夹具可用 probe_scripts（CLI 为重复 --probe-script）显式指定。probe_script_sha256 对仓内文本脚本统一使用 LF，其余路径按原始字节计算。门禁在每次执行前后核验，红绿两次探针在同一个 worktree 内还须具有相同的 probe_runtime_sha256 原始字节摘要。验证节点也支持相同声明。
- 浏览器探针自行运行项目的浏览器脚本，断言布局、像素或帧时；Aegis 记录退出和证据。FailureDiagnosis.failure_code 区分断言失败、模块未实现、浏览器等待超时和环境不可用。HTML 内联 JS 与 CSS 引用可由 L3 原生审查，静态规则只覆盖可识别的源码模式。

## v1.2 高信噪比扫描与交付门禁

- `inspect_source` 为每条发现增加 `confidence`、`confidence_score`、`scope`、`asset_type` 和 `suppression_reason`。`scan_changed_files` 把被抑制的依赖、测试夹具、静态原型、离线脚本、生命周期清理与框架默认超时发现单独列入 `suppressed_findings`，不让它们阻塞门禁。
- `aegis ship-fleet --owner Xiaoyu155` 默认只生成完整 PR 计划；它读取 `feat/aegis-*` 的提交、变更、冲突预检和 GitHub 检查状态，并对已有 PR 做幂等复用。只有显式 `--execute` 且检查全绿时才创建 PR。`nova-agent`、`Checkpoint`、`edu-typing` 默认排除。
- `aegis world-model-bench` 用锦标赛正负样本和已验证因果轨迹做分组留出校准。只有同一阈值同时达到正确补丁误拦率低于 1%、风险样本召回率至少 85%，并提供至少 16 字节 `AEGIS_CALIBRATION_SIGNING_KEY` 时，报告才签发 `observe -> enforce` 许可。

## 第十二阶段工厂

`aegis wish` 可以从带有可执行规则的单句验收需求生成并冻结 pytest 或 Node.js 22+ 探针。例如 `src/limits.py::clamp clamps integers to [0, 10] and returns null for invalid inputs.`。事件集合规则可用 `src/events.py::count counts distinct string events and returns -1 for non-string events.`。规格通过 `ExpandedSpec` 记录正常、边界和回退行为。生成的探针必须在独立基线上出现真实断言失败。在候选补丁通过静态探针后，门禁按固定种子生成不同输入，失败记为 `MUTATION_PROBE_FAILED`。自定义验收语义可继续使用显式冻结探针。

`aegis wish` 和 `aegis plan` 接受 `--max-token-budget`、`--max-cost-usd` 与 `--max-wall-clock-sec`。DAG 可在各节点声明同名下划线字段。父预算汇总子工单开销，子预算可单独熔断。账本记录模型请求及重试、探针进程与结果奖励。API 开销按固定混合费率估算，优先使用供应商返回的 token 数。没有返回用量时按 UTF-8 字节保守估计。离线补丁的 API token 和费用为零，仍消耗时钟预算。

`aegis batch --config factory.json --out report.json` 读取 `aegis.factory.v1` 配置。每个 `jobs` 条目需要 `id`、整数 `priority`、`aegis attach` 生成的 `profile` 路径，以及一个 `contract` 或 `dag` 路径。相对路径从配置文件目录解析。大优先级先执行，同优先级按配置顺序。单条失败后继续执行后续工作，最终退出码与汇总报告反映所有工作结果。

```bash
python scripts/run_factory_v12_drill.py
```

## 四个原语

- `ledger.Ledger`：SQLite WAL、NORMAL 同步、15 秒锁等待；每个操作独立连接。
  工单创建记录 step 0；状态流转与轨迹写入在 `BEGIN IMMEDIATE` 中同时提交。
  抢占返回不透明租约令牌，过期后重新抢占会更换令牌，旧执行者不能提交。
  允许 `queued -> running -> succeeded/failed`，终态不可再抢占。
  `state_s0` 必须与上一条实际状态一致，奖励必须有限、token 成本必须是非负整数。
  轨迹禁止 UPDATE/DELETE，JSONL 采用一致查询快照并通过临时文件替换目标文件。
- `slicer.extract_causal_slice`：解析目标目录所有 `.py`，保留目标完整源码与装饰器；
  提取直接被调用函数、类及方法的签名、类型注解和 Docstring；汇总上游文件、
  调用作用域、行号和调用表达式。支持导入别名、相对导入、模块别名、self/cls
  方法及 `Class().method()`。无法静态解析的动态调用列在 `unresolved_calls`，
  局部遮蔽名称不擅自连边。`compression_ratio` 为相对目录源码 UTF-8 字节数
  节省比例（目标源码加去重下游存根，不含结果元数据）。
- `sandbox.Sandbox`：创建 `.worktrees/<ticket_id>` 和 `aegis/<ticket_id>` 分支；
  `git apply --check` 后原子应用补丁。`exam` 捕获输出、退出码和前后文件树哈希，
  只有退出码为 0 且文件树不变才奖励 +1，其余奖励 -1。
  `merge` 要求本实例持有的当前文件树通过结果、主工作区干净且基线未前进，
  然后提交并 fast-forward 合并；`cleanup` 删除该实例创建的工作树和分支。
- `interface_schema`：三个严格 JSON 契约：`MissingConstraintQuestion`、
  `CausalRolloutPlan`、`FaultAttribution`；导出 `JSON_SCHEMA` 并通过
  `validate_message` 验证未知字段、类型、非有限数字、连续推演与 DAG 无环约束。

## 最小账本闭环

```python
from pathlib import Path
from aegis_core.ledger import Ledger
from aegis_core.sandbox import Sandbox

ledger = Ledger("aegis.sqlite")
ledger.create_ticket("task-1", "实现目标函数", {"goal": "pass tests"},
                     "def target(x: int) -> int: ...")
token = ledger.acquire_lease("task-1", "worker-1", ttl=300)
if token is None:
    raise RuntimeError("lease unavailable")
ledger.transition("task-1", token, "queued", "running",
    state_s0={"status": "queued"}, action_a0={"op": "start"},
    actual_state_s1={"status": "running"}, reward=0)
sandbox = Sandbox(Path("/path/to/repo"))
worktree = sandbox.create("task-1")
# sandbox.apply_patch("task-1", validated_unified_diff)
gate = sandbox.exam("task-1")
status = "succeeded" if gate["passed"] else "failed"
ledger.transition("task-1", token, "running", status,
    state_s0={"status": "running"}, action_a0={"op": "exam_gate"},
    predicted_next_state={"status": "succeeded"},
    actual_state_s1={"status": status, "gate": gate}, reward=gate["reward"])
ledger.export_jsonl(Path("training.jsonl"))
if gate["passed"]:
    sandbox.merge("task-1", "Implement target")
sandbox.cleanup("task-1")
```

JSONL 同时包含数据库原始 `actual_state_s1` 与训练字段别名 `state_s1`，
每行具备 `(state_s0, action_a0, state_s1, reward)`、预测状态、成本和时间戳。
租约抢占是协调元数据；执行器必须在 TTL 内完成，否则需要重新抢占后提交。
失败重试使用新工单，以保留每次独立执行的终态和因果反馈。

## 边界

Worktree 提供 Git 文件隔离，执行命令继承宿主权限；不提供不可信代码的 OS 隔离。
单个工作树应由一个控制器独占，外部同时修改文件不在原子性保证范围内。
门禁哈希忽略 `.git`、`.worktrees`、虚拟环境和 Python/pytest 缓存，涵盖其他
文件内容、路径、目录、可执行位与符号链接目标。验收使用独立 Python 缓存路径，
防止同秒同大小补丁复用旧字节码。测试超时返回 124；子进程生成的后台进程
不在 V1 的进程隔离范围内。合并提交失败时保留工作树，便于修复或显式清理。

AST 分析范围是目标同目录；不求解反射、动态绑定、跨目录包图或全程序数据流。
存根保持静态语法接口，装饰器与默认值不执行。SQLite WAL 适用于本地文件系统；
NORMAL 模式不承诺掉电时最近提交绝不丢失。零运行时第三方依赖不意味着系统
预装 Python、Git 或 pytest；移动端需要提供这些工具的 Python/终端环境。

测试覆盖多进程并发写、竞争租约与过期令牌、事务回滚、JSONL、跨文件切片、
压缩率、遮蔽名称、补丁原子失败、正负奖励、缓存失效、基线前进及门禁合并闭环。
CI 配置覆盖 Linux/Windows 与 Python 3.11/3.12；本地云端验证见 PR 说明。

## Crucible 第二阶段

第二阶段新增确定性介质编译、双探针与防作弊门禁、模型配额路由/复读熔断、
稠密奖励以及离线业务靶场。实战 Prompt、四组完整 JSON、资料核实边界与协议参数见
[Crucible 实战说明](docs/crucible_practical.md)。

```bash
python -m aegis_core.simulator --output datasets/causal_traces_v1.jsonl
pytest -v
```

`config/models.yaml` 是 JSON 语法的 YAML 1.2 子集，核心代码不依赖 PyYAML。
默认验收与仿真不连接任何交易所或外部模型。模型槽位需配置实际端点、型号与
宿主凭据引用后才可调用。数据集用于训练数据交换，训练框架与 GPU 调优不在本仓库执行。

## 统一 CLI 与本地接管

```bash
aegis --help
python -m aegis_core --help
```

完整流程、退出码、账本位置及数据来源见 [控制塔说明](docs/control_tower.md)；
Windows 环境初始化与 QLoRA/GRPO 启动见 [三条接管命令](LOCAL_HANDOFF.md)。
