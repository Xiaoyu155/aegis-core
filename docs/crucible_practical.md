# Crucible 实战配置与样例

本文是本阶段新增的可执行配置，不是未提供的《Aegis v2.2.0 白皮书》复原。
历史 Prompt、讨论中原始 Wish 和生产报错堆栈在当前记录中不可核实，故不冠以“历史真实样例”。

## 1. System Prompt、反问规则、完整 JSON

完整 [System Prompt](../examples/causal_medium_system_prompt.txt) 与代码中的
`aegis_core.mediator.SYSTEM_PROMPT` 一致。四组完整的
Wish → 结构化选择题 → 用户补充约束 → Contract Stub JSON 在
[mediator_cases.json](../examples/mediator_cases.json)，单元测试重新执行编译器逐组核对。

```python
from aegis_core.mediator import compile_request
result = compile_request(
    ticket_id="fix-cat", wish="猫咪遮挡顺序不对，修好它",
    repo_skeleton=["engine/iso_renderer.py", "engine/cat_kinematics.py"],
    proposal={"target_files": ["engine/iso_renderer.py"],
              "contract_stub": 'def depth_order(cats: list) -> list[str]:\n    ...'},
    candidates=[{"path": "engine/iso_renderer.py", "score": .82},
                {"path": "engine/cat_kinematics.py", "score": .73}],
)
assert result["status"] == "MISSING_CONSTRAINT"
```

确定性宿主规则优先于模型：

| 代码 | 条件 | 推荐选项 |
|---|---|---|
| A_CREDENTIALS | 生产/实盘 API，非显式 dry_run，缺少 `credential_available=true` 与 `env:`/`vault:` 引用 | 离线 dry_run |
| B_FINANCE | 生产资金动作，未同时给出 `0<max_drawdown<=1`、`max_notional>0`、one_way/hedge | 停止资金动作 |
| B_MIGRATION | 破坏性迁移未给出存在于骨架中的路径白名单和非空回滚边界 | 保持挂起 |
| C_AMBIGUITY | 排名前两候选分差 `<0.15` 且未显式选择 | 排名第一，但必须由人提交选择 |

推荐值不是授权，不自动提交。反问只询问可用性和引用，禁止输入原始密钥。
Wish 关键词仅提供保守提示，生产调用方还应从可信任务元数据设置
`production_api`、`financial_effect`、`destructive_migration`；不把自然语言分类当权限系统。
所有路径必须是骨架中存在的规范相对路径，唯一目标不能是 tests/tools/check/config。
存根必须是 Python 函数/类接口、类型注解、Docstring 与 `...`，不含实现或可执行表达式。
在传入 `ledger` 和 `lease_token` 时，状态、编译结果和轨迹同事务落盘。
V1 的 MissingConstraintQuestion 保持兼容；介质多问项协议使用独立的
`mediator.validate_compilation`，不把不同结构强塞进 V1 schema。

## 2. 真实仓库资料与离线业务契约

已读取的资料（2026-11-17，当前默认分支）：

- [crypto-quant-mvp README](https://github.com/Xiaoyu155/crypto-quant-mvp/blob/experiment-harness/README.md)：
  Binance 公共 K 线、DCA/RSI/Grid、回测/模拟盘/强制风控；实盘默认禁用。
  主力/稀缺/本地三层 provider，稀缺调用预算按 UTC 持久化。本地示例为
  Ollama `quant-qwen3:4b`、4096 上下文；README 记录的显存与耗时是该项目历史测量，
  不是本次对 RTX 5060 的实测。本次不复用或修改该项目业务实现。
- [palm-cat-cafe README](https://github.com/Xiaoyu155/palm-cat-cafe/blob/main/README.md)：
  当前入口根目录 `preview.html`；`miniprogram/`、`game/catcafe/systems.js`、
  `game/catcafe/src/`、`宠物小程序/`；Node.js 20+，核心预览/测试不需 npm 依赖。
  美术生成与旧版 Python/浏览器验收存在 Pillow/NumPy/Playwright 等可选依赖，
  本控制面不安装它们，也不把 Python 仿真说成真实 JS Canvas 引擎。

用户指定、由本阶段 original Python fixtures 实现的契约如下；不是从上述 README
推断出的生产参数：

| 靶场 | 分层与接口 | 固定不变量 | 故障与修复 |
|---|---|---|---|
| live_trading_v1 | signal_engine → execution_router；intraday_guardian → execution_router | 信号周期 28800s；守护间隔 60s；回撤 `(peak-equity)/peak >= .05` | One-Way 传 long 导致 51000；修为 net；守护触发时 reduceOnly=True |
| ws-cat-engine | cat_kinematics.Cat → iso_renderer.depth_order / render_center_roi | 以 foot_anchor_y 升序绘制，后绘制的近处物体覆盖远处；中心 ROI 总体方差 >=18.5 | 误用 body_anchor_y；平坦像素使画布不变量失败 |

深度层级：场景远景/地面可先绘制，动态猫咪层按脚底锚点排序，UI 最后绘制；
当前仿真只断言动态层顺序。中心 ROI 是 64 个灰度值的 `statistics.pvariance`，
不是截图识别器；18.5 是用户给定门槛，不是跨设备校准结果。

实际仿真生成的失败堆栈见 [arena_failure_stacks.txt](../examples/arena_failure_stacks.txt)，
示意调用链为：

```text
tests/test_invariant.py -> signal_engine.execute_signal
  -> execution_router.submit_order
ExchangeAPIError: code=51000: posSide invalid in One-Way mode

intraday_guardian.guard(1000, 950, 60)
AssertionError: reduceOnly must be True

iso_renderer.depth_order([Cat('back',10,99), Cat('front',20,1)])
AssertionError: ['front', 'back'] != ['back', 'front']

pvariance(iso_renderer.render_center_roi()) >= 18.5
AssertionError: 0 >= 18.5
```

这些是离线模拟故障，不是曾发生在真实业务仓库的历史栈。

## 3. models.yaml 与 Sub2API 适配

[config/models.yaml](../config/models.yaml) 使用 **JSON 语法的 YAML 1.2 子集**，
以标准库 `json` 加载，无 PyYAML。分层、并发与超时是本阶段默认配置，非历史配置复原。

| 档位 / 节点 | timeout_sec | 并发与权重 |
|---|---:|---|
| L1 local_3b / local_9b | 45 / 90 | 各1；共享 GPU 总额1；权重3:1 |
| L2 qwen_28b / deepseek / glm | 120 / 120 / 100 | 各1；Sub2API 总额3；权重3:2:1 |
| L3 codex_architect / claude_architect / gemini_4 / codex_direct | 180 / 180 / 150 / 180 | 各1；两条 Codex 路由共享总额1 |

“Qwen 28B”“Gemini 4”和 9B 是需求中的槽位标签，不保证服务商存在对应型号。
除本地 `qwen2.5-coder:3b` 示例外，实际 model ID 必须用环境变量配置。
9B 在 8GB 上是否可用取决于量化、KV cache 与上下文长度；配置槽位不表示已实测可运行。

```bash
export SUB2API_BASE_URL='https://your-sub2api-host/v1'
export SUB2API_QWEN28B_MODEL='实际可用的模型ID'
export SUB2API_DEEPSEEK_MODEL='实际可用的模型ID'
export SUB2API_GLM_MODEL='实际可用的模型ID'
# SUB2API_API_KEY 由宿主 Secrets 注入，不写入文件或命令历史。
```

OpenAI-compatible Sub2API 请求：

```http
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer <宿主注入>

{"model":"实际模型ID","messages":[{"role":"user","content":"..."}],"temperature":0,"stream":false}
```

Claude 原生 Messages 协议使用 `ANTHROPIC_BASE_URL`（以 `/v1` 结尾）、
`x-api-key`、`anthropic-version: 2023-06-01` 与 `/messages`；
Gemini 槽位使用显式配置的 OpenAI-compatible 网关，不假设原生协议兼容；
Codex 调用本机 `codex exec --json --model MODEL -`，必须已安装并完成宿主认证。

```python
import time
from pathlib import Path
from aegis_core.router import Router, load_models
router = Router(load_models(Path('config/models.yaml')))
result = router.call('L1_LOCAL_CAUSAL', [{'role': 'user', 'content': '...'}],
                     deadline=time.monotonic() + 180)
```

档位链：`L1_LOCAL_CAUSAL -> L2_WORKER_POOL -> L3_ARCHITECT_AND_DIRECT -> blocked_needs_human`。
先在同档按权重分配空闲槽位，额度满/不可用再向后走；失败槽位在当前 Router
实例中隔离。截止时间使用同进程 monotonic 时基，预留 `min_verify_reserve_sec=30`，
每次调用前计算 `timeout=min(节点阈值, deadline-now-reserve)`；响应超预算不得放行。
跨进程应传 remaining duration 后重建本机 deadline，不传跨机器 monotonic 数字。
HTTP/socket timeout 不是对恶意慢流的 OS 级终止器，V2 不声称提供进程级资源隔离。

并发锁和熔断状态是当前进程内的线程安全配额；多进程部署应使用单一 Router
调度服务，不能把它误当跨机器分布式锁。被拒补丁登记后，连续两次完全相同的
UTF-8 SHA-256 输出会剔除该槽位并返回下一档；末档进入 blocked_needs_human。
中间出现不同输出会清零连续计数。独立使用 `record_output` 时，升级决定由调用方执行；`Router.call` 则自动校验
输出（可识别 JSON 中的 patch 字段）并执行配置链升级，始终沿用原 deadline。

## 导出与训练读取

```bash
python -m aegis_core.simulator --output datasets/causal_traces_v1.jsonl --seed 0
pytest -v
```

四种故障 × 正负补丁各一组，共 8 个修复样本、32 条全生命周期轨迹。
`seed` 是保留的来源标签，本版本不引入随机扰动。日志保留真实时间/墙钟耗时；
离线 Reward 使用显式标记的合成 `reward_elapsed_sec=1.0`，保证标签可复算，
不把机器快慢误当策略优劣。线上可用实测耗时替换该输入。

```python
import json
rows = [json.loads(line) for line in open('datasets/causal_traces_v1.jsonl', encoding='utf-8')]
repair = [row for row in rows if row['trainable']]
sft = [row['messages'] for row in repair if row['reward'] > 0]
groups = {name: [row for row in repair if row['group_id'] == name]
          for name in {row['group_id'] for row in repair}}
```

修复样本的 prompt 含冻结源码、验收断言和 S0；completion 是真实补丁动作，
同一 group 的正负样本共享输入，保留 R 和 S1。字段有 `messages`、`prompt`、
`completion`、`group_id`、`reward`、`state_s0/action_a0/state_s1`。
这是框架中立的 SFT/偏好/奖励数据，不冒称某个训练器的唯一“标准格式”。
GRPO 仍需训练器在线采样并调用奖励函数；静态正负对不能替代 on-policy 采样。
QLoRA/GRPO 的 PyTorch/PEFT/TRL/CUDA 训练栈不属于零依赖控制面，未执行训练，
也未承诺特定 8GB 训练配置一定可行。

## 门禁与稠密奖励的精确定义

`Sandbox.repair` 的命令和 `source_dirs` 必须来自宿主冻结契约，不能采用补丁自报的
验收命令。先要求创建时基线 HEAD 未变化且没有待提交修改，再运行 Fail-Before；
只有退出码1且文件树未变才认定复现。补丁后统计 `git diff BASE --numstat`（包括
已暂存和已提交差异）与未跟踪文件；核心源目录没有变更行，或修改了测试/配置/
排除规则，均视为作弊。Pass-After 必须退出0且文件树未变。失败结果不能通过
再次调用 V1 单探针 `exam` 绕过。

```text
R_gate = +1.0  if baseline_failed and after_passed
         -0.5 if baseline_failed and not after_passed
         -1.0 otherwise
R_scope = |target_files ∩ changed_files| / |target_files ∪ changed_files|
          # 两集合均为空时取0
R_concise = max(0, 1 - diff_lines / concise_budget_lines)  # 默认80
P_latency = min(1, elapsed_sec / latency_budget_sec)
P_cheat = 1 if cheat_detected or repeated_patch else 0   # 两者同时出现不双扣
R = clip(R_gate + .3*R_scope + .2*R_concise - .2*P_latency - P_cheat, -2, 1.5)
```

这里的延迟项是预算占用比例，在达到/超过预算时扣满；它不是额外推断的模型奖励。
修复类门禁直接返回的 V1 兼容 reward 是门禁基础分，训练导出使用
`compute_step_reward` 的完整稠密分数。
