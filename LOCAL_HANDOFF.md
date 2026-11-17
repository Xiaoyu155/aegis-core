# Windows 11 本地接管：三条命令

先在仓库目录打开 PowerShell，确保当前分支无未提交修改。以下第一条命令会把
交付分支合并进当前本地分支；建议先切到个人集成分支。不会合并 GitHub 的 PR。

```powershell
# 1. 拉取并合并整个堆叠分支（含 V1、V2、本次数据与训练脚手架）
git pull --no-rebase origin feat/control-tower-v4

# 2. 建立 .venv、安装核心和 pytest、执行全部测试（包括离线 CLI 端到端闭环）
& .\scripts\bootstrap_local.ps1

# 3. 显式安装可选 CUDA/训练栈，并启动一个 QLoRA SFT 步骤；改 -Train grpo 即在线 GRPO
& .\scripts\bootstrap_local.ps1 -InstallTraining -Train sft -MaxSteps 1
```

需要：Git for Windows、64 位 Python 3.11+、可用的 NVIDIA 驱动、模型下载网络与
足够磁盘空间。优先使用 `py -3.11`，已有 .venv 会复用。脚本不修改系统执行策略；
若本机策略阻止脚本，按组织策略允许当前脚本后执行。所有命令失败立即停止，
不会把依赖安装失败或测试失败报告成初始化成功。

第二条仅安装标准库控制面与 pytest，并将 .venv\Scripts 加入当前 PowerShell 的 PATH（不修改系统 PATH）；第三条才安装 torch/transformers/PEFT/TRL/
bitsandbytes。RTX 50 系列从官方 cu128 索引安装 CUDA 12.8 wheel，预检 CUDA
可用性；不静默降级成全精度或 CPU。模型权重首次下载需要额外网络。
Windows 的 bitsandbytes 后端可用性受版本/驱动影响；失败时保留错误，推荐 WSL2
使用相同训练入口。未在实际 RTX 5060 上测量显存，不承诺“一定装得下”。

SFT 默认：Qwen2.5-Coder-3B-Instruct、4-bit NF4、LoRA r8、batch1、累积8、
序列1024、梯度检查点。GRPO：LoRA r4、2候选、batch2、累积4、prompt768/
completion256、无 vLLM；在线奖励在离线 Git 靶场重新执行模型补丁，不读取模型
自报分数。若提示超出序列预算会明确报错，需增大长度或缩短样本；显存不足应停止
并调整，不能把失败当成完成。本次云端只验证数据、控制面和奖励，不执行 GPU 训练。

先检查数据格式无需任何训练依赖：

```powershell
.\.venv\Scripts\python.exe -m training.train_sft --validate-only
.\.venv\Scripts\python.exe -m training.train_grpo --validate-only
```

## 从 Wish 到本地修复验收

```powershell
# 可先用 --target 指定源码，或省略让编译器按文件名/模块名选择并反问歧义。
aegis wish "修复 signal_engine.py 单向持仓参数" --repo C:\repos\target --target signal_engine.py --out C:\tasks\contract.json --fail-before '{python} -m pytest -q' --pass-after '{python} -m pytest -q' --patch C:\tasks\fix.diff
# 若返回 MISSING_CONSTRAINT，用同一个 --ticket-id 和 --constraints 回答后重新编译。
aegis run --contract C:\tasks\contract.json
aegis stats --repo C:\repos\target
aegis export-traces --repo C:\repos\target --out C:\tasks\latest.jsonl
```

使用 argv JSON 数组避免跨平台命令引号差异；`{python}` 替换为当前虚拟环境
解释器。离线 `--patch` 可重复以提供多个候选。模型调度需在 wish 时显式
`--models config/models.yaml --allow-model`，默认不访问网络或模型CLI。
契约绑定 Git HEAD 和现有源码路径；通过后补丁保存在 WAL 轨迹中，工作树清理，
不会直接修改或合并业务仓库。默认账本位于 Git common-dir 的 `aegis/ledger.sqlite`，
可用所有命令的 `--db` 选项指定显式路径。状态/trace输出为JSON；失败退出非零。

## 数据来源与范围

原 V2 的32条数据原样保留。新增12种可执行故障模板；每个模板正负补丁各物理
验收一次，再生成520条介质SFT（468训练/52评估）与520候选/260组GRPO配对。
GRPO文件是合成指令变体，不是520次独立物理执行，也不是生产采集数据。
SFT由确定性编译器逐条产生并校验；划分无完全相同的请求，但不是跨故障家族的
泛化评测。`datasets/manifest_v3.json` 保存计数、模板验证状态和文件哈希。

重新构建（不覆盖 V2 causal_traces_v1.jsonl）：

```powershell
python -m aegis_core.dataset_builder --output-dir datasets
```
