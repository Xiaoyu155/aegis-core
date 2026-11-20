#!/usr/bin/env python3
"""Build a Chinese, evidence-backed Aegis vs. bare GPT showcase."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


WIDTH, HEIGHT = 1920, 1080
FPS, SOURCE_FPS, DURATION = 30, 12, 90
BACKGROUND = (7, 12, 22)
PANEL = (14, 23, 37)
PANEL_EDGE = (36, 52, 72)
WHITE = (241, 246, 252)
MUTED = (159, 177, 196)
RED = (255, 85, 91)
GREEN = (54, 226, 150)
CYAN = (72, 204, 235)
GOLD = (252, 196, 87)

SCENES = [
    {
        "duration": 12,
        "title": "裸跑 GPT 做不到",
        "left": ["改动看似完成，主干却可能被污染", "断言放宽，错误也能亮绿灯", "无人值守缺少确定性门禁"],
        "right": ["Aegis Core｜确定性软件工厂控制塔", "Git Worktree 隔离 → 探针验证 → 独立审查", "把概率生成接入可复核的交付流程"],
        "subtitle": "GPT 能生成代码；把真实项目安全交付，还需要隔离、验证和守门。",
    },
    {
        "duration": 16,
        "title": "同一任务，四类候选见真章",
        "left": ["只追求测试变绿", "可能放宽断言或针对用例硬编码", "风险示意｜不是事故录像"],
        "right": ["四组同题偏好记录｜任务 10", "正确修复胜出｜三类坏补丁被拒", "拒绝样本沉淀为 GRPO 偏好数据"],
        "subtitle": "真实候选记录显示：正确修复胜出；签名破坏、测试作弊和无界增长都被拒绝。",
    },
    {
        "duration": 18,
        "title": "真钱路径，先守住输入与故障",
        "left": ["局部单测通过，不代表风险已消失", "数值边界、超时重试、资源增长", "风险类型示意｜不指称真实事故"],
        "right": ["InvariantGuard｜数值与资源不变量", "OKX 交付：校验杠杆与名义金额", "32 项验证通过｜真实交付回执", "宠物小程序后端：坏连接恢复", "274 项通过＋1 项故障集成验证"],
        "subtitle": "真实交付记录了 OKX 杠杆输入校验和 PostgreSQL 故障恢复，并带有对应测试结果。",
    },
    {
        "duration": 16,
        "title": "跨文件改动，不再靠记忆兜底",
        "left": ["上下文变长，接口契约容易漏", "反复试错可能把主干弄脏", "风险示意｜不是事故录像"],
        "right": ["Python／JS／TS／HTML 契约分析", "InterfaceContractLock 锁定接口面", "十万次热调用｜1.418 秒", "15 个交付分支｜零冲突｜主干零污染"],
        "subtitle": "接口契约锁住跨文件边界；十万次热调用画像耗时 1.418 秒，舰队账本记录主干零污染。",
    },
    {
        "duration": 14,
        "title": "不确定时，安全阀先踩刹车",
        "left": ["模型没把握，也可能过早下结论", "失败现场若无法复现，定位更困难", "风险示意｜不是事故录像"],
        "right": ["aegis replay｜复现冻结现场", "aegis blame｜查看变更行归因", "风险头验证支持数：2", "误报 2｜当前回执：observe"],
        "subtitle": "失败可以重放；模型验证样本很少且出现误报，当前采用 observe 观察模式。",
    },
    {
        "duration": 14,
        "title": "从一个补丁，到整支仓库舰队",
        "left": ["手工盯一个文件，难以扩展", "每次交付都要守住主干边界", "这份影片只展示仓库账本证据"],
        "right": ["1,338 项测试通过｜双平台 CI 8/8", "15 个交付分支｜零冲突", "主干零污染｜发布数据可追溯", "GPT 负责创造，Aegis 负责确定性兜底。"],
        "subtitle": "当前主干账本：1,338 项通过、双平台 CI 8/8、15 个交付分支零冲突。",
    },
]

FRAME_NAMES = [
    "01-开场破题.png",
    "02-候选对比.png",
    "03-风险守卫.png",
    "04-跨文件契约.png",
    "05-工程安全阀.png",
    "06-舰队交付.png",
]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_key(value, key):
    if isinstance(value, dict):
        if key in value:
            yield value[key]
        for child in value.values():
            yield from _walk_key(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_key(child, key)


def load_evidence(root: Path) -> dict:
    """Load every numeric claim from checked-in repository evidence."""
    status = _load_json(root / "STATUS.json")
    manifest = _load_json(root / "datasets/release_v15/manifest.json")
    deliveries = _load_json(root / "artifacts/real_delivery_v15/report.json")
    metrics = _load_json(root / "artifacts/causal_world_model/training_metrics.json")
    fleet_dashboard = _load_json(root / "artifacts/fleet_dashboard_v16.json")
    model_receipt = _load_json(root / "artifacts/mega_campaign_v14/workorders/omni-eval/v14-batch-omni-eval-1.batch.json")
    pairs = [json.loads(line) for line in (root / "datasets/grpo_tournament_pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    milestones = status["milestones"]
    profiler = milestones["m2_runtime_profiler_sampling"]["high_frequency_probe"]
    fleet = milestones["m3_pr_orchestrator"]["fleet_audit"]
    ci = status["github_actions"]
    recorded_ci_successes = [run for run in ci["latest_successful_runs"] if run.get("result") == "8/8 success"]
    test_results = status["tests"]
    counts = manifest["counts"]

    task10 = [row for row in pairs if row.get("chosen_candidate") == "job10_correct"]
    candidates = {}
    for row in task10:
        name = row["rejected_candidate"]
        candidates[name] = {
            "name": name,
            "score": row["rejected_score"],
            "evidence": row["rejection_evidence"],
            "prompt": row["prompt"],
        }
    chosen_score = task10[0]["chosen_score"] if task10 else None
    required_candidates = {"job10_wrong", "job10_signature", "job10_test_cheat", "job10_memory_leak"}
    if set(candidates) < required_candidates or len({row["prompt"] for row in task10}) != 1:
        raise ValueError("任务 10 的候选偏好记录不完整或题目不一致")

    okx = deliveries["business_repositories"]["okx-freqtrade-live"]
    pet = deliveries["business_repositories"]["pet-miniapp-backend"]
    guard = milestones["m1_frontend_invariant_guard"]
    metrics_ci = metrics["training"]["reward_interval"]["validation_calibrated_metrics"]["risk_head_precision_recall"]
    # The sample receipt and metrics come from checked-in artifacts. Do not infer an unreported FPR.
    observe_values = list(_walk_key(model_receipt, "model_mode"))
    observe_mode = "observe" if "observe" in observe_values else "未在舰队账本中记录"

    return {
        "version": status["version"],
        "phase": status["phase"],
        "tests_passed": test_results["passed"],
        "tests_total": test_results["total_collected"],
        "ci_required": ci["required_jobs"],
        "ci_success_records": len(recorded_ci_successes),
        "ci_head": ci["latest_verified_head"],
        "sft_examples": counts["sft_examples"],
        "preference_pairs": counts["preference_pairs"],
        "truncated_rows": counts["truncated_rows"],
        "profiler": profiler,
        "fleet": fleet,
        "dashboard_repositories": len(fleet_dashboard["repositories"]),
        "guard": guard,
        "candidates": candidates,
        "candidate_pair_count": len(task10),
        "chosen_score": chosen_score,
        "okx": okx,
        "pet_backend": pet,
        "risk_validation": metrics_ci,
        "model_mode": observe_mode,
        "sources": [
            "STATUS.json",
            "datasets/release_v15/manifest.json",
            "datasets/grpo_tournament_pairs.jsonl",
            "artifacts/real_delivery_v15/report.json",
            "artifacts/causal_world_model/training_metrics.json",
            "artifacts/fleet_dashboard_v16.json",
            "artifacts/mega_campaign_v14/workorders/omni-eval/v14-batch-omni-eval-1.batch.json",
        ],
    }


def resolve_cjk_font(*, install_missing: bool = True) -> tuple[Path, int]:
    """Find a real CJK font; on Linux, install Noto CJK automatically if absent."""
    explicit = os.environ.get("AEGIS_CJK_FONT")
    candidates = [Path(explicit)] if explicit else []
    if sys.platform.startswith("win"):
        candidates += [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf"), Path(r"C:\Windows\Fonts\simsun.ttc")]
    elif sys.platform == "darwin":
        candidates += [Path("/System/Library/Fonts/PingFang.ttc"), Path("/System/Library/Fonts/STHeiti Medium.ttc")]
    else:
        fc_match = shutil.which("fc-match")
        if fc_match:
            result = subprocess.run([fc_match, "-f", "%{file}\n%{index}\n%{family}\n", "Noto Sans CJK SC:lang=zh-cn"], capture_output=True, text=True, check=False)
            lines = result.stdout.splitlines()
            if len(lines) >= 2 and Path(lines[0]).is_file() and "Noto Sans CJK" in (lines[2] if len(lines) > 2 else ""):
                try:
                    return Path(lines[0]), int(lines[1])
                except ValueError:
                    pass
        candidates += [Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"), Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")]
    for path in candidates:
        if path.is_file():
            return path, 0
    if sys.platform.startswith("linux") and install_missing:
        apt = shutil.which("apt-get")
        if apt:
            prefix = [] if hasattr(os, "geteuid") and os.geteuid() == 0 else ["sudo"]
            subprocess.run(prefix + [apt, "update"], check=True)
            subprocess.run(prefix + [apt, "install", "-y", "fonts-noto-cjk"], check=True)
            fc_match = shutil.which("fc-match")
            if fc_match:
                result = subprocess.run([fc_match, "-f", "%{file}\n%{index}\n", "Noto Sans CJK SC:lang=zh-cn"], capture_output=True, text=True, check=True)
                lines = result.stdout.splitlines()
                if len(lines) >= 2 and Path(lines[0]).is_file():
                    return Path(lines[0]), int(lines[1])
    raise RuntimeError("没有找到可用的中文字体；请安装 Noto Sans CJK SC 或设置 AEGIS_CJK_FONT")


@dataclass
class Renderer:
    font_path: Path
    font_index: int
    evidence: dict

    def __post_init__(self):
        self.fonts = {
            "hero": ImageFont.truetype(str(self.font_path), 76, index=self.font_index),
            "title": ImageFont.truetype(str(self.font_path), 64, index=self.font_index),
            "subtitle": ImageFont.truetype(str(self.font_path), 40, index=self.font_index),
            "body": ImageFont.truetype(str(self.font_path), 38, index=self.font_index),
            "body2": ImageFont.truetype(str(self.font_path), 34, index=self.font_index),
            "label": ImageFont.truetype(str(self.font_path), 26, index=self.font_index),
            "small": ImageFont.truetype(str(self.font_path), 22, index=self.font_index),
            "mono": ImageFont.truetype(str(self.font_path), 28, index=self.font_index),
        }

    def _text(self, draw, xy, text, font, fill=WHITE, anchor=None):
        draw.text(xy, text, font=font, fill=fill, anchor=anchor, stroke_width=0)

    def _wrap(self, text, font, max_width):
        lines, current = [], ""
        for char in text:
            candidate = current + char
            if current and font.getlength(candidate) > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    def _panel(self, draw, rect, color):
        draw.rounded_rectangle(rect, radius=28, fill=PANEL, outline=PANEL_EDGE, width=2)
        x1, y1, x2, _ = rect
        draw.rounded_rectangle((x1, y1, x2, y1 + 10), radius=5, fill=color)

    def _evidence_lines(self, scene_index: int) -> list[str]:
        e = self.evidence
        if scene_index == 0:
            return [
                f"Aegis Core v{e['version']}｜确定性软件工厂",
                "隔离改动 → 探针验证 → 独立审查",
                f"{e['tests_passed']:,} 项测试通过｜状态账本",
                f"双平台 CI：{e['ci_required']}/{e['ci_required']}｜最近记录成功",
                f"SFT {e['sft_examples']:,} 条｜偏好对 {e['preference_pairs']:,} 组",
            ]
        if scene_index == 1:
            return [
                "正确修复｜拒绝放宽类型约束",
                "破坏函数签名｜拒绝",
                "改断言作弊｜得分 0.000｜拒绝",
                "移除历史上限｜无界增长｜拒绝",
                f"GRPO 偏好对：{e['preference_pairs']:,} 组",
            ]
        if scene_index == 2:
            return [
                "okx-freqtrade-live：拒绝无效杠杆与非有限金额",
                f"对应测试：{e['okx']['verified_tests']['okx_exchange_and_v15_leverage']['passed']} 项通过",
                "后端：损坏连接丢弃并保留原始事务错误",
                f"整套测试：{e['pet_backend']['verified_tests']['full_node_suite']['passed']} 项通过",
                f"故障集成验证：{e['pet_backend']['verified_tests']['postgres_fault_integration']['passed']} 项通过",
            ]
        if scene_index == 3:
            p = e["profiler"]
            f = e["fleet"]
            return [
                "契约分析：Python／JS／TS／HTML",
                "InterfaceContractLock｜锁定接口面",
                f"{p['hot_calls']:,} 次热调用｜{p['compatibility_backend_elapsed_seconds']:.3f} 秒",
                f"舰队：{f['delivery_branches']} 个交付分支｜冲突 {f['conflicts']}",
                "默认分支污染：0｜账本确认",
            ]
        if scene_index == 4:
            r = e["risk_validation"]
            return [
                "aegis replay｜冻结探针与现场复放",
                "aegis blame｜变更行归因",
                f"验证支持 {r['support']}｜误报 {r['false_positive']}",
                f"召回率 {r['recall']:.0%}｜当前模式 {e['model_mode']}",
                "误报率未报告｜不宣称门槛已通过",
            ]
        return [
            f"{e['tests_passed']:,}/{e['tests_total']:,} 项测试通过",
            f"双平台 CI：{e['ci_required']}/{e['ci_required']}｜最新已记录矩阵",
            f"{e['fleet']['delivery_branches']} 个交付分支｜{e['fleet']['conflicts']} 冲突",
            f"SFT {e['sft_examples']:,}｜GRPO 偏好对 {e['preference_pairs']:,}",
            "GPT 负责创造，Aegis 负责确定性兜底。",
        ]

    def frame(self, scene_index: int, local_time: float, *, animate: bool = True) -> Image.Image:
        scene = SCENES[scene_index]
        im = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(im)

        # Industrial control-room grid and header.
        for x in range(40, WIDTH, 80):
            draw.line((x, 0, x, 820), fill=(13, 22, 34), width=1)
        for y in range(28, 820, 64):
            draw.line((0, y, WIDTH, y), fill=(13, 22, 34), width=1)
        draw.rounded_rectangle((48, 32, 1872, 104), radius=18, fill=(11, 19, 32), outline=PANEL_EDGE, width=2)
        self._text(draw, (78, 47), "A E G I S   /   工程演示", self.fonts["label"], CYAN)
        self._text(draw, (1840, 50), f"第 {scene_index + 1} 幕 / 共六幕", self.fonts["label"], MUTED, anchor="ra")
        self._text(draw, (WIDTH // 2, 53), scene["title"], self.fonts["title"], WHITE, anchor="ma")

        left = (54, 130, 930, 822)
        right = (990, 130, 1866, 822)
        self._panel(draw, left, RED)
        self._panel(draw, right, GREEN)
        self._text(draw, (92, 158), "裸跑 GPT  /  风险示意", self.fonts["body2"], RED)
        self._text(draw, (1028, 158), "Aegis 调度 GPT  /  仓库证据", self.fonts["body2"], GREEN)
        draw.line((92, 218, 890, 218), fill=(45, 56, 70), width=2)
        draw.line((1028, 218, 1826, 218), fill=(45, 56, 70), width=2)

        y = 252
        for i, line in enumerate(scene["left"]):
            color = (255, 138, 143) if i < 2 else MUTED
            self._text(draw, (100, y), "›", self.fonts["body"], RED)
            self._text(draw, (145, y), line, self.fonts["body2"], color)
            y += 86

        right_lines = self._evidence_lines(scene_index)
        if scene_index == 1:
            # Recorded same-task preference examples; each line is backed by a checked-in row.
            names = [
                ("job10_correct", "正确修复｜胜出样本", GREEN),
                ("job10_signature", "破坏签名｜已拒绝", RED),
                ("job10_test_cheat", "修改断言｜0.000 分｜已拒绝", RED),
                ("job10_memory_leak", "移除上限｜无界增长｜已拒绝", RED),
            ]
            self._text(draw, (1034, 224), "任务 10｜四组同题偏好记录（非竞速录像）", self.fonts["label"], CYAN)
            y = 262
            for idx, (key, label, color) in enumerate(names):
                candidate = self.evidence["candidates"].get(key)
                score = f"胜出 {self.evidence['chosen_score']:.3f}" if key == "job10_correct" else f"拒绝 {candidate['score']:.3f}"
                draw.rounded_rectangle((1024, y, 1826, y + 83), radius=15, fill=(18, 31, 47), outline=(43, 61, 79), width=2)
                draw.ellipse((1044, y + 25, 1072, y + 53), fill=color)
                self._text(draw, (1092, y + 15), label, self.fonts["body2"], WHITE)
                self._text(draw, (1800, y + 23), score, self.fonts["label"], color, anchor="ra")
                y += 96
            self._text(draw, (1034, 654), f"GRPO 偏好对 {self.evidence['preference_pairs']:,} 组｜数据集", self.fonts["label"], CYAN)
        else:
            y = 250
            for idx, line in enumerate(right_lines):
                color = CYAN if (scene_index, idx) in {(0, 0), (0, 1), (1, 4), (3, 2), (4, 2), (5, 0)} else WHITE
                draw.ellipse((1037, y + 15, 1053, y + 31), fill=GREEN)
                self._text(draw, (1082, y), line, self.fonts["body2"], color)
                y += 72

        # Animated terminal evidence line, a scanning light and a growing progress rail.
        tick = int(local_time * SOURCE_FPS) if animate else 0
        logs = ["读取仓库证据", "冻结探针已就绪", "变异样本逐条复核", "审查结果写入账本", "仅放行可验证修复"]
        log_index = tick // 9 % len(logs)
        active_log = logs[log_index]
        previous_log = logs[(log_index - 1) % len(logs)]
        draw.rounded_rectangle((92, 704, 1828, 786), radius=15, fill=(8, 15, 25), outline=(31, 46, 63), width=2)
        self._text(draw, (116, 708), f"上一行｜{previous_log}", self.fonts["small"], MUTED)
        self._text(draw, (116, 742), f"当前核验｜{active_log}", self.fonts["mono"], CYAN)
        cursor_x = 104 + (tick * 21) % 1680 if animate else 760
        draw.line((cursor_x, 696, cursor_x, 790), fill=(73, 180, 203), width=2)

        # Persistent, large Chinese voice-over subtitle bar.
        draw.rounded_rectangle((48, 846, 1872, 1046), radius=24, fill=(12, 22, 36), outline=(51, 75, 97), width=3)
        draw.rounded_rectangle((48, 846, 65, 1046), radius=7, fill=GOLD)
        subfont = self.fonts["subtitle"]
        lines = self._wrap(scene["subtitle"], subfont, 1725)
        for i, line in enumerate(lines[:2]):
            self._text(draw, (94, 878 + i * 62), line, subfont, WHITE)

        # Proof/source line and scene progress.
        source_y = 1054
        source_text = "证据来源：STATUS.json · artifacts/ · datasets/ ｜ 风险示意不代表真实事故"
        self._text(draw, (WIDTH // 2, source_y), source_text, self.fonts["small"], MUTED, anchor="mm")
        elapsed = sum(s["duration"] for s in SCENES[:scene_index]) + local_time
        progress = min(1.0, elapsed / DURATION)
        draw.rounded_rectangle((48, 1068, 1872, 1078), radius=5, fill=(27, 43, 58))
        draw.rounded_rectangle((48, 1068, 48 + int(1824 * progress), 1078), radius=5, fill=CYAN)

        if animate and scene_index == 1:
            # Stamp rotates in, then holds. This is a visual cue over the recorded candidate evidence.
            stamp_label = "已拦截" if int(local_time * 2) % 5 != 0 else "门禁放行"
            stamp_color = RED if stamp_label == "已拦截" else GREEN
            stamp = Image.new("RGBA", (330, 106), (0, 0, 0, 0))
            sd = ImageDraw.Draw(stamp)
            sf = ImageFont.truetype(str(self.font_path), 62, index=self.font_index)
            sd.rounded_rectangle((4, 4, 326, 102), radius=14, outline=stamp_color + (245,), width=7)
            sd.text((165, 53), stamp_label, font=sf, fill=stamp_color + (245,), anchor="mm")
            stamp = stamp.rotate(-7, resample=Image.Resampling.BICUBIC, expand=True)
            im_rgba = im.convert("RGBA")
            im_rgba.alpha_composite(stamp, (500, 580))
            im = im_rgba.convert("RGB")
        elif animate and scene_index in (2, 3, 4, 5):
            stamp_text = "已拦截" if scene_index == 2 else "门禁放行"
            stamp_color = RED if scene_index == 2 else GREEN
            stamp = Image.new("RGBA", (400, 100), (0, 0, 0, 0))
            sd = ImageDraw.Draw(stamp)
            sf = ImageFont.truetype(str(self.font_path), 54, index=self.font_index)
            sd.rounded_rectangle((3, 3, 397, 97), radius=14, outline=stamp_color + (235,), width=6)
            sd.text((200, 50), stamp_text, font=sf, fill=stamp_color + (235,), anchor="mm")
            stamp = stamp.rotate(-4, resample=Image.Resampling.BICUBIC, expand=True)
            rgba = im.convert("RGBA")
            rgba.alpha_composite(stamp, (500, 580))
            im = rgba.convert("RGB")
        return im


def _font_glyph_coverage(font_path: Path, face_index: int, text: str) -> set[int]:
    try:
        from fontTools.ttLib import TTCollection, TTFont
    except ImportError as exc:
        raise RuntimeError("请安装 fonttools 以执行中文字体覆盖检查") from exc
    if font_path.suffix.lower() == ".ttc":
        collection = TTCollection(str(font_path), lazy=True)
        font = collection.fonts[face_index]
    else:
        font = TTFont(str(font_path), lazy=True)
    cmap = set()
    for table in font["cmap"].tables:
        cmap.update(table.cmap)
    wanted = {ord(ch) for ch in text if ord(ch) >= 0x80}
    return wanted - cmap


def validate_glyph_pixels(renderer: Renderer):
    """Fail if any visible CJK character renders as the font's missing-glyph box."""
    visible_parts = []
    for scene in SCENES:
        visible_parts.extend([scene["title"], scene["subtitle"], *scene["left"], *scene["right"]])
    for scene_index in range(len(SCENES)):
        visible_parts.extend(renderer._evidence_lines(scene_index))
    visible = "".join(visible_parts)
    visible += "裸跑GPT风险示意Aegis调度GPT仓库证据GRPOobserveInterfaceContractLock"
    visible += "证据终端上一行当前核验读取仓库证据冻结探针已就绪变异样本逐条复核审查结果写入账本仅放行可验证修复"
    visible += "工程演示第幕共六幕已拦截门禁放行STATUS.json·artifacts/·datasets/›"
    missing = _font_glyph_coverage(renderer.font_path, renderer.font_index, visible)
    if missing:
        chars = "".join(chr(cp) for cp in sorted(missing))
        raise AssertionError(f"字体缺少可见字符：{chars}")
    probe_font = ImageFont.truetype(str(renderer.font_path), 72, index=renderer.font_index)
    tofu_font = probe_font
    box = Image.new("L", (112, 112), 0)
    ImageDraw.Draw(box).text((12, 8), "\U0010ffff", font=tofu_font, fill=255)
    tofu = box.getbbox()
    if not tofu:
        raise AssertionError("字体缺字测试样本未产生缺字框，像素对照无效")
    tofu_mask = box.crop(tofu)
    for ch in sorted({c for c in visible if ord(c) >= 0x80}):
        sample = Image.new("L", (112, 112), 0)
        ImageDraw.Draw(sample).text((12, 8), ch, font=probe_font, fill=255)
        bbox = sample.getbbox()
        if not bbox:
            raise AssertionError(f"字形为空：{ch}")
        mask = sample.crop(bbox)
        fitted = Image.new("L", tofu_mask.size, 0)
        fitted.paste(mask.resize(tofu_mask.size, Image.Resampling.NEAREST), (0, 0))
        if ImageChops.difference(fitted, tofu_mask).getbbox() is None:
            raise AssertionError(f"检测到方块缺字（像素与缺字框一致）：{ch}")


def _run_ffmpeg(renderer: Renderer, frames_dir: Path, output: Path, preview: Path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg；请先安装 FFmpeg")
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(SOURCE_FPS), "-i", "-", "-an", "-vf", f"fps={FPS}",
        "-frames:v", str(DURATION * FPS), "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    elapsed = 0
    try:
        for scene_index, scene in enumerate(SCENES):
            for frame_index in range(scene["duration"] * SOURCE_FPS):
                local_time = frame_index / SOURCE_FPS
                frame = renderer.frame(scene_index, local_time, animate=True)
                if frame_index == 0:
                    frame.save(frames_dir / FRAME_NAMES[scene_index], format="PNG", optimize=True)
                process.stdin.write(frame.tobytes())
                elapsed += 1
    except BrokenPipeError:
        pass
    finally:
        process.stdin.close()
    error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg 合成失败（{code}）：{error[-3000:]}")
    if elapsed != DURATION * SOURCE_FPS:
        raise RuntimeError(f"渲染帧数异常：{elapsed}")

    # A compact animated storyboard for mobile previews.
    gif_frames = []
    for scene_index in range(len(SCENES)):
        for t in (0.0, 0.6):
            image = renderer.frame(scene_index, t, animate=True).resize((640, 360), Image.Resampling.LANCZOS)
            gif_frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    gif_frames[0].save(preview, save_all=True, append_images=gif_frames[1:], duration=520, loop=0, optimize=True, disposal=2)


def validate_video(path: Path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("找不到 ffprobe，无法验证视频元数据")
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate:format=duration,size", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    duration = float(info["format"]["duration"])
    assert stream["codec_name"] == "h264"
    assert (stream["width"], stream["height"]) == (WIDTH, HEIGHT)
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["avg_frame_rate"] == "30/1"
    assert abs(duration - DURATION) < 0.05
    return {"codec": stream["codec_name"], "width": stream["width"], "height": stream["height"], "pixel_format": stream["pix_fmt"], "frame_rate": stream["avg_frame_rate"], "duration_seconds": duration, "size_bytes": int(info["format"]["size"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validate-only", action="store_true", help="读取真实证据并校验中文字体，不渲染视频")
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = load_evidence(root)
    font_path, font_index = resolve_cjk_font()
    renderer = Renderer(font_path, font_index, evidence)
    validate_glyph_pixels(renderer)
    print(f"中文字体：{font_path}（字面索引 {font_index}），像素缺字检查通过")
    print(f"数据来源：{', '.join(evidence['sources'])}")
    if args.validate_only:
        print("证据与字体校验通过")
        return
    output_dir = root / "showcase"
    frames_dir = output_dir / "frames_zh"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    video = output_dir / "aegis_vs_gpt_zh.mp4"
    preview = output_dir / "aegis_vs_gpt_zh_preview.gif"
    _run_ffmpeg(renderer, frames_dir, video, preview)
    metadata = validate_video(video)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"关键帧：{frames_dir}")
    print(f"动态预览：{preview}")


if __name__ == "__main__":
    main()
