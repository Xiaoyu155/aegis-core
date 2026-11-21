#!/usr/bin/env python3
"""Render the six-scene Chinese Aegis showcase in Chromium and mix AAC audio."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFont, ImageDraw


WIDTH, HEIGHT, FPS, DURATION = 1920, 1080, 30, 90
FRAME_NAMES = [
    "01-开场破题.png", "02-候选对比.png", "03-风险守卫.png",
    "04-跨文件契约.png", "05-工程安全阀.png", "06-舰队交付.png",
]
SCENE_STARTS = [0, 12, 28, 46, 62, 76]
AUDIO_PROVENANCE = {
    "voiceover": {
        "generated": False,
        "provider": "Microsoft Edge TTS",
        "voice_attempted": "zh-CN-YunxiNeural",
        "reason": "speech.platform.bing.com:443 返回网络不可达；运行环境未发现离线中文语音引擎。",
    },
    "music": "Python 与 NumPy 生成的立体声低频氛围、节拍脉冲与和弦。",
    "sound_effects": "Python 与 NumPy 生成的音效：键盘敲击、红色警报、激光拦截、节点脉冲、倒带和通关提示音。",
    "mix": "FFmpeg 混合为 48 kHz 双声道 AAC。",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _added_lines(patch: str) -> list[str]:
    return [line[1:].strip() for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++") and line[1:].strip()]


def _removed_line(patch: str, needle: str) -> str:
    for line in patch.splitlines():
        if line.startswith("-") and not line.startswith("---") and needle in line:
            return line[1:].strip()
    raise ValueError(f"真实补丁中缺少移除行：{needle}")


def load_evidence(root: Path) -> dict:
    status = _read_json(root / "STATUS.json")
    manifest = _read_json(root / "datasets/release_v15/manifest.json")
    report = _read_json(root / "artifacts/real_delivery_v15/report.json")
    okx_receipt = _read_json(root / "artifacts/real_delivery_v15/okx_autopilot.json")
    pet_receipt = _read_json(root / "artifacts/real_delivery_v15/pet_backend_autopilot.json")
    world_metrics = _read_json(root / "artifacts/causal_world_model/training_metrics.json")
    fleet_dashboard = _read_json(root / "artifacts/fleet_dashboard_v16.json")
    model_receipt = _read_json(root / "artifacts/mega_campaign_v14/workorders/omni-eval/v14-batch-omni-eval-1.batch.json")
    pairs = [json.loads(line) for line in (root / "datasets/grpo_tournament_pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    milestones = status.get("milestones", {})
    if "m2_runtime_profiler_sampling" not in milestones or "m3_pr_orchestrator" not in milestones:
        # v2.0 nests the v1.6 evidence under previous_phase_16 while keeping the
        # current release status at the top level.
        milestones = status.get("previous_phase_16", {}).get("milestones", milestones)
    profiler = milestones["m2_runtime_profiler_sampling"]["high_frequency_probe"]
    fleet = milestones["m3_pr_orchestrator"]["fleet_audit"]
    ci = status.get("github_actions", {})
    previous_ci = status.get("previous_phase_16", {}).get("github_actions", {})
    ci_jobs = ci.get("required_jobs", ci.get("expected_jobs_per_event",
                                                   previous_ci.get("required_jobs")))
    ci_head = ci.get("latest_verified_head") or ci.get("pull_request_event", {}).get("head_sha") \
        or previous_ci.get("latest_verified_head")
    counts = manifest["counts"]
    job10 = [row for row in pairs if row.get("chosen_candidate") == "job10_correct"]
    candidates = {row["rejected_candidate"]: row for row in job10}
    required = {"job10_signature", "job10_test_cheat", "job10_memory_leak", "job10_wrong"}
    if set(candidates) != required or len({row["prompt"] for row in job10}) != 1:
        raise ValueError("任务 10 的四组候选数据不完整")

    okx_patch = okx_receipt["winner"]["patch"]
    pet_patch = pet_receipt["winner"]["patch"]
    okx_added = _added_lines(okx_patch)
    pet_added = _added_lines(pet_patch)
    okx_lines = [s for s in okx_added if any(k in s for k in ("leverage = payload.get", "isinstance(leverage", "if leverage <=", "notional = payload.get", "max_notional must"))]
    pet_lines = [s for s in pet_added if any(k in s for k in ("connectionError = null", "rememberConnectionError", "client.on('error'", "removeListener('error'", "client.release(connectionError"))]
    if len(okx_lines) < 4 or len(pet_lines) < 4:
        raise ValueError("真实修复补丁中的代码行不完整")

    okx = report["business_repositories"]["okx-freqtrade-live"]["verified_tests"]["okx_exchange_and_v15_leverage"]
    pet = report["business_repositories"]["pet-miniapp-backend"]["verified_tests"]
    risk = world_metrics["training"]["reward_interval"]["validation_calibrated_metrics"]["risk_head_precision_recall"]
    model_modes = []
    for job in model_receipt.get("jobs", []):
        result = job.get("result", {})
        model_modes.extend(item.get("model_mode") for item in result.get("results", []))
        if result.get("telemetry"):
            model_modes.append(result["telemetry"].get("model_mode"))
    observe = "observe" if "observe" in model_modes else "未记录"
    repos = [row["repository"] for row in fleet_dashboard["repositories"]]
    if len(repos) != fleet["repositories_with_delivery_branches"]:
        raise ValueError("舰队仪表盘仓库数与状态账本不一致")

    return {
        "version": status["version"],
        "tests": status["tests"]["passed"],
        "ci_jobs": ci_jobs,
        "ci_head": ci_head,
        "sft": counts["sft_examples"],
        "preference_pairs": counts["preference_pairs"],
        "truncated_rows": counts["truncated_rows"],
        "profiler": profiler,
        "fleet": fleet,
        "fleet_repositories": repos,
        "candidates": {key: {"score": value["rejected_score"], "evidence": value["rejection_evidence"], "patch": value["rejected_patch"]} for key, value in candidates.items()},
        "chosen_score": job10[0]["chosen_score"],
        "candidate_pairs": len(job10),
        "okx_added": okx_lines[:6],
        "pet_added": pet_lines[:5],
        "old_okx_division": _removed_line(okx_patch, 'payload["max_notional"] / payload["leverage"]'),
        "old_db_release": _removed_line(pet_patch, "client.release();"),
        "okx_case_count": okx_receipt["synthesized_probe"]["case_count"],
        "okx_baseline_red": okx_receipt["synthesized_probe"]["baseline_true_red"],
        "okx_tests": {"passed": okx["passed"], "total": okx["passed"] + okx["failed"]},
        "pet_case_count": pet_receipt["synthesized_probe"]["case_count"],
        "pet_baseline_red": pet_receipt["synthesized_probe"]["baseline_true_red"],
        "pet_tests": {"passed": pet["full_node_suite"]["passed"], "skipped": pet["full_node_suite"]["skipped"]},
        "pet_integration": {"passed": pet["postgres_fault_integration"]["passed"], "total": pet["postgres_fault_integration"]["passed"] + pet["postgres_fault_integration"]["failed"]},
        "replay_ticket": okx_receipt["time_travel_snapshot"]["ticket_id"],
        "replay_hash": okx_receipt["time_travel_snapshot"]["archive_sha256"][:12],
        "risk_validation": risk,
        "model_mode": observe,
        "sources": [
            "STATUS.json", "datasets/release_v15/manifest.json", "datasets/grpo_tournament_pairs.jsonl",
            "artifacts/real_delivery_v15/report.json", "artifacts/real_delivery_v15/okx_autopilot.json",
            "artifacts/real_delivery_v15/pet_backend_autopilot.json", "artifacts/causal_world_model/training_metrics.json",
            "artifacts/fleet_dashboard_v16.json", "artifacts/mega_campaign_v14/workorders/omni-eval/v14-batch-omni-eval-1.batch.json",
        ],
    }


def resolve_cjk_font(*, install_missing: bool = True) -> tuple[Path, int]:
    explicit = os.environ.get("AEGIS_CJK_FONT")
    if explicit and Path(explicit).is_file():
        return Path(explicit), 0
    if sys.platform.startswith("win"):
        for candidate in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf"), Path(r"C:\Windows\Fonts\simsun.ttc")):
            if candidate.is_file():
                return candidate, 0
    elif sys.platform == "darwin":
        for candidate in (Path("/System/Library/Fonts/PingFang.ttc"), Path("/System/Library/Fonts/STHeiti Medium.ttc")):
            if candidate.is_file():
                return candidate, 0
    else:
        fc = shutil.which("fc-match")
        if fc:
            result = subprocess.run([fc, "-f", "%{file}\n%{index}\n%{family}\n", "Noto Sans CJK SC:lang=zh-cn"], capture_output=True, text=True, check=False)
            lines = result.stdout.splitlines()
            if len(lines) >= 2 and Path(lines[0]).is_file() and "Noto Sans CJK" in (lines[2] if len(lines) > 2 else ""):
                return Path(lines[0]), int(lines[1])
    if sys.platform.startswith("linux") and install_missing and shutil.which("apt-get"):
        prefix = [] if getattr(os, "geteuid", lambda: 1)() == 0 else ["sudo"]
        subprocess.run(prefix + ["apt-get", "update"], check=True)
        subprocess.run(prefix + ["apt-get", "install", "-y", "fonts-noto-cjk"], check=True)
        return resolve_cjk_font(install_missing=False)
    raise RuntimeError("未找到 Noto Sans CJK SC；请安装 fonts-noto-cjk 或设置 AEGIS_CJK_FONT")


def validate_glyph_pixels(font_path: Path, font_index: int, visible_text: str) -> int:
    """Check font cmap and compare each visible Unicode glyph to a missing-glyph pixel mask."""
    from fontTools.ttLib import TTCollection, TTFont

    if font_path.suffix.lower() == ".ttc":
        font = TTCollection(str(font_path), lazy=True).fonts[font_index]
    else:
        font = TTFont(str(font_path), lazy=True)
    cmap = {code for table in font["cmap"].tables for code in table.cmap}
    chars = sorted({ch for ch in visible_text if ord(ch) >= 128})
    absent = [ch for ch in chars if ord(ch) not in cmap]
    if absent:
        raise AssertionError("字体 cmap 缺字：" + "".join(absent))

    face = ImageFont.truetype(str(font_path), 72, index=font_index)
    def mask(ch):
        image = Image.new("L", (128, 128), 0)
        ImageDraw.Draw(image).text((12, 8), ch, font=face, fill=255)
        return image
    missing = mask("\U0010ffff")
    if not missing.getbbox():
        raise AssertionError("无法构造缺字像素基准")
    missing_box = missing.crop(missing.getbbox())
    for ch in chars:
        glyph = mask(ch)
        bbox = glyph.getbbox()
        if not bbox:
            raise AssertionError(f"可见字符字形为空：{ch}")
        fitted = Image.new("L", missing_box.size, 0)
        fitted.paste(glyph.crop(bbox).resize(missing_box.size, Image.Resampling.NEAREST), (0, 0))
        if ImageChops.difference(fitted, missing_box).getbbox() is None:
            raise AssertionError(f"检测到方块缺字：{ch}")
    return len(chars)


def _add_tone(left, right, sr, start, duration, f0, f1=None, amp=.1, pan=0.0, decay=8.0, waveform="sine"):
    i0 = max(0, int(start * sr)); n = min(int(duration * sr), len(left) - i0)
    if n <= 0:
        return
    tt = np.arange(n, dtype=np.float32) / sr
    f1 = float(f0 if f1 is None else f1)
    phase = 2 * np.pi * (f0 * tt + (f1 - f0) * tt * tt / (2 * max(duration, 1e-4)))
    if waveform == "square":
        sig = np.tanh(2.4 * np.sin(phase))
    elif waveform == "tri":
        sig = (2 / np.pi) * np.arcsin(np.sin(phase))
    else:
        sig = np.sin(phase)
    fade_in = np.clip(tt / .008, 0, 1)
    env = fade_in * np.exp(-decay * tt / max(duration, 1e-4))
    sig = (sig * env * amp).astype(np.float32)
    lg = math.sqrt((1 - max(-.98, min(.98, pan))) * .5)
    rg = math.sqrt((1 + max(-.98, min(.98, pan))) * .5)
    left[i0:i0+n] += sig * lg
    right[i0:i0+n] += sig * rg


def _add_noise_click(left, right, sr, start, duration=.025, amp=.045, pan=0.0, seed=42):
    i0 = max(0, int(start * sr)); n = min(int(duration * sr), len(left) - i0)
    if n <= 0:
        return
    rng = np.random.default_rng(seed)
    tt = np.arange(n, dtype=np.float32) / sr
    sig = (rng.standard_normal(n) * np.exp(-tt * 150) * amp).astype(np.float32)
    left[i0:i0+n] += sig * math.sqrt((1-pan)*.5)
    right[i0:i0+n] += sig * math.sqrt((1+pan)*.5)


def synthesize_audio(path: Path):
    """Create an original 90-second stereo ambient score and synced UI sound effects."""
    sr = 48_000
    n = sr * DURATION
    t = np.arange(n, dtype=np.float32) / sr
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)

    # Low, slowly breathing control-room bed and scene-changing harmonic pads.
    breathing = .78 + .22 * np.sin(2 * np.pi * .035 * t)
    for channel, detune in ((left, 0.0), (right, .22)):
        channel += .021 * np.sin(2*np.pi*(55+detune)*t) * breathing
        channel += .013 * np.sin(2*np.pi*(82.41+detune)*t + .2*np.sin(2*np.pi*.07*t)) * breathing
        channel += .008 * np.sin(2*np.pi*(110-detune)*t) * breathing
    chords = [(73.42, 110.0, 146.83), (58.27, 87.31, 116.54), (65.41, 98.0, 130.81),
              (73.42, 110.0, 164.81), (58.27, 87.31, 146.83), (65.41, 98.0, 130.81)]
    boundaries = SCENE_STARTS + [DURATION]
    for scene, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        mask = (t >= a) & (t < b)
        local = t[mask] - a
        length = b - a
        edge = np.minimum(np.clip(local / 1.8, 0, 1), np.clip((length-local) / 2.2, 0, 1))
        env = (.5 - .5 * np.cos(np.pi * edge))
        for k, hz in enumerate(chords[scene]):
            pad = np.sin(2*np.pi*hz*local + .15*np.sin(2*np.pi*.08*local)) * env * (.009 if k == 0 else .006)
            left[mask] += pad.astype(np.float32)
            right[mask] += (pad * (1 if k != 1 else .78)).astype(np.float32)

    # Soft 100 BPM pulse and alternating, restrained arpeggio.
    beat = .6
    for ix, start in enumerate(np.arange(.25, DURATION, beat)):
        _add_tone(left, right, sr, start, .29, 92, 43, .075, 0.0, 7.2)
        _add_noise_click(left, right, sr, start + .015, .026, .018, -.12, ix+10)
        if ix % 2 == 0:
            notes = (392.0, 440.0, 523.25, 659.25, 587.33, 493.88, 440.0, 349.23)
            _add_tone(left, right, sr, start + .25, .22, notes[(ix//2) % len(notes)], None, .018, -.4 if ix%4 else .4, 11, "tri")

    # Keystrokes while the opening request appears.
    for ix in range(24):
        _add_noise_click(left, right, sr, .82 + ix*.105, .018 + (ix%3)*.004, .033, -.35 + .7*(ix%8)/7, ix+1)
    # Red warning signals and the blue shield resolution.
    for j, start in enumerate((4.1, 5.1, 6.1)):
        _add_tone(left, right, sr, start, .22, 760, 560, .10, -.45 + j*.45, 4.5)
        _add_tone(left, right, sr, start+.26, .18, 560, 760, .075, -.45 + j*.45, 5.2)
    for f in (392, 523.25, 783.99): _add_tone(left, right, sr, 7.25, 1.45, f, f*1.02, .065, .16, 3.0)
    _add_tone(left, right, sr, 7.1, 1.15, 90, 880, .16, 0, 3.5)
    # Parallel lane launches, three red laser checks and one green release.
    for ix in range(4):
        _add_tone(left, right, sr, 12.3 + ix*.24, .32, 300+ix*95, 720+ix*40, .06, -.7+ix*.46, 7)
    for ix, start in enumerate((17.4, 19.7, 22.0)):
        _add_tone(left, right, sr, start, .42, 1200, 260, .15, -.5+ix*.5, 4)
        _add_noise_click(left, right, sr, start+.1, .09, .10, -.5+ix*.5, 120+ix)
    for f in (523.25, 659.25, 783.99, 1046.5): _add_tone(left, right, sr, 25.5, .92, f, f, .055, .1, 3)
    # Diff keystrokes, baseline red probe and green verified tests.
    for ix in range(26): _add_noise_click(left, right, sr, 33.0+ix*.19, .016, .026, -.5+(ix%9)*.12, 300+ix)
    for start in (30.6, 32.0):
        _add_tone(left, right, sr, start, .4, 980, 410, .12, -.25, 5)
    for f in (440, 554.37, 659.25, 880): _add_tone(left, right, sr, 42.2, 1.2, f, f, .057, .12, 3)
    # Topology pings, contract lock and profile counter completion.
    for ix in range(8): _add_tone(left, right, sr, 47.2+ix*.8, .24, 480+ix*72, 820+ix*55, .045, -.65+ix*.18, 8)
    for f in (587.33, 739.99, 880): _add_tone(left, right, sr, 58.7, .75, f, f, .055, .25, 3)
    # Rewind sweep and amber lock.
    for ix in range(4): _add_tone(left, right, sr, 63.0+ix*.22, .75, 1100-ix*120, 100+ix*15, .06, -.6+ix*.4, 4)
    _add_tone(left, right, sr, 73.5, .5, 220, 90, .10, 0, 6)
    _add_tone(left, right, sr, 74.1, 1.15, 392, 392, .065, 0, 2.5)
    # Fleet lights land one by one; final resolving chord.
    for ix in range(15):
        start=76.2+ix*.40
        for f in (659.25, 880): _add_tone(left, right, sr, start, .18, f*(1+ix*.001), f*(1+ix*.001), .037, -.6+1.2*(ix%5)/4, 8)
    for f in (261.63, 329.63, 392.0, 523.25, 659.25): _add_tone(left, right, sr, 88.0, 1.85, f, f, .055, 0, 1.8)

    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-6)
    gain = min(1.0, .89 / peak)
    stereo = np.column_stack((left*gain, right*gain))
    pcm = np.clip(stereo * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(sr); wav.writeframes(pcm.tobytes())


def _run_browser_ffmpeg(root: Path, evidence: dict, output: Path, frames_dir: Path, work: Path):
    node = shutil.which("node"); ffmpeg = shutil.which("ffmpeg")
    if not node or not ffmpeg or not Path("/usr/bin/chromium").is_file():
        raise RuntimeError("需安装 Node、Playwright、Chromium 与 FFmpeg 才能逐帧渲染")
    evidence_path = work / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")
    audio_path = work / "aegis_score.wav"
    synthesize_audio(audio_path)
    ffmpeg_args = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", str(FPS), "-i", "pipe:0",
        "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-t", str(DURATION), "-frames:v", str(FPS*DURATION),
        "-vf", "scale=in_range=full:out_range=tv,format=yuv420p", "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-color_range", "tv",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-metadata", "title=Aegis Core｜GPT 工程演示",
        "-metadata", "comment=Chromium Canvas 逐帧渲染；Python 合成音乐与音效", "-movflags", "+faststart", str(output),
    ]
    node_args = [node, str(root / "scripts/render_aegis_vs_gpt_zh_v2.cjs"), str(root / "showcase/aegis_vs_gpt_zh_v2.html"), str(evidence_path), str(frames_dir)]
    with tempfile.TemporaryFile() as node_err:
        ff = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        nr = subprocess.Popen(node_args, stdout=subprocess.PIPE, stderr=node_err, cwd=root)
        try:
            assert ff.stdin and nr.stdout
            while True:
                chunk = nr.stdout.read(1024 * 1024)
                if not chunk: break
                ff.stdin.write(chunk)
            ff.stdin.close()
            node_code = nr.wait()
            ff_stderr = ff.stderr.read().decode("utf-8", errors="replace") if ff.stderr else ""
            ff_code = ff.wait()
            node_err.seek(0); node_log = node_err.read().decode("utf-8", errors="replace")
            print(node_log, end="", file=sys.stderr)
            if node_code:
                raise RuntimeError(f"Chromium 逐帧渲染失败 ({node_code})")
            if ff_code:
                raise RuntimeError(f"FFmpeg 合成失败 ({ff_code})：{ff_stderr[-2500:]}")
        except Exception:
            for p in (nr, ff):
                if p.poll() is None: p.kill()
            raise


def validate_video(path: Path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("找不到 ffprobe")
    result = subprocess.run([
        ffprobe, "-v", "error", "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,sample_rate,channels:format=duration,size",
        "-of", "json", str(path)
    ], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
    duration = float(data["format"]["duration"])
    assert (video["codec_name"], video["width"], video["height"], video["pix_fmt"], video["avg_frame_rate"]) == ("h264", WIDTH, HEIGHT, "yuv420p", "30/1")
    assert (audio["codec_name"], audio["sample_rate"], audio["channels"]) == ("aac", "48000", 2)
    assert abs(duration - DURATION) < .05
    return {"video_codec": video["codec_name"], "width": video["width"], "height": video["height"], "pixel_format": video["pix_fmt"], "frame_rate": video["avg_frame_rate"], "audio_codec": audio["codec_name"], "sample_rate": int(audio["sample_rate"]), "channels": audio["channels"], "duration_seconds": duration, "size_bytes": int(data["format"]["size"])}


def _make_preview(ffmpeg: str, video: Path, output: Path):
    filters=[]
    starts=(1,13,29,47,63,77)
    for i,start in enumerate(starts):
        filters.append(f"[0:v]trim=start={start}:duration=2,setpts=PTS-STARTPTS,scale=480:270:flags=lanczos,fps=8,setsar=1[v{i}]")
    labels="".join(f"[v{i}]" for i in range(6))
    filters.append(f"{labels}concat=n=6:v=1:a=0,split[frames][palette_src]")
    filters.append("[palette_src]palettegen=max_colors=96:stats_mode=diff[palette]")
    filters.append("[frames][palette]paletteuse=dither=bayer")
    subprocess.run([ffmpeg,"-y","-hide_banner","-loglevel","error","-i",str(video),"-filter_complex",";".join(filters),"-loop","0",str(output)],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args();root=args.root.resolve()
    evidence=load_evidence(root)
    font_path,font_index=resolve_cjk_font()
    html=(root/"showcase/aegis_vs_gpt_zh_v2.html").read_text(encoding="utf-8")
    visible=html+json.dumps(evidence,ensure_ascii=False)
    glyph_count=validate_glyph_pixels(font_path,font_index,visible)
    print(f"中文字体：{font_path}（字面索引 {font_index}）；像素核验 {glyph_count} 个 Unicode 字符无方块")
    print("音频：未生成中文TTS；使用 Python 合成配乐与音效。")
    if args.validate_only: return

    showcase=root/"showcase";frames=showcase/"frames_zh";frames.mkdir(parents=True,exist_ok=True)
    provenance=showcase/"audio_provenance_zh.json"
    provenance.write_text(json.dumps(AUDIO_PROVENANCE,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    v2=showcase/"aegis_vs_gpt_zh_v2.mp4"
    with tempfile.TemporaryDirectory(prefix="aegis-zh-v2-") as tmp:
        work=Path(tmp)
        _run_browser_ffmpeg(root,evidence,v2,frames,work)
    # The PR's established mobile path now serves the same v2 render.
    shutil.copy2(v2,showcase/"aegis_vs_gpt_zh.mp4")
    _make_preview(shutil.which("ffmpeg") or "ffmpeg",v2,showcase/"aegis_vs_gpt_zh_preview.gif")
    metadata=validate_video(v2)
    print(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(f"视频：{v2}")
    print(f"预览：{showcase/'aegis_vs_gpt_zh_preview.gif'}")
    print(f"关键帧：{frames}")


if __name__=="__main__":
    main()
