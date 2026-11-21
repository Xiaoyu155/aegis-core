from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.build_aegis_vs_gpt_zh import (
    DURATION,
    FRAME_NAMES,
    HEIGHT,
    WIDTH,
    load_evidence,
    resolve_cjk_font,
    validate_glyph_pixels,
)


ROOT = Path(__file__).resolve().parents[1]


def test_showcase_reads_real_repository_facts_and_patch_evidence():
    evidence = load_evidence(ROOT)
    assert evidence["tests"] == json.loads((ROOT / "STATUS.json").read_text(encoding="utf-8"))["tests"]["passed"]
    assert evidence["ci_jobs"] == 8
    assert evidence["sft"] == 1462
    assert evidence["preference_pairs"] == 342
    assert evidence["profiler"]["hot_calls"] == 100000
    assert evidence["profiler"]["compatibility_backend_elapsed_seconds"] == 1.418
    assert evidence["fleet"]["delivery_branches"] == 15
    assert evidence["fleet"]["conflicts"] == 0
    assert evidence["fleet"]["default_branch_zero_pollution"] is True
    assert len(evidence["fleet_repositories"]) == 15
    assert evidence["candidate_pairs"] == 4
    assert evidence["chosen_score"] == 0.98969
    assert evidence["candidates"]["job10_test_cheat"]["score"] == 0.0
    assert evidence["candidates"]["job10_memory_leak"]["evidence"]["audit_findings"][0]["code"] == "UNBOUNDED_GROWTH"
    assert evidence["okx_baseline_red"] == evidence["pet_baseline_red"] == "ASSERTION_FAILED"
    assert evidence["old_okx_division"] == 'return {"max_stake": payload["max_notional"] / payload["leverage"]}'
    assert any("client.release(connectionError" in line for line in evidence["pet_added"])
    assert evidence["okx_tests"] == {"passed": 32, "total": 32}
    assert evidence["pet_tests"] == {"passed": 274, "skipped": 1}
    assert evidence["pet_integration"] == {"passed": 1, "total": 1}
    assert evidence["model_mode"] == "observe"
    assert evidence["risk_validation"]["support"] == 2
    assert evidence["risk_validation"]["false_positive"] == 2
    assert all((ROOT / source).is_file() for source in evidence["sources"])


def test_six_scenes_have_different_canvas_visual_structures():
    html = (ROOT / "showcase/aegis_vs_gpt_zh_v2.html").read_text(encoding="utf-8")
    for function in ("drawOpening", "drawLanes", "drawCode", "drawTopology", "drawReplay", "drawFleet"):
        assert f"function {function}(" in html
    assert "window.renderDataUrl" in html
    assert "window.validateGlyphs" in html
    assert "16 / 9" not in html


def test_chinese_font_pixels_and_browser_scene_copy_have_no_tofu():
    font_path, font_index = resolve_cjk_font(install_missing=False)
    html = (ROOT / "showcase/aegis_vs_gpt_zh_v2.html").read_text(encoding="utf-8")
    evidence = load_evidence(ROOT)
    checked = validate_glyph_pixels(font_path, font_index, html + json.dumps(evidence, ensure_ascii=False))
    assert checked >= 200


def test_six_browser_rendered_keyframes_and_animated_mobile_gif():
    frame_dir = ROOT / "showcase/frames_zh"
    for filename in FRAME_NAMES:
        path = frame_dir / filename
        assert path.is_file(), f"缺少关键帧：{path}"
        with Image.open(path) as image:
            assert image.size == (WIDTH, HEIGHT)
            assert image.format == "PNG"
            assert image.convert("RGB").getbbox() == (0, 0, WIDTH, HEIGHT)

    preview = ROOT / "showcase/aegis_vs_gpt_zh_preview.gif"
    assert preview.is_file()
    with Image.open(preview) as image:
        assert image.format == "GIF"
        assert image.size == (480, 270)
        assert image.n_frames >= 90
    assert preview.stat().st_size < 5_000_000


def _probe(path: Path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("此运行环境未安装 ffprobe")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,sample_rate,channels:format=duration,size", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in info["streams"] if stream["codec_type"] == "audio")
    assert (video["codec_name"], video["width"], video["height"], video["pix_fmt"], video["avg_frame_rate"]) == ("h264", WIDTH, HEIGHT, "yuv420p", "30/1")
    assert (audio["codec_name"], audio["sample_rate"], audio["channels"]) == ("aac", "48000", 2)
    assert abs(float(info["format"]["duration"]) - DURATION) < .05
    return info


def test_v2_and_mobile_path_mp4_have_h264_video_and_aac_audio():
    v2 = ROOT / "showcase/aegis_vs_gpt_zh_v2.mp4"
    mobile_path = ROOT / "showcase/aegis_vs_gpt_zh.mp4"
    assert v2.is_file() and mobile_path.is_file()
    _probe(v2)
    _probe(mobile_path)
    assert hashlib.sha256(v2.read_bytes()).digest() == hashlib.sha256(mobile_path.read_bytes()).digest()


def test_mp4_audio_is_audible_and_voiceover_provenance_is_truthful():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("此运行环境未安装 ffmpeg")
    video = ROOT / "showcase/aegis_vs_gpt_zh_v2.mp4"
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-ss", "15", "-t", "2", "-i", str(video), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "2", "-ar", "48000", "pipe:1"],
        capture_output=True,
        check=True,
    ).stdout
    pcm = np.frombuffer(decoded, dtype="<i2").astype(np.float32)
    assert pcm.size > 0
    assert np.sqrt(np.mean(np.square(pcm))) > 180
    source = json.loads((ROOT / "showcase/audio_provenance_zh.json").read_text(encoding="utf-8"))
    assert source["voiceover"]["generated"] is False
    assert source["voiceover"]["provider"] == "Microsoft Edge TTS"
    assert "网络不可达" in source["voiceover"]["reason"]
    assert "Python" in source["music"] and "音效" in source["sound_effects"]
