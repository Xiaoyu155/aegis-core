from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from scripts.build_aegis_vs_gpt_zh import (
    FRAME_NAMES,
    HEIGHT,
    WIDTH,
    Renderer,
    load_evidence,
    resolve_cjk_font,
    validate_glyph_pixels,
)


ROOT = Path(__file__).resolve().parents[1]


def test_showcase_uses_repository_evidence_without_invented_metrics():
    evidence = load_evidence(ROOT)
    assert evidence["tests_passed"] == evidence["tests_total"] == 1338
    assert evidence["ci_required"] == 8
    assert evidence["ci_success_records"] == 2
    assert evidence["sft_examples"] == 1462
    assert evidence["preference_pairs"] == 342
    assert evidence["profiler"]["hot_calls"] == 100000
    assert evidence["profiler"]["compatibility_backend_elapsed_seconds"] == 1.418
    assert evidence["fleet"]["delivery_branches"] == 15
    assert evidence["fleet"]["conflicts"] == 0
    assert evidence["fleet"]["default_branch_zero_pollution"] is True
    assert evidence["candidate_pair_count"] == 4
    assert evidence["chosen_score"] == 0.98969
    assert set(evidence["candidates"]) == {
        "job10_wrong", "job10_signature", "job10_test_cheat", "job10_memory_leak"
    }
    assert evidence["candidates"]["job10_test_cheat"]["score"] == 0.0
    assert evidence["candidates"]["job10_memory_leak"]["evidence"]["audit_findings"][0]["code"] == "UNBOUNDED_GROWTH"
    assert evidence["model_mode"] == "observe"
    assert evidence["risk_validation"]["support"] == 2
    assert evidence["risk_validation"]["false_positive"] == 2
    assert all((ROOT / source).is_file() for source in evidence["sources"])


def test_chinese_font_has_coverage_and_no_tofu_pixels():
    path, index = resolve_cjk_font(install_missing=False)
    renderer = Renderer(path, index, load_evidence(ROOT))
    validate_glyph_pixels(renderer)


def test_six_chinese_keyframes_and_mobile_preview_are_present_and_decodable():
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
        assert image.size == (640, 360)
        assert image.n_frames >= 12
    assert preview.stat().st_size < 3_000_000


def test_mp4_is_h264_1080p_30fps_yuv420p_when_ffprobe_is_available():
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("此运行环境未安装 ffprobe")
    video = ROOT / "showcase/aegis_vs_gpt_zh.mp4"
    assert video.is_file()
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate:format=duration",
            "-of", "default=noprint_wrappers=1", str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert metadata["codec_name"] == "h264"
    assert metadata["width"] == "1920"
    assert metadata["height"] == "1080"
    assert metadata["pix_fmt"] == "yuv420p"
    assert metadata["avg_frame_rate"] == "30/1"
    assert abs(float(metadata["duration"]) - 90.0) < 0.05
