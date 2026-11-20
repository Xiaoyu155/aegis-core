# Aegis visual showcase

`aegis_demo.mp4`, `preview.gif`, and the four keyframes are generated from the
checked-in `STATUS.json`, `artifacts/`, and `datasets/` using
`scripts/build_aegis_showcase.py`. The script also refreshes the single-file
offline dashboard at `docs/aegis_showcase.html` and the evidence summary at
`showcase/data.json`.

To rebuild the MP4, GIF, and keyframes, use Python with Pillow and FFmpeg:

```sh
python scripts/build_aegis_showcase.py
```

To regenerate only the HTML dashboard and evidence summary:

```sh
python scripts/build_aegis_showcase.py --html-only
```

The dashboard uses no external scripts, fonts, or stylesheets. Its relative
preview links work when the repository is checked out as a directory. The
video uses H.264, 1920×1080, 30 fps, and yuv420p.

Some requested claims are absent from current main's evidence. The outputs
show the recorded v1.1.0 / 1,338-test / 15-branch state and explicitly identify
the scan post-deduplication count and calibrated model false-positive rate as
unreported.
