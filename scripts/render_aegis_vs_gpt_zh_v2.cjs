#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { once } = require("node:events");
const { chromium } = require("playwright");

const WIDTH = 1920;
const HEIGHT = 1080;
const FPS = 30;
const DURATION = 90;
const CUTS = [0, 12, 28, 46, 62, 76, 90];
const KEYFRAME_TIMES = [9.5, 26, 43, 60, 74, 88];
const FRAME_NAMES = [
  "01-开场破题.png",
  "02-候选对比.png",
  "03-风险守卫.png",
  "04-跨文件契约.png",
  "05-工程安全阀.png",
  "06-舰队交付.png",
];

async function main() {
  const [htmlPathArg, evidencePathArg, framesPathArg] = process.argv.slice(2);
  if (!htmlPathArg || !evidencePathArg || !framesPathArg) {
    throw new Error("usage: render_aegis_vs_gpt_zh_v2.cjs <html> <evidence.json> <frames-dir>");
  }
  const htmlPath = path.resolve(htmlPathArg);
  const evidence = JSON.parse(fs.readFileSync(path.resolve(evidencePathArg), "utf8"));
  const framesDir = path.resolve(framesPathArg);
  fs.mkdirSync(framesDir, { recursive: true });

  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-crash-reporter", "--disable-breakpad", "--font-render-hinting=slight"],
  });
  try {
    const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
    page.on("pageerror", error => { process.stderr.write(`浏览器渲染错误：${error.stack || error}\n`); });
    await page.setContent(fs.readFileSync(htmlPath, "utf8"), { waitUntil: "load" });
    const fontInfo = await page.evaluate(data => window.setEvidence(data), evidence);
    if (fontInfo.width !== WIDTH || fontInfo.height !== HEIGHT || !fontInfo.font) {
      throw new Error(`浏览器中文字库或画布尺寸异常：${JSON.stringify(fontInfo)}`);
    }
    const glyphCheck = await page.evaluate(() => window.validateGlyphs());
    process.stderr.write(`Chromium：${WIDTH}×${HEIGHT}；非 ASCII 字形像素检查 ${glyphCheck.checked} 个字符通过\n`);

    const totalFrames = FPS * DURATION;
    for (let i = 0; i < totalFrames; i++) {
      const t = i / FPS;
      const keyframe = KEYFRAME_TIMES.findIndex(keyTime => i === Math.round(keyTime * FPS));
      if (keyframe >= 0) {
        const pngData = await page.evaluate(time => window.renderDataUrl(time, "png"), t);
        fs.writeFileSync(path.join(framesDir, FRAME_NAMES[keyframe]), Buffer.from(pngData.split(",", 2)[1], "base64"));
      }
      const jpegData = await page.evaluate(time => window.renderDataUrl(time, "jpeg"), t);
      const bytes = Buffer.from(jpegData.split(",", 2)[1], "base64");
      if (!process.stdout.write(bytes)) await once(process.stdout, "drain");
      if (i % (FPS * 15) === 0) process.stderr.write(`逐帧渲染 ${Math.round(i / FPS)} / ${DURATION} 秒\n`);
    }
    process.stdout.end();
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
