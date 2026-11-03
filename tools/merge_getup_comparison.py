"""Cut the three getup comparison clips into one stylized versus-style video.

Timing is honest: a speed tag shows the playback rate, and every stamp uses
the simulated stable time from report.json.
"""
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1] / "runs/diagnostics/getup-compare-20261103"
W, H, FPS = 1920, 1080, 30
SRC_DT = 0.04  # sim seconds per source video frame
PANEL_W, PANEL_H, PANEL_Y = 640, 480, 190
LABEL_BAR = 36
BOLD = "C:/Windows/Fonts/msyhbd.ttc"
REG = "C:/Windows/Fonts/msyh.ttc"

SCENES = [
    ("仰躺", "supine_seed10000.mp4", "场景 1/3", "仰躺起身", "seed 10000"),
    ("趴地", "prone_seed10000.mp4", "场景 2/3", "趴地起身", "seed 10000"),
    ("随机", "random_seed10003.mp4", "场景 3/3", "随机初始姿势", "seed 10003"),
]
MODELS = [
    ("原模型 197M", "原模型", "PPO 197M 步", (255, 140, 60)),
    ("从零构造 getup_manual", "从零构造", "getup_manual", (90, 170, 255)),
    ("前向微调 human-stand", "前向微调", "human-stand", (190, 110, 255)),
]
GREEN, RED, BG = (60, 230, 120), (255, 70, 80), (12, 14, 20)
_fonts = {}
BACK = None


def font(size, bold=True):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(BOLD if bold else REG, size)
    return _fonts[key]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def text_c(draw, cx, y, text, size, fill, bold=True):
    f = font(size, bold)
    draw.text((cx - draw.textlength(text, font=f) / 2, y), text, font=f, fill=fill)


def background():
    y = np.linspace(0, 1, H)[:, None, None]
    base = np.array(BG) * (1 - y) + np.array((28, 22, 44)) * y
    img = np.broadcast_to(base, (H, W, 3)).astype(np.uint8).copy()
    img[::4] = (img[::4] * 0.85).astype(np.uint8)
    return Image.fromarray(img)


def glow_text(img, cx, cy, text, size, color, scale=1.0, alpha=1.0):
    size = max(8, int(size * scale))
    f = font(size)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tw = d.textlength(text, font=f)
    pos = (cx - tw / 2, cy - size * 0.65)
    d.text(pos, text, font=f, fill=color + (int(220 * alpha),), stroke_width=max(2, size // 14), stroke_fill=color + (int(220 * alpha),))
    layer = layer.filter(ImageFilter.GaussianBlur(size / 7))
    d = ImageDraw.Draw(layer)
    d.text(pos, text, font=f, fill=(255, 255, 255, int(255 * alpha)))
    img.alpha_composite(layer)


def intro_frames():
    n = int(FPS * 3.0)
    for i in range(n):
        t = i / FPS
        img = BACK.copy().convert("RGBA")
        d = ImageDraw.Draw(img)
        a = ease(t / 0.6)
        for k, (_, short, _, color) in enumerate(MODELS):
            x0 = int(-700 + (k * 640 + 710) * ease((t - 0.15 * k) / 0.7))
            d.rectangle([x0, 640, x0 + 620, 650], fill=color)
            text_c(d, x0 + 310, 670, short, 52, color)
        glow_text(img, W // 2, 330, "getup 模型表现对比", 120, (255, 210, 80), scale=1.2 - 0.2 * a, alpha=a)
        text_c(d, W // 2, 470, "仰躺 · 趴地 · 随机初始姿势    三个模型使用同一初始状态", 40, (220, 220, 230), bold=False)
        out = img.convert("RGB")
        if t > 2.6:
            out = Image.blend(out, Image.new("RGB", out.size, (255, 255, 255)), min(1.0, (t - 2.6) / 0.4) * 0.8)
        yield out


def round_card(tag, title, seed, first):
    n = int(FPS * 1.6)
    first = first.copy()
    first[:LABEL_BAR] = first[LABEL_BAR:LABEL_BAR + 1]
    preview = Image.fromarray(first).resize((W, int(W * PANEL_H / first.shape[1]))).convert("RGB")
    canvas = BACK.copy()
    canvas.paste(preview, (0, (H - preview.height) // 2))
    base = Image.blend(BACK, canvas, 0.3).convert("RGBA")
    for i in range(n):
        t = i / FPS
        img = base.copy()
        d = ImageDraw.Draw(img)
        slide = ease(t / 0.45)
        xe = int((W + 100) * slide)
        d.polygon([(0, 380), (xe, 380), (xe - 80, 720), (0, 720)], fill=(255, 200, 60, 240))
        x = int(-900 + (W // 2 + 900) * slide)
        text_c(d, x, 395, tag, 64, (40, 26, 10))
        text_c(d, x, 470, title, 130, (20, 12, 8))
        text_c(d, x, 650, seed + "  ·  三个模型同一初始状态", 34, (70, 48, 20), bold=False)
        yield img.convert("RGB")


def timeline(events, failed):
    """Sim times to show, always at real time."""
    end = 20.0 if failed else min(20.0, max(events) + 2.5)
    out, s = [], 0.0
    while s <= end - SRC_DT:
        out.append((s, 1.0))
        s += 1.0 / FPS
    return out


def draw_hud(img, scene, s, rate):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 150], fill=(8, 9, 14))
    d.rectangle([0, 148, W, 152], fill=(255, 200, 60))
    d.text((40, 26), scene[2], font=font(34), fill=(255, 200, 60))
    d.text((40, 68), scene[3], font=font(52), fill=(240, 240, 245))
    text_c(d, W // 2, 10, f"{s:05.2f}s", 92, (255, 255, 255))
    text_c(d, W // 2, 116, "仿真时间", 22, (150, 150, 160), bold=False)
    col = (120, 240, 160)
    label = "实时播放"
    d.rounded_rectangle([W - 340, 48, W - 40, 112], 14, outline=col, width=3)
    text_c(d, W - 190, 56, label, 32, col)
    d.rectangle([0, 152, int(W * s / 20.0), 160], fill=(255, 200, 60))


def compose(frame, scene, s, rate, results, end_phase=0.0):
    img = BACK.copy().convert("RGBA")
    d = ImageDraw.Draw(img)
    ranks = sorted([k for k, r in enumerate(results) if r["success"]], key=lambda k: results[k]["stable_start_seconds"])
    for k, (_, short, sub, color) in enumerate(MODELS):
        r = results[k]
        x0 = k * PANEL_W
        panel = frame[:, x0:x0 + PANEL_W].copy()
        panel[:LABEL_BAR] = panel[LABEL_BAR:LABEL_BAR + 1]
        p = Image.fromarray(panel).convert("RGBA")
        st = r["stable_start_seconds"]
        done = r["success"] and s >= st
        failed_show = (not r["success"]) and end_phase > 0
        border = color
        if done:
            border = GREEN
            dt = s - st
            if dt < 0.35:
                p = Image.blend(p, Image.new("RGBA", p.size, (255, 255, 255, 255)), 0.7 * (1 - dt / 0.35))
        if failed_show:
            border = RED
            p = Image.blend(p, Image.new("RGBA", p.size, (90, 0, 0, 255)), 0.4 * ease(end_phase))
        img.alpha_composite(p, (x0, PANEL_Y))
        d.rectangle([x0 + 2, PANEL_Y, x0 + PANEL_W - 3, PANEL_Y + PANEL_H], outline=border, width=6)
        cx, cy = x0 + PANEL_W // 2, PANEL_Y + PANEL_H // 2
        if done:
            sc = 1 + 0.8 * (1 - ease((s - st) / 0.3))
            glow_text(img, cx, PANEL_Y + 70, "已站稳", 54, GREEN, scale=sc)
        if failed_show:
            glow_text(img, cx, cy, "× 未站起", 80, RED, scale=1 + 0.6 * (1 - ease(end_phase)), alpha=ease(end_phase))
        y0 = PANEL_Y + PANEL_H + 24
        d.rounded_rectangle([x0 + 20, y0, x0 + PANEL_W - 20, y0 + 300], 20, fill=(20, 22, 32), outline=color, width=3)
        d.rectangle([x0 + 20, y0 + 24, x0 + 32, y0 + 120], fill=color)
        d.text((x0 + 52, y0 + 14), short, font=font(56), fill=color)
        d.text((x0 + 54, y0 + 88), sub, font=font(28, False), fill=(170, 170, 185))
        if done:
            status, scol = f"站稳用时  {st:.2f}s", GREEN
        elif failed_show:
            status, scol = f"失败 · 最终高度 {r['base_height']:.2f}m", RED
        else:
            status, scol = "起身中" + "." * (1 + int(s * 3) % 3), (230, 230, 240)
        d.text((x0 + 52, y0 + 150), status, font=font(46), fill=scol)
        if done:
            human = r["arm_mse"] < 0.05
            txt = "站姿：接近人形默认站姿" if human else f"站姿：手臂偏离（MSE {r['arm_mse']:.2f}）"
            d.text((x0 + 54, y0 + 222), txt, font=font(28, False), fill=(150, 230, 180) if human else (255, 190, 120))
    draw_hud(img, scene, s, rate)
    return img.convert("RGB")


def scene_frames(scene, report):
    data = report[scene[0]]["models"]
    results = [data[m[0]] for m in MODELS]
    reader = imageio.get_reader(ROOT / scene[1])
    frames = [f for f in reader]
    reader.close()
    yield from round_card(scene[2], scene[3], scene[4], frames[0])
    start = compose(frames[0], scene, 0.0, 1.0, results)
    for i in range(int(FPS * 1.2)):
        rgba = start.convert("RGBA")
        txt = "同一初始状态"
        j = i
        glow_text(rgba, W // 2, PANEL_Y + PANEL_H // 2, txt, 110, (255, 200, 60), scale=1.3 - 0.3 * ease(j / 8))
        yield rgba.convert("RGB")
    events = [r["stable_start_seconds"] for r in results if r["success"]]
    failed = any(not r["success"] for r in results)
    tl = timeline(events, failed)
    for s, rate in tl:
        yield compose(frames[min(int(round(s / SRC_DT)), len(frames) - 1)], scene, s, rate, results)
    last_s = tl[-1][0]
    last = frames[min(int(round(last_s / SRC_DT)), len(frames) - 1)]
    for i in range(int(FPS * 2.0)):
        yield compose(last, scene, last_s, 1.0, results, end_phase=i / (FPS * 0.5) if failed else 0.0)


def scoreboard(report):
    n = int(FPS * 7)
    cols = [700, 1100, 1500]
    for i in range(n):
        t = i / FPS
        img = BACK.copy().convert("RGBA")
        d = ImageDraw.Draw(img)
        glow_text(img, W // 2, 110, "结果汇总", 96, (255, 200, 60), scale=1.2 - 0.2 * ease(t / 0.4))
        for k, (_, short, sub, color) in enumerate(MODELS):
            a = ease((t - 0.2 - 0.15 * k) / 0.4)
            y = int(230 - 40 * (1 - a))
            text_c(d, cols[k], y, short, 50, color)
            text_c(d, cols[k], y + 66, sub, 26, (160, 160, 175), bold=False)
        for r, scene in enumerate(SCENES):
            a = ease((t - 0.8 - 0.35 * r) / 0.4)
            if a <= 0:
                continue
            y = 390 + r * 170
            off = int(200 * (1 - a))
            d.rounded_rectangle([60 - off, y - 10, W - 60 - off, y + 135], 18, fill=(22, 24, 36))
            d.text((110 - off, y + 30), scene[3], font=font(50), fill=(235, 235, 240))
            res = [report[scene[0]]["models"][m[0]] for m in MODELS]
            for k in range(3):
                cx = cols[k] - off
                if res[k]["success"]:
                    text_c(d, cx, y + 8, f"{res[k]['stable_start_seconds']:.2f}s 站稳", 52, GREEN)
                    tail = "人形站姿" if res[k]["arm_mse"] < 0.05 else "手臂乱伸"
                    text_c(d, cx, y + 86, "站姿：" + tail, 26, (180, 180, 190), bold=False)
                else:
                    text_c(d, cx, y + 8, "未站起", 52, RED)
                    text_c(d, cx, y + 86, f"最终高度 {res[k]['base_height']:.2f} m", 26, (180, 180, 190), bold=False)
        if t > 2.8:
            a = ease((t - 2.8) / 0.5)
            note = "注：随机起点只有一个种子。在 39 个起点上，微调模型 17/39，微调前 19/39，本片不代表总体成功率。"
            text_c(d, W // 2, 935, note, 28, tuple(int(c * a) for c in (200, 200, 210)), bold=False)
        out = img.convert("RGB")
        if t > n / FPS - 0.6:
            out = Image.blend(out, Image.new("RGB", out.size, (0, 0, 0)), (t - (n / FPS - 0.6)) / 0.6)
        yield out


def main():
    global BACK
    BACK = background()
    report = json.loads((ROOT / "report.json").read_text(encoding="utf-8"))
    out = ROOT / "getup_comparison_all.mp4"
    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    count = 0
    try:
        for part in [intro_frames()] + [scene_frames(s, report) for s in SCENES] + [scoreboard(report)]:
            for f in part:
                writer.append_data(np.asarray(f))
                count += 1
    finally:
        writer.close()
    print(out, round(count / FPS, 1), "s")


if __name__ == "__main__":
    main()
