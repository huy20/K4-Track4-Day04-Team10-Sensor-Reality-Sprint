"""Bảng và plot chính cho báo cáo Bước 6, dựng lại từ output đã lưu trong notebook benchmark.

Script KHÔNG chạy lại model. Nó đọc bảng kết quả (df.round(3)) và ảnh failure case đã nhúng
trong notebooks/camera_degradation_benchmark.ipynb, rồi ghi vào results/:
  benchmark_table.csv                     26 dòng kết quả (thiếu cột entropy: pandas đã ẩn khi in)
  main_figure.png                         mAP@50 và health score theo mức suy giảm, % so với baseline
  fallback_rule_check.csv                 các luật cảnh báo thử trên 20 điều kiện suy giảm
  failure_case_blur_sigma5p5.png          ảnh failure case gốc của notebook (4 ví dụ)
  failure_case_blur_sigma5p5_compact.png  2 ví dụ đặt cạnh nhau, dùng trong báo cáo

Chạy:  pip install matplotlib && python scripts/make_step6_figures.py
"""
import base64
import csv
import io
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "camera_degradation_benchmark.ipynb"
OUT = ROOT / "results"

DROP_LIMIT = 0.20   # "suy giảm" = mAP@50 tụt quá 20% so với baseline
HEALTH_MIN = 0.70   # luật 1: health score thấp
CONF_MIN = 0.85     # luật 2: mean max confidence thấp
RULES = {
    f"health<{HEALTH_MIN}": lambda r: r["health"] < HEALTH_MIN,
    f"mean_conf<{CONF_MIN}": lambda r: r["mean_conf"] < CONF_MIN,
    f"health<{HEALTH_MIN} OR mean_conf<{CONF_MIN}":
        lambda r: r["health"] < HEALTH_MIN or r["mean_conf"] < CONF_MIN,
}

# key trong DEGRADATIONS của notebook -> (tên hiển thị, nhãn trục x)
FAMILIES = {
    "blur_sigma": ("Gaussian blur (lệch nét)", "sigma (px)"),
    "motion_blur": ("Motion blur (rung)", "kernel (px)"),
    "expo_dark": ("Underexposure (thiếu sáng)", "gamma"),
    "expo_bright": ("Overexposure (cháy sáng)", "gamma"),
    "gauss_noise": ("Sensor noise (nhiễu)", "sigma (mức xám 0–255)"),
}

# 2 slot categorical đầu của bảng màu mặc định (đã qua kiểm tra mù màu) + màu chữ/khung
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
C_MAP, C_HEALTH = "#2a78d6", "#eb6834"
TEXT_COLS = {"degradation", "label", "unit"}


def code_cell(nb, marker):
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and marker in "".join(cell["source"]):
            return cell
    raise LookupError(f"Không thấy code cell chứa {marker!r} trong {NB_PATH.name}")


def stream_text(cell):
    return "".join("".join(o.get("text", "")) for o in cell.get("outputs", [])
                   if o.get("output_type") == "stream")


def cell_png(cell):
    for o in cell.get("outputs", []):
        png = o.get("data", {}).get("image/png")
        if png:
            return base64.b64decode("".join(png))
    raise LookupError("Cell không có ảnh PNG trong output")


def read_results_table(cell):
    html = next(("".join(o["data"]["text/html"]) for o in cell.get("outputs", [])
                 if "text/html" in o.get("data", {})), None)
    if html is None:
        raise LookupError("Cell quét degradation chưa có bảng kết quả, hãy chạy notebook trước")
    head, body = html.split("</thead>")
    columns = re.findall(r"<th>(.*?)</th>", head)[1:]   # bỏ cột index
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", body, flags=re.S):
        row = {}
        for col, val in zip(columns, re.findall(r"<td>(.*?)</td>", tr)):
            if col != "...":                             # cột pandas ẩn khi in
                row[col] = val if col in TEXT_COLS else float(val)
        row["level"] = int(row["level"])
        rows.append(row)
    return rows


def provenance(nb):
    setup = stream_text(code_cell(nb, "ultralytics.checks()"))
    data = stream_text(code_cell(nb, "project.version(VERSION).download"))
    splits = stream_text(code_cell(nb, 'for sp in ("train", "valid", "test")'))
    sweep = stream_text(code_cell(nb, "rows = [baseline]"))
    ultra = re.search(r"Ultralytics (\S+)", setup)
    ws, proj, ver = (re.search(rf"^\s*{k}: (\S+)", data, re.M)
                     for k in ("workspace", "project", "version"))
    full_val = re.search(r"valid:\s+(\d+) images", splits)
    subset = re.search(r"^\s*all\s+(\d+)\s+(\d+)\s", sweep, re.M)   # dòng val() đầu tiên
    try:
        commit = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(NB_PATH)],
                                cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = ""
    return dict(
        ultralytics=ultra.group(1) if ultra else "?",
        dataset=f"{ws.group(1)}/{proj.group(1)} v{ver.group(1)}" if ws and proj and ver else "?",
        n_full_val=int(full_val.group(1)) if full_val else None,
        n_images=int(subset.group(1)) if subset else None,
        n_instances=int(subset.group(2)) if subset else None,
        commit=commit or "?",
    )


def split_baseline(rows):
    # Baseline đúng = level 0 (ảnh chưa suy giảm) trên CÙNG tập con với các mức lỗi.
    # Dòng "clean" của notebook đo trên toàn bộ val nên không dùng để so sánh.
    level0 = [r for r in rows if r["degradation"] != "clean" and r["level"] == 0]
    base = level0[0]
    if any(r["mAP50"] != base["mAP50"] or r["health"] != base["health"] for r in level0):
        raise ValueError("Level 0 của các họ lỗi phải trùng nhau (cùng ảnh gốc)")
    conditions = [r for r in rows if r["degradation"] != "clean" and r["level"] > 0]
    return base, conditions


def pct(value, ref):
    return round(100 * value / ref, 1)


def write_table(rows, base, prov):
    cols = ["degradation", "label", "level", "value", "unit", "n_val_images",
            "mAP50", "mAP50_vs_baseline_pct", "mAP50_95", "precision", "recall",
            "mean_conf", "mean_boxes", "health", "health_vs_baseline_pct",
            "blur_health", "exposure_health", "entropy_health",
            "lap_var", "brightness", "contrast", "clip_low", "clip_high"]
    with open(OUT / "benchmark_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            clean = r["degradation"] == "clean"
            writer.writerow({
                **r,
                "n_val_images": prov["n_full_val"] if clean else prov["n_images"],
                "mAP50_vs_baseline_pct": "" if clean else pct(r["mAP50"], base["mAP50"]),
                "health_vs_baseline_pct": "" if clean else pct(r["health"], base["health"]),
            })


def print_worst(rows, base):
    print(f"Baseline (level 0): mAP50={base['mAP50']:.3f} recall={base['recall']:.3f} "
          f"boxes={base['mean_boxes']:.2f} conf={base['mean_conf']:.3f} health={base['health']:.3f}")
    for key in FAMILIES:
        r = max((r for r in rows if r["degradation"] == key), key=lambda r: r["level"])
        print(f"  {r['label']:14s} {r['unit']}={r['value']:<5g} mAP50={r['mAP50']:.3f} "
              f"({pct(r['mAP50'], base['mAP50']) - 100:+.0f}%) recall={r['recall']:.3f} "
              f"boxes={r['mean_boxes']:.2f} conf={r['mean_conf']:.3f} health={r['health']:.3f} "
              f"({pct(r['health'], base['health']) - 100:+.0f}%)")


def check_rules(conditions, base):
    limit = (1 - DROP_LIMIT) * base["mAP50"]
    with open(OUT / "fallback_rule_check.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "value", "unit", "mAP50", "mAP50_change_pct", "degraded",
                         "health", "mean_conf", *RULES])
        for r in conditions:
            writer.writerow([r["label"], r["value"], r["unit"], r["mAP50"],
                             round(pct(r["mAP50"], base["mAP50"]) - 100, 1), r["mAP50"] < limit,
                             r["health"], r["mean_conf"], *(rule(r) for rule in RULES.values())])
    degraded = [r for r in conditions if r["mAP50"] < limit]
    healthy = [r for r in conditions if r["mAP50"] >= limit]
    print(f"Suy giảm (mAP50 < {limit:.3f}): {len(degraded)}/{len(conditions)} điều kiện")
    names = lambda rs: ", ".join("%s %g" % (r["label"], r["value"]) for r in rs) or "-"
    for name, rule in RULES.items():
        missed = [r for r in degraded if not rule(r)]
        false_alarms = [r for r in healthy if rule(r)]
        print(f"  {name:32s} bắt {len(degraded) - len(missed)}/{len(degraded)} | sót: {names(missed)}"
              f" | báo nhầm {len(false_alarms)}/{len(healthy)}: {names(false_alarms)}")


def change_label(value_pct):
    return f"{value_pct - 100:+.0f}%".replace("-", "−")


def plot_main(rows, base, prov):
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(13, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    line_style = dict(lw=1.8, marker="o", ms=6.5, mec=SURFACE, mew=1.3,
                      solid_capstyle="round", solid_joinstyle="round")
    for ax, (key, (title, xlabel)) in zip(axes, FAMILIES.items()):
        fam = sorted((r for r in rows if r["degradation"] == key), key=lambda r: r["level"])
        x = [r["level"] for r in fam]
        series = [(C_MAP, [pct(r["mAP50"], base["mAP50"]) for r in fam]),
                  (C_HEALTH, [pct(r["health"], base["health"]) for r in fam])]
        ax.set_facecolor(SURFACE)
        ax.plot([-0.35, x[-1] + 0.12], [100, 100], color=INK_2, lw=0.8, zorder=1)   # baseline
        for color, y in series:
            ax.plot(x, y, color=color, zorder=3, **line_style)
        # nhãn thay đổi ở mức nặng nhất; tách ra nếu hai điểm cuối quá gần nhau
        ends = [y[-1] for _, y in series]
        label_y = list(ends)
        if abs(ends[0] - ends[1]) < 10:
            mid, upper = sum(ends) / 2, int(ends[1] > ends[0])
            label_y[upper], label_y[1 - upper] = mid + 5, mid - 5
        for end, y in zip(ends, label_y):
            ax.text(x[-1] + 0.22, y, change_label(end), fontsize=8.5, color=INK,
                    ha="left", va="center", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold", pad=8)
        ax.set_xticks(x, [f"{r['value']:g}" for r in fam])
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK_2)
        ax.set_xlim(-0.35, 5.0)
        ax.set_ylim(0, 125)
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelsize=8, length=0, pad=4)
    axes[0].set_ylabel("% so với baseline", fontsize=8.5, color=INK_2)

    fig.text(0.012, 0.965, "Health score theo kịp blur và thiếu sáng, nhưng không thấy sensor noise",
             fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.895,
             f"YOLOv8n trên {prov['n_images']} ảnh val. Đường ngang 100% = baseline: cùng "
             f"{prov['n_images']} ảnh, chưa suy giảm (mAP@50 = {base['mAP50']:.3f}; "
             f"health = {base['health']:.3f}).\nNhãn ở cuối mỗi đường = thay đổi tại mức nặng nhất.",
             fontsize=9, color=INK_2, va="top", linespacing=1.5)
    fig.legend(handles=[Line2D([], [], color=C_MAP, label="mAP@50 (detector)", **line_style),
                        Line2D([], [], color=C_HEALTH, label="Health score (giám sát ảnh)",
                               **line_style)],
               loc="upper right", bbox_to_anchor=(0.995, 0.985), ncol=2, frameon=False,
               fontsize=9, labelcolor=INK_2, handlelength=2.2)
    fig.text(0.012, 0.025,
             f"Nguồn: notebooks/{NB_PATH.name} (commit {prov['commit']}) · ảnh thật Roboflow "
             f"{prov['dataset']} + suy giảm mô phỏng · Ultralytics {prov['ultralytics']}, Kaggle T4"
             f" · {prov['n_images']} ảnh val đầu tiên theo tên file, {prov['n_instances']} đối tượng",
             fontsize=7.5, color=INK_2)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.73, bottom=0.2, wspace=0.14)
    fig.savefig(OUT / "main_figure.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def ink_runs(mask):
    runs, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def failure_images(nb):
    png = cell_png(code_cell(nb, "cases = failure_cases("))
    (OUT / "failure_case_blur_sigma5p5.png").write_bytes(png)
    im = Image.open(io.BytesIO(png)).convert("RGB")
    # Lưới 4 hàng x 2 cột (baseline | blur); mỗi hàng là một cặp dải (tiêu đề, ảnh)
    ink = np.asarray(im.convert("L")) < 245
    bands, cols = ink_runs(ink.any(axis=1)), ink_runs(ink.any(axis=0))
    if len(bands) % 2 or len(cols) != 2:
        raise ValueError(f"Không tách được lưới failure case ({len(bands)} dải, {len(cols)} cột)")
    examples = list(zip(bands[0::2], bands[1::2]))
    # Lấy ví dụ đầu và cuối; khe trong một cặp hẹp hơn khe giữa hai cặp để không ghép nhầm
    tiles = [im.crop((c0 - 4, title[0] - 6, c1 + 4, body[1] + 6))
             for title, body in (examples[0], examples[-1]) for c0, c1 in cols]
    gaps = [12, 56, 12]
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + sum(gaps),
                              max(t.height for t in tiles)), "white")
    x = 0
    for tile, gap in zip(tiles, gaps + [0]):
        sheet.paste(tile, (x, 0))
        x += tile.width + gap
    sheet.save(OUT / "failure_case_blur_sigma5p5_compact.png", optimize=True)


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    rows = read_results_table(code_cell(nb, "rows = [baseline]"))
    prov = provenance(nb)
    base, conditions = split_baseline(rows)
    print("Nguồn:", prov)
    write_table(rows, base, prov)
    print_worst(rows, base)
    check_rules(conditions, base)
    plot_main(rows, base, prov)
    failure_images(nb)
    print(f"Đã ghi vào {OUT.relative_to(ROOT)}/:", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
