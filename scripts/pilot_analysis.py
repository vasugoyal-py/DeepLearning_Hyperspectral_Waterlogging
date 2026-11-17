"""
Pilot survey analysis for DeepWaterLog.

Reads the Google Forms CSV export, computes the descriptive statistics that go
into the pilot section of the proposal and Slide 9 of the presentation, writes a
Markdown summary and draws a four-panel PNG chart.

Design rule: this script will not compute anything from fewer than 15 valid
responses, and it always excludes example rows (respondent_id beginning with
"EX"). That prevents an unfilled template from silently becoming a result.

Usage
-----
    python scripts/pilot_analysis.py --csv data/pilot/pilot_responses.csv
    python scripts/pilot_analysis.py --csv data/pilot/pilot_responses.csv --min-respondents 15

Outputs
-------
    data/pilot/pilot_summary.md
    data/pilot/pilot_slide9.png
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

DEFAULT_CSV = BASE_DIR / "data" / "pilot" / "pilot_responses.csv"
OUT_DIR = BASE_DIR / "data" / "pilot"

REQUIRED_COLUMNS = [
    "respondent_id", "consent", "age", "farm_size_acres", "crop",
    "irrigation_source", "years_farming", "waterlog_events_5yr",
    "affected_zone", "saturation_days", "yield_loss_pct", "detect_method",
    "used_app", "map_helpful", "main_obstacle", "share_photos", "village",
]

SATURATION_BANDS = ["1-3", "3-5", "5-7", "7-10", ">10"]


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------
def _norm(value: str) -> str:
    """Normalise a free-text answer for comparison."""
    return (value or "").strip().replace("\u2013", "-").replace("\u2014", "-")


def _num(value: str) -> float | None:
    """Parse a numeric answer, tolerating '%', 'acres' and stray spaces."""
    text = _norm(value)
    digits = "".join(ch for ch in text if ch.isdigit() or ch in ".-")
    try:
        return float(digits)
    except ValueError:
        return None


def _canon_saturation(value: str) -> str:
    text = _norm(value).replace(" ", "").replace("days", "")
    for band in SATURATION_BANDS:
        if text == band:
            return band
    if text.lower().startswith("morethan") or text in {">10", "10+"}:
        return ">10"
    if text in {"1-3", "3-5", "5-7", "7-10"}:
        return text
    return text or "unspecified"


def load_responses(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (valid_rows, excluded_rows)."""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                "CSV is missing required columns: " + ", ".join(missing) +
                "\nUse data/pilot/pilot_responses.csv as the schema template."
            )
        rows = [dict(r) for r in reader]

    valid: list[dict] = []
    excluded: list[dict] = []
    for row in rows:
        rid = _norm(row.get("respondent_id", ""))
        if not rid:
            continue
        if rid.upper().startswith("EX"):
            excluded.append({"reason": "example row", **row})
            continue
        if _norm(row.get("consent", "")).lower() not in {"yes", "y", "true", "1"}:
            excluded.append({"reason": "no consent", **row})
            continue
        valid.append(row)
    return valid, excluded


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _share(counter: Counter, key: str, total: int) -> float:
    return 100.0 * counter.get(key, 0) / total if total else 0.0


def compute(rows: list[dict]) -> dict:
    n = len(rows)
    ages = [v for v in (_num(r["age"]) for r in rows) if v is not None]
    farms = [v for v in (_num(r["farm_size_acres"]) for r in rows) if v is not None]
    years = [v for v in (_num(r["years_farming"]) for r in rows) if v is not None]
    events = [v for v in (_num(r["waterlog_events_5yr"]) for r in rows) if v is not None]
    losses = [v for v in (_num(r["yield_loss_pct"]) for r in rows) if v is not None]

    events_counter = Counter(str(int(e)) for e in events)
    zone_counter = Counter(_norm(r["affected_zone"]) for r in rows)
    sat_counter = Counter(_canon_saturation(r["saturation_days"]) for r in rows)
    crop_counter = Counter(_norm(r["crop"]).lower() for r in rows)
    irr_counter = Counter(_norm(r["irrigation_source"]).lower() for r in rows)
    detect_counter = Counter(_norm(r["detect_method"]).lower() for r in rows)
    app_counter = Counter(_norm(r["used_app"]).lower() for r in rows)
    map_counter = Counter(_norm(r["map_helpful"]).lower() for r in rows)
    obstacle_counter = Counter(_norm(r["main_obstacle"]).lower() for r in rows)
    share_counter = Counter(_norm(r["share_photos"]).lower() for r in rows)
    villages = Counter(_norm(r["village"]).lower() for r in rows if _norm(r["village"]))

    return {
        "n": n,
        "villages": len(villages),
        "village_list": [v.title() for v, _ in villages.most_common()],
        "age_mean": statistics.fmean(ages) if ages else None,
        "farm_mean": statistics.fmean(farms) if farms else None,
        "years_mean": statistics.fmean(years) if years else None,
        "events_mean": statistics.fmean(events) if events else None,
        "events_ge3_share": 100.0 * sum(1 for e in events if e >= 3) / n if n else 0.0,
        "events_ge4_share": 100.0 * sum(1 for e in events if e >= 4) / n if n else 0.0,
        "loss_mean": statistics.fmean(losses) if losses else None,
        "no_tool_share": _share(detect_counter, "no", n),
        "visual_only_share": 100.0 * sum(
            1 for r in rows if "visual" in _norm(r["detect_method"]).lower()
        ) / n if n else 0.0,
        "app_yes_share": _share(app_counter, "yes", n),
        "map_yes_share": _share(map_counter, "yes", n),
        "map_maybe_share": _share(map_counter, "maybe", n),
        "share_yes_share": _share(share_counter, "yes", n),
        "dominant_zone": zone_counter.most_common(1)[0][0] if zone_counter else "n/a",
        "dominant_obstacle": obstacle_counter.most_common(1)[0][0] if obstacle_counter else "n/a",
        "events_counter": events_counter,
        "zone_counter": zone_counter,
        "sat_counter": sat_counter,
        "crop_counter": crop_counter,
        "irr_counter": irr_counter,
        "detect_counter": detect_counter,
        "map_counter": map_counter,
        "obstacle_counter": obstacle_counter,
    }


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------
def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _f(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _distribution(counter: Counter, total: int) -> str:
    if not counter:
        return "n/a"
    parts = [
        f"{key.title() if key.islower() else key}: {count} ({100.0 * count / total:.1f}%)"
        for key, count in counter.most_common()
    ]
    return ", ".join(parts)


def build_summary(stats: dict, excluded: list[dict]) -> str:
    n = stats["n"]
    lines = [
        "# Pilot Survey Summary",
        "",
        f"**Valid responses:** {n}  ",
        f"**Villages:** {stats['villages']}" +
        (f" ({', '.join(stats['village_list'])})" if stats["village_list"] else "") + "  ",
        f"**Rows excluded:** {len(excluded)}",
        "",
    ]

    if excluded:
        reasons = Counter(item["reason"] for item in excluded)
        lines.append(
            "Excluded: " + ", ".join(f"{k} ({v})" for k, v in reasons.items())
        )
        lines.append("")

    lines += [
        "## Slide 9 headline numbers",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Respondents | {n} |",
        f"| Face recurring waterlogging (≥ 3 events in 5 years) | {_pct(stats['events_ge3_share'])} |",
        f"| Average self-reported yield loss in a bad year | {_pct(stats['loss_mean'])} |",
        f"| Use no pre-symptom detection tool | {_pct(stats['no_tool_share'])} |",
        f"| Want mobile waterlogging maps | {_pct(stats['map_yes_share'])} |",
        "",
        "## Basic profile",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Mean age | {_f(stats['age_mean'])} years |",
        f"| Mean farm size | {_f(stats['farm_mean'])} acres |",
        f"| Mean farming experience | {_f(stats['years_mean'])} years |",
        f"| Crops | {_distribution(stats['crop_counter'], n)} |",
        f"| Irrigation sources | {_distribution(stats['irr_counter'], n)} |",
        "",
        "## Waterlogging experience",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Mean waterlogging events in 5 years | {_f(stats['events_mean'])} |",
        f"| Share reporting ≥ 4 events | {_pct(stats['events_ge4_share'])} |",
        f"| Most affected zone | {stats['dominant_zone'].title()} |",
        f"| Saturation duration | {_distribution(stats['sat_counter'], n)} |",
        f"| Mean self-reported yield loss | {_pct(stats['loss_mean'])} |",
        "",
        "## Detection and demand",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Current detection method | {_distribution(stats['detect_counter'], n)} |",
        f"| Use no pre-symptom tool | {_pct(stats['no_tool_share'])} |",
        f"| Visual scouting only | {_pct(stats['visual_only_share'])} |",
        f"| Have used an app/drone/satellite service | {_pct(stats['app_yes_share'])} |",
        f"| Would use a field-scale map | {_distribution(stats['map_counter'], n)} |",
        f"| Main obstacle to drainage | {_distribution(stats['obstacle_counter'], n)} |",
        f"| Willing to share photos/coordinates | {_pct(stats['share_yes_share'])} |",
        "",
        "---",
        "",
        f"Generated by `scripts/pilot_analysis.py` from {n} valid responses. "
        "Do not edit these numbers by hand; re-run the script after correcting the CSV.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def _load_font(size: int):
    from PIL import ImageFont

    candidates = [
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _bar_panel(draw, box, title, counter: Counter, total: int, colour, font, small,
               keep_order: bool = False):
    """Draw one titled horizontal bar panel inside box = (x, y, w, h)."""
    x, y, w, h = box
    draw.text((x, y), title, fill=(16, 38, 59), font=font)
    if not counter:
        draw.text((x, y + 26), "no data", fill=(120, 140, 160), font=small)
        return

    items = list(counter.items()) if keep_order else counter.most_common(5)
    top = y + 28
    row_h = min(26, max(18, (h - 34) // max(1, len(items))))
    label_w = 108
    bar_max = max(1, w - label_w - 54)
    peak = max(count for _, count in items)

    for index, (label, count) in enumerate(items):
        row_y = top + index * row_h
        text = str(label).title() if str(label).islower() else str(label)
        draw.text((x, row_y + 3), text[:17], fill=(70, 90, 110), font=small)
        length = int(bar_max * count / peak)
        draw.rounded_rectangle(
            [x + label_w, row_y + 3, x + label_w + max(2, length), row_y + row_h - 5],
            radius=3, fill=colour,
        )
        share = 100.0 * count / total if total else 0.0
        draw.text(
            (x + label_w + length + 6, row_y + 3),
            f"{count} ({share:.0f}%)", fill=(70, 90, 110), font=small,
        )


def draw_chart(stats: dict, path: Path) -> str:
    from PIL import Image, ImageDraw

    width, height = 1200, 760
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    title_font = _load_font(30)
    sub_font = _load_font(17)
    panel_font = _load_font(20)
    small_font = _load_font(15)

    draw.text((36, 28), "Pilot survey — agricultural waterlogging", fill=(16, 38, 59), font=title_font)
    draw.text(
        (36, 66),
        f"n = {stats['n']} valid responses · {stats['villages']} village(s) · "
        "DeepWaterLog, GD Goenka University",
        fill=(91, 114, 136), font=sub_font,
    )

    # Headline strip
    headline = [
        ("Recurring waterlogging", f"{stats['events_ge3_share']:.0f}%", "#d62828"),
        ("Mean yield loss", f"{stats['loss_mean']:.0f}%" if stats['loss_mean'] is not None else "n/a", "#f4a261"),
        ("No detection tool", f"{stats['no_tool_share']:.0f}%", "#969696"),
        ("Want a map", f"{stats['map_yes_share']:.0f}%", "#40916c"),
    ]
    card_w = (width - 72 - 3 * 14) // 4
    for index, (label, value, colour) in enumerate(headline):
        cx = 36 + index * (card_w + 14)
        draw.rounded_rectangle([cx, 104, cx + card_w, 196], radius=10, fill=(244, 247, 250))
        draw.rectangle([cx, 104, cx + 5, 196], fill=colour)
        draw.text((cx + 18, 116), label.upper(), fill=(91, 114, 136), font=small_font)
        draw.text((cx + 18, 140), value, fill=(16, 38, 59), font=_load_font(34))

    # Panels
    pad = 36
    gap = 24
    panel_w = (width - 2 * pad - gap) // 2
    panel_h = 232
    top_y = 220

    # Events are categorical-but-ordinal: keep numeric order, not frequency order.
    events_ordered = Counter(
        dict(sorted(stats["events_counter"].items(), key=lambda kv: int(kv[0])))
    )

    boxes = [
        (pad, top_y, "Waterlogging events in last 5 years", events_ordered, "#2a6fb7", True),
        (pad + panel_w + gap, top_y, "Main obstacle to drainage", stats["obstacle_counter"], "#d62828", False),
        (pad, top_y + panel_h + gap, "Would a phone map help?", stats["map_counter"], "#40916c", False),
        (pad + panel_w + gap, top_y + panel_h + gap, "Detection method today", stats["detect_counter"], "#f4a261", False),
    ]
    for bx, by, label, counter, colour, keep_order in boxes:
        draw.rounded_rectangle(
            [bx, by, bx + panel_w, by + panel_h], radius=10,
            outline=(223, 231, 238), width=1, fill=(255, 255, 255),
        )
        _bar_panel(
            draw, (bx + 16, by + 14, panel_w - 32, panel_h - 28),
            label, counter, stats["n"], colour, panel_font, small_font,
            keep_order=keep_order,
        )

    draw.text(
        (36, height - 34),
        "Computed by scripts/pilot_analysis.py from collected responses — do not retype.",
        fill=(150, 166, 180), font=small_font,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return str(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse DeepWaterLog pilot survey responses.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Google Forms CSV export.")
    parser.add_argument("--min-respondents", type=int, default=15,
                        help="Refuse to report below this many valid responses.")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"CSV not found: {path}")
        print("Copy data/pilot/pilot_responses.csv, fill it, and pass --csv.")
        return 2

    rows, excluded = load_responses(path)
    n = len(rows)
    print(f"Valid responses : {n}")
    print(f"Excluded rows   : {len(excluded)}")
    for item in excluded[:5]:
        print(f"  - {item.get('respondent_id')}: {item['reason']}")

    if n < args.min_respondents:
        print("")
        print(f"STOP: {n} valid responses, minimum is {args.min_respondents}.")
        print("No summary or chart was written. Collect more responses, then re-run.")
        print("Example rows (EX*) are always excluded and never counted.")
        return 2

    stats = compute(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUT_DIR / "pilot_summary.md"
    summary_path.write_text(build_summary(stats, excluded), encoding="utf-8")
    chart_path = draw_chart(stats, OUT_DIR / "pilot_slide9.png")

    print("")
    print("SLIDE 9 HEADLINES")
    print(f"  Recurring waterlogging (>=3 in 5 yr) : {stats['events_ge3_share']:.1f}%")
    print(f"  Mean yield loss                      : {stats['loss_mean']:.1f}%")
    print(f"  Use no pre-symptom tool              : {stats['no_tool_share']:.1f}%")
    print(f"  Want mobile waterlogging maps        : {stats['map_yes_share']:.1f}%")
    print("")
    print(f"Summary : {summary_path}")
    print(f"Chart   : {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
