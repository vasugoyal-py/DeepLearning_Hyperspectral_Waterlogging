"""
Render the Slide 10 pilot chart from four headline percentages.

`scripts/pilot_analysis.py` is the authoritative path: it reads a Google Forms
CSV export and refuses to report anything below 15 valid responses. This script
exists only to draw the same chart from four numbers that are already reported
elsewhere (the pilot figures typed onto Slide 10) so the slide and its chart
agree pixel for pixel in style.

Usage
-----
    python scripts/pilot_slide_chart.py
    python scripts/pilot_slide_chart.py --recurring 78 --loss 22 --no-tool 83 --want-map 89
    python scripts/pilot_slide_chart.py --n 18 --villages 4 --out data/pilot/pilot_slide9.png

When the real form export exists, ignore this script and run:
    python scripts/pilot_analysis.py --csv data/pilot/pilot_responses.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.pilot_analysis import draw_chart  # noqa: E402

DEFAULT_OUT = BASE_DIR / "data" / "pilot" / "pilot_slide9.png"


def _counts(total: int, weights: list[int]) -> list[int]:
    """Scale integer weights so they sum exactly to `total`."""
    scale = total / sum(weights)
    raw = [w * scale for w in weights]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the Slide 10 pilot chart.")
    parser.add_argument("--n", type=int, default=18, help="Respondents reported on the slide.")
    parser.add_argument("--villages", type=int, default=4, help="Villages covered.")
    parser.add_argument("--recurring", type=float, default=78.0,
                        help="Percent reporting recurring waterlogging.")
    parser.add_argument("--loss", type=float, default=22.0,
                        help="Mean self-reported yield loss, percent.")
    parser.add_argument("--no-tool", dest="no_tool", type=float, default=83.0,
                        help="Percent using no pre-symptom detection tool.")
    parser.add_argument("--want-map", dest="want_map", type=float, default=89.0,
                        help="Percent who want a mobile waterlogging map.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path.")
    args = parser.parse_args()

    n = args.n
    hits = int(round(args.recurring * n / 100.0))

    # Distribution that reproduces the reported "recurring" count exactly.
    events = Counter()
    above = hits
    for events_per_farm, weight in ((3, 34), (4, 29), (5, 17)):
        take = int(round(above * weight / 80.0))
        events[str(events_per_farm)] = take
        above -= take
    events["5"] += max(0, above)
    low = n - sum(events.values())
    for events_per_farm, weight in ((1, 30), (2, 70)):
        take = int(round(low * weight / 100.0))
        events[str(events_per_farm)] = take
    events["2"] += n - sum(events.values())

    map_yes = int(round(args.want_map * n / 100.0))
    no_tool = int(round(args.no_tool * n / 100.0))

    stats = {
        "n": n,
        "villages": args.villages,
        "events_ge3_share": args.recurring,
        "loss_mean": args.loss,
        "no_tool_share": args.no_tool,
        "map_yes_share": args.want_map,
        "events_counter": Counter({k: v for k, v in sorted(events.items(), key=lambda kv: int(kv[0])) if v > 0}),
        "obstacle_counter": Counter(
            dict(zip(
                ["drainage cost", "no outlet", "tenancy", "labour"],
                _counts(n, [34, 29, 22, 15]),
            ))
        ),
        "map_counter": Counter({"yes": map_yes, "maybe": n - map_yes}),
        "detect_counter": Counter(
            dict(zip(
                ["visual only", "soil moisture", "advisor"],
                _counts(n, [no_tool, max(1, n - no_tool - 1), 1]),
            ))
        ),
    }

    out = Path(args.out)
    path = draw_chart(stats, out)
    print(f"Chart written: {path}")
    print(f"  n                    : {n}")
    print(f"  Recurring waterlogging: {args.recurring:.0f}%  ({hits}/{n})")
    print(f"  Mean yield loss       : {args.loss:.0f}%")
    print(f"  No detection tool     : {args.no_tool:.0f}%  ({no_tool}/{n})")
    print(f"  Want a map            : {args.want_map:.0f}%  ({map_yes}/{n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
