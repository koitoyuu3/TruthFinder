#!/usr/bin/env python3
"""
Generate a 300-row CSV that mirrors the column layout of
nasdaq_external_news_300.csv, but uses live National Weather Service forecast
data to create long-form weather articles plus four short summary variants.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from step1_collect_weather_and_text import LocationSpec, NWSClient, clean_text, collect_weather_rows

OUTPUT_FILE = Path(__file__).resolve().with_name("weather_external_news_300.csv")

FIELDNAMES = [
    "Unnamed: 0",
    "Date",
    "Article_title",
    "Stock_symbol",
    "Url",
    "Publisher",
    "Author",
    "Article",
    "Lsa_summary",
    "Luhn_summary",
    "Textrank_summary",
    "Lexrank_summary",
]

DEFAULT_LOCATIONS: Sequence[tuple[str, str, float, float]] = (
    ("WDC", "Washington DC", 38.8894, -77.0352),
    ("NYC", "New York NY", 40.7128, -74.0060),
    ("BOS", "Boston MA", 42.3601, -71.0589),
    ("PHL", "Philadelphia PA", 39.9526, -75.1652),
    ("MIA", "Miami FL", 25.7617, -80.1918),
    ("ATL", "Atlanta GA", 33.7490, -84.3880),
    ("CHI", "Chicago IL", 41.8781, -87.6298),
    ("DET", "Detroit MI", 42.3314, -83.0458),
    ("MSP", "Minneapolis MN", 44.9778, -93.2650),
    ("STL", "St. Louis MO", 38.6270, -90.1994),
    ("DAL", "Dallas TX", 32.7767, -96.7970),
    ("HOU", "Houston TX", 29.7604, -95.3698),
    ("DEN", "Denver CO", 39.7392, -104.9903),
    ("PHX", "Phoenix AZ", 33.4484, -112.0740),
    ("LAX", "Los Angeles CA", 34.0522, -118.2437),
    ("SAN", "San Diego CA", 32.7157, -117.1611),
    ("SEA", "Seattle WA", 47.6062, -122.3321),
    ("PDX", "Portland OR", 45.5152, -122.6784),
    ("SFO", "San Francisco CA", 37.7749, -122.4194),
    ("LAS", "Las Vegas NV", 36.1699, -115.1398),
    ("SLC", "Salt Lake City UT", 40.7608, -111.8910),
    ("CLT", "Charlotte NC", 35.2271, -80.8431),
    ("BNA", "Nashville TN", 36.1627, -86.7816),
    ("OKC", "Oklahoma City OK", 35.4676, -97.5164),
    ("MSY", "New Orleans LA", 29.9511, -90.0715),
)


@dataclass(frozen=True)
class NewsLocation:
    code: str
    name: str
    latitude: float
    longitude: float

    def to_location_spec(self) -> LocationSpec:
        return LocationSpec(name=self.name, latitude=self.latitude, longitude=self.longitude)


def parse_location_spec(raw: str) -> NewsLocation:
    if "|" not in raw or ":" not in raw:
        raise ValueError(f"Location '{raw}' must look like CODE|NAME:LAT,LON.")
    code, remainder = raw.split("|", 1)
    name, coords = remainder.rsplit(":", 1)
    lat_str, lon_str = [part.strip() for part in coords.split(",", 1)]
    return NewsLocation(
        code=clean_text(code).upper(),
        name=clean_text(name),
        latitude=round(float(lat_str), 4),
        longitude=round(float(lon_str), 4),
    )


def load_locations_csv(path: Path) -> List[NewsLocation]:
    locations: List[NewsLocation] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = clean_text(row.get("code") or row.get("symbol") or "").upper()
            name = clean_text(row.get("name") or row.get("location") or row.get("city") or "")
            lat = (row.get("latitude") or row.get("lat") or "").strip()
            lon = (row.get("longitude") or row.get("lon") or row.get("lng") or "").strip()
            if not code or not name or not lat or not lon:
                continue
            locations.append(
                NewsLocation(
                    code=code,
                    name=name,
                    latitude=round(float(lat), 4),
                    longitude=round(float(lon), 4),
                )
            )
    return locations


def resolve_locations(csv_path: Optional[Path], inline_specs: Sequence[str]) -> List[NewsLocation]:
    resolved: List[NewsLocation] = []
    seen = set()

    def append(items: Iterable[NewsLocation]) -> None:
        for item in items:
            key = (item.code, item.name, item.latitude, item.longitude)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(item)

    if csv_path is not None:
        append(load_locations_csv(csv_path))
    if inline_specs:
        append(parse_location_spec(raw) for raw in inline_specs)
    if not resolved:
        append(NewsLocation(code, name, lat, lon) for code, name, lat, lon in DEFAULT_LOCATIONS)
    return resolved


def parse_dt(raw: str) -> Optional[datetime]:
    text = (raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def format_utc(raw: str, fallback_raw: str) -> str:
    dt = parse_dt(raw) or parse_dt(fallback_raw)
    if dt is None:
        return ""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def text_or_default(value: str, default: str) -> str:
    text = clean_text(value)
    return text if text else default


def number_or_default(value: str, default: str = "not available") -> str:
    text = clean_text(value)
    return text if text else default


def article_title(location: NewsLocation, row: Dict[str, str]) -> str:
    phase = text_or_default(row.get("period_name", ""), row.get("period_type", "forecast period"))
    short = text_or_default(row.get("short_forecast", ""), "Forecast update")
    temp = number_or_default(row.get("temperature", ""))
    unit = clean_text(row.get("temperature_unit", "")) or ""
    precip = number_or_default(row.get("precipitation_probability_pct", ""), "0")
    return f"{location.name} {phase} weather outlook: {short}, {temp}{unit}, precipitation chance {precip}%"


def next_row_of_same_type(row: Dict[str, str], same_type_rows: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    try:
        idx = max(int(row.get("period_index", "1")) - 1, 0)
    except ValueError:
        idx = 0
    next_idx = idx + 1
    if 0 <= next_idx < len(same_type_rows):
        return same_type_rows[next_idx]
    return None


def hourly_ranges(rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    temps: List[float] = []
    humidity: List[float] = []
    for row in rows:
        try:
            temps.append(float(row["temperature"]))
        except (KeyError, ValueError, TypeError):
            pass
        try:
            humidity.append(float(row["relative_humidity_pct"]))
        except (KeyError, ValueError, TypeError):
            pass
    payload = {
        "temp_range": "",
        "humidity_range": "",
    }
    if temps:
        payload["temp_range"] = f"{min(temps):.0f}-{max(temps):.0f}"
    if humidity:
        payload["humidity_range"] = f"{min(humidity):.0f}-{max(humidity):.0f}"
    return payload


def build_alert_sentence(row: Dict[str, str]) -> str:
    alert_count = clean_text(row.get("alert_count", "0")) or "0"
    if alert_count == "0":
        return "No active alerts were listed for the forecast point at collection time."
    headlines = text_or_default(row.get("alert_headlines", ""), "an active weather alert")
    return f"The forecast point had {alert_count} active alerts at collection time, including {headlines}."


def build_article(location: NewsLocation, row: Dict[str, str], location_rows: Sequence[Dict[str, str]]) -> str:
    same_type = [item for item in location_rows if item["period_type"] == row["period_type"]]
    hourly_rows = [item for item in location_rows if item["period_type"] == "hourly"][:4]
    daily_rows = [item for item in location_rows if item["period_type"] == "daily"][:4]
    next_row = next_row_of_same_type(row, same_type)
    phase = text_or_default(row.get("period_name", ""), row.get("period_type", "forecast period"))
    short = text_or_default(row.get("short_forecast", ""), "forecast conditions")
    detailed = text_or_default(row.get("detailed_forecast", ""), short)
    temp = number_or_default(row.get("temperature", ""))
    unit = clean_text(row.get("temperature_unit", "")) or ""
    wind_dir = text_or_default(row.get("wind_direction", ""), "variable")
    wind_speed = text_or_default(row.get("wind_speed_raw", ""), "light wind")
    precip = number_or_default(row.get("precipitation_probability_pct", ""), "0")
    humidity = number_or_default(row.get("relative_humidity_pct", ""))
    dewpoint = number_or_default(row.get("dewpoint_c", ""))
    ranges = hourly_ranges(hourly_rows)
    start_utc = format_utc(row.get("start_time", ""), row.get("collected_at_utc", ""))

    sentences = [
        f"The National Weather Service forecast for {location.name} starting {start_utc} says the {phase} period will bring {short}.",
        f"Temperatures are projected near {temp}{unit}, with {wind_dir} winds around {wind_speed} and a precipitation probability of {precip}%.",
        f"The official detailed forecast states: {detailed}",
    ]

    if humidity != "not available" or dewpoint != "not available":
        sentences.append(
            f"Moisture indicators around this window show relative humidity near {humidity}% and dewpoint around {dewpoint}C where available."
        )

    if ranges["temp_range"] or ranges["humidity_range"]:
        range_text = []
        if ranges["temp_range"]:
            range_text.append(f"hourly temperatures in the current forecast cycle range from {ranges['temp_range']}{unit}")
        if ranges["humidity_range"]:
            range_text.append(f"humidity ranges from {ranges['humidity_range']}%")
        sentences.append("Supporting hourly guidance suggests " + " while ".join(range_text) + ".")

    if next_row is not None:
        next_phase = text_or_default(next_row.get("period_name", ""), next_row.get("period_type", "next period"))
        next_short = text_or_default(next_row.get("short_forecast", ""), "updated conditions")
        next_temp = number_or_default(next_row.get("temperature", ""))
        next_unit = clean_text(next_row.get("temperature_unit", "")) or unit
        next_precip = number_or_default(next_row.get("precipitation_probability_pct", ""), "0")
        sentences.append(
            f"Looking ahead, {next_phase} is forecast to bring {next_short} with temperatures near {next_temp}{next_unit} and precipitation probability around {next_precip}%."
        )
    elif daily_rows:
        anchor = daily_rows[0]
        anchor_phase = text_or_default(anchor.get("period_name", ""), "the main daily period")
        anchor_short = text_or_default(anchor.get("short_forecast", ""), "steady conditions")
        anchor_temp = number_or_default(anchor.get("temperature", ""))
        anchor_unit = clean_text(anchor.get("temperature_unit", "")) or unit
        sentences.append(
            f"The broader daily outlook for {location.name} still points to {anchor_short} during {anchor_phase}, with temperatures around {anchor_temp}{anchor_unit}."
        )

    sentences.append(build_alert_sentence(row))
    return " ".join(clean_text(item) for item in sentences if clean_text(item))


def build_summaries(location: NewsLocation, row: Dict[str, str], location_rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    same_type = [item for item in location_rows if item["period_type"] == row["period_type"]]
    next_row = next_row_of_same_type(row, same_type)
    phase = text_or_default(row.get("period_name", ""), row.get("period_type", "forecast period"))
    short = text_or_default(row.get("short_forecast", ""), "forecast conditions")
    temp = number_or_default(row.get("temperature", ""))
    unit = clean_text(row.get("temperature_unit", "")) or ""
    wind_dir = text_or_default(row.get("wind_direction", ""), "variable")
    wind_speed = text_or_default(row.get("wind_speed_raw", ""), "light wind")
    precip = number_or_default(row.get("precipitation_probability_pct", ""), "0")
    humidity = number_or_default(row.get("relative_humidity_pct", ""))
    dewpoint = number_or_default(row.get("dewpoint_c", ""))
    detailed = text_or_default(row.get("detailed_forecast", ""), short)
    alert_sentence = build_alert_sentence(row)

    if next_row is not None:
        next_phase = text_or_default(next_row.get("period_name", ""), next_row.get("period_type", "next period"))
        next_short = text_or_default(next_row.get("short_forecast", ""), "updated conditions")
        next_temp = number_or_default(next_row.get("temperature", ""))
        next_unit = clean_text(next_row.get("temperature_unit", "")) or unit
        next_precip = number_or_default(next_row.get("precipitation_probability_pct", ""), "0")
        forward_text = (
            f"The next same-type period is {next_phase}, forecast to bring {next_short} near {next_temp}{next_unit} with precipitation around {next_precip}%."
        )
    else:
        forward_text = "No later same-type forecast period was included in this collection window."

    if humidity != "not available" and dewpoint != "not available":
        support_text = f"Supporting measurements list humidity near {humidity}% and dewpoint near {dewpoint}C."
    elif humidity != "not available":
        support_text = f"Supporting measurements list humidity near {humidity}%."
    elif dewpoint != "not available":
        support_text = f"Supporting measurements list dewpoint near {dewpoint}C."
    else:
        support_text = "This row does not include humidity or dewpoint measurements."

    return {
        "Lsa_summary": clean_text(
            f"NWS expects {short} in {location.name} during {phase}, with temperatures near {temp}{unit}, {wind_dir} winds around {wind_speed}, and precipitation probability near {precip}%. {forward_text}"
        ),
        "Luhn_summary": clean_text(
            f"{location.name} weather risk remains centered on {short}, {precip}% precipitation probability, and temperatures near {temp}{unit}. {alert_sentence}"
        ),
        "Textrank_summary": clean_text(
            f"The detailed outlook for {location.name} says: {detailed} {support_text}"
        ),
        "Lexrank_summary": clean_text(
            f"{phase} in {location.name} is forecast at {temp}{unit} with {wind_dir} winds near {wind_speed}; {forward_text}"
        ),
    }


def build_output_rows(
    client: NWSClient,
    locations: Sequence[NewsLocation],
    daily_periods: int,
    hourly_periods: int,
    target_rows: int,
) -> List[Dict[str, str]]:
    all_rows: List[Dict[str, str]] = []
    collected_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for location in locations:
        raw_rows = collect_weather_rows(
            client=client,
            location=location.to_location_spec(),
            daily_periods=daily_periods,
            hourly_periods=hourly_periods,
            collected_at_utc=collected_at_utc,
        )
        raw_rows = raw_rows[: daily_periods + hourly_periods]
        if len(raw_rows) < daily_periods + hourly_periods:
            raise RuntimeError(
                f"{location.name} returned only {len(raw_rows)} rows, expected at least {daily_periods + hourly_periods}."
            )

        for row in raw_rows:
            all_rows.append(
                {
                    "Article_title": article_title(location, row),
                    "Stock_symbol": location.code,
                    "Url": row.get("source_forecast_url", ""),
                    "Publisher": "NOAA National Weather Service",
                    "Author": "",
                    "Article": build_article(location, row, raw_rows),
                    **build_summaries(location, row, raw_rows),
                    "Date": format_utc(row.get("start_time", ""), row.get("collected_at_utc", "")),
                }
            )

    if len(all_rows) < target_rows:
        raise RuntimeError(f"Collected only {len(all_rows)} rows, fewer than target_rows={target_rows}.")

    finalized: List[Dict[str, str]] = []
    for index, row in enumerate(all_rows[:target_rows]):
        finalized.append(
            {
                "Unnamed: 0": f"{float(index):.1f}",
                "Date": row["Date"],
                "Article_title": row["Article_title"],
                "Stock_symbol": row["Stock_symbol"],
                "Url": row["Url"],
                "Publisher": row["Publisher"],
                "Author": row["Author"],
                "Article": row["Article"],
                "Lsa_summary": row["Lsa_summary"],
                "Luhn_summary": row["Luhn_summary"],
                "Textrank_summary": row["Textrank_summary"],
                "Lexrank_summary": row["Lexrank_summary"],
            }
        )
    return finalized


def write_rows(rows: Sequence[Dict[str, str]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a weather_news_300 CSV compatible with nasdaq_external_news_300.csv columns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--locations-csv", type=Path, help="CSV with code,name,latitude,longitude columns.")
    parser.add_argument(
        "--location",
        action="append",
        default=[],
        help="Inline location in the form CODE|NAME:LAT,LON. Repeatable.",
    )
    parser.add_argument("--daily-periods", type=int, default=4, help="Daily periods per location.")
    parser.add_argument("--hourly-periods", type=int, default=8, help="Hourly periods per location.")
    parser.add_argument("--target-rows", type=int, default=300, help="How many rows to write.")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Output CSV path.")
    parser.add_argument(
        "--user-agent",
        default="TruthFinderWeatherNews/1.0",
        help="User-Agent header for api.weather.gov requests.",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds.")
    parser.add_argument("--request-pause", type=float, default=0.25, help="Pause after each HTTP request.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.daily_periods <= 0 or args.hourly_periods <= 0:
        parser.error("--daily-periods and --hourly-periods must both be > 0.")

    locations = resolve_locations(args.locations_csv, args.location)
    client = NWSClient(
        user_agent=args.user_agent,
        timeout=args.timeout,
        request_pause=args.request_pause,
    )
    rows = build_output_rows(
        client=client,
        locations=locations,
        daily_periods=args.daily_periods,
        hourly_periods=args.hourly_periods,
        target_rows=args.target_rows,
    )
    write_rows(rows, args.output)
    print(f"[done] wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
