#!/usr/bin/env python3
"""
Collect a batch of US weather forecast rows from the National Weather Service
API. Each row contains numeric weather signals plus narrative text fields so
the dataset can be used in the same "numbers + text" style as price/news data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OUTPUT_FILE = Path(__file__).resolve().with_name("step1_recent_weather_and_text.csv")
ALERT_SEPARATOR = " || "

DEFAULT_LOCATIONS: Sequence[tuple[str, float, float]] = (
    ("Washington DC", 38.8894, -77.0352),
    ("New York NY", 40.7128, -74.0060),
    ("Chicago IL", 41.8781, -87.6298),
    ("Houston TX", 29.7604, -95.3698),
    ("Los Angeles CA", 34.0522, -118.2437),
)

FIELDNAMES = [
    "collected_at_utc",
    "location_name",
    "input_latitude",
    "input_longitude",
    "resolved_city",
    "resolved_state",
    "forecast_office",
    "grid_x",
    "grid_y",
    "forecast_zone",
    "county",
    "period_type",
    "period_index",
    "period_number",
    "period_name",
    "start_time",
    "end_time",
    "is_daytime",
    "temperature",
    "temperature_unit",
    "temperature_trend",
    "wind_speed_raw",
    "wind_speed_mph",
    "wind_direction",
    "precipitation_probability_pct",
    "relative_humidity_pct",
    "dewpoint_c",
    "short_forecast",
    "detailed_forecast",
    "alert_count",
    "alert_events",
    "alert_severities",
    "alert_urgencies",
    "alert_certainties",
    "alert_headlines",
    "alert_summaries",
    "forecast_updated",
    "source_point_url",
    "source_forecast_url",
    "source_alerts_url",
]


@dataclass(frozen=True)
class LocationSpec:
    name: str
    latitude: float
    longitude: float


def clean_text(value: str) -> str:
    return " ".join((value or "").split())


def parse_location_spec(raw: str) -> LocationSpec:
    if ":" not in raw:
        raise ValueError(f"Location '{raw}' must look like NAME:LAT,LON.")
    name, coords = raw.rsplit(":", 1)
    lat_str, lon_str = [part.strip() for part in coords.split(",", 1)]
    return LocationSpec(
        name=clean_text(name),
        latitude=round(float(lat_str), 4),
        longitude=round(float(lon_str), 4),
    )


def load_locations_csv(path: Path) -> List[LocationSpec]:
    locations: List[LocationSpec] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = clean_text(row.get("name") or row.get("location") or row.get("city") or "")
            lat_raw = (row.get("latitude") or row.get("lat") or "").strip()
            lon_raw = (row.get("longitude") or row.get("lon") or row.get("lng") or "").strip()
            if not name or not lat_raw or not lon_raw:
                continue
            locations.append(
                LocationSpec(
                    name=name,
                    latitude=round(float(lat_raw), 4),
                    longitude=round(float(lon_raw), 4),
                )
            )
    return locations


def resolve_locations(csv_path: Optional[Path], inline_specs: Sequence[str]) -> List[LocationSpec]:
    resolved: List[LocationSpec] = []
    seen = set()

    def append(items: Iterable[LocationSpec]) -> None:
        for item in items:
            key = (item.name, item.latitude, item.longitude)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(item)

    if csv_path is not None:
        append(load_locations_csv(csv_path))
    if inline_specs:
        append(parse_location_spec(raw) for raw in inline_specs)
    if not resolved:
        append(LocationSpec(name, lat, lon) for name, lat, lon in DEFAULT_LOCATIONS)
    return resolved


def extract_trailing_id(url: str) -> str:
    if not url:
        return ""
    return url.rstrip("/").rsplit("/", 1)[-1]


def measurement_value(payload: object) -> Optional[float]:
    if isinstance(payload, dict):
        value = payload.get("value")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def parse_wind_speed_mph(raw: str) -> Optional[float]:
    matches = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", raw or "")]
    if not matches:
        return None
    return sum(matches) / len(matches)


def format_scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.3f}"
        return text.rstrip("0").rstrip(".")
    return str(value)


def format_coordinate(value: float) -> str:
    return f"{value:.4f}"


class NWSClient:
    def __init__(self, user_agent: str, timeout: int, request_pause: float) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.request_pause = request_pause

    def get_json(self, url: str) -> dict:
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/geo+json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"{url} returned HTTP {exc.code}: {clean_text(body)}") from exc
        except URLError as exc:
            raise RuntimeError(f"{url} failed: {exc}") from exc
        if self.request_pause > 0:
            time.sleep(self.request_pause)
        return payload


def summarize_alerts(alert_payload: dict) -> Dict[str, str]:
    features = alert_payload.get("features") or []
    events: List[str] = []
    severities: List[str] = []
    urgencies: List[str] = []
    certainties: List[str] = []
    headlines: List[str] = []
    summaries: List[str] = []

    for feature in features:
        props = feature.get("properties") or {}
        event = clean_text(props.get("event") or "")
        severity = clean_text(props.get("severity") or "")
        urgency = clean_text(props.get("urgency") or "")
        certainty = clean_text(props.get("certainty") or "")
        headline = clean_text(props.get("headline") or event)
        description = clean_text(props.get("description") or "")
        instruction = clean_text(props.get("instruction") or "")

        if event:
            events.append(event)
        if severity:
            severities.append(severity)
        if urgency:
            urgencies.append(urgency)
        if certainty:
            certainties.append(certainty)
        if headline:
            headlines.append(headline)

        summary_parts = [part for part in (event, headline, description, instruction) if part]
        if summary_parts:
            summaries.append(" | ".join(summary_parts))

    return {
        "alert_count": str(len(features)),
        "alert_events": ALERT_SEPARATOR.join(events),
        "alert_severities": ALERT_SEPARATOR.join(severities),
        "alert_urgencies": ALERT_SEPARATOR.join(urgencies),
        "alert_certainties": ALERT_SEPARATOR.join(certainties),
        "alert_headlines": ALERT_SEPARATOR.join(headlines),
        "alert_summaries": ALERT_SEPARATOR.join(summaries),
    }


def build_rows_for_periods(
    *,
    location: LocationSpec,
    point_url: str,
    point_properties: dict,
    forecast_url: str,
    forecast_payload: dict,
    alerts_url: str,
    alert_fields: Dict[str, str],
    period_type: str,
    limit: int,
    collected_at_utc: str,
) -> List[Dict[str, str]]:
    if limit <= 0:
        return []

    relative = ((point_properties.get("relativeLocation") or {}).get("properties") or {})
    city = clean_text(relative.get("city") or "")
    state = clean_text(relative.get("state") or "")
    forecast_office = clean_text(point_properties.get("gridId") or "")
    grid_x = point_properties.get("gridX")
    grid_y = point_properties.get("gridY")
    forecast_zone = extract_trailing_id(point_properties.get("forecastZone") or "")
    county = extract_trailing_id(point_properties.get("county") or "")

    rows: List[Dict[str, str]] = []
    properties = forecast_payload.get("properties") or {}
    periods = properties.get("periods") or []
    forecast_updated = clean_text(properties.get("updated") or "")

    for index, period in enumerate(periods[:limit], start=1):
        short_forecast = clean_text(period.get("shortForecast") or "")
        detailed_forecast = clean_text(period.get("detailedForecast") or short_forecast)
        period_name = clean_text(period.get("name") or period.get("startTime") or f"{period_type}_{index}")
        row = {
            "collected_at_utc": collected_at_utc,
            "location_name": location.name,
            "input_latitude": format_coordinate(location.latitude),
            "input_longitude": format_coordinate(location.longitude),
            "resolved_city": city,
            "resolved_state": state,
            "forecast_office": forecast_office,
            "grid_x": format_scalar(grid_x),
            "grid_y": format_scalar(grid_y),
            "forecast_zone": forecast_zone,
            "county": county,
            "period_type": period_type,
            "period_index": str(index),
            "period_number": format_scalar(period.get("number")),
            "period_name": period_name,
            "start_time": clean_text(period.get("startTime") or ""),
            "end_time": clean_text(period.get("endTime") or ""),
            "is_daytime": format_scalar(period.get("isDaytime")),
            "temperature": format_scalar(period.get("temperature")),
            "temperature_unit": clean_text(period.get("temperatureUnit") or ""),
            "temperature_trend": clean_text(period.get("temperatureTrend") or ""),
            "wind_speed_raw": clean_text(period.get("windSpeed") or ""),
            "wind_speed_mph": format_scalar(parse_wind_speed_mph(period.get("windSpeed") or "")),
            "wind_direction": clean_text(period.get("windDirection") or ""),
            "precipitation_probability_pct": format_scalar(
                measurement_value(period.get("probabilityOfPrecipitation"))
            ),
            "relative_humidity_pct": format_scalar(measurement_value(period.get("relativeHumidity"))),
            "dewpoint_c": format_scalar(measurement_value(period.get("dewpoint"))),
            "short_forecast": short_forecast,
            "detailed_forecast": detailed_forecast,
            "forecast_updated": forecast_updated,
            "source_point_url": point_url,
            "source_forecast_url": forecast_url,
            "source_alerts_url": alerts_url,
        }
        row.update(alert_fields)
        rows.append(row)
    return rows


def collect_weather_rows(
    *,
    client: NWSClient,
    location: LocationSpec,
    daily_periods: int,
    hourly_periods: int,
    collected_at_utc: str,
) -> List[Dict[str, str]]:
    point_url = f"https://api.weather.gov/points/{location.latitude:.4f},{location.longitude:.4f}"
    alerts_url = f"https://api.weather.gov/alerts/active?point={location.latitude:.4f},{location.longitude:.4f}"

    point_payload = client.get_json(point_url)
    point_properties = point_payload.get("properties") or {}
    daily_url = point_properties.get("forecast")
    hourly_url = point_properties.get("forecastHourly")
    if not daily_url or not hourly_url:
        raise RuntimeError(f"{point_url} did not return forecast URLs.")

    daily_payload = client.get_json(daily_url)
    hourly_payload = client.get_json(hourly_url)
    alert_payload = client.get_json(alerts_url)
    alert_fields = summarize_alerts(alert_payload)

    rows: List[Dict[str, str]] = []
    rows.extend(
        build_rows_for_periods(
            location=location,
            point_url=point_url,
            point_properties=point_properties,
            forecast_url=daily_url,
            forecast_payload=daily_payload,
            alerts_url=alerts_url,
            alert_fields=alert_fields,
            period_type="daily",
            limit=daily_periods,
            collected_at_utc=collected_at_utc,
        )
    )
    rows.extend(
        build_rows_for_periods(
            location=location,
            point_url=point_url,
            point_properties=point_properties,
            forecast_url=hourly_url,
            forecast_payload=hourly_payload,
            alerts_url=alerts_url,
            alert_fields=alert_fields,
            period_type="hourly",
            limit=hourly_periods,
            collected_at_utc=collected_at_utc,
        )
    )
    return rows


def write_rows(rows: Sequence[Dict[str, str]], output_path: str) -> None:
    if output_path == "-":
        handle = sys.stdout
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-collect US weather data with both text and numeric fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--locations-csv",
        type=Path,
        help="CSV file with columns like name,latitude,longitude.",
    )
    parser.add_argument(
        "--location",
        action="append",
        default=[],
        help="Inline location spec in the form NAME:LAT,LON. Repeatable.",
    )
    parser.add_argument(
        "--daily-periods",
        type=int,
        default=4,
        help="How many daily forecast periods to keep per location.",
    )
    parser.add_argument(
        "--hourly-periods",
        type=int,
        default=8,
        help="How many hourly forecast periods to keep per location.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_FILE),
        help="CSV output path. Use '-' to print CSV to stdout.",
    )
    parser.add_argument(
        "--user-agent",
        default="TruthFinderWeatherCollector/1.0",
        help="User-Agent header for api.weather.gov requests.",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--request-pause",
        type=float,
        default=0.25,
        help="Pause in seconds after each HTTP request.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.daily_periods <= 0 and args.hourly_periods <= 0:
        parser.error("At least one of --daily-periods or --hourly-periods must be > 0.")

    locations = resolve_locations(args.locations_csv, args.location)
    if not locations:
        raise RuntimeError("No valid locations were resolved.")

    client = NWSClient(
        user_agent=args.user_agent,
        timeout=args.timeout,
        request_pause=args.request_pause,
    )
    collected_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    all_rows: List[Dict[str, str]] = []
    failures: List[str] = []

    for location in locations:
        print(
            f"[collect] {location.name} ({location.latitude:.4f}, {location.longitude:.4f})",
            file=sys.stderr,
        )
        try:
            rows = collect_weather_rows(
                client=client,
                location=location,
                daily_periods=args.daily_periods,
                hourly_periods=args.hourly_periods,
                collected_at_utc=collected_at_utc,
            )
        except Exception as exc:  # pragma: no cover - exercised in live runs.
            failures.append(f"{location.name}: {exc}")
            print(f"[warn] {location.name}: {exc}", file=sys.stderr)
            continue
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No weather rows were collected.")

    write_rows(all_rows, args.output)
    print(
        f"[done] wrote {len(all_rows)} rows across {len(locations) - len(failures)} successful locations",
        file=sys.stderr,
    )
    if failures:
        print("[done] failed locations:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
