"""Weather Risk & Outdoor Activity Planner — compact edition.
Open-Meteo (weather) + Gemini (AI advice). Native Streamlit widgets only."""

import streamlit as st
import requests
import json
import re
import os
from datetime import datetime
from dataclasses import dataclass
import google.generativeai as genai

HISTORY_FILE, FAVOURITES_FILE, PLANS_FILE = "weather_history.json", "favourite_locations.json", "activity_plans.json"
ACTIVITIES = ["⚽ Football", "🏃 Jogging", "🌾 Farming", "🧺 Picnic", "✈️ Travelling", "🎉 Outdoor Event"]
RISK_LABELS = ["✅ Safe", "⚠️ Manageable", "🚨 Risky", "❌ Avoid"]

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain",
    65: "Heavy rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers", 85: "Slight snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Thunderstorm w/ heavy hail",
}

RISK_THRESHOLDS = {
    "⚽ Football": {"max_temp": 35, "min_temp": 5, "max_wind": 50, "max_precip": 5},
    "🏃 Jogging":  {"max_temp": 32, "min_temp": 3, "max_wind": 40, "max_precip": 3},
    "🌾 Farming":  {"max_temp": 38, "min_temp": 0, "max_wind": 60, "max_precip": 20},
    "🧺 Picnic":   {"max_temp": 33, "min_temp": 10, "max_wind": 30, "max_precip": 1},
    "✈️ Travelling": {"max_temp": 45, "min_temp": -10, "max_wind": 80, "max_precip": 50},
    "🎉 Outdoor Event": {"max_temp": 33, "min_temp": 8, "max_wind": 35, "max_precip": 2},
}


# ── DATA CLASSES ─────────────────────────────────────────
@dataclass
class HourlyForecast:
    time: str; temperature: float; humidity: float; wind_speed: float
    precipitation: float; weather_code: int; uv_index: float

    @property
    def condition(self) -> str: return WMO_CODES.get(self.weather_code, "Unknown")
    @property
    def hour(self) -> int: return int(self.time[11:13])


@dataclass
class DailyForecast:
    date: str; temp_max: float; temp_min: float; precipitation_sum: float
    wind_speed_max: float; weather_code: int; sunrise: str; sunset: str
    uv_index_max: float; hourly: list[HourlyForecast]

    @property
    def condition(self) -> str: return WMO_CODES.get(self.weather_code, "Unknown")


@dataclass
class LocationInfo:
    name: str; latitude: float; longitude: float; country: str; timezone: str


# ── FILE HANDLING ────────────────────────────────────────
class FileManager:
    @staticmethod
    def load(path: str) -> list | dict:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"Could not load {path}: {e}")
        return []

    @staticmethod
    def save(path: str, data: list | dict):
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            st.error(f"Could not save to {path}: {e}")

    @staticmethod
    def append_to_list(path: str, entry: dict, max_entries: int = 20):
        data = FileManager.load(path)
        if not isinstance(data, list):
            data = []
        key = entry.get("location", "") + entry.get("date", "")
        data = [d for d in data if (d.get("location", "") + d.get("date", "")) != key]
        data.insert(0, entry)
        FileManager.save(path, data[:max_entries])


# ── WEATHER CLIENT ───────────────────────────────────────
class WeatherClient:
    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def geocode(self, location_name: str) -> LocationInfo:
        clean_name = self._clean_location(location_name)
        try:
            resp = requests.get(self.GEO_URL, params={"name": clean_name, "count": 1, "language": "en", "format": "json"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("results"):
                raise ValueError(f"Location '{location_name}' not found. Try a different spelling.")
            r = data["results"][0]
            return LocationInfo(r.get("name", clean_name), r["latitude"], r["longitude"], r.get("country", ""), r.get("timezone", "UTC"))
        except requests.exceptions.ConnectionError:
            raise ConnectionError("No internet connection. Check your network and try again.")
        except requests.exceptions.Timeout:
            raise TimeoutError("The geocoding request timed out. Try again shortly.")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Geocoding API error: {e}")

    def get_forecast(self, loc: LocationInfo, days: int = 7) -> list[DailyForecast]:
        params = {
            "latitude": loc.latitude, "longitude": loc.longitude, "timezone": loc.timezone, "forecast_days": days,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,weather_code,sunrise,sunset,uv_index_max",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code,uv_index",
        }
        try:
            resp = requests.get(self.FORECAST_URL, params=params, timeout=15)
            resp.raise_for_status()
            return self._parse_forecast(resp.json())
        except requests.exceptions.ConnectionError:
            raise ConnectionError("No internet connection. Check your network.")
        except requests.exceptions.Timeout:
            raise TimeoutError("Weather data request timed out.")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Weather API error: {e}")
        except KeyError as e:
            raise RuntimeError(f"Unexpected weather data format: missing {e}")

    def _parse_forecast(self, data: dict) -> list[DailyForecast]:
        daily, hourly, days = data["daily"], data["hourly"], []
        for i, date in enumerate(daily["time"]):
            h0, h1 = i * 24, min(i * 24 + 24, len(hourly["time"]))
            hours = [HourlyForecast(
                hourly["time"][j], hourly["temperature_2m"][j] or 0, hourly["relative_humidity_2m"][j] or 0,
                hourly["wind_speed_10m"][j] or 0, hourly["precipitation"][j] or 0,
                hourly["weather_code"][j] or 0, hourly["uv_index"][j] or 0,
            ) for j in range(h0, h1)]
            days.append(DailyForecast(
                date, daily["temperature_2m_max"][i] or 0, daily["temperature_2m_min"][i] or 0,
                daily["precipitation_sum"][i] or 0, daily["wind_speed_10m_max"][i] or 0,
                daily["weather_code"][i] or 0, daily["sunrise"][i], daily["sunset"][i],
                daily["uv_index_max"][i] or 0, hours,
            ))
        return days

    @staticmethod
    def _clean_location(name: str) -> str:
        name = re.sub(r"[^\w\s,.\-']", "", name)
        return re.sub(r"\s+", " ", name).strip()


# ── ACTIVITY RISK ANALYZER ───────────────────────────────
class ActivityRiskAnalyzer:
    def assess(self, activity: str, f: DailyForecast) -> tuple[int, list[str]]:
        t = RISK_THRESHOLDS.get(activity, {})
        score, reasons = 0, []
        checks = [
            (f.temp_max > t.get("max_temp", 40), 2, f"Too hot ({f.temp_max:.0f}°C)"),
            (t.get("max_temp", 40) - 5 < f.temp_max <= t.get("max_temp", 40), 1, f"Warm conditions ({f.temp_max:.0f}°C)"),
            (f.temp_min < t.get("min_temp", 0), 2, f"Too cold ({f.temp_min:.0f}°C)"),
            (f.wind_speed_max > t.get("max_wind", 50), 2, f"Dangerous winds ({f.wind_speed_max:.0f} km/h)"),
            (t.get("max_wind", 50) * 0.75 < f.wind_speed_max <= t.get("max_wind", 50), 1, f"Strong winds ({f.wind_speed_max:.0f} km/h)"),
            (f.precipitation_sum > t.get("max_precip", 5), 2, f"Heavy precipitation ({f.precipitation_sum:.1f} mm)"),
            (0 < f.precipitation_sum <= t.get("max_precip", 5), 1, f"Some precipitation expected ({f.precipitation_sum:.1f} mm)"),
            (f.weather_code >= 95, 3, "Thunderstorm forecasted"),
            (71 <= f.weather_code <= 77, 2, "Snow expected"),
            (f.uv_index_max >= 8, 1, f"Very high UV ({f.uv_index_max:.0f})"),
        ]
        for condition, pts, reason in checks:
            if condition:
                score += pts
                reasons.append(reason)
        return min(score // 2, 3), reasons

    def best_hours(self, activity: str, hourly: list[HourlyForecast]) -> list[HourlyForecast]:
        t = RISK_THRESHOLDS.get(activity, {})
        mid = (t.get("max_temp", 30) + t.get("min_temp", 10)) / 2

        def hour_score(h: HourlyForecast) -> float:
            s = 3 if h.weather_code < 3 else 1 if h.weather_code < 61 else -5 if h.weather_code >= 95 else 0
            s += 2 if h.precipitation == 0 else 0
            s -= abs(h.temperature - mid) * 0.1
            s += 1 if h.wind_speed < 20 else 0
            s += 1 if 7 <= h.hour <= 19 else 0
            return s
        return sorted(hourly, key=hour_score, reverse=True)[:3]


# ── RECOMMENDATION ENGINE (Gemini) ───────────────────────
class RecommendationEngine:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def analyze(self, activity, location, f: DailyForecast, risk_level, risk_reasons, best_hours) -> dict:
        risk_label = ["Safe", "Manageable", "Risky", "Avoid"][risk_level]
        hour_strs = ", ".join(f"{h.hour:02d}:00 ({h.temperature:.0f}°C, {h.condition})" for h in best_hours)
        prompt = f"""You are a weather safety expert. Analyze this weather data and give practical advice.
Activity: {activity} | Location: {location} | Date: {f.date}
Weather: {f.condition} | Temp: {f.temp_min:.0f}–{f.temp_max:.0f}°C
Rain: {f.precipitation_sum:.1f} mm | Wind: {f.wind_speed_max:.0f} km/h | UV: {f.uv_index_max:.0f}
Risk Level: {risk_label} | Risk Factors: {', '.join(risk_reasons) if risk_reasons else 'None'}
Best Hours: {hour_strs}

Respond in valid JSON only, no markdown/code fences, with exactly these keys:
{{"summary": "2-3 sentence summary", "advice": ["tip1","tip2","tip3","tip4"],
"packing": ["item1","item2","item3","item4","item5"], "best_time_explanation": "one sentence"}}"""
        try:
            text = re.sub(r"```(?:json)?|```", "", self.model.generate_content(prompt).text.strip()).strip()
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "summary": f"Weather for {activity} on {f.date} is rated {risk_label}.",
                "advice": ["Check conditions before heading out.", "Stay hydrated.", "Dress appropriately.", "Monitor weather updates."],
                "packing": ["Water bottle", "Weather-appropriate clothing", "First aid kit", "Phone", "Snacks"],
                "best_time_explanation": f"Best windows: {hour_strs}",
            }
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")


# ── STREAMLIT UI (native widgets only) ───────────────────
def main():
    st.set_page_config(page_title="Weather Activity Planner", page_icon="🌤️", layout="wide")
    st.title("🌤️ Weather Activity Planner")
    st.caption("AI-powered weather risk analysis for outdoor activities")

    with st.sidebar:
        st.header("⚙️ Setup")
        gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...",
                                    help="Get a free key at https://aistudio.google.com")
        st.divider()
        st.header("⭐ Favourite Locations")
        favs = FileManager.load(FAVOURITES_FILE)
        if favs:
            chosen = st.selectbox("Load favourite", ["— select —"] + [f["name"] for f in favs])
            if chosen != "— select —":
                st.session_state["prefill_location"] = chosen
        else:
            st.caption("No favourites saved yet.")
        st.divider()
        st.header("🕐 Recent Searches")
        history = FileManager.load(HISTORY_FILE)
        if history:
            for h in history[:5]:
                st.text(f"{h['location']} · {h['date'][:10]}")
        else:
            st.caption("No recent searches.")

    prefill = st.session_state.pop("prefill_location", "")
    col1, col2, col3 = st.columns([2, 1, 1])
    location_input = col1.text_input("📍 Location", value=prefill, placeholder="e.g. Abuja, Lagos, London")
    activity = col2.selectbox("🏃 Activity", ACTIVITIES)
    days_ahead = col3.slider("Days ahead", 0, 6, 0)

    col_a, col_b = st.columns(2)
    search_btn = col_a.button("🔍 Analyse Weather", use_container_width=True, type="primary")
    save_fav = col_b.button("⭐ Save as Favourite", use_container_width=True)

    if save_fav and location_input.strip():
        favs = FileManager.load(FAVOURITES_FILE)
        if not isinstance(favs, list):
            favs = []
        name = location_input.strip()
        if not any(f["name"] == name for f in favs):
            favs.append({"name": name, "saved": datetime.now().isoformat()})
            FileManager.save(FAVOURITES_FILE, favs)
            st.success(f"'{name}' saved to favourites!")
        else:
            st.info("Already in favourites.")

    if not search_btn:
        st.info("Enter a location and choose an activity, then click **Analyse Weather** to begin.")
        return

    if not location_input.strip():
        st.warning("Please enter a location."); st.stop()
    if not gemini_key:
        st.warning("Please enter your Gemini API key in the sidebar."); st.stop()

    with st.spinner("Fetching weather data…"):
        try:
            client = WeatherClient()
            loc = client.geocode(location_input.strip())
            forecasts = client.get_forecast(loc)
        except (ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
            st.error(str(e)); st.stop()

    forecast = forecasts[days_ahead]
    analyzer = ActivityRiskAnalyzer()
    risk_level, risk_reasons = analyzer.assess(activity, forecast)
    best_hours = analyzer.best_hours(activity, forecast.hourly)

    with st.spinner("Asking AI for personalised advice…"):
        try:
            ai = RecommendationEngine(gemini_key).analyze(activity, f"{loc.name}, {loc.country}", forecast, risk_level, risk_reasons, best_hours)
        except RuntimeError as e:
            st.error(str(e)); st.stop()

    FileManager.append_to_list(HISTORY_FILE, {
        "location": f"{loc.name}, {loc.country}", "activity": activity, "date": forecast.date,
        "risk_level": risk_level, "condition": forecast.condition, "searched_at": datetime.now().isoformat(),
    })

    st.divider()
    st.header(f"Results for {loc.name}, {loc.country}")
    st.caption(f"Activity: {activity} · {forecast.date}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Condition", forecast.condition)
    m2.metric("Temperature", f"{forecast.temp_min:.0f}–{forecast.temp_max:.0f}°C")
    m3.metric("Rain", f"{forecast.precipitation_sum:.1f} mm")
    m4.metric("Wind", f"{forecast.wind_speed_max:.0f} km/h")

    st.divider()
    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.subheader("Risk Assessment")
        label = RISK_LABELS[risk_level]
        (st.success if risk_level == 0 else st.warning if risk_level == 1 else st.error)(label)
        if risk_reasons:
            st.write("**Risk factors:**")
            for r in risk_reasons:
                st.write(f"- {r}")
    with rc2:
        st.subheader("AI Summary")
        st.info(ai.get("summary", ""))

    st.divider()
    st.subheader("⏰ Best Times of Day")
    st.caption(ai.get("best_time_explanation", ""))
    for col, h in zip(st.columns(3), best_hours):
        with col:
            st.metric(f"{h.hour:02d}:00", f"{h.temperature:.0f}°C")
            st.caption(f"{h.condition} · 💨{h.wind_speed:.0f} km/h · 💧{h.precipitation:.1f} mm")

    st.divider()
    ad_col, pk_col = st.columns(2)
    with ad_col:
        st.subheader("🛡️ Safety Advice")
        for tip in ai.get("advice", []):
            st.write(f"- {tip}")
    with pk_col:
        st.subheader("🎒 Packing Checklist")
        ck_key = f"check_{forecast.date}_{activity}"
        checks = st.session_state.get(ck_key, {})
        for item in ai.get("packing", []):
            checks[item] = st.checkbox(item, value=checks.get(item, False), key=f"chk_{item}")
        st.session_state[ck_key] = checks

    st.divider()
    st.subheader("📅 7-Day Forecast Overview")
    rows = []
    for day in forecasts:
        lvl, _ = analyzer.assess(activity, day)
        rows.append({
            "Day": datetime.fromisoformat(day.date).strftime("%a"), "Date": day.date,
            "Condition": day.condition, "Temp (°C)": f"{day.temp_min:.0f}–{day.temp_max:.0f}",
            "Risk": RISK_LABELS[lvl],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    if st.button("💾 Save This Activity Plan"):
        plan = {
            "location": f"{loc.name}, {loc.country}", "activity": activity, "date": forecast.date,
            "risk_level": risk_level, "condition": forecast.condition, "summary": ai.get("summary", ""),
            "advice": ai.get("advice", []), "packing": ai.get("packing", []), "saved_at": datetime.now().isoformat(),
        }
        FileManager.append_to_list(PLANS_FILE, plan, max_entries=10)
        st.success("Plan saved! Find it in `activity_plans.json`.")


if __name__ == "__main__":
    main()
