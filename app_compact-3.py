"""Weather Risk & Outdoor Activity Planner — Open-Meteo + Gemini AI."""
import streamlit as st, requests, json, re, os
from datetime import datetime
import google.generativeai as genai

# ── CONSTANTS ─────────────────────────────────────────────
HISTORY_FILE, FAVOURITES_FILE, PLANS_FILE = "weather_history.json", "favourite_locations.json", "activity_plans.json"
ACTIVITIES = ["⚽ Football","🏃 Jogging","🌾 Farming","🧺 Picnic","✈️ Travelling","🎉 Outdoor Event"]
RISK_LABELS = ["✅ Safe","⚠️ Manageable","🚨 Risky","❌ Avoid"]
WMO = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Foggy",48:"Icy fog",51:"Light drizzle",53:"Moderate drizzle",
    55:"Dense drizzle",61:"Slight rain",63:"Moderate rain",65:"Heavy rain",71:"Slight snow",73:"Moderate snow",75:"Heavy snow",
    77:"Snow grains",80:"Slight showers",81:"Moderate showers",82:"Violent showers",85:"Slight snow showers",
    86:"Heavy snow showers",95:"Thunderstorm",96:"Thunderstorm w/ hail",99:"Thunderstorm w/ heavy hail"}
THR = {"⚽ Football":(35,5,50,5),"🏃 Jogging":(32,3,40,3),"🌾 Farming":(38,0,60,20),
       "🧺 Picnic":(33,10,30,1),"✈️ Travelling":(45,-10,80,50),"🎉 Outdoor Event":(33,8,35,2)}

# ── FILE HELPERS ──────────────────────────────────────────
def fload(p):
    try: return json.load(open(p)) if os.path.exists(p) else []
    except Exception: return []
def fappend(p, e, n=20):
    data = [d for d in (fload(p) or []) if d.get("location","")+d.get("date","") != e.get("location","")+e.get("date","")]
    json.dump([e]+data[:n-1], open(p,"w"), indent=2, default=str)

# ── WEATHER ───────────────────────────────────────────────
def geocode(name):
    name = re.sub(r"\s+"," ", re.sub(r"[^\w\s,.\-']","",name)).strip()
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
        params={"name":name,"count":1,"language":"en","format":"json"}, timeout=20)
    r.raise_for_status(); res = r.json().get("results")
    if not res: raise ValueError(f"Location '{name}' not found.")
    d=res[0]; return d.get("name",name), d["latitude"], d["longitude"], d.get("country",""), d.get("timezone","UTC")
def get_forecast(lat, lon, tz, days=7):
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=15, params={
        "latitude":lat,"longitude":lon,"timezone":tz,"forecast_days":days,
        "daily":"temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,weather_code,sunrise,sunset,uv_index_max",
        "hourly":"temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code,uv_index"})
    r.raise_for_status(); d,h = r.json()["daily"], r.json()["hourly"]
    mk = lambda x: x or 0
    hrow = lambda j: {"time":h["time"][j],"temp":mk(h["temperature_2m"][j]),"humidity":mk(h["relative_humidity_2m"][j]),
        "wind":mk(h["wind_speed_10m"][j]),"precip":mk(h["precipitation"][j]),"code":mk(h["weather_code"][j]),
        "uv":mk(h["uv_index"][j]),"cond":WMO.get(mk(h["weather_code"][j]),"Unknown"),"hour":int((h["time"][j] or "T00")[11:13])}
    drow = lambda i: {"date":d["time"][i],"temp_max":mk(d["temperature_2m_max"][i]),"temp_min":mk(d["temperature_2m_min"][i]),
        "precip":mk(d["precipitation_sum"][i]),"wind_max":mk(d["wind_speed_10m_max"][i]),"code":mk(d["weather_code"][i]),
        "sunrise":d["sunrise"][i],"sunset":d["sunset"][i],"uv_max":mk(d["uv_index_max"][i]),
        "cond":WMO.get(mk(d["weather_code"][i]),"Unknown"),"hourly":[hrow(j) for j in range(i*24,min(i*24+24,len(h["time"])))]}
    return [drow(i) for i in range(len(d["time"]))]

# ── RISK ──────────────────────────────────────────────────
def assess(activity, f):
    mt,mn,mw,mp = THR.get(activity,(40,0,50,5)); score,reasons = 0,[]
    checks = [(f["temp_max"]>mt,2,f"Too hot ({f['temp_max']:.0f}°C)"),(mt-5<f["temp_max"]<=mt,1,f"Warm ({f['temp_max']:.0f}°C)"),
        (f["temp_min"]<mn,2,f"Too cold ({f['temp_min']:.0f}°C)"),(f["wind_max"]>mw,2,f"Dangerous winds ({f['wind_max']:.0f} km/h)"),
        (mw*0.75<f["wind_max"]<=mw,1,f"Strong winds ({f['wind_max']:.0f} km/h)"),(f["precip"]>mp,2,f"Heavy rain ({f['precip']:.1f} mm)"),
        (0<f["precip"]<=mp,1,f"Some rain ({f['precip']:.1f} mm)"),(f["code"]>=95,3,"Thunderstorm forecasted"),
        (71<=f["code"]<=77,2,"Snow expected"),(f["uv_max"]>=8,1,f"Very high UV ({f['uv_max']:.0f})")]
    for cond,pts,msg in checks:
        if cond: score+=pts; reasons.append(msg)
    return min(score//2,3), reasons
def best_hours(activity, hourly):
    mt,mn = THR.get(activity,(30,10,0,0))[:2]; mid=(mt+mn)/2
    score = lambda h: (3 if h["code"]<3 else 1 if h["code"]<61 else -5 if h["code"]>=95 else 0)+(2 if h["precip"]==0 else 0)-abs(h["temp"]-mid)*0.1+(1 if h["wind"]<20 else 0)+(1 if 7<=h["hour"]<=19 else 0)
    return sorted(hourly, key=score, reverse=True)[:3]

# ── AI ────────────────────────────────────────────────────
def ai_analyze(api_key, activity, location, f, risk_level, risk_reasons, bh):
    label = ["Safe","Manageable","Risky","Avoid"][risk_level]
    hrs = ", ".join(f"{h['hour']:02d}:00 ({h['temp']:.0f}°C, {h['cond']})" for h in bh)
    prompt = (f"Weather safety expert. Activity:{activity} Location:{location} Date:{f['date']} Cond:{f['cond']} "
              f"Temp:{f['temp_min']:.0f}–{f['temp_max']:.0f}°C Rain:{f['precip']:.1f}mm Wind:{f['wind_max']:.0f}km/h "
              f"UV:{f['uv_max']:.0f} Risk:{label} Factors:{', '.join(risk_reasons) or 'None'} BestHours:{hrs}\n"
              'Reply ONLY valid JSON: {"summary":"...","advice":["","","",""],"packing":["","","","",""],"best_time_explanation":"..."}')
    genai.configure(api_key=api_key)
    try:
        text = re.sub(r"```(?:json)?|```","", genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt).text.strip()).strip()
        return json.loads(text)
    except Exception:
        return {"summary":f"{activity} on {f['date']}: {label}.","advice":["Check conditions.","Stay hydrated.","Dress appropriately.","Monitor updates."],
                "packing":["Water bottle","Clothing","First aid kit","Phone","Snacks"],"best_time_explanation":f"Best: {hrs}"}

# ── UI ────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Weather Activity Planner", page_icon="🌤️", layout="wide")
    st.title("🌤️ Weather Activity Planner"); st.caption("AI-powered weather risk analysis for outdoor activities")
    with st.sidebar:
        gemini_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="AIza...", help="https://aistudio.google.com")
        st.divider(); st.header("⭐ Favourites"); favs = fload(FAVOURITES_FILE)
        chosen = (st.selectbox("Load", ["— select —"]+[f["name"] for f in favs]) if favs else None)
        if chosen and chosen != "— select —": st.session_state["prefill"] = chosen
        if not favs: st.caption("None saved.")
        st.divider(); st.header("🕐 Recent"); history = fload(HISTORY_FILE)
        [st.text(f"{h['location']} · {h['date'][:10]}") for h in history[:5]]
        if not history: st.caption("No recent searches.")
    prefill = st.session_state.pop("prefill","")
    c1,c2,c3 = st.columns([2,1,1])
    loc_in = c1.text_input("📍 Location", value=prefill, placeholder="e.g. Abuja, Lagos, London")
    activity = c2.selectbox("🏃 Activity", ACTIVITIES); days_ahead = c3.slider("Days ahead", 0, 6, 0)
    ca,cb = st.columns(2); search_btn = ca.button("🔍 Analyse Weather", use_container_width=True, type="primary")
    if cb.button("⭐ Save Favourite", use_container_width=True) and loc_in.strip():
        fs = [f for f in fload(FAVOURITES_FILE) if isinstance(f,dict)]; name = loc_in.strip()
        if not any(f["name"]==name for f in fs):
            fs.append({"name":name,"saved":datetime.now().isoformat()}); json.dump(fs, open(FAVOURITES_FILE,"w"), indent=2); st.success(f"'{name}' saved!")
        else: st.info("Already in favourites.")
    if not search_btn: st.info("Choose a location & activity, then click **Analyse Weather**."); return
    if not loc_in.strip(): st.warning("Please enter a location."); st.stop()
    if not gemini_key: st.warning("Please enter your Gemini API key in the sidebar."); st.stop()
    try:
        with st.spinner("Fetching weather…"):
            loc_name, lat, lon, country, tz = geocode(loc_in.strip())
            forecasts = get_forecast(lat, lon, tz)
    except Exception as e: st.error(str(e)); st.stop()
    f = forecasts[days_ahead]; risk_level, risk_reasons = assess(activity, f); bh = best_hours(activity, f["hourly"])
    try:
        with st.spinner("Getting AI advice…"):
            ai = ai_analyze(gemini_key, activity, f"{loc_name}, {country}", f, risk_level, risk_reasons, bh)
    except Exception as e: st.error(str(e)); st.stop()
    fappend(HISTORY_FILE, {"location":f"{loc_name}, {country}","activity":activity,"date":f["date"],"risk_level":risk_level,"condition":f["cond"],"searched_at":datetime.now().isoformat()})
    st.divider(); st.header(f"Results — {loc_name}, {country}"); st.caption(f"{activity} · {f['date']}")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Condition",f["cond"]); m2.metric("Temp",f"{f['temp_min']:.0f}–{f['temp_max']:.0f}°C"); m3.metric("Rain",f"{f['precip']:.1f} mm"); m4.metric("Wind",f"{f['wind_max']:.0f} km/h")
    st.divider(); rc1,rc2 = st.columns([1,2])
    with rc1:
        st.subheader("Risk"); (st.success if risk_level==0 else st.warning if risk_level==1 else st.error)(RISK_LABELS[risk_level])
        for r in risk_reasons: st.write(f"- {r}")
    with rc2: st.subheader("AI Summary"); st.info(ai.get("summary",""))
    st.divider(); st.subheader("⏰ Best Times"); st.caption(ai.get("best_time_explanation",""))
    for col,h in zip(st.columns(3),bh):
        with col: st.metric(f"{h['hour']:02d}:00",f"{h['temp']:.0f}°C"); st.caption(f"{h['cond']} · 💨{h['wind']:.0f} · 💧{h['precip']:.1f}mm")
    st.divider(); ad,pk = st.columns(2)
    with ad:
         st.subheader("🛡️ Safety Advice")
         for t in ai.get("advice",[]):
            st.write(f"- {t}") 
    with pk:
        st.subheader("🎒 Packing"); ck = st.session_state.get(f"ck_{f['date']}_{activity}",{})
        for item in ai.get("packing",[]): ck[item]=st.checkbox(item,value=ck.get(item,False),key=f"c_{item}")
        st.session_state[f"ck_{f['date']}_{activity}"] = ck
    st.divider(); st.subheader("📅 7-Day Forecast")
    st.dataframe([{"Day":datetime.fromisoformat(d["date"]).strftime("%a"),"Date":d["date"],"Condition":d["cond"],
        "Temp":f"{d['temp_min']:.0f}–{d['temp_max']:.0f}°C","Risk":RISK_LABELS[assess(activity,d)[0]]}
        for d in forecasts], use_container_width=True, hide_index=True)
    st.divider()
    if st.button("💾 Save Plan"):
        fappend(PLANS_FILE,{"location":f"{loc_name}, {country}","activity":activity,"date":f["date"],"risk_level":risk_level,"condition":f["cond"],"summary":ai.get("summary",""),"advice":ai.get("advice",[]),"packing":ai.get("packing",[]),"saved_at":datetime.now().isoformat()},n=10)
        st.success("Plan saved to `activity_plans.json`.")

if __name__ == "__main__": main()
