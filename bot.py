#!/usr/bin/env python3
"""
bot.py
Institutional Smart-Money Bot for XAU/USD (TwelveData + Telegram).
- Auto-activates during New York session (PKT 17:30 - 22:30).
- HTF bias (4H/1H), session setup (15m/5m), confirmation (1m-3m).
- Detects liquidity-sweeps, simple FVGs, MSS; builds trade plan and alerts Telegram.
- Inline buttons (Accept / Ignore) via getUpdates polling (basic).
Env vars required:
  TWELVE_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
Optional:
  MIN_IFS_SCORE (default 70), POLL_INTERVAL (default 60), LOOKBACK_BARS (default 20)
"""

import os
import time
import json
import math
import requests
import logging
import threading
from datetime import datetime, timedelta, timezone, time as dtime
import pandas as pd
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from typing import Optional, Dict, List

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# -------------------- Configuration --------------------
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "").strip()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not TWELVE_API_KEY:
    logging.error("TWELVE_API_KEY not set. Exiting.")
    raise SystemExit("TWELVE_API_KEY not set")
if not TELEGRAM_TOKEN:
    logging.error("TELEGRAM_TOKEN not set. Exiting.")
    raise SystemExit("TELEGRAM_TOKEN not set")
if not TELEGRAM_CHAT_ID:
    logging.error("TELEGRAM_CHAT_ID not set. Exiting.")
    raise SystemExit("TELEGRAM_CHAT_ID not set")

MIN_IFS_SCORE = int(os.getenv("MIN_IFS_SCORE", "70"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))  # in seconds for minute analysis
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "20"))

SYMBOL = os.getenv("SYMBOL", "XAU/USD")
TWELVE_BASE = "https://api.twelvedata.com/time_series"

# Session times (PKT)
PKT = pytz.timezone("Asia/Karachi")
NY_START_PKT = dtime(hour=17, minute=30)  # PKT 17:30
NY_END_PKT   = dtime(hour=22, minute=30)  # PKT 22:30

# Telegram API base
TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Files for offsets/UIDs (persist across restarts)
LAST_OFFSET_FILE = "/tmp/lf_offset.txt"
ALERT_UID_FILE = "/tmp/ifs_uid.txt"

# -------------------- Utility / Data fetch --------------------
def twelvedata_get_series(symbol: str, interval: str = "1min", outputsize: int = 300) -> List[Dict]:
    """Fetch time-series from TwelveData (returns oldest-first list)."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
        "apikey": TWELVE_API_KEY
    }
    r = requests.get(TWELVE_BASE, params=params, timeout=12)
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"TwelveData error: {data}")
    # values are newest-first, reverse to oldest-first
    vals = list(reversed(data["values"]))
    # coerce numeric fields
    for v in vals:
        for k in ("open","high","low","close","volume"):
            if k in v:
                try:
                    v[k] = float(v[k])
                except:
                    v[k] = 0.0
    return vals

def to_df(candles: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    if df.empty:
        return df
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    return df

# -------------------- Smart Money heuristics --------------------
def detect_htf_bias() -> Optional[str]:
    """HTF bias using 1H and 4H sweeps + SMA fallback (returns 'BULL','BEAR', or None)."""
    try:
        df1h = to_df(twelvedata_get_series(SYMBOL, interval="60min", outputsize=240))
        df4h = to_df(twelvedata_get_series(SYMBOL, interval="240min", outputsize=240))
    except Exception as e:
        logging.warning("HTF fetch error: %s", e)
        return None

    def bias_from_df(df: pd.DataFrame):
        if len(df) < 6:
            return "NEUTRAL"
        recent_high = df['high'].iloc[-20:-1].max() if len(df) >= 21 else df['high'].max()
        recent_low  = df['low'].iloc[-20:-1].min() if len(df) >= 21 else df['low'].min()
        last = df.iloc[-1]
        # sweep high then close back inside -> bearish
        if last['high'] > recent_high and last['close'] < recent_high:
            return "BEAR"
        # sweep low then close back inside -> bullish
        if last['low'] < recent_low and last['close'] > recent_low:
            return "BULL"
        # SMA fallback
        if len(df) >= 200:
            sma50 = df['close'].rolling(50).mean().iloc[-1]
            sma200 = df['close'].rolling(200).mean().iloc[-1]
            if sma50 > sma200: return "BULL"
            if sma50 < sma200: return "BEAR"
        return "NEUTRAL"

    b1 = bias_from_df(df1h)
    b4 = bias_from_df(df4h)
    if b1 == b4:
        return b1
    if b1 != "NEUTRAL":
        return b1
    return b4 if b4 != "NEUTRAL" else None

def detect_fvg_simple(df: pd.DataFrame, lookback: int = 8) -> List[Dict]:
    """
    Simple Fair Value Gap-like heuristic:
    If there is a 3-bar gap structure where middle bar leaves an imbalance,
    mark its index. Works as heuristic only.
    """
    res = []
    if len(df) < 3:
        return res
    for i in range(2, len(df)):
        # gap up FVG (bullish) => low[i-2] > high[i-1] & low[i-2] > high[i]
        try:
            if df['low'].iat[i-2] > df['high'].iat[i-1] and df['low'].iat[i-2] > df['high'].iat[i]:
                res.append({"index": i, "type": "FVG_UP"})
            if df['high'].iat[i-2] < df['low'].iat[i-1] and df['high'].iat[i-2] < df['low'].iat[i]:
                res.append({"index": i, "type": "FVG_DN"})
        except Exception:
            continue
    return res

def detect_sweep_and_confirmation(df15: pd.DataFrame, df5: pd.DataFrame, lookback=8) -> Dict:
    """
    Detect liquidity sweep (sweep high/low) on recent 15m candles and check 5m/1m confirmation.
    Conservative: look for a candle that pierces recent high/low then closes back inside,
    then a following bullish/bearish confirmation candle.
    """
    result = {"signal": False}
    if len(df15) < lookback + 2:
        result["reason"] = "not_enough_15m"
        return result

    # examine candidate sweep in last lookback candles (exclude last candle which may be incomplete)
    recent_window = df15.iloc[-(lookback+1):-1].reset_index(drop=True)
    for i in range(len(recent_window)):
        c = recent_window.iloc[i]
        # recent highs/lows excluding this candle
        prev_high = recent_window['high'].drop(index=i).max()
        prev_low  = recent_window['low'].drop(index=i).min()
        # Sweep high: this candle high > prev_high and close < prev_high (pierce then close inside)
        if c['high'] > prev_high and c['close'] < prev_high:
            # check next 15m candle is bearish confirmation or 5m shows rejection
            # use the immediate next 15m (if available)
            idx_next = i + 1
            if idx_next < len(recent_window):
                nxt = recent_window.iloc[idx_next]
                if nxt['close'] < nxt['open']:
                    result.update({"signal":True, "type":"SWEEP_HIGH", "sweep_candle":c.to_dict(), "confirm_candle":nxt.to_dict()})
                    return result
            # also check 5m micro confirmation (last few 5m)
            if len(df5) >= 3:
                last5 = df5.iloc[-3:].reset_index(drop=True)
                # if last 5m shows strong red candle
                if last5.iloc[-1]['close'] < last5.iloc[-1]['open']:
                    result.update({"signal":True, "type":"SWEEP_HIGH", "sweep_candle":c.to_dict(), "confirm_candle": last5.iloc[-1].to_dict()})
                    return result

        # Sweep low
        if c['low'] < prev_low and c['close'] > prev_low:
            idx_next = i + 1
            if idx_next < len(recent_window):
                nxt = recent_window.iloc[idx_next]
                if nxt['close'] > nxt['open']:
                    result.update({"signal":True, "type":"SWEEP_LOW", "sweep_candle":c.to_dict(), "confirm_candle":nxt.to_dict()})
                    return result
            if len(df5) >= 3:
                last5 = df5.iloc[-3:].reset_index(drop=True)
                if last5.iloc[-1]['close'] > last5.iloc[-1]['open']:
                    result.update({"signal":True, "type":"SWEEP_LOW", "sweep_candle":c.to_dict(), "confirm_candle": last5.iloc[-1].to_dict()})
                    return result

    result["reason"] = "no_sweep"
    return result

def detect_mss_on_1m(df1: pd.DataFrame, lookback=10) -> Optional[str]:
    """Simple MSS detection on 1m: if latest close > prior highs -> BULL_MSS, if < prior lows -> BEAR_MSS."""
    if len(df1) < lookback + 1:
        return None
    highs = df1['high'].iloc[-lookback-1:-1].max()
    lows  = df1['low'].iloc[-lookback-1:-1].min()
    last = df1.iloc[-1]
    if last['close'] > highs:
        return "BULL_MSS"
    if last['close'] < lows:
        return "BEAR_MSS"
    return None

def compute_ifs_score_components(has_sweep: bool, has_fvg: bool, vol_spike: bool, htf_bias: Optional[str], mss_flag: Optional[str]):
    # base scoring (heuristic)
    score = 0
    score += 30 if has_sweep else 0
    score += 25 if has_fvg else 0
    score += 15 if vol_spike else 0
    score += 10 if mss_flag else 0
    # HTF alignment bonus
    if htf_bias and mss_flag:
        if (htf_bias == "BULL" and mss_flag.startswith("BULL")) or (htf_bias == "BEAR" and mss_flag.startswith("BEAR")):
            score += 20
    return min(100, int(score))

# -------------------- Trade plan builder --------------------
def build_trade_plan(symbol: str, sweep_candle: Dict, confirm_candle: Dict, sweep_type: str, htf_bias: Optional[str]) -> Dict:
    """
    Build entry, SL and TP levels.
    - SL = sweep candle extreme (+ small buffer)
    - Entry = confirm candle close or retest mid of sweep/confirm
    - TP1 = internal liquidity (small structure) -> set as 1x distance
    - TP2 = HTF target (bigger) -> 2-3x distance
    """
    pip = 0.01  # XAU pip convention used here
    sweep_low = sweep_candle.get("low")
    sweep_high = sweep_candle.get("high")
    confirm_close = confirm_candle.get("close")
    if sweep_type == "SWEEP_LOW":
        sl = (sweep_low) - (0.5 * pip)  # small buffer below sweep low
        entry = max(confirm_close, (confirm_close + sweep_low) / 2)
        dist = entry - sl
        tp1 = entry + dist * 1.0
        tp2 = entry + dist * 2.5
        direction = "LONG"
    else:
        sl = (sweep_high) + (0.5 * pip)
        entry = min(confirm_close, (confirm_close + sweep_high) / 2)
        dist = sl - entry
        tp1 = entry - dist * 1.0
        tp2 = entry - dist * 2.5
        direction = "SHORT"

    return {
        "symbol": symbol,
        "direction": direction,
        "entry": round(entry, 3),
        "sl": round(sl, 3),
        "tp1": round(tp1, 3),
        "tp2": round(tp2, 3),
        "logic": f"HTF {htf_bias or 'N/A'} + {sweep_type} + MSS/FVG confirmation"
    }

# -------------------- Telegram helpers --------------------
def next_uid():
    try:
        if os.path.exists(ALERT_UID_FILE):
            n = int(open(ALERT_UID_FILE).read().strip()) + 1
        else:
            n = 1
        open(ALERT_UID_FILE, "w").write(str(n))
        return str(n)
    except:
        return str(int(time.time()))

def send_telegram_plan(plan: Dict, ifs_score: int):
    """Send an alert containing plan with inline keyboard."""
    uid = next_uid()
    emoji = "🟢" if plan['direction'] == "LONG" else "🔴"
    text = (f"{emoji} <b>SMART-MONEY {plan['direction']} SETUP</b>\n"
            f"Pair: <b>{plan['symbol']}</b>\n"
            f"IFS Score: {ifs_score}/100\n"
            f"Entry: <code>{plan['entry']}</code>\nSL: <code>{plan['sl']}</code>\nTP1: <code>{plan['tp1']}</code>\nTP2: <code>{plan['tp2']}</code>\n"
            f"Logic: {plan['logic']}\nSession: New York (PKT 17:30-22:30)\n\n"
            "Suggested flow: Wait 1-3m confirmation on 1m; SL = sweep extreme; TP1 internal liquidity; TP2 HTF level.")
    keyboard = {
        "inline_keyboard": [
            [{"text":"✅ Confirm (manual)","callback_data":f"ACCEPT|{uid}"},{"text":"❌ Ignore","callback_data":f"IGNORE|{uid}"}],
            [{"text":"ℹ️ Auto-exec disabled","callback_data":f"DISABLED|{uid}"}]
        ]
    }
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "reply_markup": json.dumps(keyboard)}
    try:
        r = requests.post(TELEGRAM_API_BASE + "/sendMessage", json=payload, timeout=12)
        if r.status_code != 200:
            logging.warning("Telegram send failed: %s", r.text)
    except Exception as e:
        logging.exception("Telegram send exception: %s", e)

# -------------------- Callback poller (getUpdates) --------------------
def get_last_offset() -> Optional[int]:
    try:
        if os.path.exists(LAST_OFFSET_FILE):
            return int(open(LAST_OFFSET_FILE).read().strip())
    except:
        pass
    return None

def set_last_offset(offset: int):
    try:
        open(LAST_OFFSET_FILE, "w").write(str(offset))
    except:
        pass

def poll_callbacks(stop_event):
    logging.info("Starting Telegram callback poller.")
    offset = get_last_offset()
    while not stop_event.is_set():
        try:
            params = {"timeout": 20}
            if offset:
                params["offset"] = offset + 1
            r = requests.get(TELEGRAM_API_BASE + "/getUpdates", params=params, timeout=30)
            data = r.json()
            if not data.get("ok"):
                time.sleep(2)
                continue
            for upd in data.get("result", []):
                offset = upd["update_id"]
                set_last_offset(offset)
                if "callback_query" in upd:
                    cq = upd["callback_query"]
                    data_c = cq.get("data", "")
                    action, uid = (data_c.split("|") + [""])[:2]
                    user = cq.get("from", {})
                    user_str = f"{user.get('first_name','')} ({user.get('id','')})"
                    if action == "ACCEPT":
                        ack = f"✅ Accepted by {user_str}. UID {uid}. (Auto-exec disabled - manual step required.)"
                    elif action == "IGNORE":
                        ack = f"❌ Ignored by {user_str}. UID {uid}."
                    else:
                        ack = f"ℹ Action {action} received for UID {uid} by {user_str}."
                    # answerCallbackQuery
                    try:
                        requests.post(TELEGRAM_API_BASE + "/answerCallbackQuery", json={"callback_query_id": cq.get("id"), "text": "Action received"}, timeout=10)
                    except:
                        pass
                    try:
                        requests.post(TELEGRAM_API_BASE + "/sendMessage", json={"chat_id": cq["message"]["chat"]["id"], "text": ack}, timeout=10)
                    except:
                        pass
                    logging.info("Processed callback: %s by %s", action, user_str)
        except Exception as e:
            logging.exception("Error polling telegram updates: %s", e)
            time.sleep(2)

# -------------------- Analyzer session loop --------------------
analyzer_thread = None
analyzer_stop = threading.Event()

def in_pkt_session_now() -> bool:
    now = datetime.now(PKT).time()
    return (NY_START_PKT <= now <= NY_END_PKT)

def analyzer_worker():
    logging.info("Analyzer worker started (session active). Poll every %s sec.", POLL_INTERVAL)
    while not analyzer_stop.is_set():
        try:
            # fetch candles for TFs
            df1 = to_df(twelvedata_get_series(SYMBOL, "1min", 200))
            df5 = to_df(twelvedata_get_series(SYMBOL, "5min", 200))
            df15 = to_df(twelvedata_get_series(SYMBOL, "15min", 200))
            # HTF bias
            htf = detect_htf_bias()
            # detect components
            fvg = detect_fvg_simple(df15)
            sweep = detect_sweep_and_confirmation(df15, df5, lookback=8)
            mss = detect_mss_on_1m(df1, lookback=10)
            # volume spike heuristic on 1min
            vol_spike = False
            if len(df1) >= LOOKBACK_BARS:
                vol_sma = df1['volume'].rolling(LOOKBACK_BARS).mean().iloc[-1]
                if vol_sma > 0 and df1['volume'].iloc[-1] > vol_sma * 2.0:
                    vol_spike = True

            has_sweep = sweep.get("signal", False)
            has_fvg = len(fvg) > 0

            ifs_score = compute_ifs_score_components(has_sweep, has_fvg, vol_spike, htf, mss)

            logging.info("IFS %s | sweep:%s | fvg:%s | vol:%s | htf:%s | mss:%s", ifs_score, has_sweep, bool(has_fvg), vol_spike, htf, mss)

            # if strong enough, build plan and send alert (also require sweep & confirmation)
            if ifs_score >= MIN_IFS_SCORE and has_sweep:
                # Build trade plan
                sweep_type = sweep.get("type")
                sweep_c = sweep.get("sweep_candle")
                confirm_c = sweep.get("confirm_candle")
                plan = build_trade_plan(SYMBOL, sweep_c, confirm_c, sweep_type, htf)
                send_telegram_plan(plan, ifs_score)
            # sleep until next poll
        except Exception as e:
            logging.exception("Analyzer exception: %s", e)
        # breakable sleep to respect stop_event
        for _ in range(max(1, int(POLL_INTERVAL))):
            if analyzer_stop.is_set():
                break
            time.sleep(1)
    logging.info("Analyzer worker stopped.")

# -------------------- Scheduler start/stop --------------------
scheduler = BackgroundScheduler(timezone="UTC")

def start_analyzer_job():
    global analyzer_thread, analyzer_stop
    if analyzer_thread and analyzer_thread.is_alive():
        logging.info("Analyzer already running.")
        return
    analyzer_stop.clear()
    analyzer_thread = threading.Thread(target=analyzer_worker, daemon=True)
    analyzer_thread.start()
    logging.info("Started analyzer thread.")

def stop_analyzer_job():
    global analyzer_thread, analyzer_stop
    if analyzer_thread and analyzer_thread.is_alive():
        analyzer_stop.set()
        analyzer_thread.join(timeout=5)
        logging.info("Stopped analyzer thread.")
    else:
        logging.info("Analyzer thread not running.")

def job_pre_alert():
    now_pkt = datetime.now(timezone.utc).astimezone(PKT)
    send_msg = f"🕒 Pre-NY Snapshot (PKT {now_pkt.strftime('%Y-%m-%d %H:%M')}) — scanning liquidity pools."
    try:
        requests.post(TELEGRAM_API_BASE + "/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": send_msg}, timeout=8)
    except:
        logging.exception("Pre-alert send failed.")

def job_start_session():
    logging.info("Session start triggered — starting analyzer.")
    try:
        start_analyzer_job()
        requests.post(TELEGRAM_API_BASE + "/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": "🟢 NY Session started — Analyzer active (PKT 17:30)."}, timeout=8)
    except:
        logging.exception("Failed to announce session start.")

def job_stop_session():
    logging.info("Session stop triggered — stopping analyzer.")
    try:
        stop_analyzer_job()
        requests.post(TELEGRAM_API_BASE + "/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": "🔴 NY Session ended — Analyzer paused (PKT 22:30)."}, timeout=8)
    except:
        logging.exception("Failed to announce session stop.")

def schedule_session_jobs():
    """
    Schedule 3 jobs:
      - Pre-alert (PKT 17:25) -> UTC 12:25
      - Start analyzer (PKT 17:30) -> UTC 12:30
      - Stop analyzer  (PKT 22:30) -> UTC 17:30
    Use cron style scheduling (UTC).
    """
    # PKT is UTC+5 => PKT H -> UTC H-5
    pre_utc_hour = 17 - 5  # 12
    start_utc_hour = 17 - 5  # 12
    stop_utc_hour = 22 - 5   # 17
    # pre-alert at PKT 17:25 -> UTC 12:25
    scheduler.add_job(job_pre_alert, 'cron', hour=pre_utc_hour, minute=25)
    # start at PKT 17:30 -> UTC 12:30
    scheduler.add_job(job_start_session, 'cron', hour=start_utc_hour, minute=30)
    # stop at PKT 22:30 -> UTC 17:30
    scheduler.add_job(job_stop_session, 'cron', hour=stop_utc_hour, minute=30)
    scheduler.start()
    logging.info("Scheduler started (pre,start,stop jobs scheduled).")

# -------------------- Main --------------------
if __name__ == "__main__":
    logging.info("Starting Institutional Smart-Money Bot for %s", SYMBOL)
    # schedule session jobs
    schedule_session_jobs()
    # also, if currently inside session on start, start analyzer immediately
    now_pkt = datetime.now(timezone.utc).astimezone(PKT).time()
    if NY_START_PKT <= now_pkt <= NY_END_PKT:
        start_analyzer_job()
    # start callback poller
    stop_evt = threading.Event()
    cb_thread = threading.Thread(target=poll_callbacks, args=(stop_evt,), daemon=True)
    cb_thread.start()
    # keep main thread alive
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        stop_evt.set()
        stop_analyzer_job()
        scheduler.shutdown(wait=False)
        logging.info("Exited.")
