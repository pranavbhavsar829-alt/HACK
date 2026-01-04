import aiohttp
import asyncio
import json
import sqlite3
import time
import sys
import os
from collections import deque
from datetime import datetime

# --- IMPORT ENGINE ---
try:
    from prediction_engine import ultraAIPredict
    print("[INIT] Engine Linked.")
except ImportError:
    print("[ERROR] prediction_engine.py missing.")

# --- CONFIG ---
API_URL = "https://harshpredictor.site/api/api.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Connection": "keep-alive"
}
TACTICAL_DELAY_SECONDS = 15
HISTORY_LIMIT = 1500
DB_FILE = 'ar_lottery_history.db'
DASHBOARD_FILE = 'dashboard_data.json'

RAM_HISTORY = deque(maxlen=HISTORY_LIMIT)
UI_HISTORY = deque(maxlen=50) # Updated to 50 as requested

currentbankroll = 10000.0 
last_prediction = {"issue": None, "label": "WAITING", "stake": 0, "conf": 0, "level": "---"}
last_win_status = "NONE"

# --- NEW: STATS TRACKING ---
session_wins = 0
session_losses = 0

def get_outcome_from_number(n):
    try:
        val = int(float(n))
        if 0 <= val <= 4: return "SMALL"
        if 5 <= val <= 9: return "BIG"
    except: pass
    return None

def update_dashboard(status_text="IDLE", timer_val=0):
    total = session_wins + session_losses
    acc = f"{(session_wins/total)*100:.0f}%" if total > 0 else "0%"
    
    data = {
        "period": last_prediction['issue'] if last_prediction['issue'] else "---",
        "prediction": last_prediction['label'],
        "confidence": f"{last_prediction.get('conf', 0)*100:.1f}%",
        "stake": last_prediction['stake'],
        "level": last_prediction.get('level', '---'),
        "bankroll": currentbankroll,
        "lastresult_status": last_win_status,
        "status_text": status_text,
        "timer": timer_val,
        "stats": {"wins": session_wins, "losses": session_losses, "accuracy": acc},
        "history": list(UI_HISTORY),
        "timestamp": time.time()
    }
    try:
        with open(DASHBOARD_FILE + ".tmp", "w") as f: json.dump(data, f)
        os.replace(DASHBOARD_FILE + ".tmp", DASHBOARD_FILE)
    except: pass

async def fetch_latest_data(session):
    try:
        async with session.get(API_URL, headers=HEADERS, params={'pageSize': 5, 'page': 1}, timeout=5) as response:
            if response.status == 200:
                d = await response.json()
                return d.get('data', {}).get('list', [])
    except: return None

# --- MAIN LOOP (CALLED BY SERVER.PY) ---
async def main_loop():
    global currentbankroll, last_prediction, last_win_status, session_wins, session_losses
    print("--- FETCHER ENGINE STARTED ---")
    
    last_processed_issue = None
    
    async with aiohttp.ClientSession() as session:
        while True:
            update_dashboard("FETCHING...", 0) 
            raw_list = await fetch_latest_data(session)
            
            if raw_list:
                latest = raw_list[0]
                curr_issue = str(latest['issueNumber'])
                curr_code = int(latest['number'])
                
                if curr_issue != last_processed_issue:
                    # Update History
                    new_record = {'issue': curr_issue, 'actual_number': curr_code}
                    if not RAM_HISTORY or RAM_HISTORY[-1]['issue'] != curr_issue:
                        RAM_HISTORY.append(new_record)
                    
                    # Check Result
                    if last_prediction['issue'] and last_prediction['label'] != "SKIP":
                        real_outcome = get_outcome_from_number(curr_code)
                        predicted = last_prediction['label']
                        
                        is_win = (predicted == real_outcome)
                        # Green/Red logic check
                        if predicted in ["RED", "GREEN"]:
                            real_c = "RED" if curr_code in [0,2,4,6,8] else "GREEN"
                            if curr_code == 5: real_c = "GREEN"
                            if predicted == real_c: is_win = True

                        if is_win:
                            session_wins += 1
                            last_win_status = "WIN"
                            UI_HISTORY.appendleft({"period": last_prediction['issue'], "pred": predicted, "result": "WIN"})
                            print(f"[WIN] {curr_issue}")
                        else:
                            session_losses += 1
                            last_win_status = "LOSS"
                            UI_HISTORY.appendleft({"period": last_prediction['issue'], "pred": predicted, "result": "LOSS"})
                            print(f"[LOSS] {curr_issue}")

                    # Countdown Loop
                    next_issue = str(int(curr_issue) + 1)
                    print(f"[WAIT] Target: {next_issue}")
                    for i in range(TACTICAL_DELAY_SECONDS, 0, -1):
                        update_dashboard(f"WAITING... {i}s", i)
                        await asyncio.sleep(1)

                    # Predict
                    update_dashboard("CALCULATING...", 0)
                    if len(RAM_HISTORY) > 10:
                        try:
                            ai_result = ultraAIPredict(list(RAM_HISTORY), currentbankroll)
                            decision = ai_result['finalDecision']
                            last_prediction = {
                                "issue": next_issue, 
                                "label": decision, 
                                "stake": ai_result['positionsize'],
                                "conf": ai_result['confidence'],
                                "level": ai_result['level']
                            }
                            print(f"[PRED] {decision}")
                            update_dashboard("ACTIVE", 0)
                        except Exception as e:
                            print(f"Error: {e}")
                            update_dashboard("ERROR", 0)
                    
                    last_processed_issue = curr_issue

            await asyncio.sleep(1.0)

if __name__ == '__main__':
    try:
        if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main_loop())
    except KeyboardInterrupt: pass
