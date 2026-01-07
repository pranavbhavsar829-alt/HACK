# ==============================================================================
# MODULE: FETCHER.PY (WITH HISTORY LOG)
# ==============================================================================

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
    from prediction_engine import ultraAIPredict, get_outcome_from_number, GameConstants
    print("[INIT] TITAN V102 Sniper Engine Loaded.")
except ImportError as e:
    print(f"\n[CRITICAL ERROR] prediction_engine.py not found: {e}")
    sys.exit()

# --- CONFIG ---
API_URL = "https://api-wo4u.onrender.com/api/get_history"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Connection": "keep-alive"
}

TACTICAL_DELAY_SECONDS = 15
HISTORY_LIMIT = 500
DB_FILE = 'ar_lottery_history.db'
DASHBOARD_FILE = 'dashboard_data.json'

RAM_HISTORY = deque(maxlen=HISTORY_LIMIT)
UI_HISTORY = deque(maxlen=7) # <--- STORES LAST 7 RESULTS FOR UI

current_bankroll = 10000.0 
last_prediction = {"issue": None, "label": "WAITING", "stake": 0, "conf": 0, "level": "---"}
last_win_status = "NONE"

# --- UI BRIDGE FUNCTION ---
def update_dashboard(status_text="IDLE"):
    data = {
        "period": last_prediction['issue'] if last_prediction['issue'] else "---",
        "prediction": last_prediction['label'],
        "confidence": f"{last_prediction.get('conf', 0)*100:.1f}%",
        "stake": last_prediction['stake'],
        "level": last_prediction.get('level', '---'),
        "bankroll": current_bankroll,
        "last_result_status": last_win_status,
        "status_text": status_text,
        "history": list(UI_HISTORY), # <--- SEND HISTORY TO UI
        "timestamp": time.time()
    }
    try:
        with open(DASHBOARD_FILE + ".tmp", "w") as f:
            json.dump(data, f)
        os.replace(DASHBOARD_FILE + ".tmp", DASHBOARD_FILE)
    except: pass

# --- DB FUNCTIONS ---
def ensure_db_setup():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS results (issue TEXT PRIMARY KEY, code INTEGER, fetch_time TEXT)')
        conn.commit()
        conn.close()
    except: pass

async def save_to_db_background(issue, code):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO results (issue, code, fetch_time) VALUES (?, ?, ?)", (issue, code, str(datetime.now())))
        conn.commit()
        conn.close()
    except: pass

# --- BACKFILL ---
async def perform_initial_backfill(session):
    print("[INIT] Checking DB...")
    ensure_db_setup()
    try:
        params = {'pageSize': 2000, 'page': 1} 
        async with session.get(API_URL, headers=HEADERS, params=params, timeout=30) as response:
            if response.status == 200:
                data = await response.json()
                raw_list = data.get('data', {}).get('list', [])
                if raw_list:
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    batch = []
                    for item in raw_list:
                        batch.append((str(item['issueNumber']), int(item['number']), str(datetime.now())))
                    cursor.executemany("INSERT OR IGNORE INTO results (issue, code, fetch_time) VALUES (?, ?, ?)", batch)
                    conn.commit()
                    conn.close()
                    print(f"[INIT] Backfill: {len(raw_list)} records.")
    except Exception as e:
        print(f"[INIT] Backfill error: {e}")

async def load_initial_history():
    RAM_HISTORY.clear()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(f"SELECT issue, code FROM results ORDER BY issue DESC LIMIT {HISTORY_LIMIT}")
        rows = cursor.fetchall()
        conn.close()
        for r in reversed(rows):
            RAM_HISTORY.append({'issue': str(r[0]), 'actual_number': int(r[1]), 'code': int(r[1])})
        print(f"[INIT] Loaded {len(RAM_HISTORY)} records.")
    except: pass

async def fetch_latest_data(session):
    try:
        async with session.get(API_URL, headers=HEADERS, params={'pageSize': 5, 'page': 1}, timeout=5) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('data', {}).get('list', [])
    except: return None

# --- MAIN LOOP ---
async def main_loop():
    global current_bankroll, last_prediction, last_win_status
    
    print("================================================================")
    print("   TITAN V102 - UI SERVER CONNECTED (HISTORY ON)")
    print("================================================================")
    
    last_processed_issue = None
    
    async with aiohttp.ClientSession() as session:
        await perform_initial_backfill(session)
        await load_initial_history()
        
        while True:
            update_dashboard("FETCHING...")
            raw_list = await fetch_latest_data(session)
            
            if raw_list:
                latest = raw_list[0]
                curr_issue = str(latest['issueNumber'])
                curr_code = int(latest['number'])
                
                if curr_issue != last_processed_issue:
                    # 1. Update Memory
                    new_record = {'issue': curr_issue, 'actual_number': curr_code}
                    if not RAM_HISTORY or RAM_HISTORY[-1]['issue'] != curr_issue:
                        RAM_HISTORY.append(new_record)
                        asyncio.create_task(save_to_db_background(curr_issue, curr_code))
                    
                    # 2. Check Win/Loss
                    if last_prediction['issue'] and last_prediction['label'] != GameConstants.SKIP:
                        real_outcome = get_outcome_from_number(curr_code)
                        predicted = last_prediction['label']
                        stake = last_prediction['stake']
                        
                        is_win = False
                        if predicted == real_outcome: is_win = True
                        if predicted in ["RED", "GREEN"]:
                            real_c = "RED" if curr_code in [0,2,4,6,8] else "GREEN"
                            if curr_code == 5: real_c = "GREEN"
                            if predicted == real_c: is_win = True

                        if is_win:
                            profit = stake * 0.98
                            current_bankroll += profit
                            last_win_status = "WIN"
                            # Add to UI History
                            UI_HISTORY.appendleft({
                                "period": last_prediction['issue'],
                                "pred": predicted,
                                "result": "WIN",
                                "profit": f"+{profit:.0f}"
                            })
                            print(f"\n[WIN] {curr_issue}={curr_code} | +${profit:.0f}")
                        else:
                            current_bankroll -= stake
                            last_win_status = "LOSS"
                            # Add to UI History
                            UI_HISTORY.appendleft({
                                "period": last_prediction['issue'],
                                "pred": predicted,
                                "result": "LOSS",
                                "profit": f"-{stake:.0f}"
                            })
                            print(f"\n[LOSS] {curr_issue}={curr_code} | -${stake:.0f}")
                    
                    # 3. Wait
                    next_issue = str(int(curr_issue) + 1)
                    print(f"[WAIT] Target: {next_issue}")
                    
                    for i in range(TACTICAL_DELAY_SECONDS, 0, -1):
                        update_dashboard(f"WAITING... {i}s")
                        await asyncio.sleep(1)

                    # 4. Predict
                    update_dashboard("CALCULATING...")
                    snapshot = list(RAM_HISTORY)
                    if len(snapshot) > 10:
                        ai_result = ultraAIPredict(
                            history=snapshot, 
                            current_bankroll=current_bankroll, 
                            last_result=last_prediction['label'] if last_prediction['issue'] == curr_issue else None
                        )
                        
                        decision = ai_result['finalDecision']
                        stake = ai_result['positionsize']
                        conf = ai_result['confidence']
                        level = ai_result['level']
                        
                        last_prediction = {
                            "issue": next_issue, 
                            "label": decision, 
                            "stake": stake,
                            "conf": conf,
                            "level": level
                        }
                        
                        print(f"[PRED] {decision} | {conf:.0%} | ${stake}")
                        update_dashboard("ACTIVE")
                    
                    last_processed_issue = curr_issue

            await asyncio.sleep(1.0)

if __name__ == '__main__':
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
