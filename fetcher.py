import aiohttp
import asyncio
import json
import os
import time
import sys
from collections import deque
from datetime import datetime

# --- IMPORT ENGINE ---
try:
    from prediction_engine import ultraAIPredict
    print("[INIT] TITAN V700 SOVEREIGN ENGINE LINKED SUCCESSFULLY.")
except ImportError:
    print("[ERROR] prediction_engine.py is missing! Predictions will fail.")

# --- API CONFIGURATION ---
# Updated to your new working API source
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# --- RENDER PERSISTENT STORAGE ---
if os.path.exists('/var/lib/data'):
    BASE_DIR = '/var/lib/data'
    print("[SYSTEM] Using Render Disk: /var/lib/data")
else:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    print(f"[SYSTEM] Using Local Storage: {BASE_DIR}")

DASHBOARD_FILE = os.path.join(BASE_DIR, 'dashboard_data.json')

# --- SETTINGS ---
HISTORY_LIMIT = 2000       
MIN_DATA_REQUIRED = 30  # Forces logic to wait for 30 records before predicting

# --- MEMORY STORAGE ---
RAM_HISTORY = deque(maxlen=HISTORY_LIMIT)
UI_HISTORY = deque(maxlen=50)

# --- GLOBAL STATE ---
currentbankroll = 10000.0 
last_prediction = {
    "issue": None, 
    "label": "WAITING", 
    "stake": 0, 
    "conf": 0, 
    "level": "---", 
    "reason": "Initializing System...", 
    "strategy": "BOOTING"
}
last_win_status = "NONE"
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
    
    # Custom status when collecting initial data
    if len(RAM_HISTORY) < MIN_DATA_REQUIRED:
        display_status = f"COLLECTING DATA ({len(RAM_HISTORY)}/{MIN_DATA_REQUIRED})"
    else:
        display_status = status_text

    data = {
        "period": last_prediction['issue'] if last_prediction['issue'] else "---",
        "prediction": last_prediction['label'],
        "confidence": f"{last_prediction.get('conf', 0)*100:.1f}%",
        "stake": last_prediction['stake'],
        "level": last_prediction.get('level', '---'),
        "strategy": last_prediction.get('strategy', 'SOVEREIGN'),
        "reason": last_prediction.get('reason', 'Initializing...'),
        "bankroll": currentbankroll,
        "lastresult_status": last_win_status,
        "status_text": display_status,
        "timer": timer_val,
        "data_size": len(RAM_HISTORY),
        "stats": {"wins": session_wins, "losses": session_losses, "accuracy": acc},
        "history": list(UI_HISTORY),
        "timestamp": time.time()
    }
    
    try:
        temp_file = DASHBOARD_FILE + ".tmp"
        with open(temp_file, "w") as f: 
            json.dump(data, f)
        os.replace(temp_file, DASHBOARD_FILE)
    except Exception as e: 
        print(f"[DASHBOARD ERROR] {e}")

async def fetch_api_data(session):
    try:
        # Note: Using POST as many lottery APIs require it, but GET is used here per user URL
        async with session.get(API_URL, headers=HEADERS, timeout=10) as response:
            if response.status == 200:
                json_data = await response.json()
                # Extract list based on standard lottery JSON formats
                if 'data' in json_data and 'list' in json_data['data']:
                    return json_data['data']['list']
                elif 'list' in json_data:
                    return json_data['list']
                return json_data
            else:
                print(f"[API ERROR] Status {response.status}")
    except Exception as e:
        print(f"[FETCH ERROR] {e}")
    return None

async def main_loop():
    global currentbankroll, last_prediction, last_win_status, session_wins, session_losses
    print("--- TITAN V700 SOVEREIGN FETCHER ACTIVE ---")
    
    last_processed_issue = None
    
    async with aiohttp.ClientSession() as session:
        while True:
            raw_list = await fetch_api_data(session)
            
            if raw_list:
                # Process all items in the returned list (usually 10 items)
                for item in reversed(raw_list):
                    try:
                        issue = str(item.get('issueNumber') or item.get('issue'))
                        num = int(item.get('number') or item.get('result'))
                        
                        # Add to RAM if it's a new issue
                        if not RAM_HISTORY or RAM_HISTORY[-1]['issue'] != issue:
                            RAM_HISTORY.append({'issue': issue, 'actual_number': num})
                            print(f"[DATA] Recorded Issue {issue} | Result: {num}")
                    except: continue

                # Get the absolute latest for the dashboard
                latest = raw_list[0]
                curr_issue = str(latest.get('issueNumber') or latest.get('issue'))
                curr_num = int(latest.get('number') or latest.get('result'))
                
                now = datetime.now()
                seconds_left = 60 - now.second
                update_dashboard("SYNCED", seconds_left)
                
                if curr_issue != last_processed_issue:
                    # 1. Evaluate previous prediction result
                    if last_prediction['issue'] == curr_issue:
                        real_outcome = get_outcome_from_number(curr_num)
                        predicted = last_prediction['label']
                        
                        if predicted != "SKIP" and predicted != "WAITING":
                            is_win = (predicted == real_outcome)
                            if is_win:
                                session_wins += 1
                                last_win_status = "WIN"
                                UI_HISTORY.appendleft({"period": curr_issue, "pred": predicted, "result": "WIN"})
                            else:
                                session_losses += 1
                                last_win_status = "LOSS"
                                UI_HISTORY.appendleft({"period": curr_issue, "pred": predicted, "result": "LOSS"})

                    # 2. Run Prediction Logic (Only if we have 30 records)
                    if len(RAM_HISTORY) >= MIN_DATA_REQUIRED:
                        try:
                            next_issue = str(int(curr_issue) + 1)
                            # Call the Sniper Engine
                            ai_res = ultraAIPredict(list(RAM_HISTORY), currentbankroll, get_outcome_from_number(curr_num))
                            
                            last_prediction = {
                                "issue": next_issue, 
                                "label": ai_res['finalDecision'], 
                                "stake": ai_res['positionsize'],
                                "conf": ai_res['confidence'],
                                "level": ai_res['level'],
                                "strategy": "SOVEREIGN",
                                "reason": ai_res.get('reason', 'Analyzing Pattern...')
                            }
                        except Exception as e:
                            print(f"[ENGINE ERROR] {e}")
                    else:
                        # Still collecting
                        last_prediction['label'] = "WAITING"
                        last_prediction['reason'] = f"Collecting Data: {len(RAM_HISTORY)}/30"
                    
                    last_processed_issue = curr_issue

            await asyncio.sleep(2) # Poll every 2 seconds to avoid API bans

if __name__ == '__main__':
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("System Stopped.")
