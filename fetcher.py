import aiohttp
import asyncio
import json
import time
import sys
import os
from collections import deque
from datetime import datetime

# --- IMPORT ENGINE ---
# We try to import the engine. If missing, it will still run in "Manual Mode"
try:
    from prediction_engine import ultraAIPredict
    print("[INIT] Prediction Engine Loaded Successfully.")
except ImportError:
    print("[ERROR] prediction_engine.py is missing! Predictions will be skipped.")
    # Dummy function to prevent crash
    def ultraAIPredict(h, b, l): return {'finalDecision': 'SKIP', 'confidence': 0, 'level': 'ERR'}

# --- RENDER PATH SETUP ---
# Ensures we write to the persistent disk on Render
if os.path.exists('/var/lib/data'):
    BASE_DIR = '/var/lib/data'
elif os.path.exists('/data'):
    BASE_DIR = '/data'
else:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DASHBOARD_FILE = os.path.join(BASE_DIR, 'dashboard_data.json')

# --- CONFIGURATION ---
API_URL = "https://harshpredictor.site/api/api.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Connection": "keep-alive"
}

# --- STATE VARIABLES ---
HISTORY_LIMIT = 2000
RAM_HISTORY = deque(maxlen=HISTORY_LIMIT)

# HISTORY FOR UI: Stores the last 50 results (Wins/Losses)
UI_HISTORY = deque(maxlen=50)

currentbankroll = 10000.0
session_wins = 0
session_losses = 0

last_prediction = {
    "issue": None, 
    "label": "WAITING", 
    "stake": 0, 
    "conf": 0, 
    "strategy": "BOOTING",
    "reason": ""
}

def get_outcome_from_number(n):
    """Converts 0-9 to SMALL/BIG."""
    try:
        val = int(float(n))
        if 0 <= val <= 4: return "SMALL"
        if 5 <= val <= 9: return "BIG"
    except: pass
    return None

def update_dashboard(status_text="IDLE"):
    """Writes the current state and HISTORY list to JSON."""
    total = session_wins + session_losses
    acc = f"{(session_wins/total)*100:.0f}%" if total > 0 else "0%"
    
    data = {
        "period": last_prediction['issue'] if last_prediction['issue'] else "---",
        "prediction": last_prediction['label'],
        "confidence": f"{last_prediction.get('conf', 0)*100:.1f}%",
        "strategy": last_prediction.get('strategy', 'STANDARD'),
        "status_text": status_text,
        "stats": {"wins": session_wins, "losses": session_losses, "accuracy": acc},
        "history": list(UI_HISTORY),  # <--- SENDS HISTORY TO SERVER.PY
        "timestamp": time.time()
    }
    
    # Atomic Write (Write temp then rename) to prevent corruption
    try:
        temp_file = DASHBOARD_FILE + ".tmp"
        with open(temp_file, "w") as f: 
            json.dump(data, f)
        os.replace(temp_file, DASHBOARD_FILE)
    except Exception as e: 
        print(f"[DASHBOARD ERROR] Write Failed: {e}")

async def fetch_data(session, limit=5):
    """Fetches latest results from API."""
    try:
        params = {'pageSize': limit, 'page': 1}
        async with session.get(API_URL, headers=HEADERS, params=params, timeout=10) as response:
            if response.status == 200:
                d = await response.json()
                return d.get('data', {}).get('list', [])
    except Exception as e:
        print(f"[FETCH ERROR] {e}")
    return None

async def main_loop():
    """Main Logic Loop."""
    global last_prediction, session_wins, session_losses
    print("--- TITAN V700 FETCHER STARTED ---")
    
    last_processed_issue = None
    
    async with aiohttp.ClientSession() as session:
        
        # --- PHASE 1: INITIAL LOAD ---
        print("[STARTUP] Loading history from API...")
        update_dashboard("SYNCING HISTORY...")
        startup_data = await fetch_data(session, 100)
        
        if startup_data:
            # Load roughly 100 items into memory
            for item in reversed(startup_data):
                try:
                    curr_issue = str(item['issueNumber'])
                    curr_code = int(item['number'])
                    RAM_HISTORY.append({'issue': curr_issue, 'actual_number': curr_code})
                    last_processed_issue = curr_issue
                except: continue
            print(f"[STARTUP] Loaded {len(RAM_HISTORY)} records.")
        
        # --- PHASE 2: LIVE LOOP ---
        while True:
            # 1. Fetch Data
            raw_list = await fetch_data(session, 5)
            
            if raw_list:
                latest = raw_list[0]
                curr_issue = str(latest['issueNumber'])
                curr_code = int(latest['number'])
                
                # 2. Check for New Result
                if curr_issue != last_processed_issue:
                    print(f"[NEW RESULT] {curr_issue}: {curr_code}")
                    
                    # Add to memory
                    if not RAM_HISTORY or RAM_HISTORY[-1]['issue'] != curr_issue:
                        RAM_HISTORY.append({'issue': curr_issue, 'actual_number': curr_code})

                    # 3. WIN/LOSS CHECK Logic
                    if last_prediction['issue'] and last_prediction['label'] != "SKIP":
                        real_outcome = get_outcome_from_number(curr_code)
                        predicted = last_prediction['label']
                        
                        # Standard Check
                        is_win = (predicted == real_outcome)
                        
                        # Violet/Color Logic (0=Red+Vio, 5=Green+Vio)
                        # If we predicted Red, and it came 0, it's a Win.
                        if predicted == "RED" and curr_code in [0,2,4,6,8]: is_win = True
                        if predicted == "GREEN" and curr_code in [1,3,5,7,9]: is_win = True
                        if curr_code == 5 and predicted == "GREEN": is_win = True
                        
                        result_str = "WIN" if is_win else "LOSS"
                        
                        if is_win:
                            session_wins += 1
                            print(f"   >>> WIN! ({predicted})")
                        else:
                            session_losses += 1
                            print(f"   >>> LOSS. ({predicted})")

                        # 4. RECORD TO HISTORY
                        UI_HISTORY.appendleft({
                            "period": last_prediction['issue'],
                            "pred": predicted,
                            "result": result_str
                        })

                    # 5. PREDICT NEXT ROUND
                    next_issue = str(int(curr_issue) + 1)
                    
                    # Visual "Thinking" Delay
                    for i in range(3, 0, -1):
                        update_dashboard(f"CALCULATING... {i}")
                        await asyncio.sleep(1)

                    # Call Engine
                    try:
                        ai_result = ultraAIPredict(list(RAM_HISTORY), currentbankroll, get_outcome_from_number(curr_code))
                        
                        last_prediction = {
                            "issue": next_issue,
                            "label": ai_result['finalDecision'],
                            "conf": ai_result['confidence'],
                            "strategy": ai_result.get('level', 'STD'),
                            "reason": ai_result.get('reason', '')
                        }
                    except Exception as e:
                        print(f"[ENGINE ERROR] {e}")
                        last_prediction['label'] = "SKIP"
                    
                    last_processed_issue = curr_issue
                else:
                    # No new result yet, just update status
                    update_dashboard("LIVE MONITORING")

            # Wait 1 second before next poll
            await asyncio.sleep(1.0)

if __name__ == '__main__':
    # Windows fix for asyncio
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_loop())
