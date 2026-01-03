import aiohttp
import asyncio
import json
import time
import sys
import os
from collections import deque
from datetime import datetime
from prediction_engine import ultraAIPredict, get_outcome_from_number
from database_handler import ensure_setup, save_result_to_database, DB_PATH

API_URL = "https://harshpredictor.site/api/api.php"
# SAVE DASHBOARD DATA TO PERSISTENT DISK
DASHBOARD_FILE = os.path.join(DB_PATH, 'dashboard_data.json')

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# --- STATS TRACKING ---
RAM_HISTORY = deque(maxlen=500)
UI_HISTORY = deque(maxlen=20)  # Stores last 20 results for UI
current_bankroll = 10000.0 
last_prediction = {"issue": None, "label": "WAITING", "stake": 0}

session_stats = {
    "wins": 0,
    "losses": 0,
    "accuracy": "0%"
}

def calculate_time_remaining():
    """
    Calculates seconds remaining until the next minute (00 seconds).
    Most lottery games run on 1-minute intervals.
    """
    now = time.time()
    next_minute = (int(now) // 60 + 1) * 60
    remaining = int(next_minute - now)
    return max(0, remaining)

def update_dashboard(status_text="IDLE"):
    global session_stats
    
    # Calculate Timer
    remaining_seconds = calculate_time_remaining()
    
    data = {
        "period": last_prediction['issue'] if last_prediction['issue'] else "---",
        "prediction": last_prediction['label'],
        "history": list(UI_HISTORY),
        "status_text": status_text,
        "timer": remaining_seconds,
        "stats": session_stats,
        "timestamp": time.time()
    }
    
    try:
        # Atomic write to prevent file corruption
        temp_file = DASHBOARD_FILE + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(data, f)
        os.replace(temp_file, DASHBOARD_FILE)
    except Exception as e:
        print(f"Error writing dashboard: {e}")

async def fetch_loop():
    global last_prediction, session_stats
    
    ensure_setup()
    print("Bot Started.")
    
    last_processed = None
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # 1. UPDATE TIMER & DASHBOARD FREQUENTLY (Every 1s)
                # We do this inside the loop to keep the UI timer synced even if API is slow
                update_dashboard("WAITING FOR DRAW...")
                
                async with session.get(API_URL, headers=HEADERS, params={'pageSize': 5, 'page': 1}, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw = data.get('data', {}).get('list', [])
                        
                        if raw:
                            latest = raw[0]
                            curr_issue = str(latest['issueNumber'])
                            curr_code = int(latest['number'])
                            
                            if curr_issue != last_processed:
                                print(f"New Result: {curr_issue} = {curr_code}")
                                
                                # --- 1. UPDATE MEMORY ---
                                RAM_HISTORY.append({'issue': curr_issue, 'actual_number': curr_code})
                                
                                # --- 2. CHECK WIN/LOSS ---
                                if last_prediction['issue']:
                                    actual = get_outcome_from_number(curr_code)
                                    res = "PENDING"
                                    
                                    if last_prediction['label'] == "SKIP":
                                        res = "SKIP"
                                    elif last_prediction['label'] == actual:
                                        res = "WIN"
                                        session_stats['wins'] += 1
                                    else:
                                        res = "LOSS"
                                        session_stats['losses'] += 1
                                    
                                    # Update Accuracy
                                    total = session_stats['wins'] + session_stats['losses']
                                    acc = (session_stats['wins'] / total * 100) if total > 0 else 0
                                    session_stats['accuracy'] = f"{acc:.1f}%"
                                    
                                    # Add to UI History
                                    UI_HISTORY.appendleft({
                                        "period": last_prediction['issue'],
                                        "pred": last_prediction['label'],
                                        "result": res,
                                        "code": curr_code
                                    })

                                # --- 3. PREDICT NEXT ---
                                snapshot = list(RAM_HISTORY)
                                prediction = ultraAIPredict(snapshot)
                                next_issue = str(int(curr_issue) + 1)
                                
                                last_prediction = {
                                    "issue": next_issue,
                                    "label": prediction['finalDecision'],
                                    "stake": prediction['positionsize']
                                }
                                
                                last_processed = curr_issue
                                
            except Exception as e:
                print(f"Fetch Error: {e}")
            
            # Smart Sleep: Update UI every second, but fetch API every 3 seconds
            for _ in range(3):
                update_dashboard("SYNCING...")
                await asyncio.sleep(1)

if __name__ == '__main__':
    asyncio.run(fetch_loop())
