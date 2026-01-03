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

RAM_HISTORY = deque(maxlen=500)
UI_HISTORY = deque(maxlen=7) 
current_bankroll = 10000.0 
last_prediction = {"issue": None, "label": "WAITING", "stake": 0}

def update_dashboard(status_text="IDLE"):
    data = {
        "period": last_prediction['issue'] if last_prediction['issue'] else "---",
        "prediction": last_prediction['label'],
        "history": list(UI_HISTORY),
        "status_text": status_text,
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
    # --- FIX IS HERE: DECLARE GLOBAL AT THE TOP ---
    global last_prediction 
    
    ensure_setup()
    print("Bot Started.")
    
    last_processed = None
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
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
                                
                                # 1. Update History
                                RAM_HISTORY.append({'issue': curr_issue, 'actual_number': curr_code})
                                
                                # 2. Check Win/Loss
                                if last_prediction['issue']:
                                    actual = get_outcome_from_number(curr_code)
                                    res = "WIN" if last_prediction['label'] == actual else "LOSS"
                                    if last_prediction['label'] == "SKIP": res = "SKIP"
                                    
                                    UI_HISTORY.appendleft({
                                        "period": last_prediction['issue'],
                                        "pred": last_prediction['label'],
                                        "result": res
                                    })

                                # 3. Predict Next
                                snapshot = list(RAM_HISTORY)
                                prediction = ultraAIPredict(snapshot)
                                next_issue = str(int(curr_issue) + 1)
                                
                                last_prediction = {
                                    "issue": next_issue,
                                    "label": prediction['finalDecision'],
                                    "stake": prediction['positionsize']
                                }
                                
                                update_dashboard("ACTIVE")
                                last_processed = curr_issue
                                
            except Exception as e:
                print(f"Fetch Error: {e}")
                
            await asyncio.sleep(5)

if __name__ == '__main__':
    asyncio.run(fetch_loop())
