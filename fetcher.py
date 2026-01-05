"""
TITAN V700 - SOVEREIGN FETCHER ENGINE (V2026.6 - OCTET-STREAM FIX)
================================================================
A highly robust, asynchronous data acquisition layer designed to 
synchronize with the draw.ar-lottery01.com API.
================================================================
"""

import aiohttp
import asyncio
import json
import logging
import os
import sys
import time
import sqlite3
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional, Any

# --- CUSTOM LOGGING CONFIGURATION ---
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
logger = logging.getLogger("TITAN_FETCHER")

# --- CORE SYSTEM ENGINE IMPORT ---
try:
    from prediction_engine import ultraAIPredict
    # Integrating helper functions if available in your prediction_engine
    def get_outcome_from_number(n):
        try:
            val = int(float(n))
            return "SMALL" if 0 <= val <= 4 else "BIG"
        except: return None
    logger.info("Logic Engine (TITAN V700) linked successfully.")
except ImportError as e:
    logger.critical(f"FATAL: Prediction engine missing! Details: {e}")
    sys.exit(1)

# --- API CONFIGURATION ---
TARGET_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://draw.ar-lottery01.com",
    "Referer": "https://draw.ar-lottery01.com/",
    "Connection": "keep-alive"
}

# --- PERSISTENT STORAGE PATHS ---
BASE_STORAGE_PATH = os.path.abspath(os.path.dirname(__file__))
DB_FILE = os.path.join(BASE_STORAGE_PATH, 'ar_lottery_history.db')
DASHBOARD_PATH = os.path.join(BASE_STORAGE_PATH, 'dashboard_data.json')

# --- OPERATIONAL PARAMETERS ---
HISTORY_RETENTION_LIMIT = 2000
MIN_BRAIN_CAPACITY = 10   
LIVE_POLL_INTERVAL = 2.0  

# --- SHARED GLOBAL STATE ---
class FetcherState:
    def __init__(self):
        self.bankroll = 10000.0
        self.ram_history = deque(maxlen=HISTORY_RETENTION_LIMIT)
        self.ui_history = deque(maxlen=50)
        self.wins = 0
        self.losses = 0
        self.last_processed_issue = None
        self.active_prediction = {
            "issue": None, "label": "WAITING", "stake": 0, "conf": 0,
            "level": "---", "reason": "Initializing system...", "strategy": "BOOTING"
        }
        self.last_win_status = "NONE"

state = FetcherState()

# =============================================================================
# DATABASE LAYER
# =============================================================================

def ensure_db_setup():
    """Initializes SQLite to ensure data persists across script restarts."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE IF NOT EXISTS results 
                    (issue TEXT PRIMARY KEY, code INTEGER, fetch_time TEXT)''')
    conn.commit()
    conn.close()

async def save_to_db(issue: str, code: int):
    """Saves a new record to the local SQLite database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT OR IGNORE INTO results (issue, code, fetch_time) VALUES (?, ?, ?)", 
                       (str(issue), int(code), str(datetime.now())))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB Save Error: {e}")

async def load_db_to_ram():
    """Loads historical data from DB into RAM for the AI engine."""
    state.ram_history.clear()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(f"SELECT issue, code FROM results ORDER BY issue DESC LIMIT {HISTORY_RETENTION_LIMIT}")
        rows = cursor.fetchall()
        conn.close()
        for r in reversed(rows):
            state.ram_history.append({'issue': str(r[0]), 'actual_number': int(r[1])})
        return len(state.ram_history)
    except Exception as e:
        logger.error(f"DB Load Error: {e}")
        return 0

# =============================================================================
# UI & ENGINE SYNC
# =============================================================================

async def sync_dashboard(status: str, timer: int):
    """Updates the JSON bridge file for server.py."""
    total = state.wins + state.losses
    acc = f"{(state.wins / total) * 100:.1f}%" if total > 0 else "0.0%"
    
    payload = {
        "period": state.active_prediction['issue'] or "---",
        "prediction": state.active_prediction['label'],
        "confidence": f"{state.active_prediction.get('conf', 0)*100:.1f}%",
        "stake": state.active_prediction['stake'],
        "level": state.active_prediction.get('level', '---'),
        "bankroll": state.bankroll,
        "lastresult_status": state.last_win_status,
        "status_text": status,
        "timer": timer,
        "data_size": len(state.ram_history),
        "stats": {"wins": state.wins, "losses": state.losses, "accuracy": acc},
        "history": list(state.ui_history),
        "timestamp": time.time()
    }

    try:
        temp_path = DASHBOARD_PATH + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_path, DASHBOARD_PATH)
    except Exception as e:
        logger.error(f"Dashboard Sync Error: {e}")

# =============================================================================
# NETWORK LAYER
# =============================================================================

async def execute_api_fetch(session: aiohttp.ClientSession, limit: int) -> List[Dict]:
    """
    Fetches data using GET.
    CRITICAL FIX: content_type=None allows decoding 'application/octet-stream'.
    """
    params = {
        "pageSize": limit,
        "pageNo": 1,
        "typeId": 1,
        "language": 0,
        "timestamp": int(time.time() * 1000)
    }

    try:
        async with session.get(TARGET_URL, headers=HEADERS, params=params, timeout=10) as response:
            if response.status == 200:
                # content_type=None bypasses the mimetype validation error
                json_data = await response.json(content_type=None)
                return json_data.get('data', {}).get('list', []) or json_data.get('list', [])
            else:
                logger.warning(f"API Rejection: Status {response.status}")
    except Exception as e:
        logger.error(f"Network Error: {e}")
    
    return []

# =============================================================================
# MAIN OPERATIONAL LOOP
# =============================================================================

async def run_fetcher_engine():
    """Main loop for data synchronization and AI prediction."""
    ensure_db_setup()
    await load_db_to_ram()
    
    logger.info(f"--- TITAN V700 STARTED | {len(state.ram_history)} RECORDS LOADED ---")
    
    async with aiohttp.ClientSession() as session:
        while True:
            raw_list = await execute_api_fetch(session, 10)
            
            if raw_list:
                # 1. Update Database and RAM with any new rounds in the batch
                for item in reversed(raw_list):
                    iss = str(item.get('issueNumber') or item.get('issue'))
                    num = int(item.get('number') or item.get('result'))
                    
                    if not any(d['issue'] == iss for d in state.ram_history):
                        await save_to_db(iss, num)
                        state.ram_history.append({'issue': iss, 'actual_number': num})
                        logger.info(f"[DATABASE] Stored Round {iss}")

                # 2. Process Latest Round
                latest = raw_list[0]
                curr_issue = str(latest.get('issueNumber') or latest.get('issue'))
                curr_num = int(latest.get('number') or latest.get('result'))
                
                seconds_remaining = 60 - datetime.now().second
                await sync_dashboard("LIVE", seconds_remaining)
                
                if curr_issue != state.last_processed_issue:
                    logger.info(f"[NEW ROUND] {curr_issue} | Result: {curr_num}")
                    
                    # 3. Verify Previous Prediction Result
                    if state.active_prediction['issue'] == curr_issue:
                        real_outcome = get_outcome_from_number(curr_num)
                        pred_label = state.active_prediction['label']
                        
                        if pred_label not in ["WAITING", "SKIP"]:
                            win = (pred_label == real_outcome)
                            state.last_win_status = "WIN" if win else "LOSS"
                            if win: state.wins += 1
                            else: state.losses += 1
                            
                            state.ui_history.appendleft({
                                "period": curr_issue, "pred": pred_label, "result": state.last_win_status
                            })

                    # 4. Generate Next Prediction
                    if len(state.ram_history) >= MIN_BRAIN_CAPACITY:
                        next_issue = str(int(curr_issue) + 1)
                        try:
                            ai_output = ultraAIPredict(list(state.ram_history), state.bankroll, get_outcome_from_number(curr_num))
                            state.active_prediction = {
                                "issue": next_issue,
                                "label": ai_output['finalDecision'],
                                "stake": ai_output['positionsize'],
                                "conf": ai_output['confidence'],
                                "level": ai_output['level'],
                                "reason": ai_output.get('reason', 'Analyzing patterns...')
                            }
                            logger.info(f"[AI] Target: {next_issue} | Decision: {ai_output['finalDecision']}")
                        except Exception as e:
                            logger.error(f"Engine Error: {e}")
                    
                    state.last_processed_issue = curr_issue
            
            await asyncio.sleep(LIVE_POLL_INTERVAL)

if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(run_fetcher_engine())
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
