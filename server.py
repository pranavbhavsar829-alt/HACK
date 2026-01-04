from flask import Flask, render_template_string, jsonify, request, session, redirect, url_for
import sqlite3
import os
import json
import functools
import uuid
import hmac
import hashlib
import base64
import time
import threading
import asyncio
from datetime import datetime, timedelta

# --- CRITICAL FIX: PREVENT ENCODING CRASH ---
import encodings.idna 

import fetcher

app = Flask(__name__)

# --- SECURITY ---
app.secret_key = "TITAN_SECURE_KEY_CHANGE_THIS" 
ADMIN_PASSWORD = "admin" 
OFFLINE_SECRET = "TITAN_OFFLINE_SECRET_CODE_123" 
MASTER_KEY = "TITAN-PERM-ADMIN" 

# --- CRITICAL FIX: MATCH FETCHER PATH ---
if os.path.exists('/var/lib/data'):
    BASE_DIR = '/var/lib/data'
else:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DB_PATH = os.path.join(BASE_DIR, 'titan_db.sqlite')
DASHBOARD_FILE = os.path.join(BASE_DIR, 'dashboard_data.json') # <--- NOW MATCHES FETCHER

def create_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_tables():
    conn = create_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS access_keys (
                    key_code TEXT PRIMARY KEY,
                    note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 1,
                    expires_at TEXT,
                    max_devices INTEGER DEFAULT 1,
                    bound_device_id TEXT
                )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS active_sessions (
                    session_id TEXT PRIMARY KEY,
                    key_code TEXT,
                    last_seen TIMESTAMP,
                    ip_address TEXT
                )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS blacklisted_keys (
                    key_code TEXT PRIMARY KEY, 
                    reason TEXT, 
                    banned_at TEXT
                )''')
    conn.commit()
    conn.close()

def cleanup_inactive_sessions():
    conn = create_connection()
    limit = datetime.now() - timedelta(minutes=5)
    conn.execute("DELETE FROM active_sessions WHERE last_seen < ?", (limit,))
    conn.commit()
    conn.close()

try: ensure_tables()
except: pass

# --- BACKGROUND WORKER ---
def start_fetcher_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try: loop.run_until_complete(fetcher.main_loop())
    except Exception as e: print(f"Fetcher Error: {e}")

if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
    t = threading.Thread(target=start_fetcher_loop, daemon=True)
    t.start()

# --- AUTH ---
def get_name_from_key(key):
    if key == MASTER_KEY: return "ADMIN"
    if not key.startswith("TITAN-"): return "Legacy/Unknown"
    try:
        payload_b64 = key.split('-')[1]
        padding = len(payload_b64) % 4
        if padding: payload_b64 += '=' * (4 - padding)
        payload = base64.urlsafe_b64decode(payload_b64).decode()
        return payload.split('|')[2]
    except: return "Unknown User"

def login_required(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('authenticated'): return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapped

def admin_required(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('is_admin'): return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return wrapped

# --- ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        key = request.form.get('key', '').strip()
        conn = create_connection()
        banned = conn.execute("SELECT * FROM blacklisted_keys WHERE key_code = ?", (key,)).fetchone()
        conn.close()
        
        if banned:
            error = "ACCESS DENIED: KEY IS BANNED"
        elif key.startswith("TITAN-") or key == MASTER_KEY:
            session['authenticated'] = True
            session['user_key'] = key
            conn = create_connection()
            try:
                conn.execute("INSERT OR REPLACE INTO active_sessions (session_id, key_code, last_seen) VALUES (?, ?, ?)", 
                             (str(uuid.uuid4()), key, datetime.now()))
                conn.commit()
            except: pass
            conn.close()
            return redirect(url_for('index'))
        else:
            error = "INVALID KEY FORMAT"
    return f"""<body style="background:#000; color:#0f0; display:flex; justify-content:center; align-items:center; height:100vh; font-family:monospace; flex-direction:column;">
            <h1>TITAN V700 ACCESS</h1><p style="color:red">{error if error else ''}</p>
            <form method="post"><input type="text" name="key" placeholder="PASTE ACCESS KEY" style="padding:10px; width:250px; text-align:center;"><br><br><button style="padding:10px 20px; font-weight:bold; cursor:pointer;">UNLOCK SYSTEM</button></form></body>"""

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    return jsonify({"status": "ok"})

@app.route('/data')
@login_required
def data():
    try:
        if os.path.exists(DASHBOARD_FILE):
            with open(DASHBOARD_FILE, 'r') as f: return jsonify(json.load(f))
    except: pass
    return jsonify({"prediction": "LOADING...", "status_text": "WAITING FOR FETCHER..."})

# --- ADMIN PANEL ---
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form.get('password') == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('admin_panel'))
    return '<body style="background:#111; color:white; display:flex; justify-content:center; align-items:center; height:100vh;"><form method="post" style="text-align:center;"><input type="password" name="password" placeholder="Admin Password" style="padding:10px;"><button style="padding:10px; cursor:pointer;">Login</button></form></body>'

@app.route('/admin', methods=['GET', 'POST'])
@admin_required
def admin_panel():
    cleanup_inactive_sessions()
    conn = create_connection()
    if request.method == 'POST' and 'ban_key' in request.form:
        b_key = request.form.get('ban_key').strip()
        conn.execute("INSERT OR REPLACE INTO blacklisted_keys (key_code, reason, banned_at) VALUES (?, ?, ?)", (b_key, "Admin Ban", datetime.now()))
        conn.commit()
        conn.execute("DELETE FROM active_sessions WHERE key_code = ?", (b_key,))
        conn.commit()
    active = conn.execute("SELECT * FROM active_sessions ORDER BY last_seen DESC").fetchall()
    banned = conn.execute("SELECT * FROM blacklisted_keys ORDER BY banned_at DESC").fetchall()
    conn.close()
    active_rows = ""
    for s in active:
        k = s['key_code']
        name = get_name_from_key(k)
        active_rows += f"<tr><td style='color:#00ff41; font-weight:bold;'>{name}</td><td style='font-size:12px;'>{k[:20]}...</td><td>{s['last_seen']}</td><td><form method='POST' style='margin:0;'><input type='hidden' name='ban_key' value='{k}'><button style='background:red; color:white; border:none; padding:5px; cursor:pointer;'>BAN</button></form></td></tr>"
    ban_rows = "".join([f"<tr><td>{b[0][:30]}...</td><td style='color:red'>BANNED</td></tr>" for b in banned])
    return f"""<body style="font-family:monospace; padding:20px; background:#f0f0f0;">
            <h1>ADMIN PANEL - TITAN V700</h1>
            <div style="background:white; padding:15px; border:1px solid #ccc; margin-bottom:20px;">
                <h3 style="margin-top:0; color:green;">LIVE ACTIVE USERS ({len(active)})</h3>
                <table border="1" cellpadding="5" style="width:100%; border-collapse:collapse;">
                    <tr style="background:#eee;"><th>User</th><th>Key</th><th>Last Seen</th><th>Action</th></tr>
                    {active_rows if active_rows else "<tr><td colspan='4' style='text-align:center'>No users online</td></tr>"}
                </table>
            </div>
            <div style="background:#ffdddd; padding:15px; border:1px solid red;">
                <h3 style="margin-top:0; color:darkred;">BANNED KEYS</h3>
                <form method="POST" style="margin-bottom:10px;">
                    <input type="text" name="ban_key" placeholder="Paste Key to ban..." style="width:300px; padding:5px;">
                    <button style="padding:5px; cursor:pointer;">Ban Key</button>
                </form>
                <table border="1" cellpadding="5" style="width:100%; background:white;">{ban_rows}</table>
            </div>
            <br><a href="/">Back</a> | <a href="/logout">Logout</a></body>"""

@app.route('/')
@login_required
def index():
    return render_template_string(HTML_TEMPLATE)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>TITAN V700 SOVEREIGN</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
        :root { --bg: #050505; --card: #111; --text: #fff; --accent: #00ff41; --loss: #ff0055; }
        body { background-color: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; margin: 0; padding: 10px; display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
        .header { width: 100%; max-width: 480px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #222; }
        .dashboard-grid { width: 100%; max-width: 480px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 15px; }
        .stat-card { background: var(--card); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #222; }
        .stat-val { font-size: 16px; font-weight: 900; margin-top: 4px; font-family: 'JetBrains Mono'; }
        .main-card { width: 100%; max-width: 480px; background: var(--card); border-radius: 16px; padding: 20px; text-align: center; border: 1px solid #222; box-shadow: 0 0 40px rgba(0,0,0,0.5); margin-bottom: 20px; position: relative; overflow: hidden; }
        .prediction-display { font-size: 64px; font-weight: 900; text-transform: uppercase; margin: 15px 0; line-height: 1; letter-spacing: -2px; }
        .res-big { color: var(--accent); } .res-small { color: var(--loss); } .res-wait { color: #333; }
        .history-list { width: 100%; max-width: 480px; background: var(--card); border-radius: 12px; border: 1px solid #222; overflow-y: auto; max-height: 300px; }
        .history-item { display: flex; justify-content: space-between; padding: 10px 15px; border-bottom: 1px solid #1a1a1a; font-size: 12px; font-family: 'JetBrains Mono'; }
        .h-win { color: var(--accent); } .h-loss { color: var(--loss); }
    </style>
</head>
<body>
    <div class="header">
        <div style="font-weight:900; font-size:20px;">TITAN <span style="color:var(--accent)">V700</span></div>
        <a href="/logout" style="color:#444; text-decoration:none; font-size:12px;">EXIT</a>
    </div>

    <div class="dashboard-grid">
        <div class="stat-card"><div style="font-size:10px; color:#666;">WINS</div><div class="stat-val" style="color:var(--accent)" id="s-wins">0</div></div>
        <div class="stat-card"><div style="font-size:10px; color:#666;">LOSS</div><div class="stat-val" style="color:var(--loss)" id="s-loss">0</div></div>
        <div class="stat-card"><div style="font-size:10px; color:#666;">ACCURACY</div><div class="stat-val" id="s-acc">0%</div></div>
    </div>

    <div class="main-card">
        <div style="width:100%; height:4px; background:#222; position:absolute; top:0; left:0;"><div style="height:100%; background:var(--accent); width:0%; transition:width 1s linear;" id="t-bar"></div></div>
        <div style="font-family:'JetBrains Mono'; color:#555; font-size:13px; margin-top:10px;" id="period">PERIOD: ---</div>
        <div id="prediction" class="prediction-display res-wait">---</div>
        <div style="font-family:'JetBrains Mono'; font-size:28px; font-weight:bold; color:#fff; margin-bottom:5px;" id="countdown">00</div>
        <div style="font-size:10px; color:#444;" id="status-text">SYNCING...</div>
        <div style="background:#0a0a0a; border:1px solid #222; border-radius:6px; padding:8px; font-size:10px; color:#888; font-family:'JetBrains Mono'; margin-top:15px;">
            LOGIC: <span id="reason-text" style="color:#ccc;">Waiting...</span>
        </div>
        <div style="margin-top:10px; font-size:10px; color:#555;">
            BRAIN CAPACITY: <span id="mem-count" style="color:var(--accent)">0</span> RECORDS
        </div>
    </div>
    
    <div class="history-list" id="history-box">
        <div class="history-item" style="justify-content:center; color:#444;">No history available</div>
    </div>

    <script>
        function update() {
            fetch('/data').then(r => r.json()).then(d => {
                document.getElementById('period').innerText = "PERIOD: " + d.period;
                document.getElementById('status-text').innerText = d.status_text;
                document.getElementById('s-wins').innerText = d.stats.wins;
                document.getElementById('s-loss').innerText = d.stats.losses;
                document.getElementById('s-acc').innerText = d.stats.accuracy;
                document.getElementById('mem-count').innerText = d.data_size || 0;
                
                const p = document.getElementById('prediction');
                p.innerText = d.prediction;
                p.className = 'prediction-display';
                if (d.prediction === 'BIG') p.classList.add('res-big');
                else if (d.prediction === 'SMALL') p.classList.add('res-small');
                else p.classList.add('res-wait');

                document.getElementById('reason-text').innerText = d.reason;

                const sec = d.timer || 0;
                document.getElementById('countdown').innerText = sec < 10 ? "0"+sec : sec;
                document.getElementById('t-bar').style.width = (sec/60 * 100) + "%";

                if (d.history && d.history.length > 0) {
                    let html = "";
                    d.history.forEach(h => {
                        let cls = h.result === 'WIN' ? 'h-win' : 'h-loss';
                        html += `<div class="history-item"><span style="color:#666">${h.period}</span><span style="font-weight:bold">${h.pred}</span><span class="${cls}">${h.result}</span></div>`;
                    });
                    document.getElementById('history-box').innerHTML = html;
                }
            });
        }
        setInterval(update, 1000);
        update();
    </script>
</body>
</html>
