from flask import Flask, render_template_string, jsonify, request, session, redirect, url_for
import sqlite3
import os
import json
import functools
import uuid
import secrets
import threading
import asyncio
from datetime import datetime, timedelta

# --- IMPORT FETCHER ---
# This ensures the prediction logic runs in the background
import fetcher

app = Flask(__name__)

# --- CONFIGURATION ---
# CHANGE THIS TO A RANDOM SECRET STRING FOR SECURITY
app.secret_key = "TITAN_SECURE_KEY_CHANGE_THIS" 
ADMIN_PASSWORD = "admin" 

# --- RENDER DISK SETUP ---
# Render specific paths to ensure database is not deleted on restart
if os.path.exists('/var/lib/data'):
    BASE_DIR = '/var/lib/data'
elif os.path.exists('/data'):
    BASE_DIR = '/data'
else:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DB_PATH = os.path.join(BASE_DIR, 'titan_db.sqlite')
DASHBOARD_FILE = os.path.join(BASE_DIR, 'dashboard_data.json')

print(f"[SYSTEM] Running with Storage Path: {BASE_DIR}")

def create_connection():
    """Creates a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_tables():
    """Sets up the database tables for Keys and Sessions."""
    conn = create_connection()
    
    # 1. ACCESS KEYS TABLE
    # Stores the key, the user note, expiration, and the LOCKED DEVICE ID
    conn.execute('''CREATE TABLE IF NOT EXISTS access_keys (
                    key_code TEXT PRIMARY KEY,
                    note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    bound_device_id TEXT,
                    is_active INTEGER DEFAULT 1
                )''')
    
    # 2. ACTIVE SESSIONS TABLE
    # Tracks who is currently logged in
    conn.execute('''CREATE TABLE IF NOT EXISTS active_sessions (
                    session_id TEXT PRIMARY KEY,
                    key_code TEXT,
                    ip_address TEXT,
                    last_heartbeat TIMESTAMP
                )''')
    
    conn.commit()
    conn.close()

# Initialize Database
ensure_tables()

# --- BACKGROUND WORKER ---
def start_fetcher_loop():
    """Runs the fetcher in a background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try: 
        loop.run_until_complete(fetcher.main_loop())
    except Exception as e: 
        print(f"Fetcher Error: {e}")

# Only start the background thread if this is the main process
if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
    t = threading.Thread(target=start_fetcher_loop, daemon=True)
    t.start()

# --- AUTHENTICATION HELPERS ---
def cleanup_sessions():
    """Removes sessions that have been inactive for >60 seconds."""
    conn = create_connection()
    limit = datetime.now() - timedelta(seconds=60)
    conn.execute("DELETE FROM active_sessions WHERE last_heartbeat < ?", (limit,))
    conn.commit()
    conn.close()

def login_required(f):
    """Decorator to protect routes."""
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapped

def admin_required(f):
    """Decorator to protect Admin Panel."""
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('is_admin'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return wrapped

# --- ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        key_input = request.form.get('key', '').strip()
        device_fingerprint = request.form.get('device_id', '').strip() # Received from JavaScript
        
        cleanup_sessions()
        
        conn = create_connection()
        # 1. Look up the key
        key_data = conn.execute("SELECT * FROM access_keys WHERE key_code = ?", (key_input,)).fetchone()
        
        if not key_data:
            error = "ERROR: INVALID KEY"
        elif key_data['is_active'] == 0:
            error = "ERROR: KEY IS BANNED"
        else:
            # 2. Check Expiration
            expire_dt = datetime.strptime(key_data['expires_at'], '%Y-%m-%d %H:%M:%S')
            if datetime.now() > expire_dt:
                error = "ERROR: KEY EXPIRED"
            else:
                # 3. DEVICE LOCK CHECK (CRITICAL)
                current_bound_id = key_data['bound_device_id']
                
                # If key has no owner yet, bind it to this phone PERMANENTLY
                if not current_bound_id:
                    conn.execute("UPDATE access_keys SET bound_device_id = ? WHERE key_code = ?", 
                                 (device_fingerprint, key_input))
                    conn.commit()
                # If key is already owned, check if it matches this phone
                elif current_bound_id != device_fingerprint:
                    error = "ACCESS DENIED: Key is locked to a different device."
                
                if not error:
                    # SUCCESSFUL LOGIN
                    new_sess_id = str(uuid.uuid4())
                    session['authenticated'] = True
                    session['user_key'] = key_input
                    session['session_uuid'] = new_sess_id
                    
                    # Log the session
                    conn.execute("INSERT INTO active_sessions (session_id, key_code, ip_address, last_heartbeat) VALUES (?, ?, ?, ?)",
                                 (new_sess_id, key_input, request.remote_addr, datetime.now()))
                    conn.commit()
                    conn.close()
                    return redirect(url_for('index'))

        conn.close()

    # LOGIN SCREEN HTML (Includes Device Fingerprinting Script)
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ background-color: #050505; color: #00ff41; font-family: 'Courier New', monospace; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
            input {{ background: #111; border: 1px solid #333; color: white; padding: 12px; width: 280px; text-align: center; font-size: 16px; margin-bottom: 20px; outline: none; }}
            button {{ background: #00ff41; color: black; border: none; padding: 12px 30px; font-weight: bold; font-size: 16px; cursor: pointer; }}
            .error {{ color: #ff0055; margin-bottom: 20px; font-weight: bold; }}
            h1 {{ margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <h1>TITAN V700</h1>
        <div style="color:#666; margin-bottom:30px; font-size:12px;">SECURE ACCESS PORTAL</div>
        
        <div class="error">{error if error else ''}</div>
        
        <form method="post" id="loginForm">
            <input type="hidden" name="device_id" id="device_id_field">
            <input type="text" name="key" placeholder="ENTER LICENSE KEY" autocomplete="off">
            <br>
            <button type="submit">AUTHENTICATE</button>
        </form>

        <script>
            // 1. DEVICE FINGERPRINT GENERATION
            // This runs silently to create a unique ID for this phone
            let dId = localStorage.getItem('titan_device_id');
            if (!dId) {{
                // Generate a random ID if one doesn't exist
                dId = 'DEV-' + Math.random().toString(36).substr(2, 9).toUpperCase() + '-' + Date.now();
                localStorage.setItem('titan_device_id', dId);
            }}
            // 2. Inject ID into the form so the server receives it
            document.getElementById('device_id_field').value = dId;
        </script>
    </body>
    </html>
    """

@app.route('/logout')
def logout():
    # Kill the session in the DB
    if 'session_uuid' in session:
        conn = create_connection()
        conn.execute("DELETE FROM active_sessions WHERE session_id = ?", (session['session_uuid'],))
        conn.commit()
        conn.close()
    session.clear()
    return redirect(url_for('login'))

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    """Called by the client every few seconds to keep the session alive."""
    if 'session_uuid' in session:
        conn = create_connection()
        conn.execute("UPDATE active_sessions SET last_heartbeat = ? WHERE session_id = ?", (datetime.now(), session['session_uuid']))
        conn.commit()
        conn.close()
    return jsonify({"status": "ok"})

@app.route('/data')
@login_required
def data():
    """Serves the prediction data."""
    # Mini-heartbeat
    if 'session_uuid' in session:
        conn = create_connection()
        conn.execute("UPDATE active_sessions SET last_heartbeat = ? WHERE session_id = ?", (datetime.now(), session['session_uuid']))
        conn.close()

    try:
        if os.path.exists(DASHBOARD_FILE):
            with open(DASHBOARD_FILE, 'r') as f: return jsonify(json.load(f))
    except: pass
    return jsonify({"prediction": "LOADING...", "status_text": "SYSTEM INITIALIZING..."})

# --- ADMIN PANEL ---

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form.get('password') == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('admin_panel'))
    return '<body style="background:#111; color:white; display:flex; justify-content:center; align-items:center; height:100vh;"><form method="post"><input type="password" name="password" placeholder="Admin Password" style="padding:10px;"><button>Login</button></form></body>'

@app.route('/admin', methods=['GET', 'POST'])
@admin_required
def admin_panel():
    cleanup_sessions()
    conn = create_connection()
    msg = ""

    # ACTION: GENERATE NEW KEY
    if request.method == 'POST' and 'create_key' in request.form:
        name = request.form.get('note', 'User')
        days = int(request.form.get('days', 30))
        
        # Create Key: TITAN-USER-RANDOM
        rand_str = secrets.token_hex(4).upper()
        clean_name = "".join(c for c in name if c.isalnum())
        new_key = f"TITAN-{clean_name.upper()}-{rand_str}"
        
        expire_str = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            conn.execute("INSERT INTO access_keys (key_code, note, expires_at) VALUES (?, ?, ?)", 
                         (new_key, name, expire_str))
            conn.commit()
            msg = f"SUCCESS: Generated {new_key}"
        except Exception as e:
            msg = f"ERROR: {e}"

    # ACTION: DELETE KEY
    if request.method == 'POST' and 'delete_key' in request.form:
        target_key = request.form.get('delete_key')
        conn.execute("DELETE FROM access_keys WHERE key_code = ?", (target_key,))
        conn.commit()
        msg = f"Deleted key {target_key}"

    # ACTION: RESET DEVICE (If user changes phone)
    if request.method == 'POST' and 'reset_device' in request.form:
        target_key = request.form.get('reset_device')
        conn.execute("UPDATE access_keys SET bound_device_id = NULL WHERE key_code = ?", (target_key,))
        conn.commit()
        msg = f"Device Lock RESET for {target_key}. User can now link a new phone."

    # FETCH DATA FOR TABLE
    keys = conn.execute("SELECT * FROM access_keys ORDER BY created_at DESC").fetchall()
    
    # Check Online Status
    online_map = {}
    sessions = conn.execute("SELECT key_code FROM active_sessions").fetchall()
    for s in sessions: online_map[s['key_code']] = True
    
    conn.close()
    
    # BUILD HTML TABLE
    rows = ""
    for k in keys:
        is_online = online_map.get(k['key_code'], False)
        status_dot = "🟢" if is_online else "⚪"
        
        # Check if device is locked
        device_status = f"<span style='color:orange'>LOCKED ({k['bound_device_id'][:8]}...)</span>" if k['bound_device_id'] else "<span style='color:green'>OPEN (No Device)</span>"
        
        rows += f"""
        <tr style="border-bottom:1px solid #ddd; height:40px;">
            <td style="padding:10px;"><strong>{k['key_code']}</strong></td>
            <td>{k['note']}</td>
            <td>{status_dot}</td>
            <td style="font-size:12px;">{device_status}</td>
            <td style="font-size:12px;">{k['expires_at']}</td>
            <td>
                <form method="POST" style="display:inline;" onsubmit="return confirm('Allow new device?');">
                    <input type="hidden" name="reset_device" value="{k['key_code']}">
                    <button style="font-size:10px; cursor:pointer; background:#eee;">RESET ID</button>
                </form>
                <form method="POST" style="display:inline;" onsubmit="return confirm('Delete Key?');">
                    <input type="hidden" name="delete_key" value="{k['key_code']}">
                    <button style="color:white; background:red; border:none; font-size:10px; cursor:pointer; padding:2px 5px;">DEL</button>
                </form>
            </td>
        </tr>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>Titan Admin</title></head>
    <body style="font-family:sans-serif; background:#f4f4f4; padding:20px;">
        <div style="max-width:1000px; margin:0 auto;">
            <h1 style="margin-bottom:5px;">TITAN V700 MANAGER</h1>
            <p style="color:blue; font-weight:bold;">{msg}</p>
            
            <div style="background:white; padding:20px; border-radius:8px; box-shadow:0 2px 5px rgba(0,0,0,0.1); margin-bottom:20px;">
                <h3 style="margin-top:0;">Generate New License</h3>
                <form method="POST" style="display:flex; gap:10px;">
                    <input type="hidden" name="create_key" value="1">
                    <input type="text" name="note" placeholder="User Name (e.g. Client1)" required style="padding:8px;">
                    <input type="number" name="days" value="30" style="width:60px; padding:8px;" title="Days">
                    <button style="padding:8px 20px; background:black; color:white; border:none; cursor:pointer;">CREATE KEY</button>
                </form>
            </div>

            <table style="width:100%; border-collapse:collapse; background:white; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
                <thead style="background:#222; color:white;">
                    <tr>
                        <th style="padding:10px; text-align:left;">License Key</th>
                        <th style="text-align:left;">User</th>
                        <th style="text-align:left;">Status</th>
                        <th style="text-align:left;">Device Lock</th>
                        <th style="text-align:left;">Expires</th>
                        <th style="text-align:left;">Actions</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
            <br>
            <a href="/">Go to Dashboard</a> | <a href="/logout">Logout</a>
        </div>
    </body>
    </html>
    """

@app.route('/')
@login_required
def index():
    return render_template_string(HTML_TEMPLATE)

# --- CLIENT DASHBOARD TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>TITAN V700 LIVE</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;800&display=swap');
        body { background-color: #050505; color: #fff; font-family: 'JetBrains Mono', monospace; margin: 0; padding: 10px; display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
        
        .card { width: 100%; max-width: 400px; background: #111; border: 1px solid #222; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
        
        .header { display: flex; justify-content: space-between; width: 100%; max-width: 400px; margin-bottom: 20px; border-bottom: 1px solid #222; padding-bottom: 10px; }
        .logo { font-weight: 800; font-size: 18px; }
        .accent { color: #00ff41; }
        
        .period { color: #666; font-size: 14px; margin-top: 10px; }
        .prediction { font-size: 56px; font-weight: 800; margin: 15px 0; line-height: 1; }
        .big { color: #00ff41; text-shadow: 0 0 30px rgba(0,255,65,0.3); }
        .small { color: #ff0055; text-shadow: 0 0 30px rgba(255,0,85,0.3); }
        .wait { color: #444; }
        
        .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 20px; }
        .info-box { background: #1a1a1a; padding: 10px; border-radius: 6px; font-size: 12px; }
        .info-label { color: #666; font-size: 10px; display: block; margin-bottom: 4px; }
        
        .timer-bar { width: 100%; height: 4px; background: #222; margin-bottom: 20px; border-radius: 2px; overflow: hidden; }
        .fill { height: 100%; background: #00ff41; width: 0%; transition: width 0.5s linear; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">TITAN <span class="accent">V700</span></div>
        <a href="/logout" style="color:#666; text-decoration:none; font-size:12px;">LOGOUT</a>
    </div>

    <div class="card">
        <div class="timer-bar"><div class="fill" id="timer"></div></div>
        
        <div class="period" id="period">WAITING FOR SERVER...</div>
        
        <div id="prediction" class="prediction wait">---</div>
        
        <div style="color:#888; font-size:12px;" id="status">CONNECTING...</div>

        <div class="info-grid">
            <div class="info-box">
                <span class="info-label">CONFIDENCE</span>
                <span id="conf" style="font-weight:bold; color:white;">0%</span>
            </div>
            <div class="info-box">
                <span class="info-label">STRATEGY</span>
                <span id="strat" style="font-weight:bold; color:white;">---</span>
            </div>
        </div>
        
        <div style="margin-top:15px; font-size:10px; color:#444;">
            SESSION WINS: <span id="wins" style="color:#00ff41">0</span> | LOSS: <span id="losses" style="color:#ff0055">0</span>
        </div>
    </div>

    <script>
        // --- HEARTBEAT SYSTEM (Keeps session alive) ---
        setInterval(() => {
            fetch('/heartbeat', {method: 'POST'});
        }, 5000); // Ping every 5 seconds

        function update() {
            fetch('/data').then(r => r.json()).then(d => {
                // Period
                document.getElementById('period').innerText = "PERIOD " + d.period;
                
                // Prediction Display
                const p = document.getElementById('prediction');
                p.innerText = d.prediction;
                p.className = 'prediction'; // reset
                if (d.prediction === 'BIG') p.classList.add('big');
                else if (d.prediction === 'SMALL') p.classList.add('small');
                else p.classList.add('wait');
                
                // Details
                document.getElementById('status').innerText = d.status_text;
                document.getElementById('conf').innerText = d.confidence || "0%";
                document.getElementById('strat').innerText = d.strategy || "---";
                document.getElementById('wins').innerText = d.stats ? d.stats.wins : 0;
                document.getElementById('losses').innerText = d.stats ? d.stats.losses : 0;

                // Timer Visual
                const time = d.timer || 0;
                const pct = (time / 60) * 100;
                document.getElementById('timer').style.width = pct + "%";
            });
        }
        
        setInterval(update, 1000); // Update data every second
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
