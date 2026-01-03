from flask import Flask, render_template_string, jsonify, request, session, redirect, url_for
import sqlite3
import secrets
import os
import json
import functools
import uuid
from datetime import datetime, timedelta
from database_handler import create_connection, DB_PATH, ensure_setup, cleanup_inactive_sessions

app = Flask(__name__)
app.secret_key = "TITAN_SECURE_KEY_CHANGE_THIS" 
ADMIN_PASSWORD = "pranav1920"  # <--- SET YOUR PASSWORD

DASHBOARD_FILE = os.path.join(DB_PATH, 'dashboard_data.json')
ensure_setup()

# --- AUTH DECORATORS ---
def login_required(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapped

def admin_required(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('is_admin'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return wrapped

# --- KEY & SESSION LOGIC ---
def validate_key_and_login(key_input):
    cleanup_inactive_sessions() # Remove stale users first
    
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM access_keys WHERE key_code = ? AND is_active = 1", (key_input,))
    row = cur.fetchone()
    
    if not row:
        conn.close()
        return False, "INVALID KEY"
        
    # Check Expiration (Index 4)
    expires_at = row[4]
    if expires_at:
        exp_date = datetime.strptime(expires_at, '%Y-%m-%d %H:%M:%S.%f')
        if datetime.now() > exp_date:
            conn.close()
            return False, "KEY EXPIRED"

    # Check Device Limit (Index 5 is max_devices)
    max_devices = row[5] if len(row) > 5 else 1
    
    # Count current active sessions for this key
    cur.execute("SELECT COUNT(*) FROM active_sessions WHERE key_code = ?", (key_input,))
    current_users = cur.fetchone()[0]
    
    # If the user is already logged in (same session), don't block them
    my_session_id = session.get('session_uuid')
    
    # Allow login if: Slots open OR I am already one of the active users
    if current_users < max_devices or (my_session_id and check_is_active(my_session_id)):
        # Register Session
        new_uuid = my_session_id if my_session_id else str(uuid.uuid4())
        session['session_uuid'] = new_uuid
        conn.execute("INSERT OR REPLACE INTO active_sessions (session_id, key_code, last_seen) VALUES (?, ?, ?)", 
                     (new_uuid, key_input, datetime.now()))
        conn.commit()
        conn.close()
        return True, "OK"
    else:
        conn.close()
        return False, f"MAX DEVICES REACHED ({current_users}/{max_devices})"

def check_is_active(sess_id):
    conn = create_connection()
    row = conn.execute("SELECT * FROM active_sessions WHERE session_id = ?", (sess_id,)).fetchone()
    conn.close()
    return row is not None

def generate_key(days=30, max_dev=1, note="Generated"):
    key = "KEY-" + secrets.token_hex(4).upper()
    expiration_date = datetime.now() + timedelta(days=int(days))
    conn = create_connection()
    conn.execute("INSERT INTO access_keys (key_code, note, expires_at, max_devices) VALUES (?, ?, ?, ?)", 
                 (key, note, expiration_date, int(max_dev)))
    conn.commit()
    conn.close()

def revoke_key(key):
    conn = create_connection()
    conn.execute("DELETE FROM access_keys WHERE key_code = ?", (key,))
    conn.execute("DELETE FROM active_sessions WHERE key_code = ?", (key,)) # Kick users
    conn.commit()
    conn.close()

# --- ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        key = request.form.get('key', '').strip()
        is_valid, msg = validate_key_and_login(key)
        if is_valid:
            session['authenticated'] = True
            session['user_key'] = key
            return redirect(url_for('index'))
        else:
            error = msg
    
    return f"""
    <body style="background:#000; color:#0f0; display:flex; justify-content:center; align-items:center; height:100vh; font-family:monospace;">
        <div style="text-align:center; border:1px solid #333; padding:40px; border-radius:10px;">
            <h1>🔒 SYSTEM LOCKED</h1>
            <p style="color:red">{error if error else ''}</p>
            <form method="post">
                <input type="text" name="key" placeholder="ENTER LICENSE KEY" style="padding:10px; width:250px; text-align:center;">
                <br><br>
                <button style="padding:10px 20px; font-weight:bold; cursor:pointer;">UNLOCK</button>
            </form>
        </div>
    </body>
    """

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    if session.get('authenticated') and session.get('session_uuid'):
        conn = create_connection()
        conn.execute("UPDATE active_sessions SET last_seen = ? WHERE session_id = ?", 
                     (datetime.now(), session['session_uuid']))
        conn.commit()
        conn.close()
        return jsonify({"status": "ok"})
    return jsonify({"status": "ignored"})

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form.get('password') == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('admin_panel'))
    return '<form method="post" style="text-align:center; margin-top:50px;"><input type="password" name="password" placeholder="Admin Pass"><button>Login</button></form>'

@app.route('/admin')
@admin_required
def admin_panel():
    cleanup_inactive_sessions() # Clean list before showing
    conn = create_connection()
    keys = conn.execute("SELECT * FROM access_keys ORDER BY created_at DESC").fetchall()
    
    # Get active counts
    active_counts = {}
    sessions = conn.execute("SELECT key_code, COUNT(*) FROM active_sessions GROUP BY key_code").fetchall()
    for s in sessions:
        active_counts[s[0]] = s[1]
    conn.close()
    
    rows = ""
    for k in keys:
        # k[0]=key, k[1]=note, k[4]=expires, k[5]=max_devices
        expiry = k[4][:10] if k[4] else "Lifetime"
        max_dev = k[5] if len(k) > 5 else 1
        current = active_counts.get(k[0], 0)
        
        rows += f"""
        <tr>
            <td>{k[0]}</td>
            <td>{k[1]}</td>
            <td>{expiry}</td>
            <td style="text-align:center;">{current} / {max_dev}</td>
            <td><a href='/admin/del/{k[0]}' style='color:red'>REVOKE</a></td>
        </tr>"""
        
    return f"""
    <body style="font-family:monospace; padding:20px;">
        <h1>ADMIN PANEL</h1>
        
        <div style="background:#eee; padding:15px; margin-bottom:20px; border:1px solid #ccc;">
            <h3>GENERATE KEY</h3>
            <form action="/admin/gen" method="POST">
                Validity (Days): <input type="number" name="days" value="30" style="width:50px;">
                Max Users: <input type="number" name="max_dev" value="1" style="width:50px;">
                Note: <input type="text" name="note" placeholder="Client Name" style="width:150px;">
                <button type="submit" style="background:#00ff41; border:none; padding:5px 10px; font-weight:bold; cursor:pointer;">CREATE</button>
            </form>
        </div>

        <a href="/"><button>VIEW DASHBOARD</button></a>
        
        <table border="1" cellpadding="10" style="border-collapse:collapse; margin-top:20px; width:100%;">
            <tr style="background:#ddd;"><th>KEY</th><th>NOTE</th><th>EXPIRES</th><th>USERS</th><th>ACTION</th></tr>
            {rows}
        </table>
    </body>
    """

@app.route('/admin/gen', methods=['POST'])
@admin_required
def gen():
    days = request.form.get('days', 30)
    max_dev = request.form.get('max_dev', 1)
    note = request.form.get('note', 'Manual')
    generate_key(days, max_dev, note)
    return redirect(url_for('admin_panel'))

@app.route('/admin/del/<k>')
@admin_required
def delete(k):
    revoke_key(k)
    return redirect(url_for('admin_panel'))

@app.route('/logout')
def logout():
    if session.get('session_uuid'):
        conn = create_connection()
        conn.execute("DELETE FROM active_sessions WHERE session_id = ?", (session['session_uuid'],))
        conn.commit()
        conn.close()
    session.clear()
    return redirect(url_for('login'))

@app.route('/data')
@login_required
def data():
    try:
        if os.path.exists(DASHBOARD_FILE):
            with open(DASHBOARD_FILE, 'r') as f:
                return jsonify(json.load(f))
    except: pass
    return jsonify({"period": "---", "prediction": "LOADING"})

@app.route('/')
@login_required
def index():
    return render_template_string(HTML_TEMPLATE)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TITAN V4</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap');
        body { background-color: #000; color: #fff; font-family: 'JetBrains Mono', monospace; display: flex; flex-direction: column; align-items: center; min-height: 100vh; margin: 0; padding: 20px; }
        .card { background: #0a0a0a; border: 1px solid #222; border-radius: 12px; padding: 20px; text-align: center; width: 100%; max-width: 500px; margin-bottom: 15px; }
        .pred-box { font-size: 64px; font-weight: 900; margin: 10px 0; }
        .big { color: #00ff41; text-shadow: 0 0 20px rgba(0,255,65,0.3); }
        .small { color: #ff0055; text-shadow: 0 0 20px rgba(255,0,85,0.3); }
        .wait { color: #444; }
        .logout { position: fixed; top: 10px; right: 10px; color: red; text-decoration: none; font-size: 12px; border: 1px solid red; padding: 5px; }
    </style>
</head>
<body>
    <a href="/logout" class="logout">LOGOUT</a>
    <div class="card">
        <div style="font-size:12px; color:#555;">TITAN V4 // LOCKED SYSTEM</div>
        <div id="period" style="font-size:18px; color:#888; margin-top:5px;">PERIOD: ---</div>
        <div id="prediction" class="pred-box wait">---</div>
        <div id="status" style="font-size:12px; color:#444;">CONNECTING...</div>
    </div>
    
    <div class="card" style="text-align:left;">
        <div style="font-size:12px; color:#666; margin-bottom:10px;">RECENT HISTORY</div>
        <div id="history-list" style="font-size:11px; line-height:1.6;"></div>
    </div>

    <script>
        // HEARTBEAT SYSTEM (Keeps session alive)
        setInterval(() => {
            fetch('/heartbeat', { method: 'POST' });
        }, 30000); // Ping every 30 seconds

        function update() {
            fetch('/data').then(r => r.json()).then(d => {
                document.getElementById('period').innerText = "PERIOD: " + d.period;
                document.getElementById('status').innerText = d.status_text || 'ACTIVE';
                const p = document.getElementById('prediction');
                p.innerText = d.prediction;
                p.className = 'pred-box ' + (d.prediction === 'BIG' ? 'big' : d.prediction === 'SMALL' ? 'small' : 'wait');
                let histHtml = "";
                if(d.history) {
                    d.history.forEach(h => {
                        histHtml += `${h.period} | ${h.pred} | ${h.result}<br>`;
                    });
                }
                document.getElementById('history-list').innerHTML = histHtml;
            });
        }
        setInterval(update, 1000);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
