from flask import Flask, render_template_string, jsonify, request, session, redirect, url_for
import sqlite3
import secrets
import os
import json
import functools
from database_handler import create_connection, DB_PATH, ensure_setup

app = Flask(__name__)
# SECURITY SETTINGS
app.secret_key = "TITAN_SECURE_KEY_CHANGE_THIS" 
ADMIN_PASSWORD = "admin" # <--- SET YOUR ADMIN PASSWORD HERE

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

# --- DB HELPERS ---
def validate_key(key):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM access_keys WHERE key_code = ? AND is_active = 1", (key,))
    valid = cur.fetchone() is not None
    conn.close()
    return valid

def generate_key():
    key = "KEY-" + secrets.token_hex(4).upper()
    conn = create_connection()
    conn.execute("INSERT INTO access_keys (key_code, note) VALUES (?, ?)", (key, "Manual Gen"))
    conn.commit()
    conn.close()

def revoke_key(key):
    conn = create_connection()
    conn.execute("DELETE FROM access_keys WHERE key_code = ?", (key,))
    conn.commit()
    conn.close()

# --- ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        key = request.form.get('key', '').strip()
        if validate_key(key):
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            error = "INVALID LICENSE KEY"
    
    return f"""
    <body style="background:#000; color:#0f0; display:flex; justify-content:center; align-items:center; height:100vh; font-family:monospace;">
        <div style="text-align:center; border:1px solid #333; padding:40px; border-radius:10px;">
            <h1>🔒 SYSTEM LOCKED</h1>
            <p style="color:red">{error if error else ''}</p>
            <form method="post">
                <input type="text" name="key" placeholder="ENTER LICENSE KEY" style="padding:10px; width:250px;">
                <br><br>
                <button style="padding:10px 20px; font-weight:bold; cursor:pointer;">UNLOCK</button>
            </form>
        </div>
    </body>
    """

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
    conn = create_connection()
    keys = conn.execute("SELECT * FROM access_keys ORDER BY created_at DESC").fetchall()
    conn.close()
    rows = "".join([f"<tr><td>{k[0]}</td><td>{k[2]}</td><td><a href='/admin/del/{k[0]}' style='color:red'>REVOKE</a></td></tr>" for k in keys])
    return f"""
    <body style="font-family:monospace; padding:20px;">
        <h1>ADMIN PANEL</h1>
        <a href="/admin/gen"><button>+ GENERATE NEW KEY</button></a>
        <a href="/"><button>VIEW DASHBOARD</button></a>
        <table border="1" cellpadding="10" style="border-collapse:collapse; margin-top:20px; width:100%;">
            <tr><th>KEY</th><th>CREATED</th><th>ACTION</th></tr>
            {rows}
        </table>
    </body>
    """

@app.route('/admin/gen')
@admin_required
def gen():
    generate_key()
    return redirect(url_for('admin_panel'))

@app.route('/admin/del/<k>')
@admin_required
def delete(k):
    revoke_key(k)
    return redirect(url_for('admin_panel'))

@app.route('/logout')
def logout():
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

# --- UI TEMPLATE ---
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