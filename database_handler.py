import sqlite3
import os

# RENDER CONFIGURATION
# If '/data' exists (Render Persistent Disk), use it. Otherwise, use local folder.
DB_PATH = '/data' if os.path.exists('/data') else '.'
DATABASE_FILE = os.path.join(DB_PATH, 'ar_lottery_history.db')

def create_connection():
    """Create a database connection."""
    return sqlite3.connect(DATABASE_FILE)

def setup_database():
    """Create tables for Results and License Keys."""
    conn = create_connection()
    cursor = conn.cursor()
    
    # 1. Historical Results Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            issue TEXT PRIMARY KEY, 
            code INTEGER NOT NULL,
            api_timestamp REAL,
            fetch_timestamp REAL
        )
    """)

    # 2. License Keys Table (THE LOCK SYSTEM)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS access_keys (
            key_code TEXT PRIMARY KEY,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    """)
    
    conn.commit()
    conn.close()

async def save_result_to_database(data_to_save):
    """Bulk save results (called by fetcher)."""
    conn = create_connection()
    cursor = conn.cursor()
    new_records = False
    try:
        batch = []
        for record in data_to_save:
            batch.append((
                str(record['issue']), 
                int(record['code']), 
                record.get('api_timestamp'), 
                record.get('fetch_timestamp')
            ))
        cursor.executemany(
            "INSERT OR IGNORE INTO results (issue, code, api_timestamp, fetch_timestamp) VALUES (?, ?, ?, ?)",
            batch
        )
        if cursor.rowcount > 0:
            new_records = True
        conn.commit()
    except Exception as e:
        print(f"[DB ERROR] {e}")
    finally:
        conn.close()
    return new_records

def ensure_setup():
    setup_database()