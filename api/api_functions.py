import sqlite3
from datetime import datetime, date

conn = sqlite3.connect("dbs/api/api_keys.db")
cursor = conn.cursor()

# # Create table if it doesn't exist
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS api_keys (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     api_key TEXT UNIQUE,
#     tries_today INTEGER,
#     last_used DATE,
#     available BOOL
# )
# """)
# conn.commit()


# --- Helper functions ---

def add_api_key(new_key: str):
    conn = sqlite3.connect("dbs/api/api_keys.db")
    cursor = conn.cursor()

    """Add a new API key to the database."""
    today_str = date.today().isoformat()
    try:
        cursor.execute("INSERT INTO api_keys (api_key, tries_today, last_used, available) VALUES (?, ?, ?, ?)",
                       (new_key, 0, today_str, True))
        conn.commit()
        print(f"API key '{new_key}' added successfully!")
    except sqlite3.IntegrityError:
        print("This API key already exists.")
    finally:
        conn.close()





# update 
def get_api_key(key: str):
    """Fetch API key info, reset tries_today if last_used is before today."""
    cursor.execute("SELECT id, tries_today, last_used FROM api_keys WHERE api_key = ?", (key,))
    result = cursor.fetchone()
    
    if not result:
        print("API key not found.")
        return None
    
    id, tries_today, last_used_str = result
    last_used_date = datetime.strptime(last_used_str, "%Y-%m-%d").date()
    today = date.today()

    # Reset tries_today if last_used is before today
    if last_used_date < today:
        tries_today = 0
        last_used_str = today.isoformat()
        cursor.execute("UPDATE api_keys SET tries_today = ?, last_used = ?, available = ? WHERE api_key = ?",
                       (tries_today, last_used_str, True, key))
        conn.commit()

    return {"id": id, "api_key": key, "tries_today": tries_today, "last_used": last_used_str}





def increment_tries(key: str):
    """Increment the try counter for an API key."""
    available = True
    api_info = get_api_key(key)
    if api_info is None:
        return
    
    new_tries = api_info['tries_today'] + 1
    if api_info['tries_today'] >= 25:
        available = False
    cursor.execute("UPDATE api_keys SET tries_today = ?, last_used = ?, available = ? WHERE api_key = ?",
                (new_tries, date.today().isoformat(), available, key))
    conn.commit()
    print(f"API key '{key}' used. Tries today: {new_tries}")





def disable_key(key):
    conn = sqlite3.connect("dbs/api/api_keys.db")
    cursor = conn.cursor()

    cursor.execute("UPDATE api_keys SET tries_today = ?, available = ? WHERE api_key = ?",
                        (25, False, key))
    conn.commit()
    conn.close()





def find_available():
    conn = sqlite3.connect("dbs/api/api_keys.db")
    cursor = conn.cursor()

    cursor.execute("SELECT api_key FROM api_keys ORDER BY id ASC")
    keys = cursor.fetchall()

    for (key,) in keys:
        info = get_api_key(key)
        if info['tries_today'] < 25:
            return f"""#####################################################################################\n
            Available key has been found: key: {info['api_key']} number: {info['id']}
            \n#####################################################################################
            """
        
    conn.close()
    return "!!!   No available key right now   !!!"



# --- Example Usage ---



# # Using a key
# increment_tries("MY_SECRET_KEY_1")

# # Fetch key info
# info = get_api_key("MY_SECRET_KEY_1")
# print(info)

