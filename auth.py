import sqlite3
import bcrypt

# Create DB + table
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password BLOB
        )
    """)
    conn.commit()
    conn.close()

# SIGNUP
def create_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                  (username, hashed_pw))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

# LOGIN
def verify_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()

    if result:
        return bcrypt.checkpw(password.encode(), result[0])
    return False