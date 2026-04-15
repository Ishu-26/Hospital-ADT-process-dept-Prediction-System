import bcrypt

# Dummy admin (you can store in DB later)
ADMIN_USERNAME = "admin"

# Hashed password for: admin123
HASHED_PASSWORD = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt())

def verify_login(username, password):
    if username == ADMIN_USERNAME:
        return bcrypt.checkpw(password.encode(), HASHED_PASSWORD)
    return False