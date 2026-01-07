import bcrypt
from sqlalchemy.orm import Session
from src.database import User

def get_password_hash(password):
    """Hash a password using bcrypt."""
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(pwd_bytes, salt)
    return hashed_password.decode('utf-8')

def verify_password(plain_password, hashed_password):
    """Verify a password against a hash."""
    password_byte_enc = plain_password.encode('utf-8')
    hashed_password_byte_enc = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_byte_enc, hashed_password_byte_enc)

def signup_user(db: Session, username, password, role='user'):
    """Create a new user."""
    # Check if user already exists
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        return None, "Username already exists"
    
    hashed_pwd = get_password_hash(password)
    new_user = User(username=username, password_hash=hashed_pwd, role=role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user, "User created successfully"

def login_user(db: Session, username, password):
    """Authenticate a user."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None, "Invalid username or password"
    
    if not verify_password(password, user.password_hash):
        return None, "Invalid username or password"
    
    return user, "Login successful"
