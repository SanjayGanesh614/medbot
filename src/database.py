import os
from datetime import datetime
import json
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Define database file path
DB_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(DB_FOLDER, "medbot_users.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default='user')  # 'user' or 'admin'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    records = relationship("PatientRecord", back_populates="user")

class PatientRecord(Base):
    __tablename__ = 'patient_records'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # Store minimal searchable info as columns, rest as JSON
    patient_name = Column(String, nullable=True) # Optional, strictly for UI listing if needed
    risk_score = Column(Float)
    risk_category = Column(String)
    
    # JSON strings for complex data
    patient_data_json = Column(Text, nullable=False) # The full input data
    prediction_result_json = Column(Text, nullable=False) # The full output including SHAP etc
    clinical_recommendations = Column(Text, nullable=True) # AI Generated text
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="records")

# Database setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize the database tables."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
