import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

load_dotenv()

Database_URL = os.getenv("DATABASE_URL")

if not Database_URL:
    raise ValueError("DATABASE_URL is not set")

engine = create_engine(Database_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()

    try :
        yield db
    finally:
        db.close()