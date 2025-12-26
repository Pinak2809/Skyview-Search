# backend/db.py
from sqlalchemy import create_engine, Column, String, Integer, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

Base = declarative_base()

class ImageRecord(Base):
    __tablename__ = "images"
    uuid = Column(String, primary_key=True)
    filepath = Column(String, nullable=False)
    category = Column(String)
    caption = Column(String)
    image_metadata = Column(JSON)
    embedding_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_db(db_path="sqlite:///../data/processed/skyview/skyview.db"):
    engine = create_engine(db_path, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

if __name__ == "__main__":
    Session = init_db()
    session = Session()
    print("DB ready")
