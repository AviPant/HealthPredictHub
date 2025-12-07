import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Use environment variable if set, otherwise use local SQLite database
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    # Default to local SQLite database
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'healthpredict.db')
    DATABASE_URL = f'sqlite:///{db_path}'

engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False} if 'sqlite' in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    disease_type = Column(String(50), index=True)
    prediction = Column(Integer)
    probability = Column(Float)
    risk_level = Column(String(20))
    input_features = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String(100), index=True)
    notes = Column(Text, nullable=True)


class BatchPrediction(Base):
    __tablename__ = "batch_predictions"

    id = Column(Integer, primary_key=True, index=True)
    batch_name = Column(String(100))
    disease_type = Column(String(50))
    total_records = Column(Integer)
    high_risk_count = Column(Integer)
    medium_risk_count = Column(Integer)
    low_risk_count = Column(Integer)
    results_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String(100), index=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


def save_prediction(disease_type,
                    prediction,
                    probability,
                    risk_level,
                    input_features,
                    session_id,
                    notes=None):
    db = get_db()
    try:
        pred = PredictionHistory(disease_type=disease_type,
                                 prediction=prediction,
                                 probability=probability,
                                 risk_level=risk_level,
                                 input_features=input_features,
                                 session_id=session_id,
                                 notes=notes)
        db.add(pred)
        db.commit()
        db.refresh(pred)
        return pred.id
    except Exception as e:
        db.rollback()
        print(f"Error saving prediction: {e}")
        return None
    finally:
        db.close()


def get_prediction_history(session_id=None, disease_type=None, limit=100):
    db = get_db()
    try:
        query = db.query(PredictionHistory)
        if session_id:
            query = query.filter(
                PredictionHistory.session_id == session_id)
        if disease_type:
            query = query.filter(
                PredictionHistory.disease_type == disease_type)
        return query.order_by(
            PredictionHistory.created_at.desc()).limit(limit).all()
    finally:
        db.close()


def get_all_predictions(limit=1000):
    db = get_db()
    try:
        return db.query(PredictionHistory).order_by(
            PredictionHistory.created_at.desc()).limit(limit).all()
    finally:
        db.close()


def save_batch_prediction(batch_name, disease_type, results_data, session_id):
    db = get_db()
    try:
        high_risk = sum(1 for r in results_data
                        if r.get('risk_level') == 'High')
        medium_risk = sum(1 for r in results_data
                          if r.get('risk_level') == 'Medium')
        low_risk = sum(1 for r in results_data
                       if r.get('risk_level') == 'Low')

        batch = BatchPrediction(batch_name=batch_name,
                                disease_type=disease_type,
                                total_records=len(results_data),
                                high_risk_count=high_risk,
                                medium_risk_count=medium_risk,
                                low_risk_count=low_risk,
                                results_data=results_data,
                                session_id=session_id)
        db.add(batch)
        db.commit()
        db.refresh(batch)
        return batch.id
    except Exception as e:
        db.rollback()
        print(f"Error saving batch prediction: {e}")
        return None
    finally:
        db.close()


def get_batch_predictions(session_id=None, limit=50):
    db = get_db()
    try:
        query = db.query(BatchPrediction)
        if session_id:
            query = query.filter(BatchPrediction.session_id == session_id)
        return query.order_by(
            BatchPrediction.created_at.desc()).limit(limit).all()
    finally:
        db.close()


def get_prediction_stats():
    db = get_db()
    try:
        from sqlalchemy import func

        total = db.query(func.count(PredictionHistory.id)).scalar() or 0
        by_disease = db.query(PredictionHistory.disease_type,
                              func.count(PredictionHistory.id)).group_by(
                                  PredictionHistory.disease_type).all()

        by_risk = db.query(PredictionHistory.risk_level,
                           func.count(PredictionHistory.id)).group_by(
                               PredictionHistory.risk_level).all()

        return {
            'total': total,
            'by_disease': {row[0]: row[1] for row in by_disease},
            'by_risk': {row[0]: row[1] for row in by_risk}
        }
    finally:
        db.close()


init_db()
