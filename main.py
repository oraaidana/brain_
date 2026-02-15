import os
import base64
import shutil
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from nilearn import plotting
from PIL import Image

# --- 1. DATABASE CONFIGURATION ---
DATABASE_URL = DATABASE_URL = "mysql+pymysql://avnadmin:AVNS_3yPXZnCr3LR2e7jo0ps@mysql-1fd02b41-careon.a.aivencloud.com:26047/defaultdb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security Setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PatientRecord(Base):
    __tablename__ = "patient_history"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), index=True)
    full_name = Column(String(255))
    age = Column(Integer)
    password_hash = Column(String(255))
    risk_score = Column(Float)
    diagnostic_status = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# --- 2. APP INITIALIZATION ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 3. ANALYTICAL UTILITIES (SOCR Logic) ---
def analyze_signal_variance(file_bytes):
    img = Image.open(BytesIO(file_bytes)).convert('L')
    return np.var(np.array(img))


# --- 4. API ENDPOINTS ---

@app.post("/register")
async def register(full_name: str = Form(...), email: str = Form(...), age: int = Form(...), password: str = Form(...)):
    db = SessionLocal()
    try:
        hashed_pw = pwd_context.hash(password[:72])
        new_user = PatientRecord(full_name=full_name, email=email, age=age, password_hash=hashed_pw, risk_score=0.0,
                                 diagnostic_status="Pending")
        db.add(new_user)
        db.commit()
        return {"message": "User registered successfully"}
    finally:
        db.close()


@app.post("/analyze")
async def analyze_diagnostic(
        email: str = Form(...),
        mri_baseline: UploadFile = File(...),
        mri_current: UploadFile = File(...),
        ecg_strip: UploadFile = File(None),
        eeg_strip: UploadFile = File(None)
):
    db = SessionLocal()
    # Paths for Nibabel processing
    path_b = f"temp_b_{mri_baseline.filename}"
    path_c = f"temp_c_{mri_current.filename}"

    try:
        # Save MRI files temporarily
        for path, file in [(path_b, mri_baseline), (path_c, mri_current)]:
            with open(path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 1. Structural Analysis (MRI)
        img_p = nib.load(path_b).get_fdata()
        img_n = nib.load(path_c).get_fdata()
        wm_p, wm_n = np.sum(img_p > 0), np.sum(img_n > 0)
        wm_change = ((wm_n - wm_p) / wm_p) * 100 if wm_p > 0 else 0
        mri_base_risk = 1 / (1 + np.exp(-(abs(wm_change) - 3))) * 100

        # 2. Electrophysiology Analysis (ECG/EEG)
        ecg_mult = 1.7 if ecg_strip and analyze_signal_variance(await ecg_strip.read()) > 1000 else 1.1
        eeg_mult = 1.4 if eeg_strip and analyze_signal_variance(await eeg_strip.read()) > 800 else 1.0

        # 3. Final Probability
        total_risk = min((mri_base_risk * ecg_mult * eeg_mult), 99.9)
        status = "🚨 CRITICAL" if total_risk > 50 else "🟡 MODERATE" if total_risk > 25 else "🟢 LOW"
        # Calculate current_risk (Simulated base logic)
        current_risk = min((mri_base_risk * ecg_mult * eeg_mult), 99.9)

        # --- NEW: RISK PROJECTION DATA (The Trend) ---
        # Baseline, Current, Year 1, Year 2, Year 3
        risk_trend = {
            "labels": ["Baseline", "Current", "Year 1", "Year 2", "Year 3"],
            "unmanaged": [20, current_risk, round(current_risk * 1.12, 1), round(current_risk * 1.28, 1),
                          round(current_risk * 1.45, 1)],
            "managed": [20, current_risk, round(current_risk * 0.85, 1), round(current_risk * 0.70, 1),
                        round(current_risk * 0.55, 1)]
        }
        # 4. Generate Ortho-view Image for React
        plt.switch_backend('Agg')
        fig = plt.figure(figsize=(10, 4))
        plotting.plot_anat(path_c, display_mode='ortho', figure=fig, title="Neural Mapping Result")

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        img_str = base64.b64encode(buf.getvalue()).decode()

        # 5. Database Update
        user = db.query(PatientRecord).filter(PatientRecord.email == email).first()
        if user:
            user.risk_score = round(total_risk, 2)
            user.diagnostic_status = status
            db.commit()

        report_fields = {
            "mri_change": f"{round(wm_change, 2)}%" if mri_current else "N/A",
            "ecg_status": "🚨 ABNORMAL" if ecg_mult > 1.2 else "✅ STABLE",
            "eeg_status": "⚠️ SLOWING" if eeg_mult > 1.2 else "✅ STABLE",
            "z_score": round((wm_change - (-0.5)) / 0.4, 2) if mri_current else 0.0
        }

        return {
            "probability": round(current_risk, 1),
            "status": "CRITICAL" if current_risk > 50 else "STABLE",
            "image": f"data:image/png;base64,{img_str}" if img_str else None,
            "trend": risk_trend,
            "metrics": report_fields,
            "recommendations": [
                "Initiate neuro-protective protocol.",
                "Schedule follow-up MRI in 6 months.",
                "Monitor heart rate variability daily."
            ]
        }

    finally:
        db.close()
        for p in [path_b, path_c]:
            if os.path.exists(p): os.remove(p)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)