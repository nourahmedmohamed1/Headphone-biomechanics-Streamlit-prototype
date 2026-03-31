# 🎧 CervicalIQ — Headphone Weight & Neck Biomechanics Analyzer

Real-time craniovertebral angle (CVA) measurement and cervical spine load calculator built with **MediaPipe**, **OpenCV**, and **Streamlit**.

---

## 🚀 Deploy to Streamlit Cloud (Recommended)

1. Push this folder to a **GitHub repository** (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, set **Main file** = `app.py`.
4. Click **Deploy** — Streamlit Cloud auto-reads `requirements.txt` and `packages.txt`.

> **Note:** `packages.txt` installs system-level apt packages (OpenCV/MediaPipe runtime libs, ffmpeg). This is only supported on Streamlit Cloud, not plain Vercel.

---

## 🖥️ Run Locally

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install Python deps
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

---

## 📁 Project Structure

```
headphone_biomechanics/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── packages.txt            # System apt packages (Streamlit Cloud)
├── .streamlit/
│   └── config.toml         # Theme & server config
└── README.md
```

---

## 🔬 Biomechanics Model

Based on **Hansraj (2014) — "Assessment of Stresses in the Cervical Spine Caused by Posture and Position of the Head"**:

| Head Tilt | Cervical Spine Load |
|-----------|-------------------|
| 0° (neutral) | ~10–12 lbs |
| 15° | ~27 lbs |
| 30° | ~40 lbs |
| 45° | ~49 lbs |
| 60° | ~60 lbs |

**Formula used:**
```
forward_tilt = 90° - CVA
base_load_lbs = 10 + (tilt / 60) × 50
headphone_load = (weight_g / 453.592) × 0.3 × (1 + tilt/30)
total_load = base_load + headphone_load
```

**CVA Measurement:** MediaPipe Pose estimates C7 via the shoulder midpoint and uses the ear landmark as a Tragus proxy. The angle between the horizontal at C7 and the C7→Ear line = CVA.

---

## ⚠️ Vercel Deployment Note

Vercel's serverless functions have a **250MB bundle limit** and no apt package support. MediaPipe + OpenCV exceed this. **Streamlit Cloud is the recommended cloud target.** For Vercel, consider:
- Hosting only the frontend on Vercel
- Running the analysis backend separately (Railway, Render, Fly.io)

---

## 📋 Risk Levels

| Badge | CVA | Interpretation |
|-------|-----|----------------|
| 🟢 LOW | ≥ 50° | Good posture, minimal strain |
| 🟡 MEDIUM | 35–50° | Moderate forward head posture |
| 🔴 HIGH | < 35° | Severe text neck, high spinal load |

---

*For educational and research purposes only. Not a medical device.*
