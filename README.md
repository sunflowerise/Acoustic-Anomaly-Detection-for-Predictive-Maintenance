## Acoustic Anomaly Detection for Predictive Maintenance

This project implements an end-to-end AI/ML pipeline for **acoustic anomaly detection** on multiple machine types (`fan`, `pump`, `slider`, `ToyCar`, `ToyConveyor`).

- **Model**: Isolation Forest anomaly detection on MFCC audio features extracted with `librosa`
- **Backend**: FastAPI service with `/predict` and `/mel-difference` endpoints
- **Frontend**: Web UI served by FastAPI for uploading audio and visualizing predictions

### Quick Start

**1. Install dependencies** (using virtual environment):

```bash
cd "/Users/akanksha/Downloads/final dataset-4"
source venv_mac/bin/activate
pip install -r requirements.txt
```

**2. Run the API server**:

```bash
cd "/Users/akanksha/Downloads/final dataset-4"
source venv_mac/bin/activate
uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000
```

**3. Open the frontend**:

Open in your browser: `http://localhost:8000`

> **Note:** The frontend is served by the FastAPI server to avoid CORS issues. Access it via `http://localhost:8000` (not by opening the HTML file directly).

### Alternative: Use Scripts

**Start:**
```bash
cd "/Users/akanksha/Downloads/final dataset-4" && ./run.sh
```

**Stop:**
```bash
cd "/Users/akanksha/Downloads/final dataset-4" && ./stop.sh
```

### API Endpoints

- **Frontend:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health
- **Predict:** POST http://localhost:8000/predict
- **Mel Difference:** POST http://localhost:8000/mel-difference
