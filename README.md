# Smart Pest Detection System

A small Python demonstration project for exploring crop pest identification. It provides a Streamlit interface where users can upload a crop or pest image, view a predicted pest category, and read example treatment and prevention information. The interface also includes sample outbreak and weather panels.

> **Project status:** This is an educational/demo application. The current image classification functions use simple image statistics and hand-written rules. They do not identify pests using a trained computer vision model, so the displayed pest label and confidence must not be used to make real crop-management decisions.

## Features

- Upload JPG, JPEG, and PNG images in the Streamlit app.
- Predict one of five categories: Aphids, Whitefly, Bollworm, Leaf Miner, or Thrips.
- Show example descriptions, affected crops, severity, cost estimates, and treatment/prevention suggestions.
- Browse hard-coded sample outbreak reports for locations in Madhya Pradesh.
- Browse hard-coded weather conditions and pest activity forecasts.
- Includes a separate Flask module with a `/predict` JSON endpoint and an `/outbreak-data` endpoint.

## Project layout

```text
.
├── app.py             # Streamlit user interface and demo prediction logic
├── model.py           # Flask application, pest information, and demo prediction logic
├── requirements.txt   # Python dependencies
└── README.md
```

The repository does not include a trained `pest_model.h5` file or a training dataset. The Flask root route refers to `templates/index.html`, which is not included, so the Flask web page is not currently available without adding that template. The Streamlit app is the ready-to-run interface.

## Requirements

- Python 3.9 or newer is recommended.
- `pip` and a virtual environment.
- TensorFlow may require a platform-specific Python version or installation; consult the TensorFlow installation guide if its installation fails on your operating system.

The dependencies listed in `requirements.txt` are Flask, TensorFlow, NumPy, Pillow, and Streamlit. The Streamlit interface only needs Streamlit, Pillow, and NumPy; TensorFlow is imported by the Flask module.

## Setup

Clone the repository and enter its directory:

```bash
git clone <repository-url>
cd pest-detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Streamlit app

From the repository root, run:

```bash
streamlit run app.py
```

Streamlit prints a local URL in the terminal (usually `http://localhost:8501`). Open it in a browser.

### Using the app

1. Open the **Detect Pest** tab.
2. Upload a JPG, JPEG, or PNG image.
3. Select **Detect Pest** to display a category, demo confidence score, and the associated information.
4. Open **Outbreak Map** or **Weather Alert** to view the sample dashboard panels.

The outbreak counts, weather metrics, and forecast in the interface are static examples embedded in the source. They are not fetched from live services.

## Flask module and API

`model.py` defines a Flask application. With dependencies installed, start it using:

```bash
python model.py
```

The application listens on port `5000`. Its `/` route currently expects a missing `templates/index.html`; the JSON endpoints can still be used independently.

### `POST /predict`

Upload an image as multipart form data using the field name `file`:

```bash
curl -X POST -F "file=@path/to/crop.jpg" http://localhost:5000/predict
```

A successful response has this shape:

```json
{
  "success": true,
  "pest": "Aphids",
  "confidence": 89.0,
  "data": {
    "description": "...",
    "severity": "Medium",
    "crops": ["Cotton", "Wheat", "Rice", "Vegetables"],
    "organic": [],
    "chemical": [],
    "preventive": [],
    "cost": "₹200-500 per acre"
  }
}
```

The returned pest and confidence values are examples; they are generated from image color/brightness/variance rules, not model probabilities. Missing or invalid uploads return an error response.

### `GET /outbreak-data`

Returns the hard-coded outbreak sample as JSON:

```bash
curl http://localhost:5000/outbreak-data
```

## How the prediction currently works

In `app.py`, the uploaded image is resized to 224 × 224 pixels. The code calculates average RGB values, brightness, color variance, and color ratios, then maps a derived score to one of the five pest labels. The confidence number is also derived from those image statistics. This is placeholder/demo logic and has not been validated against labeled images.

`model.py` contains a MobileNetV2 model-construction/loading path, but its active `predict_pest` function likewise assigns a label using hand-written image-statistic rules. The loaded model is not used to produce that prediction. If no saved model is present, model initialization may also attempt to obtain MobileNetV2 ImageNet weights. No training workflow or trained pest classifier is supplied in this repository.

## Limitations and responsible use

- Predictions can be incorrect and should not be treated as agricultural diagnosis.
- Confidence values are heuristic display values, not calibrated probabilities or measured accuracy.
- Treatment recommendations and cost ranges are static examples. Verify any pesticide, dosage, crop compatibility, and local regulatory guidance with a qualified agricultural expert and the product label before use.
- The sample outbreak and weather information is not live or location-aware.
- The Flask app runs with debug mode enabled in its current development entry point. Do not expose it publicly as a production service without appropriate changes.

## Development notes

- The Streamlit app entry point is `app.py`.
- The Flask app and API entry point is `model.py`.
- Pest labels and display information are currently defined separately in both files; keep them aligned if updating the categories or recommendations.
- No automated test suite or model-training scripts are included.

## License

No license file is currently included. Add a license before redistributing or reusing this project if you need to specify usage rights.
