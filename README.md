# Skin Cancer Model Flask 

This repository provides a minimal Flask server to serve a trained Keras model (a `.keras` model file) for skin cancer detection. It includes a demo HTML page for manual testing and a small test script for programmatic calls.

Assumptions
- The trained model file is in Keras format and loadable with `tensorflow.keras.models.load_model` (filename `model.keras` by default).
- The model expects RGB images resized to a square (default 224x224). You can change this with the `IMAGE_SIZE` environment variable.
- If the model outputs a single probability (shape `(1,1)`), the app treats it as a binary classifier (classes default to `benign,melanoma`). For multi-class outputs it maps probabilities to class names supplied via `CLASSES` env var.

Quick start (Windows PowerShell)

1. Create a virtual environment and activate it

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Put your model file in the project root named `model.keras`, or set the `MODEL_PATH` env var to your .keras file path

```powershell
$env:MODEL_PATH = "C:\path\to\your\model.keras"
```

4. Run the server

```powershell
python app.py
```

5. Open http://127.0.0.1:5000 in your browser and try uploading an image, or use the test script:

```powershell
python test_predict.py path\to\example.jpg
```

Environment variables
- MODEL_PATH: path to your `.keras` model (default: `./model.keras`)
- IMAGE_SIZE: integer image size to resize input images to (default: 224)
- CLASSES: comma-separated list of class names (default: `benign,melanoma`)

Notes
- TensorFlow can be large; ensure you have an appropriate Python version and GPU drivers if using GPU builds.
- If your model uses custom layers, you may need to provide `custom_objects` to `load_model`. The `load_model_safe` helper in `app.py` can be extended for that.

