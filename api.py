"""
Flask REST API for Malaria Cell Classification.

Provides web interface and REST endpoints for image classification.
"""

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from pathlib import Path

from src.inference import MalariaClassifier

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Initialize classifier
try:
    classifier = MalariaClassifier()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    classifier = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Page d'accueil"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Malaria Cell Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .result { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .parasitized { background: #ffebee; border-left: 4px solid #f44336; }
            .uninfected { background: #e8f5e8; border-left: 4px solid #4caf50; }
        </style>
    </head>
    <body>
        <h1>🦠 Classificateur de Malaria</h1>
        <p>Uploadez une image de cellule sanguine pour détecter la présence de parasites de malaria.</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <input type="file" id="fileInput" name="file" accept="image/*" required>
                <p>Formats supportés: PNG, JPG, JPEG, GIF, BMP</p>
            </div>
            <button type="submit">Analyser l'image</button>
        </form>
        
        <div id="results"></div>
        
        <script>
            document.getElementById('uploadForm').onsubmit = function(e) {
                e.preventDefault();
                const formData = new FormData();
                const fileInput = document.getElementById('fileInput');
                formData.append('file', fileInput.files[0]);
                
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    const resultsDiv = document.getElementById('results');
                    if (data.error) {
                        resultsDiv.innerHTML = '<div class="result">Erreur: ' + data.error + '</div>';
                    } else {
                        const className = data.class.toLowerCase();
                        const confidence = (data.confidence * 100).toFixed(1);
                        resultsDiv.innerHTML = `
                            <div class="result ${className}">
                                <h3>Résultat: ${data.class}</h3>
                                <p>Confiance: ${confidence}%</p>
                                <p>Parasitized: ${(data.probabilities.Parasitized * 100).toFixed(1)}%</p>
                                <p>Uninfected: ${(data.probabilities.Uninfected * 100).toFixed(1)}%</p>
                            </div>
                        `;
                    }
                });
            };
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for single image classification.
    
    Returns:
        JSON with prediction results
    """
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File format not supported'}), 400
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            file.save(tmp_file.name)
            
            # Prediction
            result = classifier.predict(tmp_file.name)
            
            # Cleanup
            os.unlink(tmp_file.name)
            
            return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if classifier else 'unhealthy',
        'model_loaded': classifier is not None
    })


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    API endpoint for batch image classification.
    
    Returns:
        JSON with list of prediction results
    """
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    file.save(tmp_file.name)
                    result = classifier.predict(tmp_file.name)
                    result['filename'] = secure_filename(file.filename)
                    results.append(result)
                    os.unlink(tmp_file.name)
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
