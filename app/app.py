from flask import Flask, render_template
from flask import request, jsonify
from feat import Detector
import tempfile
import os

app = Flask(__name__, template_folder='templates')
detector = Detector()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    files = request.files.getlist('images')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        
        # Save all uploaded files to the temp directory
        for file in files:
            if file.filename:
                # Create a full path inside the temp folder
                path = os.path.join(temp_dir, file.filename)
                file.save(path)
                file_paths.append(path)
        try:
            results = detector.detect(file_paths, output_size=(256,256))
        except Exception as e:
            print(f"Error during detection: {e}")
            return jsonify({'error': str(e)}), 500
    
        return jsonify({'results': results.to_csv(index=False)})

if __name__ == '__main__':
    app.run(debug=True)