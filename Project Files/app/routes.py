import os, uuid, numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from model.infer import load_trained_model, predict_image

ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'bmp' }
UPLOAD_DIR = os.path.join('app', 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = load_trained_model()
class_names = model.class_names  # attribute added in load_trained_model

main_bp = Blueprint('main', __name__, template_folder='templates', static_folder='static')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    file_url = None

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(UPLOAD_DIR, unique_name)
            file.save(filepath)
            file_url = url_for('main.static', filename=f'uploads/{unique_name}')

            prediction, confidence = predict_image(model, filepath)

    return render_template('index.html', prediction=prediction, confidence=confidence, img_url=file_url, classes=class_names)