from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import joblib
import os
from scripts.utils import extract_text_from_pdf  # Import correct

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Charger les modèles et le LabelEncoder
vectorizer = joblib.load('models/vectorizer.pkl')
decision_tree_model = joblib.load('models/decision_tree_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the CV
            text = extract_text_from_pdf(file_path)
            vectorized_text = vectorizer.transform([text])

            try:
                prediction = decision_tree_model.predict(vectorized_text)
                domain = label_encoder.inverse_transform(prediction)[0]
            except ValueError as e:
                known_labels = label_encoder.classes_
                error_message = f"Erreur: {str(e)}. Labels connus: {known_labels}"
                return render_template('result.html', prediction=error_message)

            return render_template('result.html', prediction=domain)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
