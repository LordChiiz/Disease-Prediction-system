from flask import Flask, render_template, request, flash, session, make_response, redirect, url_for
import joblib
import numpy as np
from fpdf import FPDF
import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func # NEW: Needed for counting

app = Flask(__name__)
app.secret_key = "chiiz_secret_key"

# DATABASE CONFIG
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_storage.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# DATABASE MODEL
class MedicalReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    date = db.Column(db.String(50), nullable=False)
    symptoms = db.Column(db.String(500), nullable=False)

with app.app_context():
    db.create_all()

# LOAD AI MODELS
model = joblib.load('disease_model.pkl')
symptom_list = joblib.load('symptoms_list.pkl')

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptom_list)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        
        if not selected_symptoms:
            flash('Please select at least one symptom', 'error')
            return render_template('index.html', symptoms=symptom_list)

        input_data = [0] * len(symptom_list)
        for symptom in selected_symptoms:
            if symptom in symptom_list:                     #One hotdog                                                                                                                 
                index = symptom_list.index(symptom)
                input_data[index] = 1
        
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        probabilities = model.predict_proba(input_array)
        confidence = np.max(probabilities) * 100
        
        session['prediction'] = prediction
        session['confidence'] = f"{confidence:.2f}"
        session['symptoms'] = selected_symptoms
        session['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return render_template('index.html', 
                               prediction=prediction, 
                               confidence=f"{confidence:.2f}",
                               symptoms=symptom_list,
                               selected=selected_symptoms)

@app.route('/save_to_db', methods=['POST'])
def save_to_db():
    if 'prediction' in session:
        name = request.form.get('patient_name')
        disease = session['prediction']
        conf = float(session['confidence'])
        date = session['date']
        syms = ", ".join(session['symptoms'])

        new_report = MedicalReport(patient_name=name, disease=disease, confidence=conf, date=date, symptoms=syms)
        db.session.add(new_report)
        db.session.commit()
        
        flash("Report saved successfully!")
        return redirect(url_for('history'))
    return redirect(url_for('home'))

@app.route('/history')
def history():
    reports = MedicalReport.query.order_by(MedicalReport.date.desc()).all()
    return render_template('history.html', reports=reports)

@app.route('/download_report')
def download_report():
    if 'prediction' not in session:
        return redirect(url_for('home'))
    
    disease = session['prediction']
    conf = session['confidence']
    symptoms = session['symptoms']
    date = session['date']

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Medi-AI System Medical Report", ln=True, align='C')
    pdf.line(10, 20, 200, 20)
    pdf.ln(20)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {date}", ln=True)
    pdf.cell(200, 10, txt=f"Diagnosis: {disease}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {conf}%", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Symptoms:", ln=True)
    for sym in symptoms:
        pdf.cell(200, 8, txt=f"- {sym}", ln=True)
    
    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=Report_{disease}.pdf'
    return response

# === NEW: DASHBOARD ROUTE ===
@app.route('/dashboard')
def dashboard():
    # 1. Count Total Patients
    total_patients = MedicalReport.query.count()
    
    # 2. Get Disease Counts (e.g., Malaria: 5, Typhoid: 2)
    # This acts like a "Pivot Table" in Excel
    disease_counts = db.session.query(MedicalReport.disease, func.count(MedicalReport.disease)).group_by(MedicalReport.disease).all()
    
    # Separate the data for the chart (Labels vs. Numbers)
    diseases = [row[0] for row in disease_counts]
    counts = [row[1] for row in disease_counts]
    
    return render_template('dashboard.html', 
                           total=total_patients, 
                           diseases=diseases, 
                           counts=counts)

@app.route('/delete/<int:id>', methods=['POST'])
def delete_patient(id):
    record = MedicalReport.query.get_or_404(id)
    
    try:
        db.session.delete(record)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting record: {e}")
        
    return redirect(url_for('history')) 

if __name__ == '__main__':
    app.run(debug=True)