from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load('./Best_model.pkl')
scaler=joblib.load('./scaler.pkl')
app = Flask(__name__,template_folder='template')

@app.route('/')
def home():
	return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age= float(request.form['age'])
        sex= int(request.form['sex'])
        creatinine_phosphokinase= float(request.form['creatinine_phosphokinase'])
        diabetes = int(request.form['diabetes'])
        ejection_fraction = float(request.form['ejection_fraction'])
        high_blood_pressure = int(request.form['high blood pressure'])
        platelets = float(request.form['platelets'])
        anaemia = int(request.form['anaemia'])
        serum_creatinine = float(request.form['serum_creatinine'])
        serum_sodium= float(request.form['serum_sodium'])
        smoking= int(request.form['smoking'])
        time = float(request.form['time'])
        
        array=np.asarray([[age,creatinine_phosphokinase,ejection_fraction,platelets,serum_creatinine,serum_sodium,time]])
        nm_array=scaler.transform(array)
        data = np.array([[nm_array[0][0],anaemia,nm_array[0][1],diabetes,nm_array[0][2],high_blood_pressure,nm_array[0][3],nm_array[0][4],nm_array[0][5],sex,smoking,nm_array[0][6]]])
        prediction = model.predict(data)
        
        return render_template('Result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

