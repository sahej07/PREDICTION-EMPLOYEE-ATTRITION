import pandas as pd
from flask import Flask, render_template, request
import pickle
import json 
import numpy as np

app = Flask(__name__)
data = pd.read_csv('cleaned.csv')
pipe = pickle.load(open("rf.pickle",'rb'))


__data_columns = None

def load_saved_artifacts():
    print("Loading the saved artifacts...start !")
    global __data_columns
    global pipe

    with open("columns.json") as f:
        __data_columns = json.loads(f.read())["data_columns"]
load_saved_artifacts()
def get_attrition(input_json):
    try:
        job_index = __data_columns.index(input_json['JobRole'].lower())
        travel_index = __data_columns.index(input_json['BusinessTravel'].lower())
        education_index = __data_columns.index(input_json['EducationField'].lower())
        marital_index = __data_columns.index(input_json['MaritalStatus'].lower())
        overtime_index = __data_columns.index(input_json['OverTime'].lower())
    except:
        job_index = -1
        travel_index = -1
        education_index = -1
        marital_index = -1
        overtime_index = -1
    x = np.zeros(35)
    x[0] = input_json['Age']
    x[1] = input_json['DistanceFromHome']
    x[2] = input_json['EnvironmentSatisfaction']
    x[3] = input_json['MonthlyIncome']
    x[4] = input_json['NumCompaniesWorked'] 
    x[5] = input_json['PercentSalaryHike']
    x[6] = input_json['StockOptionLevel']
    x[7] = input_json['TotalWorkingYears']
    x[8] = input_json['YearsAtCompany']
    x[9] = input_json['YearsInCurrentRole']
    x[10] = input_json['YearsSinceLastPromotion']
    x[11] = input_json['YearsWithCurrManager']
    
    if job_index >= 0:
        x[job_index] = 1
    if travel_index >= 0:
        x[travel_index] = 1 
    if education_index >= 0:
        x[education_index] = 1 
    if marital_index >= 0:
        x[marital_index] = 1
    if overtime_index >= 0:
        x[overtime_index] = 1
    result = pipe.predict([x])[0]
    return result


# Here we have started to do proper html and page calling the first one being home 
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')
@app.route('/test')
def test():
    return render_template('test.html')
@app.route('/index')
def index():
    travel = sorted(data["BusinessTravel"].unique())
    role = sorted(data["JobRole"].unique())
    education = sorted(data["EducationField"].unique())
    status = sorted(data["MaritalStatus"].unique())
    overtime = sorted(data["OverTime"].unique())
    return render_template('index.html', travel=travel, role=role, status=status, education=education, overtime=overtime)
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_json = {
            "JobRole" : request.form.get('role'),
            "BusinessTravel" : request.form.get('travel'),
            "EducationField" : request.form.get('education'),
            "MaritalStatus" : request.form.get('status'),
            "Age" : request.form.get('age'),
            "DistanceFromHome" : request.form.get('dist'),
            "EnvironmentSatisfaction" : request.form.get('env'),
            "MonthlyIncome" : request.form.get('mon'),
            "NumCompaniesWorked" : request.form.get('company'),
            "OverTime" : request.form.get('overtime'),
            "PercentSalaryHike" : request.form.get('PercentSalaryHike'),
            "StockOptionLevel" : request.form.get('StockOptionLevel'),
            "TotalWorkingYears" : request.form.get('TotalWorkingYears'),
            "YearsAtCompany" : request.form.get('YearsAtCompany'),
            "YearsInCurrentRole" : request.form.get('YearsInCurrentRole'),
            "YearsSinceLastPromotion" : request.form.get('YearsSinceLastPromotion'),
            "YearsWithCurrManager" : request.form.get('YearsWithCurrManager')
        }
        result = get_attrition(input_json)
        

        if str(result) == '1':
            return "Employee is likely to leave (Attrition: Yes)."
        else:
            return "Employee is likely to stay (Attrition: No)."




