from flask import Flask, request, jsonify
import json
import pickle
import pandas as pd
from models.movie_recommendation.model import predict

app = Flask(__name__)


def convertJSON(json):
    for key, value in json.items():
        json[key] = [value]
    return json


@app.route("/wine-quality", methods=["POST"])
def wineQuality():
    def interpret(x):
        if x == 1:
            return "Good"
        else:
            return "Bad"

    input = convertJSON(request.json)
    input_pd = pd.DataFrame(input)
    loaded_model = pickle.load(open("./models/wine-quality/model.sav", "rb"))
    predicted = loaded_model.predict(input_pd)
    return jsonify({"result": [interpret(output) for output in predicted.tolist()][0]})
    # {
    #     "fixed acidity": 8.5,
    #     "volatile acidity": 0.47,
    #     "citric acid": 0.27,
    #     "chlorides": 0.058,
    #     "free sulfur dioxide": 18.0,
    #     "total sulfur dioxide": 38.0,
    #     "density": 0.99518,
    #     "pH": 3.16000,
    #     "sulphates": 0.85,
    #     "alcohol": 11.10000,
    # }


@app.route("/medical-insurance", methods=["POST"])
def medicalInsurance():
    input = convertJSON(request.json)
    input_pd = pd.DataFrame(input)
    input_pd = input_pd.replace(
        pickle.load(open("./models/medical-insurance/replace.pkl", "rb"))
    )
    sc = pickle.load(open("./models/medical-insurance/scaler.pkl", "rb"))
    input_pd = sc.transform(input_pd)
    model = pickle.load(open("./models/medical-insurance/model.pkl", "rb"))
    prediction = model.predict(input_pd)
    return {"result": int(prediction[0])}
    # {
    #     "age": 19,
    #     "sex": "male",
    #     "bmi": 27.9,
    #     "children": 0,
    #     "smoker": "yes",
    #     "region": "southwest",
    # }


@app.route("/movie-recommendation", methods=["POST"])
def movieRecommendation():
    input = convertJSON(request.json)
    input_pd = pd.DataFrame(input)
    prediction = predict(input)
    return {"result": prediction}


# {"movie_name": "Spider-man"}


@app.route("/loan-status-prediction", methods=["POST"])
def loanStatusPrediction():
    input = convertJSON(request.json)
    input_pd = pd.DataFrame(input)
    input_pd = input_pd.replace(
        pickle.load(open("./models/loan_prediction/replace.pkl", "rb"))
    )
    model = pickle.load(open("./models/loan_prediction/model.pkl", "rb"))
    prediction = model.predict(input_pd)
    return {"result": "No" if prediction[0] == 0 else "Yes"}


# {
#     "Gender": "Male",
#     "Married": "Yes",
#     "Dependents": 2,
#     "Education": "Graduate",
#     "Self_Employed": "No",
#     "ApplicantIncome": 3100,
#     "CoapplicantIncome": 1400,
#     "LoanAmount": 113,
#     "Loan_Amount_Term": 360,
#     "Credit_History": 1,
#     "Property_Area": "Urban",
# }


@app.route("/heart-disease-prediction", methods=["POST"])
def heartDiseasePrediction():
    input = convertJSON(request.json)
    input_pd = pd.DataFrame(input)
    input_pd = input_pd.replace(
        {
            "sex": {"male": 1, "female": 0},
            "exang": {"yes": 1, "no": 0},
            "fbs": {"yes": 1, "no": 0},
        }
    )
    input_pd = pickle.load(open("./models/heart_disease/scaler.pkl", "rb")).transform(
        input_pd
    )
    model = pickle.load(open("./models/heart_disease/model.pkl", "rb"))
    prediction = model.predict(input_pd)
    return {"result": "No" if prediction[0] == 0 else "Yes"}
    # {
    #     "age": 52,
    #     "sex": "male",
    #     "cp": 0,
    #     "trestbps": 125,
    #     "chol": 212,
    #     "fbs": "no",
    #     "restecg": 1,
    #     "thalach": 168,
    #     "exang": "no",
    #     "oldpeak": 1,
    #     "slope": 2,
    #     "ca": 2,
    #     "thal": 3,
    # }


@app.route("/gold-price-prediction", methods=["POST"])
def goldPricePrediction():
    input = convertJSON(request.json)
    input_pd = pd.DataFrame(input)
    scaler = pickle.load(open("./models/gold_price_prediction/scaler.pkl", "rb"))
    input_pd = scaler.transform(input_pd)
    model = pickle.load(open("./models/gold_price_prediction/model.pkl", "rb"))
    prediction = model.predict(input_pd)
    return {"result": prediction[0]}


# {"SPX": 1447.16, "USO": 78.47, "SLV": 15.18, "EUR/USD": 1.4716}


@app.route("/diabetes-prediction", methods=["POST"])
def diabetesPrediction():
    input = convertJSON(request.json)
    input_pd = pd.DataFrame(input)
    scaler = pickle.load(open("./models/diabetes_prediction/scaler.pkl", "rb"))
    input_pd = scaler.transform(input_pd)
    model = pickle.load(open("./models/diabetes_prediction/model.pkl", "rb"))
    prediction = model.predict(input_pd)
    return {"result": "Diabetic" if prediction[0] == 1 else "Not Diabetic"}


# {
#     "Pregnancies": 6,
#     "Glucose": 148,
#     "BloodPressure": 72,
#     "SkinThickness": 35,
#     "Insulin": 0,
#     "BMI": 33.6,
#     "DiabetesPedigreeFunction": 0.627,
#     "Age": 50,
# }


@app.route("/car-price-prediction", methods=["POST"])
def carPricePrediction():
    input = convertJSON(request.json)
    input_pd = pd.DataFrame(input)
    replace_values = pickle.load(
        open("./models/car_price_prediction/replace.pkl", "rb")
    )
    input_pd = input_pd.replace(replace_values)
    model = pickle.load(open("./models/car_price_prediction/model.pkl", "rb"))
    prediction = model.predict(input_pd)
    return {"result": round(prediction[0], 3)}


# {
#     "Year": 2014,
#     "Present_Price": 5.59,
#     "Kms_Driven": 27000,
#     "Fuel_Type": "Petrol",
#     "Seller_Type": "Dealer",
#     "Transmission": "Manual",
#     "Owner": 0,
# }

if __name__ == "__main__":
    app.run(debug=True)
