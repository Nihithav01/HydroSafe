from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import joblib
scaler = joblib.load("scaler.save")
app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

# @app.route("/home")
@app.route("/")
def login():
    return render_template("login.html")

@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/index", methods = ["POST"])
def index2():
    if request.method == "POST":
        # ph = request.form['ph']
        # print(ph)
        input_features = [float(x) for x in request.form.values()]
        # print(request.form['ph'])
        print(input_features)
        features_value = [np.array(input_features)]

        feature_names = ["ph", "Hardness" , "Solids", "Chloramines", "Sulfate",
                         "Conductivity", "Organic_carbon","Trihalomethanes", "Turbidity"]

        df = pd.DataFrame(features_value, columns = feature_names)
        print(df)
        df = scaler.transform(df)
        output = model.predict(df)
        print(output)
        if output[0] == 1:           
            prediction = "safe"
            return render_template('safe.html', prediction_text= "{}".format(prediction), input_features=input_features, prediction=prediction)
        else:
            prediction = "not safe"
            return render_template('notsafe.html', prediction_text= "{}".format(prediction), input_features=input_features, prediction=prediction)

@app.route("/alt")
def alt():
    return render_template("alt.html")


if __name__ == "__main__":
    app.run(debug=True)