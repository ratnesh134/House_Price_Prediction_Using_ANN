from flask import Flask,request,render_template
import  pickle
import numpy as np
import pandas as pd
import keras

# loading models
app = Flask(__name__)
model = keras.models.load_model("ANN_model_House_pred.keras")
scaler = pickle.load(open('scaler.pkl','rb'))

# creating routes 
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/house",methods=['POST','GET'])
def house():
    if request.method=='POST':
       longitude = request.form['longitude']
       latitude = request.form['latitude']
       houseage = request.form['houseage']
       houserooms = request.form['houserooms']
       totlabedrooms = request.form['totlabedrooms']
       population = request.form['population']
       households = request.form['households']
       medianincome = request.form['medianincome']
       oceanproximity = request.form['oceanproximity']

       features = np.array([longitude,latitude,houseage,houserooms,totlabedrooms,population,households,
                            medianincome,oceanproximity], dtype=float)

       features_scaled = scaler.transform([features])

       result = model.predict(features_scaled).reshape(1,-1)
       return render_template('index.html',result = result)



if __name__ == "__main__":
    app.run(debug=True)