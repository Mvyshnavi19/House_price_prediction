import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd 

app = Flask(__name__)

model = pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    scaler = pickle.load(open('./house_scaler.pkl','rb'))
    data_new = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    prediction = model.predict(data_new)
    print(prediction[0])
    return jsonify(prediction[0])

if __name__ == "__main__":
    app.run(debug=True)


