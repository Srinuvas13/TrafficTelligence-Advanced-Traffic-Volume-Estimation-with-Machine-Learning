import numpy as np 
import pickle 
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time 
import pandas
import os
from flask import Flask ,request, jsonify,render_template 


app = Flask(__name__)
print("Templates path:", app.template_folder)
import pickle
scale = pickle.load(open('encoder.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
          return render_template("index.html", prediction_text=...)

  
@app.route('/predict',methods=["POST","GET"])
def predict():
        input_feature = [float(x) for x in request.form.values()] 
        features_values= [np.array(input_feature)]
        names=[['holiday','temp','rain','snow','weather','year','mounth','day','hours','minutes','secounds']]
        data = pandas.DataFrame(features_values,columns=names)
        
        # data = scale.fit_transfrom(data)
        date = pandas.DataFrame(data,columns = names)
        prediction = model.predict(data)
        print(prediction)
        text = "Estimated traffic Volume is :"
        return render_template("output.html", Result=text + str(prediction[0]) + " units")



if __name__ == "__main__":
        port = int(os.environ.get("PORT",5000))
        app.run(port=port, debug=True, use_reloader=False)  # ✅ "use_reloader"

