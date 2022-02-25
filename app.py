from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model01.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    Experiance1 = float(request.form['Experiance'])
    Test_score1 = float(request.form['Test_score'])
    Interview_score1 = float(request.form['Interview_score'])

    arr = np.array([[Experiance1,Test_score1,Interview_score1]])
    pred = model.predict(arr)
    output = np.round(pred,decimals=2)
    #output=round(pred[0],2)


    return render_template('after.html', data="Predicted Salary is : {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
