import pandas as pd
from flask import Flask, request, render_template
import os
from joblib import load

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)
app = Flask(__name__)


#loading model 
clf= load(MODEL_PATH)


@app.route('/', methods=['GET', 'POST'])
def index():
    result=''
    text=''
    #print(clf.predict(["I am angry"]))
    if request.method == 'POST':
        text=request.form.get("textTocompute")
        result = clf.predict([text])
    return render_template("index.html",result=result,text=text)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
