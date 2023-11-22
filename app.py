from flask import Flask, request, render_template
from simpletransformers.ner import NERModel
from joblib import load

app = Flask(__name__)

model = load('/static/model.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        if model:
            predictions, raw_outputs = model.predict([text])
            return render_template('result.html', predictions=predictions, text=text)
        else:
            return "Model not loaded properly", 500
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
