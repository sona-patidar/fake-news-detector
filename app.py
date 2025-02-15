from flask import Flask, render_template, request
import pandas as pd
import string
import re
import joblib

app = Flask(__name__)

# Load pre-trained models
vectorization = joblib.load('models/vectorizer.pkl')
LR = joblib.load('models/logistic_regression.pkl')
DT = joblib.load('models/decision_tree.pkl')
GB = joblib.load('models/gradient_boosting.pkl')
RF = joblib.load('models/random_forest.pkl')


# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# Function to get prediction label
def output_label(n):
    return "Fake News" if n == 0 else "Real News"


@app.route('/')
def home():
    return render_template('index.html', predictions={})


@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']

    if not news.strip():
        return render_template('index.html', predictions={}, error="Please enter some text to check.")

    new_data = pd.DataFrame({"text": [news]})
    new_data['text'] = new_data['text'].apply(wordopt)
    new_x_test = vectorization.transform(new_data['text'])

    predictions = {
        "Logistic Regression": output_label(LR.predict(new_x_test)[0]),
        "Decision Tree": output_label(DT.predict(new_x_test)[0]),
        "Gradient Boosting": output_label(GB.predict(new_x_test)[0]),
        "Random Forest": output_label(RF.predict(new_x_test)[0])
    }

    return render_template('index.html', predictions=predictions, news=news)


if __name__ == '__main__':
    app.run(debug=True)