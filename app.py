import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = load_model("bilstm_sentiment_analysis_model.keras")

# with open("tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)

tokenizer = pickle.load(open("tokenizer.pickle","rb"))

max_len = 100

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)[0][0]
    return "Positive 😊" if pred > 0.5 else "Negative 😞"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    result = predict_sentiment(text)
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)

