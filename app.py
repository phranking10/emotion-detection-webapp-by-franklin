from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)

# model
model = load_model("emotion_model.h5")

# classes mapping
emotions = ["angry", "sad", "happy", "fear"]

# upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# create database if not exist
db = "emotion.db"
conn = sqlite3.connect(db, check_same_thread=False)
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS records(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT,
    image_path TEXT,
    prediction TEXT,
    date_time TEXT
)""")
conn.commit()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    user_name = request.form['username']
    img_file = request.files['image']

    if img_file:
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], img_file.filename)
        img_file.save(img_path)

        img = load_img(img_path, target_size=(48,48), color_mode="grayscale")
        img = img_to_array(img)
        img = img/255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        pred_emotion = emotions[np.argmax(prediction)]

        # save to DB
        cur.execute("INSERT INTO records(user_name, image_path, prediction, date_time) VALUES (?,?,?,?)",
                    (user_name, img_path, pred_emotion, str(datetime.now())))
        conn.commit()

        return render_template("result.html", emotion=pred_emotion, img_path=img_path, username=user_name)
    else:
        return "No image uploaded."


@app.route("/history")
def history():
    cur.execute("SELECT user_name, image_path, prediction, date_time FROM records ORDER BY id DESC")
    data = cur.fetchall()
    return render_template("history.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)
