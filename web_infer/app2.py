from flask import Flask, render_template, request
import uuid
import os
from infer_api import generate_tts

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        gender = request.form["gender"]
        speaker_id = 0 if gender == "남성" else 1

        filename = f"{uuid.uuid4().hex}.wav"
        output_path = os.path.join("web_infer", "static", "audio", filename)
        generate_tts(text, speaker_id, output_path)
        return render_template("index.html", audio_file=filename)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
