from flask import Flask, request, jsonify, render_template
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import io
import time

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/whisper-audio-to-text"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
processor = WhisperProcessor.from_pretrained(model_name)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]

    # Read file into memory
    audio_bytes = file.read()
    audio_stream = io.BytesIO(audio_bytes)

    # Load audio
    audio_array, _ = librosa.load(audio_stream, sr=16000)

    input_features = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            task="transcribe",
            language="en"
        )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    start_time = time.time()

    # your transcription code here...

    end_time = time.time()

    return jsonify({
        "text": transcription,
        "time_taken": f"{round(end_time - start_time, 2)} seconds"
    })
if __name__ == "__main__":
    app.run(debug=True)