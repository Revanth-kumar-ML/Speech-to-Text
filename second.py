from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once
model_name = "AventIQ-AI/whisper-audio-to-text"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
processor = WhisperProcessor.from_pretrained(model_name)

# Load and process audio file
def load_audio(file_path, target_sampling_rate=16000):
    # librosa automatically converts to mono and resamples
    audio, _ = librosa.load(file_path, sr=target_sampling_rate)
    return audio

# Give correct path (IMPORTANT: use raw string for Windows path)
input_audio_path = r"D:\ML_intern\Speech-to-Text\fromapp.wav"

# Load audio
audio_array = load_audio(input_audio_path)

# Convert to model input
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

# Decode output
transcription = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True
)[0]

print(f"Transcribed Text: {transcription}")