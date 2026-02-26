from transformers import WhisperProcessor, WhisperForConditionalGeneration # pip install transformers
import torch # pip install torch
import torchaudio # pip install torchaudio and pip install torchcodec

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/whisper-audio-to-text"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
processor = WhisperProcessor.from_pretrained(model_name)

# Load and process audio file
def load_audio(file_path, target_sampling_rate=16000):
    # Load audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sampling_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sampling_rate)(waveform)

    return waveform.squeeze(0).numpy()

input_audio_path = "/home/avinash/Downloads/fromapp.wav"  # Change this to your audio file
audio_array = load_audio(input_audio_path)

input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
input_features = input_features.to(device)

forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

with torch.no_grad():
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

# Decode output
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(f"Transcribed Text: {transcription}")