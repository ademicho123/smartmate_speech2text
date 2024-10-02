import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import numpy as np

def load_model(model_name='facebook/wav2vec2-base-960h'):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model

def transcribe(audio, processor, model):
    # Check if audio is too short (less than 0.5 seconds)
    if len(audio) < 8000:  # Assuming 16kHz sample rate
        # Pad the audio to 0.5 seconds
        padding = np.zeros(8000 - len(audio))
        audio = np.concatenate([audio, padding])
    
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    
    # Get the attention mask
    attention_mask = torch.ones(input_values.shape, dtype=torch.long, device=input_values.device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription
'''
# This is used to test the model above
if __name__ == "__main__":
    # Load the model and processor
    processor, model = load_model()

    # Load an audio file
    audio_path = "/workspaces/smartmate_speech2text/Recording (online-audio-converter.com).wav"  # Update with your audio file path
    audio, sample_rate = sf.read(audio_path)

    # Ensure audio is mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        # You may want to use a proper resampling method here
        audio = np.interp(np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate)), np.arange(len(audio)), audio)

    # Perform transcription
    result = transcribe(audio, processor, model)
    print("Transcription:", result)
'''