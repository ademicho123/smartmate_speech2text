import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import numpy as np
from jiwer import wer
from model import load_model, transcribe

def load_audio(audio_path, target_sample_rate=16000):
    audio, sample_rate = sf.read(audio_path)
    
    # Ensure audio is mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample to target sample rate if necessary
    if sample_rate != target_sample_rate:
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * target_sample_rate / sample_rate)),
            np.arange(len(audio)),
            audio
        )
    
    return audio

def evaluate_model(model_name, test_data):
    processor, model = load_model(model_name)
    
    total_wer = 0
    num_samples = len(test_data)
    
    for audio_path, reference_text in test_data:
        audio = load_audio(audio_path)
        predicted_text = transcribe(audio, processor, model)
        
        sample_wer = wer(reference_text, predicted_text)
        total_wer += sample_wer
        
        print(f"Audio: {audio_path}")
        print(f"Reference: {reference_text}")
        print(f"Predicted: {predicted_text}")
        print(f"WER: {sample_wer:.4f}\n")
    
    average_wer = total_wer / num_samples
    print(f"Average WER: {average_wer:.4f}")
    
    return average_wer

if __name__ == "__main__":
    model_name = 'facebook/wav2vec2-base-960h'
    
    # Example test data: (audio_path, reference_text)
    test_data = [
        ("//workspaces/smartmate_speech2text/harvard.wav", "The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun."),
        #("/path/to/audio2.wav", "reference transcription two"),
        
    ]
    
    evaluate_model(model_name, test_data)
