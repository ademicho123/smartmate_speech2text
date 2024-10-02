import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def load_model(model_name='facebook/wav2vec2-base-960h'):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model

def transcribe(audio, processor, model):
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription