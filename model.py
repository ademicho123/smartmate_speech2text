import whisper

def load_model(model_name='base'):
    model = whisper.load_model(model_name)
    return model