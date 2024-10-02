from flask import Flask, request, jsonify
import soundfile as sf
from model import load_model

# Initialize Flask app and load Whisper model
app = Flask(__name__)
model = load_model('base')  

@app.route('/upload', methods=['POST'])
def transcribe():
    # Load audio file from request
    audio_file = request.files['audio_file']
    audio, sample_rate = sf.read(audio_file)

    # Whisper requires WAV audio; ensure format compatibility
    transcription = model.transcribe(audio)

    return jsonify({"transcription": transcription['text']})

if __name__ == '__main__':
    app.run(debug=True)
