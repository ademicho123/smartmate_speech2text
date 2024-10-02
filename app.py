from flask import Flask, request, jsonify, render_template
import soundfile as sf
import numpy as np
import logging
from model import load_model, transcribe
import torch

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
processor, model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400

    try:
        # Read the audio file
        audio, sample_rate = sf.read(audio_file)
        logging.info(f"Audio file read. Shape: {audio.shape}, Sample rate: {sample_rate}")
        
        # Convert stereo to mono if necessary
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            logging.info(f"Converted to mono. New shape: {audio.shape}")
        
        # Ensure audio is in float32 format and normalize
        audio = audio.astype(np.float32)
        audio = audio / np.max(np.abs(audio))
        logging.info(f"Converted to float32 and normalized. Data type: {audio.dtype}, Max value: {np.max(np.abs(audio))}")

        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            # You may want to use a proper resampling method here
            audio = np.interp(np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate)), np.arange(len(audio)), audio)
            logging.info(f"Resampled to 16kHz. New shape: {audio.shape}")

        # Transcribe the audio
        logging.info("Starting transcription...")
        
        result = transcribe(audio, processor, model)
        
        logging.info("Transcription completed.")
        
        return jsonify({
            "transcription": result,
        })
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)