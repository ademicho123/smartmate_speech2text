from flask import Flask, request, jsonify, render_template
import whisper
import soundfile as sf
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
model = whisper.load_model("small")

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

        # Resample to 16kHz
        if sample_rate != 16000:
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio)
            _, probs = model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            logging.info(f"Detected language: {detected_lang}")
        
        # Transcribe the audio with different configurations
        logging.info("Starting transcription...")
        
        result1 = model.transcribe(audio, language='en', temperature=0.0)
        logging.info(f"Transcription 1 (temp=0.0): {result1['text']}")
        
        result2 = model.transcribe(audio, language='en', temperature=0.7)
        logging.info(f"Transcription 2 (temp=0.7): {result2['text']}")
        
        result3 = model.transcribe(audio, language=None, temperature=0.7)
        logging.info(f"Transcription 3 (auto lang): {result3['text']}")
        
        # Choose the non-empty result, preferring higher temperature
        result = result3 if result3['text'].strip() else (result2 if result2['text'].strip() else result1)
        
        logging.info("Transcription completed.")
        
        return jsonify({
            "transcription": result["text"],
            "detected_language": detected_lang,
            "transcription1": result1["text"],
            "transcription2": result2["text"],
            "transcription3": result3["text"]
        })
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)