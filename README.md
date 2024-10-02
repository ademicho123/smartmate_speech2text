# Speech Recognition Tool

This project is a simple web-based speech recognition tool that uses Facebook AI's Wav2Vec 2.0 model to transcribe English speech from audio files. It provides a user-friendly interface for uploading audio files and receiving text transcriptions.

## Features

- Upload audio files through a web interface
- Transcribe English speech to text using Wav2Vec 2.0
- Display transcription results in real-time

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/speech-recognition-tool.git
   cd speech-recognition-tool
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`

3. Use the web interface to upload an audio file and transcribe it:
   - Click the "Choose File" button to select an audio file
   - Click the "Transcribe" button to start the transcription process
   - Wait for the transcription to complete
   - View the transcription result on the page

## Project Structure

- `app.py`: Main Flask application file
- `model.py`: Contains functions for loading the Wav2Vec 2.0 model and performing transcription
- `templates/index.html`: HTML template for the web interface
- `requirements.txt`: List of Python dependencies

## Technologies Used

- Flask: Web framework for the backend
- PyTorch: Deep learning framework
- Transformers: Library for state-of-the-art Natural Language Processing
- Wav2Vec 2.0: Speech recognition model by Facebook AI
- SoundFile: Library for reading audio files
- NumPy: Library for numerical computations

## Contributing

Contributions to this project are welcome. Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook AI for the Wav2Vec 2.0 model
- Hugging Face for the Transformers library

## Support

If you encounter any problems or have any questions, please open an issue in the GitHub repository.