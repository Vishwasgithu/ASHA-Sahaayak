import os
import tempfile
from typing import Any

from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper 

from healthcare_engine import process_healthcare_input
from rag.pipeline import RAGPipeline
from utils.helpers import get_logger


logger = get_logger("Flask.API")

app = Flask(__name__)
# Enable CORS for all routes so our local static frontend pages can query the endpoints
CORS(app)

# RAG components load their database and models lazily. Keeping one pipeline
# instance avoids rebuilding clients and provides thread-safe access.
rag_pipeline = RAGPipeline()

# Load Whisper model at startup
print("Loading OpenAI Whisper 'base' model... This may take a few seconds.")
try:
    model = whisper.load_model("base")
    print("Whisper model loaded successfully!")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    print("Warning: Voice transcription will run in mock fallback mode if model fails.")
    model = None


def _legacy_guidance_fallback(analysis: dict[str, Any]) -> str:
    """Return the pre-RAG guidance only when RAG is operationally unavailable."""
    symptoms = set(analysis.get("symptoms") or [])

    if "bleeding" in symptoms:
        return (
            "Possible emergency maternal condition detected. "
            "Immediate medical attention required."
        )
    if {"headache", "swelling", "blurred vision"}.issubset(symptoms):
        return (
            "Possible maternal hypertension or preeclampsia risk detected. "
            "Blood pressure monitoring and PHC referral recommended."
        )
    if symptoms.intersection({"severe abdominal pain", "reduced fetal movement"}):
        return (
            "A maternal warning sign was detected. "
            "Prompt assessment by a qualified healthcare professional is recommended."
        )
    if "fever" in symptoms:
        return (
            "Monitor temperature and hydration. "
            "Refer to PHC if symptoms persist."
        )
    return "No immediate maternal risk detected."


def analyze_healthcare_text(text: str) -> dict[str, Any]:
    """Run rule-based screening followed by evidence-grounded RAG generation."""
    analysis = process_healthcare_input(text)
    rag_query = analysis["recommended_rag_query"]

    try:
        rag_result = rag_pipeline.run(rag_query)
        recommendation = rag_result["answer"]
        sources = rag_result["retrieved_sources"]
        rag_metadata = dict(rag_result["metadata"])
        rag_metadata.update(
            {
                "rag_available": True,
                "fallback_used": False,
            }
        )
    except Exception as exc:
        # Do not expose provider, database, or credential details to clients.
        logger.exception(
            "RAG processing unavailable; returning legacy guidance fallback."
        )
        recommendation = _legacy_guidance_fallback(analysis)
        sources = []
        rag_metadata = {
            "rag_available": False,
            "fallback_used": True,
            "failure_type": type(exc).__name__,
        }

    return {
        "risk_level": analysis["risk_level"],
        "symptoms": analysis["symptoms"],
        "pregnancy_month": analysis["pregnancy_month"],
        "clinical_recommendation": recommendation,
        "sources": sources,
        "metadata": rag_metadata,
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to test connection status from frontend settings."""
    return jsonify({
        "status": "healthy",
        "whisper_loaded": model is not None
    }), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Process symptoms text and calculate maternal risk factors."""
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({"error": "No healthcare text provided"}), 400
        
    try:
        analysis_report = analyze_healthcare_text(text)
        return jsonify(analysis_report), 200
    except Exception as e:
        return jsonify({"error": f"Internal processing error: {str(e)}"}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Receive audio file, transcribe using Whisper, and analyze symptoms."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded in the 'audio' field"}), 400
        
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "Selected audio file is empty"}), 400

    # Save to a temporary file
    temp_dir = tempfile.gettempdir()
    # Keep original file extension (e.g. webm, wav) so Whisper/ffmpeg can decode it correctly
    ext = os.path.splitext(audio_file.filename)[1] or '.webm'
    temp_path = os.path.join(temp_dir, f"asha_upload_{os.urandom(4).hex()}{ext}")
    
    try:
        audio_file.save(temp_path)
        print(f"Saved temporary upload file to: {temp_path}")
        
        # Verify Whisper model is loaded
        if not model:
            raise RuntimeError("Whisper model is not initialized on this server.")
            
        print("Starting Whisper transcription...")
        result = model.transcribe(temp_path)
        transcribed_text = result.get("text", "").strip()
        print(f"Transcribed Text: {transcribed_text}")
        
        # Process transcribed text through the same screening and RAG flow as
        # typed input.
        analysis_report = analyze_healthcare_text(transcribed_text)
        return jsonify(analysis_report), 200
        
    except Exception as e:
        print(f"Transcription/Processing Error: {e}")
        # Clean up in case of error
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass
            
        return jsonify({
            "error": f"Failed to transcribe and analyze audio: {str(e)}",
            "fallback_suggested": True
        }), 500
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Deleted temporary upload file: {temp_path}")
            except Exception as cleanup_err:
                print(f"Error cleaning up temporary file: {cleanup_err}")

if __name__ == '__main__':
    # Run Flask server locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
