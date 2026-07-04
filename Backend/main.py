import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

from healthcare_engine import process_healthcare_input

# Load Whisper model
model = whisper.load_model("base")

sample_rate = 44100

print("Choose Input Method:")
print("1. Voice Input")
print("2. Text Input")

choice = input("Enter choice: ")

# -------------------------------
# VOICE INPUT
# -------------------------------

if choice == "1":

    print("\nPress ENTER to start recording...")
    input()

    print("Recording... Press ENTER to stop.")

    recording = []

    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        callback=callback
    ):
        input()

    print("Recording stopped.")

    audio = np.concatenate(recording, axis=0)

    write("input.wav", sample_rate, audio)

    result = model.transcribe("input.wav")

    text = result["text"]

# -------------------------------
# TEXT INPUT
# -------------------------------

elif choice == "2":

    text = input("\nEnter healthcare text: ")

else:
    print("Invalid choice.")
    exit()

# -------------------------------
# PROCESS HEALTHCARE INPUT
# -------------------------------

output = process_healthcare_input(text)

# -------------------------------
# DISPLAY OUTPUT
# -------------------------------

print("\n--- Healthcare Analysis Report ---")

print("\nInput:")
print(output["input_text"])

print("\nDetected Symptoms:")
for symptom in output["symptoms"]:
    print("-", symptom)

if output["pregnancy_month"]:
    print("\nPregnancy Month:")
    print("-", output["pregnancy_month"])

print("\nRisk Level:")
print("-", output["risk_level"])

print("\nRecommended RAG Query:")
print(output["recommended_rag_query"])
