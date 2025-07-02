import os
import torch
import whisper
import tempfile
import torchaudio
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# Load environment variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Ensure the audio file path is correct
AUDIO_FILE = os.path.join(os.path.dirname(__file__), "clip2.wav")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Optionally disable TF32 warnings for reproducibility (pyannote suggestion)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Load models
print("Loading Whisper model...")
whisper_model = whisper.load_model("base").to(device)

print("Loading speaker diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HUGGINGFACE_TOKEN
)
pipeline.to(device)

# Run speaker diarization
print("Running speaker diarization...")
diarization = pipeline(AUDIO_FILE)

# Transcribe each speaker segment
print("Running Whisper transcription...")
transcript = []
audio, sr = torchaudio.load(AUDIO_FILE)

for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_time = turn.start
    end_time = turn.end
    print(f"{speaker}: {start_time:.2f}s - {end_time:.2f}s")

    # Extract the speaker segment
    segment = audio[:, int(start_time * sr):int(end_time * sr)]

    # Save temporary segment and transcribe
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        torchaudio.save(tmp.name, segment, sr)
        result = whisper_model.transcribe(tmp.name, fp16=True)
        text = result["text"].strip()
        transcript.append(f"{speaker} [{start_time:.1f}s - {end_time:.1f}s]: {text}")
        os.remove(tmp.name)

# Print final transcript
print("\n===== FINAL TRANSCRIPT =====")
for line in transcript:
    print(line)
