# Astra Backend

FastAPI backend for Astra, the voice-first multi-agent college AI assistant.

## Run

```powershell
.\start-backend.ps1
```

Add keys in `.env` when ready. The app runs in setup mode without keys.

## Local voice

Piper TTS is enabled by default for free local Astra speech and accurate waveform sync.
The configured high-quality voice is downloaded into `.cache/piper` and warmed when
the API starts; later requests reuse the model and cached WAV files.
