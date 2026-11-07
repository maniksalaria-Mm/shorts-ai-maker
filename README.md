# Shorts AI Maker

A local Streamlit app that turns videos you own or have permission to reuse into vertical, captioned Shorts. It runs on Windows and can be opened from an iPhone browser on the same Wi-Fi.

## Features
- YouTube URL ingestion via yt-dlp, or direct video upload
- Local speech transcription with faster-whisper
- Claude transcript-based clip selection when an API key is configured
- Local heuristic clip selection fallback when no API key is configured
- 9:16, 1080×1920 MP4 rendering with FFmpeg
- Word-timed captions and optional OpenCV face-aware crop
- Preview, individual downloads, ZIP export, and local export history

## Requirements
- Windows 10/11
- Python 3.10 or 3.11 recommended
- FFmpeg installed and available in PATH
- 8 GB RAM minimum recommended; 16 GB is better
- NVIDIA GPU is optional. CPU transcription works but can be slower.

## Windows installation
1. Install Python from python.org. During setup, enable **Add Python to PATH**.
2. Install FFmpeg and add its `bin` folder to PATH. Verify in a new PowerShell window:
   ```powershell
   ffmpeg -version
   ffprobe -version
   ```
3. Extract this project folder.
4. Open PowerShell in the project folder:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Optional Claude clip selection: copy `.env.example` to `.env` and add your Anthropic API key.
   Without a key, the app still runs and selects moments using a basic local transcript scoring fallback.
6. Start the app:
   ```powershell
   streamlit run app.py
   ```

Open the address printed by Streamlit, usually `http://localhost:8501`.

## Open it on iPhone
1. Connect the iPhone and Windows laptop to the same Wi-Fi.
2. In PowerShell, run `ipconfig` and find the laptop's IPv4 address.
3. Start Streamlit so it listens on the network:
   ```powershell
   streamlit run app.py --server.address 0.0.0.0
   ```
4. On iPhone Safari, open `http://LAPTOP-IP:8501` (replace with your laptop's IPv4 address).
5. If Windows Firewall asks, allow Python on your private network only.

For access away from home, use a private network tool such as Tailscale. Do not expose Streamlit directly to the public internet.

## Use
- Choose YouTube URL or Upload a video.
- Set the number and duration of clips in the sidebar.
- Click Generate Shorts and wait for transcription and rendering.
- Preview and download individual clips or a ZIP.

## Notes and limitations
- Only download/process YouTube videos that you own or are authorized to reuse. Follow YouTube's terms and applicable copyright law. Uploading your own source file is the most reliable route.
- Face-aware crop samples a frame near the middle of the clip and shifts the crop toward the largest detected face. It is not continuous AI tracking. If no face is detected, it uses a centered crop.
- The no-API fallback is a simple heuristic, not equivalent to an LLM. Claude API usage is billed by Anthropic and requires your own key.
- The first Whisper run downloads the selected model and may take time.
- Large videos can take a long time and use substantial disk space. Delete old files from `uploads` and `outputs` when you no longer need them.
- This is a local starter app, not a hosted service. Keep your laptop awake while processing.

## GitHub repository notes

- Do not commit `.env`, API keys, virtual environments, downloaded videos, or rendered outputs.
- `pipeline.py` uses Faster-Whisper on CPU (`int8`) to avoid CUDA/cuBLAS DLL dependency issues on Windows.
