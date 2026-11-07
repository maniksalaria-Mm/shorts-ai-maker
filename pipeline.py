import os, re, sys, json, subprocess, shutil, math
from pathlib import Path

# Set by choose_clips() so the UI can tell the user which selector actually ran.
LAST_SELECTOR = {"mode": "local", "note": ""}
_WHISPER_CACHE = {}

def run(cmd, cwd=None):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
    if p.returncode:
        raise RuntimeError(p.stderr[-4000:] or "Command failed")
    return p.stdout

def ensure_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "FFmpeg/ffprobe not found. On Streamlit Cloud, add a packages.txt file containing 'ffmpeg' "
            "to the repo root and reboot the app. On Windows, install FFmpeg and add it to PATH.")

def probe_duration(path):
    out = run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)])
    return float(out.strip())

def download_video(url, folder):
    # yt-dlp is used only when the user has rights/permission to download the source.
    Path(folder).mkdir(parents=True, exist_ok=True)
    template = str(Path(folder) / "%(id)s.%(ext)s")
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress",
           "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
           "--merge-output-format", "mp4", "-o", template, url]
    cookies = os.getenv("YTDLP_COOKIES_FILE")
    if cookies and Path(cookies).exists():
        cmd[3:3] = ["--cookies", cookies]
    try:
        run(cmd)
    except RuntimeError as e:
        msg = str(e)
        if "Sign in to confirm" in msg or "not a bot" in msg:
            raise RuntimeError(
                "YouTube blocked this download (bot check). This commonly happens on cloud servers. "
                "Download the video yourself and use 'Upload a video', or run the app on your own laptop.")
        raise
    videos = [p for p in Path(folder).glob("*") if p.suffix.lower() in [".mp4", ".mkv", ".webm", ".mov"]]
    if not videos:
        raise RuntimeError("yt-dlp completed but no video file was found.")
    return max(videos, key=lambda p: p.stat().st_mtime)

def save_upload(uploaded, folder):
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    path = Path(folder) / f"upload_{os.urandom(4).hex()}{suffix}"
    path.write_bytes(uploaded.getbuffer())
    return path

def extract_audio(video_path, wav_path):
    """16 kHz mono WAV: small, fast to decode, and avoids pushing a large video through PyAV."""
    try:
        run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(wav_path)])
    except RuntimeError as e:
        raise RuntimeError("Could not read audio from this video (it may have no audio track). " + str(e)[-300:])
    return wav_path

def transcribe_video(video_path, model_name, language=None):
    from faster_whisper import WhisperModel

    # CPU on purpose: CTranslate2 may detect CUDA even when cuBLAS DLLs are missing.
    if model_name not in _WHISPER_CACHE:
        _WHISPER_CACHE.clear()  # keep only one model in RAM (Streamlit Cloud is memory-limited)
        _WHISPER_CACHE[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
    model = _WHISPER_CACHE[model_name]
    wav = Path(video_path).with_suffix(".16k.wav")
    try:
        extract_audio(video_path, wav)
        segments, info = model.transcribe(str(wav), language=language, word_timestamps=True, vad_filter=True)
        result = []
        for seg in segments:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append({"start": float(w.start), "end": float(w.end), "word": w.word})
            result.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip(), "words": words})
    finally:
        try:
            wav.unlink()
        except OSError:
            pass
    if not result:
        raise RuntimeError("No speech detected. Try another Whisper model or check the audio.")
    return {"language": info.language, "segments": result}

def _cuda_available():
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False

def choose_clips(transcript, duration, count, min_len, max_len, api_key=None):
    segments = transcript["segments"]
    LAST_SELECTOR.update(mode="local", note="")
    clips = None
    if api_key:
        try:
            clips = _claude_clips(segments, duration, count, min_len, max_len, api_key)
            if clips:
                LAST_SELECTOR.update(mode="claude", note="")
        except Exception as e:
            # A usable local fallback is preferable to failing the entire job.
            LAST_SELECTOR.update(mode="local", note=f"Claude selection failed ({type(e).__name__}); used local selection instead.")
    if not clips:
        clips = _heuristic_clips(segments, duration, count, min_len, max_len)
    return _snap_to_speech(clips, segments, duration, min_len, max_len)

def _snap_to_speech(clips, segments, duration, min_len, max_len):
    """Move each clip start/end to the nearest sentence boundary so cuts don't land mid-sentence."""
    if not segments:
        return clips
    starts = [s["start"] for s in segments]; ends = [s["end"] for s in segments]
    out = []
    for c in clips:
        a = min(starts, key=lambda t: abs(t - c["start"]))
        b = min(ends, key=lambda t: abs(t - c["end"]))
        a = max(0.0, a - 0.15); b = min(duration, b + 0.25)
        if b - a < min_len * 0.8 or b - a > max_len * 1.1:
            a, b = c["start"], c["end"]
        out.append({**c, "start": a, "end": b})
    out.sort(key=lambda x: x["start"])
    clean = []
    for c in out:  # snapping can create overlaps; drop any
        if not clean or c["start"] >= clean[-1]["end"]:
            clean.append(c)
    return clean

def _claude_clips(segments, duration, count, min_len, max_len, api_key):
    import anthropic
    compact = [{"start":round(s["start"],1),"end":round(s["end"],1),"text":s["text"]} for s in segments]
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-haiku-5-5"), max_tokens=3000,
        messages=[{"role":"user","content":
        "Select the best short-form video moments from this timestamped transcript. "
        f"Return exactly up to {count} non-overlapping clips, each {min_len}-{max_len} seconds, "
        f"within 0-{duration:.1f} seconds. Prefer complete thoughts, strong hooks, useful, surprising, "
        "funny or emotionally engaging moments. Start/end should align to speech boundaries. "
        "Return ONLY valid JSON array: [{start:number,end:number,title:string,reason:string}]. "
        "Transcript: "+json.dumps(compact, ensure_ascii=False)}]
    )
    text = msg.content[0].text
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    data = json.loads(text)
    return _validate_clips(data, duration, min_len, max_len, count)

def _validate_clips(data, duration, min_len, max_len, count):
    clips=[]
    for x in data:
        try:
            start=max(0,float(x["start"])); end=min(duration,float(x["end"]))
            if min_len <= end-start <= max_len:
                clips.append({"start":start,"end":end,"title":str(x.get("title","Best moment"))[:90],
                              "reason":str(x.get("reason",""))[:300]})
        except Exception:
            continue
    clips.sort(key=lambda x:x["start"])
    selected=[]
    for c in clips:
        if all(c["end"] <= s["start"] or c["start"] >= s["end"] for s in selected):
            selected.append(c)
        if len(selected)>=count: break
    return selected

def _heuristic_clips(segments, duration, count, min_len, max_len):
    # Fallback: rank transcript windows using hook/energy vocabulary and speech density.
    keywords = ["why","secret","mistake","never","best","worst","important","imagine","because",
                "surprising","amazing","actually","problem","solution","truth","learn","tip","number"]
    windows=[]
    step=max(8,min_len//2)
    t=0
    while t < duration:
        end=min(duration,t+max_len)
        if end-t >= min_len:
            text=" ".join(s["text"] for s in segments if s["end"]>t and s["start"]<end)
            if text.strip():
                lower=text.lower()
                score=sum(lower.count(k) for k in keywords)*2 + min(len(text.split())/35,8)
                windows.append({"start":max(0,t),"end":end,"title":" ".join(text.split()[:7])[:75],
                                "reason":"Selected using local transcript relevance scoring.","_score":score})
        t+=step
    windows.sort(key=lambda x:x["_score"],reverse=True)
    chosen=[]
    for c in windows:
        if all(c["end"]<=s["start"] or c["start"]>=s["end"] for s in chosen):
            c.pop("_score",None); chosen.append(c)
        if len(chosen)>=count: break
    return sorted(chosen,key=lambda x:x["start"])

def _ass_time(seconds):
    h=int(seconds//3600); m=int((seconds%3600)//60); s=seconds%60
    return f"{h}:{m:02d}:{s:05.2f}"

def _escape_ass(text):
    return text.replace("\\","\\\\").replace("{","\\{").replace("}","\\}").replace("\n"," ")

def _make_ass(path, transcript, start, end, style):
    if style=="Yellow highlight":
        primary="&H0000D7FF&"
    elif style=="Minimal":
        primary="&H00FFFFFF&"
    else:
        primary="&H00FFFFFF&"
    header=f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,66,{primary},&H000000FF,&H90000000,&H60000000,-1,0,1,4,2,2,90,90,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    rows=[]
    for seg in transcript["segments"]:
        for w in seg.get("words",[]):
            a=max(start,w["start"]); b=min(end,w["end"])
            if b>a:
                # Word-by-word timing gives a readable karaoke-like caption rhythm.
                rows.append(f"Dialogue: 0,{_ass_time(a-start)},{_ass_time(b-start)},Default,,0,0,0,,{_escape_ass(w['word'].strip())}")
    if not rows:
        for seg in transcript["segments"]:
            a=max(start,seg["start"]); b=min(end,seg["end"])
            if b>a:
                rows.append(f"Dialogue: 0,{_ass_time(a-start)},{_ass_time(b-start)},Default,,0,0,0,,{_escape_ass(seg['text'])}")
    Path(path).write_text(header+"\n".join(rows),encoding="utf-8")

def render_clip(video, output, clip, transcript, caption_style, use_face):
    output = Path(output)
    start = float(clip["start"]); end = float(clip["end"]); duration = end - start
    ass = output.with_suffix(".ass")
    _make_ass(ass, transcript, start, end, caption_style)
    # Crop width never exceeds the source width (portrait sources would otherwise fail).
    crop_w = "min(iw\\,ih*9/16)"
    x_expr = "(iw-ow)/2"
    if use_face:
        try:
            x_expr = f"min(iw-ow\\,max(0\\,{_face_crop_x(video, start, end)}))"
        except Exception:
            pass
    # Run ffmpeg inside the output folder and reference the .ass by bare filename:
    # no drive letters / backslashes / colons to escape, so it behaves the same on Windows and Linux.
    vf = f"crop={crop_w}:ih:{x_expr}:0,scale=1080:1920,setsar=1,subtitles={ass.name}"
    try:
        run(["ffmpeg", "-y", "-ss", str(start), "-i", str(Path(video).resolve()), "-t", str(duration), "-vf", vf,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", output.name],
            cwd=str(output.parent.resolve()))
    finally:
        try:
            ass.unlink()
        except OSError:
            pass
    return output

def _face_crop_x(video, start, end):
    import cv2
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError("Face model not found")
    cap = cv2.VideoCapture(str(video))
    centers = []; frame_w = frame_h = 0
    try:
        # Sample several frames across the clip and use the median face position (steadier than one frame).
        for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
            cap.set(cv2.CAP_PROP_POS_MSEC, (start + (end - start) * frac) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_h, frame_w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                centers.append(x + w / 2)
    finally:
        cap.release()
    if not centers:
        raise RuntimeError("No face")
    centers.sort()
    center = centers[len(centers) // 2]
    crop_w = frame_h * 9 / 16
    return str(round(max(0, min(frame_w - crop_w, center - crop_w / 2)), 2))
