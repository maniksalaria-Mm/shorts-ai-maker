import os, re, json, subprocess, shutil, math
from pathlib import Path

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-4000:] or "Command failed")
    return p.stdout

def ensure_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg/ffprobe not found. Install FFmpeg and add it to your PATH, then restart the terminal.")

def probe_duration(path):
    out = run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)])
    return float(out.strip())

def download_video(url, folder):
    # yt-dlp is used only when the user has rights/permission to download the source.
    template = str(Path(folder) / "%(title).80s_%(id)s.%(ext)s")
    run(["yt-dlp","--no-playlist","-f","bv*+ba/b","--merge-output-format","mp4","-o",template,url])
    candidates = sorted(Path(folder).glob("*"))
    videos = [p for p in candidates if p.suffix.lower() in [".mp4",".mkv",".webm",".mov"]]
    if not videos:
        raise RuntimeError("yt-dlp completed but no video file was found.")
    return max(videos, key=lambda p:p.stat().st_mtime)

def save_upload(uploaded, folder):
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    path = Path(folder) / f"upload_{os.urandom(4).hex()}{suffix}"
    path.write_bytes(uploaded.getbuffer())
    return path

def transcribe_video(video_path, model_name, language=None):
    from faster_whisper import WhisperModel

    # Use CPU deliberately. CTranslate2 may detect a CUDA device even when
    # the required CUDA/cuBLAS DLLs are missing on Windows.
    print("Loading Faster-Whisper on CPU (int8)...")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(video_path), language=language, word_timestamps=True, vad_filter=True)
    result = []
    for seg in segments:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({"start":float(w.start),"end":float(w.end),"word":w.word})
        result.append({"start":float(seg.start),"end":float(seg.end),"text":seg.text.strip(),"words":words})
    if not result:
        raise RuntimeError("No speech detected. Try another Whisper model or check the audio.")
    return {"language":info.language,"segments":result}

def _cuda_available():
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False

def choose_clips(transcript, duration, count, min_len, max_len, api_key=None):
    segments = transcript["segments"]
    if api_key:
        try:
            return _claude_clips(segments, duration, count, min_len, max_len, api_key)
        except Exception:
            # A usable local fallback is preferable to failing the entire job.
            pass
    return _heuristic_clips(segments, duration, count, min_len, max_len)

def _claude_clips(segments, duration, count, min_len, max_len, api_key):
    import anthropic
    compact = [{"start":round(s["start"],1),"end":round(s["end"],1),"text":s["text"]} for s in segments]
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-3-5-haiku-latest", max_tokens=3000,
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
    start=float(clip["start"]); end=float(clip["end"]); duration=end-start
    ass=output.with_suffix(".ass")
    _make_ass(ass,transcript,start,end,caption_style)
    # Center crop to 9:16. Face-aware option attempts to locate a face near the middle
    # and offsets the crop horizontally; safe fallback is a centered crop.
    x_expr="(iw-ih*9/16)/2"
    if use_face:
        try:
            x_expr=f"({_face_crop_x(video,start, end)})"
        except Exception:
            pass
    ass_filter = str(ass).replace("\\\\", "/").replace(":", "\\\\:").replace("\\'", "\\\\\\'")
    vf=f"crop=ih*9/16:ih:{x_expr}:0,scale=1080:1920,setsar=1,subtitles='{ass_filter}'"
    run(["ffmpeg","-y","-ss",str(start),"-i",str(video),"-t",str(duration),"-vf",vf,
         "-c:v","libx264","-preset","veryfast","-crf","21","-c:a","aac","-b:a","160k",
         "-movflags","+faststart",str(output)])
    try:
        ass.unlink()
    except OSError:
        pass
    return output

def _face_crop_x(video,start,end):
    import cv2
    cap=cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, (start+end)*500)
    ok, frame=cap.read(); cap.release()
    if not ok: raise RuntimeError("No frame")
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    faces=cascade.detectMultiScale(gray,1.1,4,minSize=(50,50))
    if len(faces)==0: raise RuntimeError("No face")
    x,y,w,h=max(faces,key=lambda f:f[2]*f[3])
    frame_w=frame.shape[1]; frame_h=frame.shape[0]
    crop_w=frame_h*9/16
    center=x+w/2
    left=max(0,min(frame_w-crop_w,center-crop_w/2))
    return str(round(left,2))
