import os, json, shutil, zipfile, uuid, subprocess
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from pipeline import (
    ensure_ffmpeg, download_video, save_upload, transcribe_video,
    choose_clips, render_clip, probe_duration
)

load_dotenv()
BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

st.set_page_config(page_title="Shorts AI Maker", page_icon="🎬", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
.hero {padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#17152f,#29205c); color:white; margin-bottom:1.2rem;}
.hero h1 {font-size:2.2rem; margin:0 0 .3rem 0;}
.hero p {color:#d8d5f5; margin:0;}
div[data-testid="stMetric"] {background:#f5f4fb; padding:12px; border-radius:12px;}
</style>
<div class="hero">
<h1>🎬 Shorts AI Maker</h1>
<p>Turn videos you own or have permission to reuse into vertical, captioned Shorts.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Whisper model", ["small", "base", "medium", "large-v3"], index=0,
                         help="Small is a good starting point. Larger models are slower but may be more accurate.")
    language = st.text_input("Language (optional)", placeholder="auto-detect")
    count = st.slider("Number of clips", 1, 10, 5)
    min_len = st.slider("Minimum clip length (sec)", 15, 45, 30)
    max_len = st.slider("Maximum clip length (sec)", 30, 90, 60)
    caption_style = st.selectbox("Caption style", ["Bold white", "Yellow highlight", "Minimal"])
    use_face = st.checkbox("Try face-aware vertical crop", value=True,
                           help="Uses OpenCV face detection when available; otherwise uses a centered crop.")
    st.caption("AI clip selection uses Claude when ANTHROPIC_API_KEY is configured.")

tab1, tab2 = st.tabs(["✨ Create Shorts", "📁 Previous exports"])
with tab1:
    source_mode = st.radio("Choose video source", ["YouTube URL", "Upload a video"], horizontal=True)
    url = ""
    uploaded = None
    if source_mode == "YouTube URL":
        url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=...")
        st.caption("Only process videos you created, own, or have explicit permission to reuse.")
    else:
        uploaded = st.file_uploader("Choose a video", type=["mp4", "mov", "mkv", "webm", "avi"])

    c1, c2 = st.columns([1, 3])
    with c1:
        run = st.button("🚀 Generate Shorts", type="primary", use_container_width=True)
    with c2:
        st.caption("Pipeline: ingest → transcribe → AI moment selection → vertical edit → captions → export.")

    if run:
        try:
            ensure_ffmpeg()
            if source_mode == "YouTube URL":
                if not url.strip():
                    st.error("Paste a video URL first.")
                    st.stop()
                with st.spinner("Downloading video…"):
                    video_path = download_video(url.strip(), UPLOADS)
            else:
                if uploaded is None:
                    st.error("Choose a video first.")
                    st.stop()
                video_path = save_upload(uploaded, UPLOADS)

            st.success(f"Video ready: {video_path.name}")
            duration = probe_duration(video_path)
            st.info(f"Video duration: {duration/60:.1f} minutes")

            progress = st.progress(0, text="Transcribing audio…")
            transcript = transcribe_video(video_path, model, language.strip() or None)
            progress.progress(35, text="Finding the strongest moments…")
            clips = choose_clips(
                transcript, duration, count, min_len, max_len,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            if not clips:
                st.error("No suitable clips were found. Try a longer video or a smaller minimum duration.")
                st.stop()

            job = uuid.uuid4().hex[:8]
            job_dir = OUTPUTS / job
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "transcript.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")
            results = []
            for i, clip in enumerate(clips, 1):
                progress.progress(35 + int(60*i/len(clips)), text=f"Rendering clip {i}/{len(clips)}…")
                out = job_dir / f"short_{i:02d}.mp4"
                render_clip(video_path, out, clip, transcript, caption_style, use_face)
                results.append((out, clip))
            progress.progress(100, text="Finished!")
            st.session_state["latest_results"] = [(str(p), c) for p,c in results]
            st.session_state["latest_job"] = str(job_dir)
            st.success(f"Created {len(results)} Shorts.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.exception(e)

    if "latest_results" in st.session_state:
        st.subheader("Your generated Shorts")
        for path, clip in st.session_state["latest_results"]:
            p = Path(path)
            with st.container(border=True):
                left, right = st.columns([1, 2])
                with left:
                    st.video(str(p))
                with right:
                    st.markdown(f"**{clip.get('title','Untitled clip')}**")
                    st.write(clip.get("reason", ""))
                    st.caption(f"{clip['start']:.1f}s – {clip['end']:.1f}s")
                    st.download_button("⬇️ Download MP4", data=p.read_bytes(), file_name=p.name,
                                       mime="video/mp4", key=f"dl_{p.name}")

        job_dir = Path(st.session_state["latest_job"])
        zip_path = job_dir / "shorts_export.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in job_dir.glob("short_*.mp4"):
                z.write(p, p.name)
        st.download_button("📦 Download all Shorts (ZIP)", data=zip_path.read_bytes(),
                           file_name="shorts_export.zip", mime="application/zip")

with tab2:
    folders = sorted([p for p in OUTPUTS.iterdir() if p.is_dir()], key=lambda p:p.stat().st_mtime, reverse=True)
    if not folders:
        st.info("Your exported videos will appear here after your first generation.")
    for folder in folders:
        vids = sorted(folder.glob("short_*.mp4"))
        if vids:
            with st.expander(f"Export {folder.name} — {len(vids)} clips"):
                for v in vids:
                    st.video(str(v))
                    st.download_button(f"Download {v.name}", data=v.read_bytes(), file_name=v.name,
                                       mime="video/mp4", key=f"hist_{folder.name}_{v.name}")
