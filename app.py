from pathlib import Path
import json, html
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model" / "persian_model.keras"
LABELS_PATH = BASE / "model" / "label_mapping.json"
IMG_SIZE = (160, 160)

PERSIAN_TO_LATIN = {
    "Alef":"A", "Be":"B", "Pe":"P", "Te":"T", "Se":"S / Th",
    "Jim":"J", "Che":"Ch", "He":"H", "Khe":"Kh", "Dal":"D",
    "Zal":"Z / Dh", "Re":"R", "Ze":"Z", "Zhe":"Zh", "Sin":"S",
    "Shin":"Sh", "Sad":"S", "Zad":"Z", "Ta":"T", "Za":"Z",
    "Ayn":"'A", "Ghayn":"Gh", "Fe":"F", "Qaf":"Q", "Kaf":"K",
    "Gaf":"G", "Lam":"L", "Mim":"M", "Nun":"N", "Vav":"V / O / U",
    "Heh":"H", "Ye":"Y / I"
}

st.set_page_config(page_title="Persian Letter Recognition", page_icon="✍️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Naskh+Arabic:wght@400;500;600;700&display=swap');
:root { --navy:#071629; --navy2:#0b2540; --panel:#0d2036; --gold:#e9b85f; --gold2:#f6d58c; --cream:#fff5df; --muted:#bdc9d8; }
.stApp { background:radial-gradient(ellipse at top right,#173653 0%,#091a2d 42%,#050e1b 100%); color:#f7f2e8; font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(5,14,27,.92); }
.block-container { max-width:1400px; padding-top:1.5rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#06101e,#0b2540); border-right:1px solid #9a753d; }
[data-testid="stSidebar"] * { color:#f8f0df !important; }
.hero { padding:1.7rem 2rem; border:1px solid #b58b4b; border-radius:20px; background:linear-gradient(115deg,rgba(5,17,32,.98),rgba(12,42,67,.95)); box-shadow:0 12px 34px #0005; margin-bottom:1.25rem; position:relative; overflow:hidden; }
.hero:after { content:'خط فارسی  ✦  ش  ✦  الفبای فارسی'; position:absolute; right:1.5rem; top:1.3rem; color:#e9b85f; font-family:'Noto Naskh Arabic',serif; font-size:1.35rem; opacity:.8; }
.eyebrow { text-transform:uppercase; letter-spacing:.17em; font-size:.72rem; font-weight:700; color:var(--gold); }
.hero h1 { color:#fff5df; margin:.35rem 0 .4rem; font-family:Georgia,serif; font-size:2.35rem; line-height:1.2; }
.hero p { margin:0; color:#c7d3df; font-size:1rem; max-width:75%; }
.section-title { font-family:Georgia,serif; font-size:1.18rem; font-weight:700; color:var(--gold2); margin:.35rem 0 .75rem; }
.panel,.result-card { background:linear-gradient(145deg,rgba(15,39,63,.97),rgba(7,24,42,.98)); border:1px solid #8f6d3b; border-radius:17px; padding:1.1rem 1.25rem; box-shadow:0 7px 22px #0003; color:#f7f2e8; }
.muted,.small-note { color:var(--muted); font-size:.9rem; }
.result-card { text-align:center; min-height:320px; padding:1.3rem; }
.result-label { color:#d3b77e; font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; }
.big-letter { font-family:'Noto Naskh Arabic',serif; direction:rtl; font-size:6rem; line-height:1.3; color:#111; margin:.3rem auto; background:var(--cream); border:1px solid #e9c77e; border-radius:13px; max-width:270px; }
.translit { color:var(--gold2); font-weight:700; font-size:1.2rem; }
.latin-result { color:#fff5df; font-weight:800; font-size:1.55rem; margin-top:.1rem; }
.pill { display:inline-block; margin-top:.75rem; padding:.4rem .85rem; border-radius:99px; background:linear-gradient(90deg,#dba64d,#f5d58b); color:#102036; font-size:.8rem; font-weight:800; }
[data-testid="stFileUploader"] { background:#0b1d31; border:1px dashed #c29b56; border-radius:14px; padding:.7rem; }
[data-testid="stFileUploader"] section { background:transparent; }
div[data-testid="stMetric"] { background:#0d2036; border:1px solid #8f6d3b; padding:.85rem; border-radius:14px; }
div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#fff5df !important; }
.stProgress > div > div { background:linear-gradient(90deg,#b78337,#f6d58c); }
.stButton button { border:1px solid #c49b55; }
hr { border-color:#6e5838; }
[data-testid="stImage"] img { border-radius:12px; border:1px solid #5e6e7e; }
@media(max-width:700px) { .hero { padding:1.2rem; } .hero h1 {font-size:1.7rem;} .hero:after {display:none;} .hero p {max-width:100%;} .big-letter {font-size:4.7rem;} }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    mapping = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, mapping

def prepare_image(pil_image):
    image = ImageOps.exif_transpose(pil_image).convert("RGB")
    # Match the synthetic dataset: white background, dark glyphs; preserve aspect ratio.
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side-image.width)//2, (side-image.height)//2))
    canvas = canvas.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(canvas, dtype=np.float32)[None, ...]

with st.sidebar:
    st.markdown("<div style='text-align:center;font-family:Georgia,serif;font-size:1.35rem;color:#f6d58c'>✦ خط فارسی ✦</div>", unsafe_allow_html=True)
    st.markdown("## Persian Letter Recognition")
    st.caption("Computer Vision Prototype · 32 Persian letters")
    st.divider()
    st.markdown("**How to use**")
    st.markdown("1. Upload one Persian letter image.\\n2. Use a clear, high-contrast image.\\n3. Review the top predictions.")
    st.divider()
    st.markdown("**Recognition pipeline**")
    st.markdown("Image → consistent resizing → TensorFlow classifier")
    st.caption("Generated-font evaluation does not guarantee handwritten-letter accuracy.")

st.markdown("""<div class="hero"><div class="eyebrow">✦ Persian calligraphy · Computer vision · 32 classes ✦</div><h1>Persian Letter Recognition</h1><p>Upload a Persian character to explore the model's prediction, transliteration, and processed image.</p></div>""", unsafe_allow_html=True)

model, mapping = load_model()
if model is None:
    st.warning("The model has not been trained yet.")
    st.markdown("Open the included `README_FIRST.txt` and follow the setup steps. The training script creates the model files automatically.")
    st.code("python generate_dataset.py\npython train_model_tf.py\nstreamlit run app.py", language="powershell")
    st.stop()

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown('<div class="section-title">✦ 01 · Upload a Persian letter</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose one character image (PNG or JPG)", type=["png", "jpg", "jpeg"])
    st.markdown("<div class='panel'><b style='color:#f6d58c'>For best results</b><p class='muted'>Upload a single centered character with a plain background. Avoid images containing multiple letters or decorative text.</p></div>", unsafe_allow_html=True)

if uploaded:
    try:
        original = Image.open(uploaded).convert("RGB")
        arr = prepare_image(original)
        probs = model.predict(arr, verbose=0)[0]
        order = np.argsort(probs)[::-1][:3]
        by_id = {int(item["class_id"]): item for item in mapping}
        best = by_id[int(order[0])]
        latin = PERSIAN_TO_LATIN.get(best["name"], best["name"])
        with right:
            st.markdown('<div class="section-title">✦ 02 · Prediction result</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="result-card"><div class="result-label">Predicted Persian character</div><div class="big-letter">{html.escape(best['letter'])}</div><div class="translit">{html.escape(best['name'])}</div><div style="margin-top:.65rem;color:#d5c8ad;font-size:.85rem;font-weight:600;">English transliteration</div><div class="latin-result">{html.escape(latin)}</div><div class="pill">TensorFlow transfer learning</div></div>""", unsafe_allow_html=True)
            st.markdown("#### ✦ Top matching letters")
            for rank, idx in enumerate(order, start=1):
                item = by_id[int(idx)]
                a, b = st.columns([4, 1])
                with a:
                    st.markdown(f"**{rank}. {html.escape(item['letter'])} — {html.escape(item['name'])}**")
                    st.progress(float(np.clip(probs[idx], 0, 1)))
                with b:
                    st.markdown(f"**{probs[idx]:.1%}**")
            st.caption("Softmax scores are model scores, not guaranteed calibrated probabilities.")
        st.markdown("---")
        st.markdown('<div class="section-title">✦ 03 · Image preview</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div class='panel'><b>Original upload</b></div>", unsafe_allow_html=True)
            st.image(original, use_container_width=True)
        with c2:
            st.markdown("<div class='panel'><b>Image resized for prediction</b></div>", unsafe_allow_html=True)
            st.image(arr[0].astype("uint8"), use_container_width=True)
        st.markdown("---")
        st.markdown('<div class="section-title">✦ 04 · Letter information</div>', unsafe_allow_html=True)
        x, y, z = st.columns(3)
        with x:
            st.markdown(f"<div class='panel'><span class='muted'>Letter name</span><h3 style='color:#f6d58c'>{html.escape(best['name'])}</h3></div>", unsafe_allow_html=True)
        with y:
            st.markdown(f"<div class='panel'><span class='muted'>English equivalent</span><h3 style='color:#f6d58c'>{html.escape(latin)}</h3></div>", unsafe_allow_html=True)
        with z:
            st.markdown(f"<div class='panel'><span class='muted'>Unicode</span><h3 style='color:#f6d58c'>U+{ord(best['letter']):04X}</h3></div>", unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.markdown("---")
summary_path = BASE / "evaluation" / "evaluation_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
st.markdown('<div class="section-title">✦ 05 · Model information</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Classifier", summary.get("selected_model", "MobileNetV2"))
with m2: st.metric("Input size", "160 × 160")
with m3: st.metric("Letter classes", len(mapping))
with m4:
    acc = summary.get("test_accuracy")
    st.metric("Held-out font accuracy", f"{acc:.2%}" if isinstance(acc, (float, int)) else "Not available")
st.info("Evaluation is based on the split described in the report. Generated-font performance is not the same as handwritten recognition.")
st.markdown("<div style='text-align:center;color:#c7a96d;padding:1rem 0;font-family:Georgia,serif'>✦  بسم الله  ·  Persian Letter Recognition  ·  خط فارسی  ✦</div>", unsafe_allow_html=True)
