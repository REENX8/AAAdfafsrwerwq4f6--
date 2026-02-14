import io
import os
import zipfile
import shutil
import numpy as np
import streamlit as st
import torch
import gdown
import pydicom
from scipy.ndimage import zoom
from PIL import Image
from torchvision.models import resnet18

# =========================
# Model definition
# =========================
class CNNGRUClassifier(torch.nn.Module):
    def __init__(self, cnn_name: str = "resnet18", hidden_size: int = 256, num_classes: int = 1) -> None:
        super().__init__()
        if cnn_name == "resnet18":
            model = resnet18(weights=None)
            self.cnn = torch.nn.Sequential(*(list(model.children())[:-1]))  # (batch, 512, 1, 1)
            cnn_out_channels = 512
        else:
            raise ValueError(f"Unsupported cnn_name: {cnn_name}")
        self.gru = torch.nn.GRU(
            input_size=cnn_out_channels, hidden_size=hidden_size, num_layers=1, batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.sigmoid = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, c, h, w = x.shape
        x = x.view(batch_size * time_steps, c, h, w)
        features = self.cnn(x)
        features = features.view(batch_size, time_steps, -1)
        output, _ = self.gru(features)
        last_hidden = output[:, -1, :]
        logits = self.fc(last_hidden)
        return logits.squeeze(dim=-1)


# =========================
# CONFIG
# =========================
MODEL_PATH = "best_bowel_injury_model.pth"
GDRIVE_FILE_ID = "1-awchgMTBa9Ra7jYzlKzccN8MKeUvOs_"

DEFAULT_NUM_STEPS = 32
DEFAULT_NUM_SLICES_PER_STEP = 3
TARGET_SHAPE = (96, 256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DICOM_SLICES = None


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    tmp_path = f"/tmp/{MODEL_PATH}"

    with st.spinner("กำลังดาวน์โหลดโมเดลจาก Google Drive..."):
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        gdown.download(url, tmp_path, quiet=False, fuzzy=True)

    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1024 * 100:
        raise RuntimeError("ดาวน์โหลดโมเดลไม่สำเร็จ / ไฟล์เล็กผิดปกติ (อาจเป็น permission/HTML)")

    with open(tmp_path, "rb") as f:
        head = f.read(200).lower()
    if b"<html" in head or b"google drive" in head:
        raise RuntimeError("ดาวน์โหลดได้ไฟล์ HTML แทนโมเดล → เช็คว่าไฟล์ Drive เป็น Anyone with the link")

    shutil.move(tmp_path, MODEL_PATH)


# =========================
# CSS (FIX ALL FADED TEXT)
# =========================
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* App background + global text */
[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at top left, #1f2937 0%, #0f172a 100%) !important;
  font-family: 'Poppins', sans-serif !important;
  color: #e5e7eb !important;
}

/* Force default text to be readable everywhere */
html, body, [data-testid="stAppViewContainer"] * {
  color: #e5e7eb;
  opacity: 1 !important;
  filter: none !important;
  text-shadow: none !important;
}

/* Headings gradient */
h1, h2, h3 {
  background: -webkit-linear-gradient(315deg, #5eead4 0%, #22d3ee 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700 !important;
  margin-bottom: 0.5rem !important;
}

/* Buttons */
button {
  background-image: linear-gradient(145deg, #06b6d4, #3b82f6) !important;
  color: #ffffff !important;
  border-radius: 30px !important;
  padding: 14px 28px !important;
  border: none !important;
  font-weight: 600 !important;
  box-shadow: 0 10px 20px rgba(3, 169, 244, 0.25), 0 6px 6px rgba(0, 0, 0, 0.1) !important;
  transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}
button:hover {
  transform: translateY(-3px) scale(1.02) !important;
  box-shadow: 0 14px 24px rgba(3, 169, 244, 0.35), 0 8px 8px rgba(0, 0, 0, 0.1) !important;
}

/* Cards / containers */
.stMetric, div.element-container, .stExpander {
  background: rgba(255, 255, 255, 0.05) !important;
  backdrop-filter: blur(10px) !important;
  border-radius: 20px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background-color: rgba(17, 24, 39, 0.92) !important;
  border-right: 1px solid rgba(255,255,255,0.12) !important;
}

/* Force sidebar content readable (fix faded labels) */
section[data-testid="stSidebar"] * {
  color: #e5e7eb !important;
  opacity: 1 !important;
  filter: none !important;
}

/* Inputs / sliders / radios / selectboxes text */
.stRadio *, .stSelectbox *, .stNumberInput *, .stSlider *, .stTextInput *, .stTextArea * {
  color: #e5e7eb !important;
  opacity: 1 !important;
}

/* Radio/Select containers */
.stRadio > div, .stSelectbox > div > div {
  background-color: rgba(255,255,255,0.06) !important;
  border-radius: 12px !important;
  padding: 12px !important;
}

/* ===== File Uploader: make user view crisp ===== */
.stFileUploader {
  border: 2px dashed #38bdf8 !important;
  border-radius: 20px !important;
  padding: 30px !important;
  background-color: rgba(30, 41, 59, 0.92) !important;
}

/* Dropzone itself */
[data-testid="stFileUploaderDropzone"] {
  background-color: rgba(30, 41, 59, 0.94) !important;
  border-radius: 20px !important;
  opacity: 1 !important;
}

/* EVERYTHING inside uploader must be visible (fix faded "อัปโหลด..." / "Drag and drop...") */
.stFileUploader *,
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploaderDropzone"] path,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] label,
[data-testid="stFileUploaderDropzone"] div {
  color: #e5e7eb !important;
  opacity: 1 !important;
  filter: none !important;
}

/* Hover */
.stFileUploader:hover {
  border-color: #7dd3fc !important;
}

/* Images */
.stImage {
  border-radius: 20px !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35) !important;
  transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}
.stImage:hover {
  transform: scale(1.02) !important;
  box-shadow: 0 14px 40px rgba(0,0,0,0.4) !important;
}

/* Tabs text */
.stTabs * {
  opacity: 1 !important;
  color: #e5e7eb !important;
}

/* Captions / markdown text */
.stCaption, .stMarkdown, .stMarkdown * {
  opacity: 1 !important;
}

/* Alerts: keep your bold black text (optional) */
.stAlert p { color: #000000 !important; font-weight: 700 !important; }
</style>
"""


# =========================
# Processing functions
# =========================
def load_dicom_series_from_bytes(dicom_bytes_list: list[bytes]) -> np.ndarray:
    dcm_list = []
    for b in dicom_bytes_list:
        try:
            dcm = pydicom.dcmread(io.BytesIO(b), force=True)
            dcm_list.append(dcm)
        except Exception as e:
            st.warning(f"Skipping invalid DICOM: {e}")
    if not dcm_list:
        raise ValueError("No valid DICOM files")

    def inst(d): return int(getattr(d, "InstanceNumber", 0))
    dcm_list.sort(key=inst)

    vol = np.stack([d.pixel_array.astype(np.int16) for d in dcm_list], axis=0).astype(np.float32)
    slope = float(getattr(dcm_list[0], "RescaleSlope", 1.0))
    inter = float(getattr(dcm_list[0], "RescaleIntercept", 0.0))
    return vol * slope + inter


def window01(hu: np.ndarray, wl=-175, ww=425) -> np.ndarray:
    lo, hi = wl - ww / 2, wl + ww / 2
    return np.clip((hu - lo) / (hi - lo), 0, 1)


def body_bbox(x01: np.ndarray, thr=0.05):
    m = x01 > thr
    if not m.any():
        return 0, x01.shape[0], 0, x01.shape[1], 0, x01.shape[2]
    zz = np.where(m.any(axis=(1, 2)))[0]
    yy = np.where(m.any(axis=(0, 2)))[0]
    xx = np.where(m.any(axis=(0, 1)))[0]
    return zz.min(), zz.max() + 1, yy.min(), yy.max() + 1, xx.min(), xx.max() + 1


def crop_resize_to_target(x01: np.ndarray, target=TARGET_SHAPE) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = body_bbox(x01)
    x = x01[z0:z1, y0:y1, x0:x1]
    tz, th, tw = target
    return zoom(
        x,
        (tz / max(x.shape[0], 1), th / max(x.shape[1], 1), tw / max(x.shape[2], 1)),
        order=1,
    ).astype(np.float16)


def volume_to_sequence(
    vol01: np.ndarray, num_steps: int = DEFAULT_NUM_STEPS, num_slices_per_step: int = DEFAULT_NUM_SLICES_PER_STEP
) -> torch.Tensor:
    z, h, w = vol01.shape
    centers = np.linspace(0, z - 1, num_steps, dtype=int)
    half = num_slices_per_step // 2
    frames: list[np.ndarray] = []

    for c in centers:
        start = max(c - half, 0)
        end = min(c + half + 1, z)
        slc = vol01[start:end]

        if slc.shape[0] < num_slices_per_step:
            pad_pre = half - (c - start)
            pad_post = (c + half) - (end - 1)
            pre = np.repeat(vol01[[start]], max(pad_pre, 0), axis=0) if pad_pre > 0 else np.empty((0, h, w))
            post = np.repeat(vol01[[end - 1]], max(pad_post, 0), axis=0) if pad_post > 0 else np.empty((0, h, w))
            slc = np.concatenate([pre, slc, post], axis=0)
            if slc.shape[0] > num_slices_per_step:
                slc = slc[:num_slices_per_step]
        else:
            slc = slc[:num_slices_per_step]

        if num_slices_per_step == 3:
            slc3 = slc
        elif num_slices_per_step == 1:
            slc3 = np.repeat(slc, 3, axis=0)
        else:
            idx = np.linspace(0, slc.shape[0] - 1, 3).round().astype(int)
            slc3 = slc[idx]

        frames.append(slc3)

    seq = np.stack(frames, axis=0).astype(np.float32)  # (T,3,H,W)
    return torch.from_numpy(seq).unsqueeze(0)  # (1,T,3,H,W)


def image_to_demo_sequence(
    img: Image.Image, num_steps: int = DEFAULT_NUM_STEPS, num_slices_per_step: int = DEFAULT_NUM_SLICES_PER_STEP
) -> torch.Tensor:
    g = img.convert("L").resize((256, 256))
    arr = np.array(g, dtype=np.float32)
    arr = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
    frame = np.stack([arr] * 3, axis=0)
    seq = np.stack([frame] * num_steps, axis=0).astype(np.float32)
    return torch.from_numpy(seq).unsqueeze(0)


@st.cache_resource
def load_model():
    ensure_model_downloaded()
    model = CNNGRUClassifier(cnn_name="resnet18", hidden_size=256).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_prob_from_seq(model, seq: torch.Tensor) -> float:
    seq = seq.to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(seq)).item()
    return float(prob)


def risk_bucket(prob: float, thresh: float):
    if prob >= thresh:
        return "HIGH RISK", "#ff0000"
    if prob >= 0.5:
        return "MEDIUM RISK", "#ffa500"
    return "LOW RISK", "#00ff00"


# =========================
# Saliency
# =========================
def compute_saliency_map(model: torch.nn.Module, seq: torch.Tensor) -> np.ndarray:
    seq = seq.to(DEVICE).clone().detach().requires_grad_(True)
    model.zero_grad()
    prob = torch.sigmoid(model(seq))
    prob.backward()

    sal = seq.grad.detach().abs().cpu().numpy()[0]  # (T,3,H,W)
    sal_per_slice = sal.sum(axis=1)  # (T,H,W)
    sal_max = sal_per_slice.max() if sal_per_slice.max() != 0 else 1.0
    return sal_per_slice / (sal_max + 1e-8)


def describe_saliency_location(saliency_map: np.ndarray, percentile: float = 99.0, border: int = 10) -> str:
    if saliency_map.size == 0:
        return "ไม่สามารถระบุตำแหน่งได้"

    agg = saliency_map.mean(axis=0)  # (H,W)
    if not np.isfinite(agg).all() or agg.max() <= 0:
        return "ไม่สามารถระบุตำแหน่งได้"

    H, W = agg.shape
    agg2 = agg.copy()
    if border > 0 and (H > 2 * border) and (W > 2 * border):
        agg2[:border, :] = 0
        agg2[-border:, :] = 0
        agg2[:, :border] = 0
        agg2[:, -border:] = 0

    if agg2.max() <= 0:
        return "ไม่สามารถระบุตำแหน่งได้"

    thr = np.percentile(agg2[agg2 > 0], percentile) if (agg2 > 0).any() else agg2.max()
    mask = agg2 >= thr
    if not mask.any():
        return "ไม่สามารถระบุตำแหน่งได้"

    weights = agg2[mask]
    coords = np.argwhere(mask)
    row_mean = (coords[:, 0] * weights).sum() / (weights.sum() + 1e-8)
    col_mean = (coords[:, 1] * weights).sum() / (weights.sum() + 1e-8)

    if row_mean < H / 3:
        v_pos = "ด้านบน"
    elif row_mean < 2 * H / 3:
        v_pos = "ตรงกลาง"
    else:
        v_pos = "ด้านล่าง"

    if col_mean < W / 3:
        h_pos = "ซ้าย"
    elif col_mean < 2 * W / 3:
        h_pos = "กลาง"
    else:
        h_pos = "ขวา"

    return f"บริเวณ {v_pos}-{h_pos}"


def vis_boost(sal: np.ndarray, p_low: float = 70.0, p_high: float = 99.7, gamma: float = 0.4) -> np.ndarray:
    sal = np.maximum(sal, 0.0)
    if not np.isfinite(sal).all():
        sal = np.nan_to_num(sal, nan=0.0, posinf=0.0, neginf=0.0)
    if sal.max() <= 1e-8:
        return np.zeros_like(sal, dtype=np.float32)

    lo = np.percentile(sal, p_low)
    hi = np.percentile(sal, p_high)
    if hi <= lo + 1e-8:
        hi = float(sal.max())

    sal = np.clip(sal, lo, hi)
    sal = (sal - lo) / (hi - lo + 1e-8)
    sal = sal ** gamma
    return np.clip(sal, 0.0, 1.0).astype(np.float32)


def show_saliency_block(model: torch.nn.Module, seq: torch.Tensor, volume01: np.ndarray | None) -> None:
    st.subheader("🔍 Saliency / Localization (Explainability)")
    saliency = compute_saliency_map(model, seq)
    T, H, W = saliency.shape
    t_mid = T // 2

    st.caption("Debug: ถ้ามืดมาก แปลว่า saliency อ่อน/กระจาย (ปกติของ gradient-based)")
    st.write("saliency step", t_mid, "min/max:", float(saliency[t_mid].min()), float(saliency[t_mid].max()))

    colA, colB = st.columns(2)
    with colA:
        st.image(saliency[t_mid], caption=f"Raw saliency (step {t_mid})", clamp=True, width="stretch")
    with colB:
        sal_boost = vis_boost(saliency[t_mid])
        st.image(sal_boost, caption=f"Boosted saliency (step {t_mid})", clamp=True, width="stretch")

    if isinstance(volume01, np.ndarray) and volume01.ndim == 3:
        z = volume01.shape[0]
        z_idx = int(np.clip(round((t_mid / max(T - 1, 1)) * (z - 1)), 0, z - 1))
        base = np.clip(volume01[z_idx].astype(np.float32), 0.0, 1.0)
        rgb = np.stack([base, base, base], axis=-1)
        alpha = 0.6
        rgb[..., 0] = np.clip((1 - alpha) * rgb[..., 0] + alpha * sal_boost, 0.0, 1.0)
        st.image(rgb, caption=f"CT + saliency overlay (z={z_idx}, step {t_mid})", width="stretch")

        try:
            buf = io.BytesIO()
            Image.fromarray((rgb * 255).astype(np.uint8)).save(buf, format="PNG")
            st.download_button(
                label="ดาวน์โหลดภาพ overlay",
                data=buf.getvalue(),
                file_name=f"saliency_overlay_z{z_idx}_step{t_mid}.png",
                mime="image/png",
            )
        except Exception:
            pass

    loc_desc = describe_saliency_location(saliency)
    st.info(f"โมเดลให้ความสนใจกับ **{loc_desc}**")


# =========================
# UI
# =========================
st.set_page_config(page_title="CT Bowel Injury Demo", layout="wide", page_icon="🩺")
st.markdown(custom_css, unsafe_allow_html=True)

st.title("🩺 CT Bowel Injury Detection Demo")
st.markdown(
    "<div style='font-size:1.05rem; color:#e5e7eb; margin-bottom:1.5rem; opacity:1;'>"
    "ต้นแบบเพื่อการศึกษาและวิจัย: ใช้โมเดล AI ช่วยประเมินความเสี่ยงการบาดเจ็บลำไส้จากภาพ CT"
    "</div>",
    unsafe_allow_html=True,
)
st.error("⚠️ โปรเจกต์นี้เพื่อการศึกษาเท่านั้น")

with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    threshold = st.slider("เกณฑ์ทำนาย HIGH RISK", 0.5, 0.9, 0.7, 0.05)
    num_steps_input = st.number_input(
        "จำนวนสเต็ป (num_steps)", min_value=8, max_value=96, value=DEFAULT_NUM_STEPS, step=2
    )
    num_slices_input = st.radio(
        "สไลซ์ต่อสเต็ป", options=[1, 3, 5], index=[1, 3, 5].index(DEFAULT_NUM_SLICES_PER_STEP)
    )
    st.caption(f"Device: **{DEVICE.upper()}**")
    page = st.radio("เมนู", ["🖼️ Gallery", "🧠 AI Prediction (Real)", "🖼️ Demo (Single Image)"])

model = load_model()

if page == "🖼️ Gallery":
    st.header("🖼️ Gallery")
    imgs = st.file_uploader("อัปโหลดภาพ JPG/PNG (หลายไฟล์ได้)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if imgs:
        cols = st.columns(3)
        for i, f in enumerate(imgs[:12]):
            cols[i%3].image(f, width="stretch")

elif page == "🧠 AI Prediction (Real)":
    st.header("🧠 AI Prediction — ใช้โมเดลจริง")
    tab1, tab2 = st.tabs(["📁 .npy (แนะนำ)", "🗜️ ZIP DICOM"])
    progress = st.progress(0)

    with tab1:
        up = st.file_uploader("อัปโหลด .npy (96×256×256)", type="npy")
        if up:
            vol = np.load(up)
            vol01 = np.clip(vol.astype(np.float32), 0, 1)
            mid = vol01.shape[0] // 2
            st.image(vol01[mid], "Preview Slice", clamp=True)
            progress.progress(50)

            seq = volume_to_sequence(vol01, num_steps=int(num_steps_input), num_slices_per_step=int(num_slices_input))
            prob = predict_prob_from_seq(model, seq)
            progress.progress(100)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probability", f"{prob:.4f}")
            with col2:
                label, bg_color = risk_bucket(prob, threshold)
                st.markdown(
                    f'<div style="background-color:{bg_color}; border-radius:15px; padding:20px; '
                    f'color:#000; font-weight:800; text-align:center;">Prediction: {label}</div>',
                    unsafe_allow_html=True,
                )

            show_heatmap = st.checkbox("แสดงแผนที่ความสนใจ (saliency map)", key="saliency_npy")
            if show_heatmap:
                show_saliency_block(model, seq, volume01=vol01)

    with tab2:
        up = st.file_uploader("อัปโหลด ZIP ที่มี DICOM slices", type="zip")
        if up:
            with zipfile.ZipFile(up) as zf:
                bytes_list = [zf.read(n) for n in zf.namelist() if n.lower().endswith(".dcm")]
            if not bytes_list:
                st.error("ไม่พบไฟล์ .dcm")
                st.stop()

            progress.progress(20)
            hu = load_dicom_series_from_bytes(bytes_list)
            progress.progress(50)
            vol01 = window01(hu)
            vol = crop_resize_to_target(vol01)
            progress.progress(80)

            mid = vol.shape[0] // 2
            st.image(vol[mid], "Processed Preview", clamp=True)

            seq = volume_to_sequence(vol, num_steps=int(num_steps_input), num_slices_per_step=int(num_slices_input))
            prob = predict_prob_from_seq(model, seq)
            progress.progress(100)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probability", f"{prob:.4f}")
            with col2:
                label, bg_color = risk_bucket(prob, threshold)
                st.markdown(
                    f'<div style="background-color:{bg_color}; border-radius:15px; padding:20px; '
                    f'color:#000; font-weight:800; text-align:center;">Prediction: {label}</div>',
                    unsafe_allow_html=True,
                )

            show_heatmap = st.checkbox("แสดงแผนที่ความสนใจ (saliency map)", key="saliency_zip")
            if show_heatmap:
                show_saliency_block(model, seq, volume01=vol)

else:
    st.header("🖼️ Demo Mode (JPEG/PNG)")
    st.warning("⚠️ เดโมเท่านั้น — ใช้ภาพเดียว ไม่แม่นยำ")
    up = st.file_uploader("อัปโหลดภาพ", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up)
        st.image(img, width="stretch")
        seq = image_to_demo_sequence(img, num_steps=int(num_steps_input), num_slices_per_step=int(num_slices_input))
        prob = predict_prob_from_seq(model, seq)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Demo Probability", f"{prob:.4f}")
        with col2:
            label, bg_color = risk_bucket(prob, threshold)
            st.markdown(
                f'<div style="background-color:{bg_color}; border-radius:15px; padding:20px; '
                f'color:#000; font-weight:800; text-align:center;">Demo: {label}</div>',
                unsafe_allow_html=True,
            )

st.markdown("---")
st.caption("Educational prototype • Developed with ❤️ by REEN • Not for medical use")
