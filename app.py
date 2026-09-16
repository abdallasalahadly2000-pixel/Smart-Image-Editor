import streamlit as st
import numpy as np
import pandas as pd
import cv2 as cv

from PIL import Image

try:
    from cvzone.SelfiSegmentationModule import SelfiSegmentation
    CVZONE_OK = True
except Exception:
    CVZONE_OK = False

from streamlit_drawable_canvas import st_canvas


# ============================================================
#  Page config
# ============================================================
st.set_page_config(
    page_title="    Smart-Image-Editor",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================ canvas
#  Design system — "Carbon" dark studio theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, [class*="css"], .stMarkdown, p, span, label, h1, h2, h3 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; max-width: 1500px; }

/* ---------- App background ---------- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1200px 500px at 85% -10%, rgba(34,211,238,.06), transparent 60%),
        radial-gradient(900px 400px at -10% 110%, rgba(34,211,238,.05), transparent 60%),
        #0b0e14;
}
[data-testid="stHeader"] { background: transparent; }

/* ---------- Header ---------- */
.pf-header {
    display: flex; align-items: center; gap: 18px;
    background: linear-gradient(180deg, #12161f, #0e1219);
    border: 1px solid #1f2633;
    border-left: 4px solid #22d3ee;
    border-radius: 14px;
    padding: 20px 26px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,.45);
}
.pf-logo {
    width: 52px; height: 52px; border-radius: 12px; flex-shrink: 0;
    background: #0b0e14; border: 1px solid #22d3ee33;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    box-shadow: inset 0 0 18px rgba(34,211,238,.12);
}
.pf-title { color: #e6edf3; font-size: 1.5rem; font-weight: 800; margin: 0; letter-spacing: -.5px; }
.pf-sub { color: #8b98a9; font-size: .86rem; margin: 2px 0 0 0; }
.pf-chip {
    margin-left: auto; text-align: right;
    font-family: 'JetBrains Mono', monospace;
    color: #22d3ee; font-size: .72rem; font-weight: 600;
    letter-spacing: .14em; text-transform: uppercase;
    border: 1px solid #22d3ee33; border-radius: 8px;
    padding: 8px 12px; background: rgba(34,211,238,.05);
}

/* ---------- Section titles ---------- */
.pf-section {
    font-size: .95rem; font-weight: 700; color: #e6edf3;
    margin: 4px 0 12px 0; display: flex; align-items: center; gap: 8px;
    text-transform: uppercase; letter-spacing: .1em; font-size: .8rem;
}
.pf-section::before {
    content: ""; width: 3px; height: 14px; border-radius: 2px;
    background: #22d3ee; box-shadow: 0 0 8px rgba(34,211,238,.6);
}
.pf-sub2 {
    font-size: .72rem; font-weight: 700; color: #22d3ee;
    text-transform: uppercase; letter-spacing: .12em;
    margin: 16px 0 8px 0;
}
.pf-hint { color: #8b98a9; font-size: .8rem; margin: -6px 0 12px 0; }

/* ---------- Info card ---------- */
.pf-card {
    background: #12161f; border: 1px solid #1f2633;
    border-radius: 12px; padding: 14px 16px;
}
.pf-card h4 {
    margin: 0 0 10px 0; font-size: .72rem; font-weight: 700;
    color: #22d3ee; text-transform: uppercase; letter-spacing: .12em;
}
.pf-stat { display: flex; justify-content: space-between; padding: 5px 0;
    border-bottom: 1px dashed #1f2633; font-size: .82rem; }
.pf-stat:last-child { border-bottom: none; }
.pf-stat .k { color: #8b98a9; }
.pf-stat .v { color: #e6edf3; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: .76rem; }

/* ---------- Widgets dark styling ---------- */
section[data-testid="stSidebar"] {
    background: #0e1219; border-right: 1px solid #1f2633;
}
section[data-testid="stSidebar"] * { color: #c9d4e0 !important; }
section[data-testid="stSidebar"] .pf-side-title {
    color: #e6edf3 !important; font-weight: 700; font-size: .95rem;
    display: flex; align-items: center; gap: 8px; margin: 6px 0 2px 0;
}
section[data-testid="stSidebar"] .pf-side-sub { color: #8b98a9 !important; font-size: .78rem; margin-bottom: 12px; }

.stTabs [data-baseweb="tab-list"] {
    gap: 2px; flex-wrap: wrap;
    background: #12161f; border: 1px solid #1f2633;
    border-radius: 10px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; font-weight: 600; font-size: .8rem;
    padding: 7px 12px; color: #8b98a9 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(34,211,238,.12) !important; color: #22d3ee !important;
}

div[data-testid="stExpander"] {
    border: 1px solid #1f2633 !important; border-radius: 12px !important;
    background: #12161f;
}

div[data-testid="stButton"] > button {
    width: 100%; border-radius: 9px !important;
    font-weight: 600 !important; font-size: .82rem !important;
    padding: .45rem .7rem !important;
    border: 1px solid #26303f !important;
    background: #161c26 !important; color: #c9d4e0 !important;
    transition: all .15s ease !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #22d3ee66 !important; color: #22d3ee !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(34,211,238,.15) !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #0891b2, #22d3ee) !important;
    color: #04121a !important; border: none !important; font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(34,211,238,.3) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 6px 22px rgba(34,211,238,.45) !important;
}

/* ---------- Canvas panel ---------- */
.pf-canvas-wrap {
    background: #0e1219; border: 1px solid #1f2633; border-radius: 14px;
    padding: 12px; box-shadow: 0 8px 26px rgba(0,0,0,.4);
}
.pf-canvas-head {
    display: flex; justify-content: space-between; align-items: center;
    padding: 2px 6px 10px 6px;
}
.pf-canvas-title { font-weight: 700; color: #e6edf3; font-size: .9rem; }
.pf-step {
    background: rgba(34,211,238,.1); color: #22d3ee;
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;
    border: 1px solid #22d3ee33;
}

/* ---------- Empty state / footer ---------- */
.pf-empty {
    border: 1px dashed #26303f; border-radius: 14px;
    padding: 80px 30px; text-align: center; color: #8b98a9;
    background: #0e1219;
}
.pf-empty .big { font-size: 1.05rem; font-weight: 600; color: #c9d4e0; margin: 10px 0 4px 0; }
.pf-footer { text-align: center; color: #55607280; font-size: .74rem; padding: 24px 0 6px 0;
    font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  Session state
# ============================================================
def init_state():
    defaults = {
        "original": None,
        "current": None,
        "history": [],
        "history_index": -1,
        "uploaded_name": None,
        "crop_canvas_version": 0,
        "object_canvas_version": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_history(img):
    st.session_state.history = st.session_state.history[:st.session_state.history_index + 1]
    st.session_state.history.append(img.copy())
    st.session_state.history_index += 1


def undo():
    if st.session_state.history_index > 0:
        st.session_state.history_index -= 1
        st.session_state.current = st.session_state.history[st.session_state.history_index].copy()
        return True
    return False


def reset_image():
    if st.session_state.original is not None:
        st.session_state.current = st.session_state.original.copy()
        st.session_state.history = [st.session_state.original.copy()]
        st.session_state.history_index = 0
        return True
    return False


def apply_transform(fn, toast, warn="Operation failed — nothing changed."):
    result = fn(st.session_state.current)
    if result is not None:
        st.session_state.current = result
        add_history(st.session_state.current)
        st.toast(toast, icon="✅")
        return True
    st.warning(warn)
    return False


# ============================================================
#  Safe widgets (prevents out-of-bounds session values
#  after crop/resize shrinks the image)
# ============================================================
def safe_slider(label, lo, hi, default, key, **kw):
    val = st.session_state.get(key)
    if val is not None and not (lo <= val <= hi):
        st.session_state.pop(key, None)
    return st.slider(label, lo, hi, default, key=key, **kw)


def safe_number(label, lo, hi, default, key, **kw):
    if key not in st.session_state:
        st.session_state[key] = default

    val = st.session_state[key]

    if not (lo <= val <= hi):
        st.session_state[key] = default

    return st.number_input(
        label,
        min_value=lo,
        max_value=hi,
        value=st.session_state[key],
        key=key,
        **kw
    )


# ============================================================
#  I/O helpers
# ============================================================
def load_image_file(f):
    if f is not None:
        arr = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is not None:
            return img
    return None


def to_rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


# ============================================================
#  Filters
# ============================================================
def gray_scale(img):
    g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.cvtColor(g, cv.COLOR_GRAY2BGR)


def apply_sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return np.clip(cv.transform(img, kernel), 0, 255).astype(np.uint8)


def apply_blur(img, k=15):
    k = k if k % 2 == 1 else k + 1
    return cv.GaussianBlur(img, (k, k), 0)


def apply_sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv.filter2D(img, -1, kernel)


def apply_invert(img):
    return cv.bitwise_not(img)


def apply_emboss(img):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    out = cv.filter2D(img, -1, kernel) + 128
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_cool(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + 8, 0, 179)
    return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)


def apply_warm(img):
    b, g, r = cv.split(img.astype(np.float32))
    r = np.clip(r + 18, 0, 255)
    b = np.clip(b - 12, 0, 255)
    return cv.merge([b, g, r]).astype(np.uint8)


# ============================================================
#  Light
# ============================================================
def adjust_brightness(img, value=0):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = cv.split(hsv)
    v = np.clip(v + value, 0, 255)
    return cv.cvtColor(cv.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)]),
                       cv.COLOR_HSV2BGR)


def adjust_contrast(img, alpha=1.0):
    return cv.convertScaleAbs(img, alpha=alpha, beta=0)


def adjust_gamma(img, gamma=1.0):
    inv = 1.0 / max(gamma, 0.05)
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv.LUT(img, lut)


def auto_fix(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cv.cvtColor(cv.merge([clahe.apply(l), a, b]), cv.COLOR_LAB2BGR)


# ============================================================
#  Transform — rotate / flip / crop / resize / perspective
# ============================================================
def rotate_90_cw(img):
    return cv.rotate(img, cv.ROTATE_90_CLOCKWISE)


def rotate_90_ccw(img):
    return cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)


def rotate_180(img):
    return cv.rotate(img, cv.ROTATE_180)


def rotate_custom(img, angle):
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR,
                         borderMode=cv.BORDER_REFLECT)


def flip_horizontal(img):
    return cv.flip(img, 1)


def flip_vertical(img):
    return cv.flip(img, 0)


def flip_both(img):
    return cv.flip(img, -1)


def crop_image(img, left, top, right, bottom):
    h, w = img.shape[:2]
    left, top = max(0, int(left)), max(0, int(top))
    right, bottom = min(w, int(right)), min(h, int(bottom))
    if right - left < 5 or bottom - top < 5:
        return None
    return img[top:bottom, left:right].copy()


def resize_image(img, new_w, new_h):
    new_w, new_h = int(new_w), int(new_h)
    if new_w < 8 or new_h < 8:
        return None
    interp = cv.INTER_AREA if (new_w < img.shape[1] and new_h < img.shape[0]) else cv.INTER_CUBIC
    return cv.resize(img, (new_w, new_h), interpolation=interp)


def perspective_correct(img, tl, tr, br, bl):
    pts = np.array([tl, tr, br, bl], dtype=np.float32)

    # Convert the four points from tuples to NumPy arrays
    tl, tr, br, bl = pts

    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    max_w, max_h = max(int(w1), int(w2), 8), max(int(h1), int(h2), 8)
    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
                   dtype=np.float32)
    M = cv.getPerspectiveTransform(pts, dst)
    return cv.warpPerspective(img, M, (max_w, max_h))


# ============================================================
#  Color Studio
# ============================================================
def color_studio(img, hue=0, saturation=100, lightness=0,
                 temperature=0, tint=0, vibrance=0,
                 r_curve=(0, 128, 255), g_curve=(0, 128, 255),
                 b_curve=(0, 128, 255)):
    """RGB curves -> temperature/tint -> HSL + vibrance."""
    out = img.astype(np.float32)

    xs = np.array([0, 128, 255], dtype=np.float32)
    identity = (0, 128, 255)
    for ch, curve in enumerate((r_curve, g_curve, b_curve)):
        if tuple(curve) != identity:
            lut = np.interp(np.arange(256), xs, np.asarray(curve, dtype=np.float32))
            out[:, :, ch] = lut[np.clip(out[:, :, ch].astype(np.int32), 0, 255)]

    if temperature != 0:
        out[:, :, 2] += temperature * 0.6
        out[:, :, 0] -= temperature * 0.6
    if tint != 0:
        out[:, :, 1] -= tint * 0.5
    out = np.clip(out, 0, 255).astype(np.uint8)

    hsv = cv.cvtColor(out, cv.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv.split(hsv)

    if hue != 0:
        h = np.mod(h + hue, 180)
    s = s * (saturation / 100.0)
    if vibrance != 0:
        s_norm = s / 255.0
        s = s + vibrance * (1.0 - s_norm) * s_norm * 4.0
    v = v + lightness

    hsv2 = cv.merge((np.mod(h, 180), np.clip(s, 0, 255), np.clip(v, 0, 255)))
    return cv.cvtColor(hsv2.astype(np.uint8), cv.COLOR_HSV2BGR)


def histogram_df(img):
    data = {}
    for ch, name in ((0, "Blue"), (1, "Green"), (2, "Red")):
        data[name] = cv.calcHist([img], [ch], None, [256], [0, 256]).flatten()
    return pd.DataFrame(data)


# ============================================================
#  Object removal (inpainting)
# ============================================================
def remove_object(image, user_mask, inpaint_radius=3, mask_expansion=2):
    if image is None or user_mask is None:
        return None
    if len(user_mask.shape) == 3:
        user_mask = cv.cvtColor(user_mask, cv.COLOR_BGR2GRAY)
    mask = np.where(user_mask > 20, 255, 0).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    if mask_expansion > 0:
        mask = cv.dilate(mask, kernel, iterations=mask_expansion)
    mask = cv.GaussianBlur(mask, (3, 3), 0)
    mask = np.where(mask > 30, 255, 0).astype(np.uint8)
    return cv.inpaint(image, mask, float(inpaint_radius), cv.INPAINT_TELEA)


# ============================================================
#  Background replacement
# ============================================================
@st.cache_resource
def get_segmentor():
    return SelfiSegmentation()


def replace_background(img, mode, color_hex, bg_file):
    if not CVZONE_OK:
        return None
    seg = get_segmentor()
    if mode == "Solid color":
        hx = color_hex.lstrip("#")
        r, g, b = (int(hx[i:i + 2], 16) for i in (0, 2, 4))
        return seg.removeBG(img, (b, g, r), cutThreshold=0.65)
    bg = load_image_file(bg_file)
    if bg is None:
        return None
    bg = cv.resize(bg, (img.shape[1], img.shape[0]))
    return seg.removeBG(img, bg, cutThreshold=0.65)


# ============================================================
#  UI blocks
# ============================================================
def render_header():
    st.markdown("""
    <div class="pf-header">
        <div class="pf-logo">🎛️</div>
        <div>
            <p class="pf-title">PixelForge Studio</p>
            <p class="pf-sub">Professional editing suite — crop &amp; rotate, color grading,
            curves, perspective, retouch &amp; AI background replacement.</p>
        </div>
        <div class="pf-chip">Studio&nbsp;Build<br>v4.0</div>
    </div>
    """, unsafe_allow_html=True)


def render_info_card(img):
    h, w, ch = img.shape
    means = [float(np.mean(img[:, :, c])) for c in range(3)] if ch == 3 else [None] * 3
    st.markdown(f"""
    <div class="pf-card">
        <h4>Image Properties</h4>
        <div class="pf-stat"><span class="k">Width</span><span class="v">{w:,} px</span></div>
        <div class="pf-stat"><span class="k">Height</span><span class="v">{h:,} px</span></div>
        <div class="pf-stat"><span class="k">Channels</span><span class="v">{ch}</span></div>
        <div class="pf-stat"><span class="k">Megapixels</span><span class="v">{w*h/1e6:.2f} MP</span></div>
        <div class="pf-stat"><span class="k">Mean R / G / B</span>
            <span class="v">{means[2]:.0f} · {means[1]:.0f} · {means[0]:.0f}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#  Tool tabs
# ============================================================ 
def tab_crop_rotate():
    img = st.session_state.current
    h, w = img.shape[:2]

    # ---------- Rotate ----------
    st.markdown('<p class="pf-sub2">⟳ Rotate</p>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("⟲ 90° Left", use_container_width=True):
            apply_transform(rotate_90_ccw, "Rotated 90° left.")
    with r2:
        if st.button("⟳ 180°", use_container_width=True):
            apply_transform(rotate_180, "Rotated 180°.")
    with r3:
        if st.button("⟳ 90° Right", use_container_width=True):
            apply_transform(rotate_90_cw, "Rotated 90° right.")

    c1, c2 = st.columns([3, 1])
    with c1:
        angle = st.slider("Custom angle", -180, 180, 0, 5)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Apply", use_container_width=True):
            apply_transform(lambda im: rotate_custom(im, angle), f"Rotated by {angle}°.")

    st.divider()

    # ---------- Crop (drag a rectangle on the canvas, Photoshop-style) ----------
    st.markdown('<p class="pf-sub2">✂️ Crop</p>', unsafe_allow_html=True)
    st.caption("Drag a rectangle on the image below, then apply the crop.")

    max_w = 520
    scale = min(1.0, max_w / w)
    cw, chh = int(w * scale), int(h * scale)
    disp = cv.resize(to_rgb(img), (cw, chh), interpolation=cv.INTER_AREA)
    
    canvas = st_canvas(
        fill_color="rgba(34, 211, 238, 0.12)",
        stroke_width=2,
        stroke_color="rgba(34, 211, 238, 1.0)",
        background_image=Image.fromarray(disp),
        background_color="#000000",
        update_streamlit=True,
        height=chh, width=cw,
        drawing_mode="rect",
        key=f"crop_canvas_{st.session_state.crop_canvas_version}",
    )

    crop_box = None
    if canvas.json_data is not None:
        objs = canvas.json_data.get("objects", [])
        rects = [o for o in objs if o.get("type") == "rect"]
        if rects:
            o = rects[-1]
            left = int(o["left"] / scale)
            top = int(o["top"] / scale)
            right = int((o["left"] + o["width"]) / scale)
            bottom = int((o["top"] + o["height"]) / scale)
            crop_box = (left, top, right, bottom)

    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("🧹 Clear Selection", use_container_width=True):
            st.session_state.crop_canvas_version += 1
            st.rerun()
    with bc2:
        disabled = crop_box is None
        if st.button("✂️ Apply Crop", type="primary", use_container_width=True,
                     disabled=disabled):
            left, top, right, bottom = crop_box
            ok = apply_transform(
                lambda im: crop_image(im, left, top, right, bottom),
                f"Cropped to {max(right-left,0)} × {max(bottom-top,0)} px.",
                "Crop area too small — drag a larger rectangle.",
            )
            if ok:
                st.session_state.crop_canvas_version += 1
                st.rerun()

    if crop_box:
        left, top, right, bottom = crop_box
        st.caption(f"Selection: {max(right-left,0)} × {max(bottom-top,0)} px "
                   f"@ ({left}, {top})")


def tab_resize():
    img = st.session_state.current
    h, w = img.shape[:2]

    st.markdown(
        '<p class="pf-hint">Scale the canvas to exact pixel dimensions.</p>',
        unsafe_allow_html=True
    )

    preset = st.selectbox(
        "Quick preset",
        [
            "Custom",
            "1920 × 1080 (FHD)",
            "1080 × 1080 (Square)",
            "1080 × 1350 (Portrait)",
            "1200 × 628 (Banner)"
        ]
    )

    lock = st.checkbox("Lock aspect ratio", value=True)

    # --------------------------------------------------
    # Apply preset BEFORE creating rz_w / rz_h widgets
    # --------------------------------------------------
    if preset != "Custom":
        pw, ph = {
            "1920 × 1080 (FHD)": (1920, 1080),
            "1080 × 1080 (Square)": (1080, 1080),
            "1080 × 1350 (Portrait)": (1080, 1350),
            "1200 × 628 (Banner)": (1200, 628)
        }[preset]

        st.session_state["rz_w"] = pw

        if not lock:
            st.session_state["rz_h"] = ph

        st.caption(
            f"Preset loaded: {pw} × {ph} px — press Apply Resize."
        )

    # --------------------------------------------------
    # Now create the widgets
    # --------------------------------------------------
    rs1, rs2 = st.columns(2)

    with rs1:
        new_w = safe_number(
            "Width (px)",
            8,
            8000,
            w,
            key="rz_w"
        )

    with rs2:
        if lock:
            new_h = max(8, round(new_w * h / w))

            st.number_input(
                "Height (px) — auto",
                min_value=8,
                max_value=8000,
                value=new_h,
                disabled=True
            )
        else:
            new_h = safe_number(
                "Height (px)",
                8,
                8000,
                h,
                key="rz_h"
            )

    st.caption(f"Result: {int(new_w)} × {int(new_h)} px")

    if st.button(
        "📐 Apply Resize",
        type="primary",
        use_container_width=True
    ):
        apply_transform(
            lambda im: resize_image(im, new_w, new_h),
            f"Resized to {int(new_w)} × {int(new_h)} px.",
            "Size too small — minimum is 8 × 8 px."
        )


def tab_flip():
    st.markdown('<p class="pf-hint">Mirror the image on either axis.</p>',
                unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        if st.button("⬌ Horizontal", use_container_width=True):
            apply_transform(flip_horizontal, "Flipped horizontally.")
    with f2:
        if st.button("⬍ Vertical", use_container_width=True):
            apply_transform(flip_vertical, "Flipped vertically.")
    with f3:
        if st.button("⇄ Both", use_container_width=True):
            apply_transform(flip_both, "Flipped both ways.")


def tab_perspective():
    img = st.session_state.current
    h, w = img.shape[:2]

    st.markdown('<p class="pf-hint">Straighten a tilted plane (document, sign, screen) '
                'by setting its four corners.</p>', unsafe_allow_html=True)

    st.markdown('<p class="pf-sub2">Top edge</p>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        tl_x = safe_number("Top-Left x", 0, w, 0, key="p_tlx")
        tl_y = safe_number("Top-Left y", 0, h, 0, key="p_tly")
    with p2:
        tr_x = safe_number("Top-Right x", 0, w, w, key="p_trx")
        tr_y = safe_number("Top-Right y", 0, h, 0, key="p_try")

    st.markdown('<p class="pf-sub2">Bottom edge</p>', unsafe_allow_html=True)
    p3, p4 = st.columns(2)
    with p3:
        bl_x = safe_number("Bottom-Left x", 0, w, 0, key="p_blx")
        bl_y = safe_number("Bottom-Left y", 0, h, h, key="p_bly")
    with p4:
        br_x = safe_number("Bottom-Right x", 0, w, w, key="p_brx")
        br_y = safe_number("Bottom-Right y", 0, h, h, key="p_bry")

    pc1, pc2 = st.columns(2)

    with pc1:
        st.button(
            "↺ Reset Points",
            use_container_width=True,
            on_click=reset_perspective_points
        )

    with pc2:
        if st.button("🧊 Apply Perspective", type="primary", use_container_width=True):
            pts = ((tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y))
            apply_transform(
                lambda im: perspective_correct(im, *pts),
                "Perspective corrected.",
                "Invalid corner points."
            )
# safe_number
def reset_perspective_points():
    img = st.session_state.current
    h, w = img.shape[:2]

    st.session_state["p_tlx"] = 0
    st.session_state["p_tly"] = 0
    st.session_state["p_trx"] = w
    st.session_state["p_try"] = 0
    st.session_state["p_brx"] = w
    st.session_state["p_bry"] = h
    st.session_state["p_blx"] = 0
    st.session_state["p_bly"] = h
    
def tab_light():
    st.markdown('<p class="pf-hint">Exposure pipeline with live preview — tune, then apply.</p>',
                unsafe_allow_html=True)

    b = st.slider("Brightness", -100, 100, 0, 1)
    c = st.slider("Contrast", -100, 100, 0, 1, help="Negative lowers, positive raises contrast")
    g = st.slider("Gamma", 40, 220, 100, 1, help="Midtone lift (>1.0) or deepen (<1.0)")

    alpha = 1.0 + c / 100.0
    preview = adjust_brightness(adjust_contrast(adjust_gamma(st.session_state.current, g / 100.0), alpha), b)

    st.image(to_rgb(preview), caption="Live preview", use_container_width=True)

    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        if st.button("↺ Reset", use_container_width=True):
            for key in ("Brightness", "Contrast", "Gamma"):
                st.session_state.pop(key, None)
            st.rerun()
    with lc2:
        if st.button("🪄 Auto Fix", use_container_width=True):
            apply_transform(auto_fix, "Auto exposure applied.")
    with lc3:
        if st.button("✨ Apply Light", type="primary", use_container_width=True):
            st.session_state.current = preview
            add_history(st.session_state.current)
            st.toast("Light adjustments applied!", icon="💡")

    st.divider()
    st.markdown('<p class="pf-sub2">Quick steps</p>', unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    with q1:
        if st.button("☀️ +Brightness", use_container_width=True):
            apply_transform(lambda im: adjust_brightness(im, 30), "Brightness increased.")
        if st.button("◐ +Contrast", use_container_width=True):
            apply_transform(lambda im: adjust_contrast(im, 1.4), "Contrast increased.")
    with q2:
        if st.button("🌙 −Brightness", use_container_width=True):
            apply_transform(lambda im: adjust_brightness(im, -30), "Brightness decreased.")
        if st.button("◑ −Contrast", use_container_width=True):
            apply_transform(lambda im: adjust_contrast(im, 0.7), "Contrast decreased.")

def reset_color_studio():
    st.session_state["temperature"] = 0
    st.session_state["tint"] = 0
    st.session_state["hue"] = 0
    st.session_state["saturation"] = 100
    st.session_state["lightness"] = 0
    st.session_state["vibrance"] = 0

    st.session_state["r_s"] = 0
    st.session_state["r_m"] = 128
    st.session_state["r_h"] = 255

    st.session_state["g_s"] = 0
    st.session_state["g_m"] = 128
    st.session_state["g_h"] = 255

    st.session_state["b_s"] = 0
    st.session_state["b_m"] = 128
    st.session_state["b_h"] = 255


def tab_color_studio():
    st.markdown('<p class="pf-hint">Pro color grading — histogram, balance, HSL & RGB curves '
                'with live preview.</p>', unsafe_allow_html=True)

    # ---------------- Histogram ----------------
    with st.expander("📈 Histogram", expanded=True):
        st.bar_chart(histogram_df(st.session_state.current),
                     color=["#3b82f6", "#22c55e", "#ef4444"], height=170)

    # ---------------- Color balance ----------------
    st.markdown('<p class="pf-sub2">🌡️ Color Balance</p>', unsafe_allow_html=True)
    cb1, cb2 = st.columns(2)
    with cb1:
        temperature = st.slider(
            "Temperature", -100, 100, 0, 1,
            key="temperature",
            help="Cool (blue) ← → Warm (orange)"
        )
    with cb2:
        tint = st.slider(
            "Tint", -100, 100, 0, 1,
            key="tint",
            help="Green ← → Magenta"
        )

    # ---------------- HSL ----------------
    st.markdown('<p class="pf-sub2">🎚️ Hue · Saturation · Lightness</p>',
                unsafe_allow_html=True)
    hue = st.slider("Hue shift", -90, 90, 0, 1, key="hue")
    saturation = st.slider("Saturation (%)", 0, 200, 100, 1, key="saturation")
    lightness = st.slider("Lightness", -100, 100, 0, 1, key="lightness")

    st.markdown('<p class="pf-sub2">💎 Vibrance</p>', unsafe_allow_html=True)
    vibrance = st.slider(
            "Vibrance", -100, 100, 0, 1,
            key="vibrance",
            help="Boosts muted colors, protects saturated areas"
        )

    # ---------------- RGB curves ----------------
    with st.expander("📉 RGB Curves", expanded=False):
        st.caption("Shadows / Midtones / Highlights output level per channel.")
        cr1, cr2, cr3 = st.columns(3)
        with cr1:
            st.markdown('<p class="pf-hint" style="color:#ef4444">Red</p>',
                        unsafe_allow_html=True)
            r_s = st.slider("R shadows", 0, 255, 0, key="r_s")
            r_m = st.slider("R midtones", 0, 255, 128, key="r_m")
            r_h = st.slider("R highlights", 0, 255, 255, key="r_h")
        with cr2:
            st.markdown('<p class="pf-hint" style="color:#22c55e">Green</p>',
                        unsafe_allow_html=True)
            g_s = st.slider("G shadows", 0, 255, 0, key="g_s")
            g_m = st.slider("G midtones", 0, 255, 128, key="g_m")
            g_h = st.slider("G highlights", 0, 255, 255, key="g_h")
        with cr3:
            st.markdown('<p class="pf-hint" style="color:#3b82f6">Blue</p>',
                        unsafe_allow_html=True)
            b_s = st.slider("B shadows", 0, 255, 0, key="b_s")
            b_m = st.slider("B midtones", 0, 255, 128, key="b_m")
            b_h = st.slider("B highlights", 0, 255, 255, key="b_h")

    # ---------------- Live preview ----------------
    preview = color_studio(
        st.session_state.current,
        hue=hue, saturation=saturation, lightness=lightness,
        temperature=temperature, tint=tint, vibrance=vibrance,
        r_curve=(r_s, r_m, r_h), g_curve=(g_s, g_m, g_h), b_curve=(b_s, b_m, b_h),
    )
    st.image(to_rgb(preview), caption="Live preview", use_container_width=True)

    a1, a2 = st.columns(2)
    with a1:
        st.button(
            "↺ Reset Studio",
            use_container_width=True,
            on_click=reset_color_studio
        )
    with a2:
        if st.button("✨ Apply Color Grade", type="primary", use_container_width=True):
            st.session_state.current = preview
            add_history(st.session_state.current)
            st.toast("Color grade applied!", icon="🌈")


def tab_filters():
    st.markdown('<p class="pf-hint">One-click artistic looks.</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="pf-sub2">Classic</p>', unsafe_allow_html=True)
    f1, f2= st.columns(2)
    with f1:
        if st.button("⚫ Black & White", use_container_width=True):
            apply_transform(gray_scale, "Black & white applied.")
        if st.button("🟤 Sepia", use_container_width=True):
            apply_transform(apply_sepia, "Sepia applied.")
    with f2:
        if st.button("🌫️ Blur", use_container_width=True):
            apply_transform(lambda im: apply_blur(im, 15), "Blur applied.")
        if st.button("✨ Sharpen", use_container_width=True):
            apply_transform(apply_sharpen, "Sharpen applied.")

    st.markdown('<p class="pf-sub2">Stylized</p>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        if st.button("🔄 Invert", use_container_width=True):
            apply_transform(apply_invert, "Colors inverted.")
        if st.button("🧊 Cool Tone", use_container_width=True):
            apply_transform(apply_cool, "Cool tone applied.")
    with s2:
        if st.button("🗿 Emboss", use_container_width=True):
            apply_transform(apply_emboss, "Emboss applied.")
        if st.button("🔥 Warm Tone", use_container_width=True):
            apply_transform(apply_warm, "Warm tone applied.")


def tab_object_removal():
    st.markdown('<p class="pf-hint">Paint over the unwanted object with the cyan brush, '
                'preview the repair, then apply.</p>', unsafe_allow_html=True)

    img = st.session_state.current
    oh, ow = img.shape[:2]

    max_w = 520
    scale = min(1.0, max_w / ow)
    cw, chh = int(ow * scale), int(oh * scale)
    disp = cv.resize(to_rgb(img), (cw, chh), interpolation=cv.INTER_AREA)

    c1, c2, c3 = st.columns(3)
    with c1:
        brush = st.slider("Brush size", 5, 100, 30, 5)
    with c2:
        expansion = st.slider("Mask expansion", 0, 5, 2, 1)
    with c3:
        radius = st.slider("Repair strength", 1, 8, 3, 1)

    cc, ic = st.columns([1, 2])
    with cc:
        if st.button("🧹 Clear Mask", use_container_width=True):
            st.session_state.object_canvas_version += 1
            st.rerun()
    with ic:
        st.caption("Fully cover the object for best results.")

    canvas = st_canvas(
    fill_color="rgba(34, 211, 238, 0.0)",
    stroke_width=brush,
    return_image_data=True,
    stroke_color="rgba(34, 211, 238, 1.0)",
    background_image=Image.fromarray(disp),
    background_color="#000000",
    update_streamlit=True,
    height=chh,
    width=cw,
    drawing_mode="freedraw",
    key=f"object_canvas_{st.session_state.object_canvas_version}",
)

    if canvas.image_data is None:
        return

    rgba = canvas.image_data
    r, g, b = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2]
    # Detect the cyan brush (image is opaque, so use color, not alpha)
    mask_small = ((b > 170) & (g > 170) & (r < 140)).astype(np.uint8) * 255

    full_mask = cv.resize(mask_small, (ow, oh), interpolation=cv.INTER_NEAREST)
    if np.count_nonzero(full_mask) == 0:
        st.info("Paint over the object you want to remove.")
        return

    preview = remove_object(img, full_mask, radius, expansion)

    st.markdown('<p class="pf-sub2">Preview</p>', unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    with pc1:
        st.image(to_rgb(img), caption="Before", use_container_width=True)
    with pc2:
        st.image(to_rgb(preview), caption="After removal", use_container_width=True)

    if st.button("✨ Apply Object Removal", type="primary", use_container_width=True):
        st.session_state.current = preview
        add_history(st.session_state.current)
        st.session_state.object_canvas_version += 1
        st.toast("Object removed!", icon="✨")
        st.rerun()


def tab_background():
    st.markdown('<p class="pf-hint">AI subject cut-out &amp; background swap.</p>',
                unsafe_allow_html=True)

    if not CVZONE_OK:
        st.warning("Background module unavailable — install `cvzone` and `mediapipe` "
                   "to enable AI cut-out.")
        return

    mode = st.radio("Background source", ["Solid color", "Upload image"],
                    horizontal=True, label_visibility="collapsed")

    bg_color, bg_file = "#0e7490", None
    if mode == "Solid color":
        bg_color = st.color_picker("Background color", "#0e7490")
    else:
        bg_file = st.file_uploader("New background image",
                                   type=["jpg", "jpeg", "png", "webp"])

    if st.button("✂️ Apply Background", type="primary", use_container_width=True):
        if mode == "Upload image" and bg_file is None:
            st.warning("Upload a background image first.")
        else:
            with st.spinner("Cutting out subject…"):
                try:
                    result = replace_background(st.session_state.current, mode, bg_color, bg_file)
                except Exception as e:
                    result = None
                    st.error(f"Segmentation failed: {e}")
            if result is not None:
                st.session_state.current = result
                add_history(st.session_state.current)
                st.toast("Background replaced!", icon="🖼️")


def tab_export():
    st.markdown('<p class="pf-hint">Export the current canvas.</p>',
                unsafe_allow_html=True)

    fmt = st.radio("Format", ["PNG (lossless)", "JPEG (quality)"], horizontal=True)
    img = st.session_state.current

    if fmt.startswith("JPEG"):
        q = st.slider("JPEG quality", 10, 100, 92, 5)
        data = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
        mime, name = "image/jpeg", "pixelforge_output.jpg"
    else:
        data = cv.imencode(".png", img)[1].tobytes()
        mime, name = "image/png", "pixelforge_output.png"

    st.download_button("⬇️ Download Image", data=data, file_name=name,
                       mime=mime, type="primary", use_container_width=True)
    st.caption("Export reflects the latest state of your canvas.")


# ============================================================
#  Main
# ============================================================
def main():
    init_state()
    render_header()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown('<p class="pf-side-title">📥 Import</p>', unsafe_allow_html=True)
        st.markdown('<p class="pf-side-sub">Upload an image to open it in the studio.</p>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop your image here",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            label_visibility="collapsed",
        )

        if uploaded is not None and uploaded.name != st.session_state.uploaded_name:
            img = load_image_file(uploaded)
            if img is not None:
                st.session_state.original = img.copy()
                st.session_state.current = img.copy()
                st.session_state.history = [img.copy()]
                st.session_state.history_index = 0
                st.session_state.uploaded_name = uploaded.name
                st.toast("Image loaded!", icon="🖼️")

        if st.session_state.current is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            render_info_card(st.session_state.current)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="pf-side-title">🕘 History</p>', unsafe_allow_html=True)
            step = st.session_state.history_index + 1
            total = len(st.session_state.history)
            st.markdown(f'<p class="pf-side-sub">Step {step} of {total}</p>',
                        unsafe_allow_html=True)
            hc1, hc2 = st.columns(2)
            with hc1:
                if st.button("↩️ Undo", use_container_width=True):
                    if undo():
                        st.toast("Undone.", icon="↩️")
                    else:
                        st.warning("Nothing to undo.")
            with hc2:
                if st.button("🔄 Reset", use_container_width=True):
                    if reset_image():
                        st.toast("Restored to original.", icon="🔄")

    # ---------------- Workspace ----------------
    if st.session_state.current is None:
        st.markdown("""
        <div class="pf-empty">
            <div style="font-size:44px">🎛️</div>
            <p class="big">Your canvas is empty</p>
            <p>Upload an image from the sidebar to start editing.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<p class="pf-footer">    Smart-Image-Editor · OpenCV · Streamlit</p>',
                    unsafe_allow_html=True)
        return

    col_canvas, col_tools = st.columns([3, 2], gap="large")

    with col_tools:
        st.markdown('<p class="pf-section">Editing Tools</p>', unsafe_allow_html=True)
        (t_crop, t_color, t_light, t_resize, t_flip, t_persp,
         t_filt, t_obj, t_bg, t_exp) = st.tabs([
            "✂️ Crop & Rotate", "🌈 Color Studio", "💡 Light",
            "📐 Resize", "⇄ Flip", "🧊 Perspective",
            "🖌️ Filters", "🪄 Object Removal", "🖼️ Background", "📤 Export",
        ])
        with t_crop:
            tab_crop_rotate()
        with t_color:
            tab_color_studio()
        with t_light:
            tab_light()
        with t_resize:
            tab_resize()
        with t_flip:
            tab_flip()
        with t_persp:
            tab_perspective()
        with t_filt:
            tab_filters()
        with t_obj:
            tab_object_removal()
        with t_bg:
            tab_background()
        with t_exp:
            tab_export()

    with col_canvas:
        step = st.session_state.history_index + 1
        total = len(st.session_state.history)
        st.markdown(f"""
        <div class="pf-canvas-wrap">
            <div class="pf-canvas-head">
                <span class="pf-canvas-title">🎯 Canvas</span>
                <span class="pf-step">STEP {step} / {total}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.image(to_rgb(st.session_state.current), use_container_width=True)

        with st.expander("👁️ View original"):
            st.image(to_rgb(st.session_state.original), use_container_width=True)

    st.markdown('<p class="pf-footer">PixelForge Studio · OpenCV · Streamlit</p>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
