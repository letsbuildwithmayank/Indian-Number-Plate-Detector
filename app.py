import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import json
import time
import os

st.set_page_config(
    page_title="Indian Number Plate Detector",
    page_icon="",
    layout="wide"
)

MODEL_PATH = 'plate_model.pt'

# ---------------- Styles ----------------
st.markdown("""
<style>
    /* upar ka khali space hatao */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    header[data-testid="stHeader"] {
        height: 0;
        background: transparent;
    }

    .title-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }
    .title-main {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .title-by {
        font-size: 0.85rem;
        opacity: 0.55;
        white-space: nowrap;
    }
    .title-sub {
        font-size: 0.85rem;
        opacity: 0.65;
        margin-top: 2px;
        margin-bottom: 4px;
    }

    .step {
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #444;
        background: rgba(128,128,128,0.08);
        font-size: 14px;
    }
    .step-active {
        border-left-color: #ffa500;
        background: rgba(255,165,0,0.12);
    }
    .step-done {
        border-left-color: #00c853;
        background: rgba(0,200,83,0.10);
    }
    .step-fail {
        border-left-color: #ff5252;
        background: rgba(255,82,82,0.10);
    }
    .step-title { font-weight: 600; }
    .step-sub { font-size: 12px; opacity: 0.7; margin-top: 2px; }
    .arrow { text-align: center; opacity: 0.35; margin: -4px 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def render_step(placeholder, steps):
    """Pipeline flow chart render karo"""
    html = ""
    for i, (title, sub, state) in enumerate(steps):
        cls = {
            'wait': 'step',
            'active': 'step step-active',
            'done': 'step step-done',
            'fail': 'step step-fail'
        }[state]

        icon = {'wait': '○', 'active': '◐', 'done': '●', 'fail': '✕'}[state]

        html += f"""<div class="{cls}">
            <div class="step-title">{icon} {title}</div>
            <div class="step-sub">{sub}</div>
        </div>"""

        if i < len(steps) - 1:
            html += '<div class="arrow">↓</div>'

    placeholder.markdown(html, unsafe_allow_html=True)


# ---------------- Header ----------------
st.markdown("""
<div class="title-row">
    <div class="title-main">Indian Number Plate Detector</div>
    <div class="title-by">Built with Mayank</div>
</div>
<div class="title-sub">
    YOLOv8m fine-tuned on 1,858 Indian vehicle images ·
    mAP50 <b>0.967</b> · Precision 0.964 · Recall 0.935
</div>
""", unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found in the project folder.")
    st.stop()

model = load_model()

st.divider()

left, right = st.columns([3, 2], gap="large")

# ================= RIGHT: Pipeline =================
with right:
    st.subheader("Pipeline")
    flow_box = st.empty()
    st.markdown("")
    stats_box = st.empty()

    STEPS_INIT = [
        ("Image loaded", "waiting for upload", 'wait'),
        ("Preprocess", "resize + normalise", 'wait'),
        ("YOLO inference", "single-class detector", 'wait'),
        ("Filter boxes", "confidence + aspect ratio", 'wait'),
        ("Crop plates", "extract regions", 'wait'),
    ]
    render_step(flow_box, STEPS_INIT)

# ================= LEFT: Input =================
with left:
    st.subheader("Input")

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader(
            "Upload a vehicle photo",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
    with c2:
        conf = st.slider("Confidence", 0.10, 0.90, 0.30, 0.05)

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    W, H = image.size

    steps = [
        ("Image loaded", f"{W} × {H} px", 'done'),
        ("Preprocess", "resize + normalise", 'active'),
        ("YOLO inference", "single-class detector", 'wait'),
        ("Filter boxes", "confidence + aspect ratio", 'wait'),
        ("Crop plates", "extract regions", 'wait'),
    ]
    render_step(flow_box, steps)
    time.sleep(0.3)

    # --- inference ---
    steps[1] = ("Preprocess", "640 × 640 letterbox", 'done')
    steps[2] = ("YOLO inference", "running...", 'active')
    render_step(flow_box, steps)

    t0 = time.time()
    results = model(np.array(image), conf=conf, verbose=False)[0]
    elapsed = time.time() - t0

    boxes = results.boxes
    steps[2] = ("YOLO inference",
                f"{elapsed*1000:.0f} ms · {len(boxes)} raw box(es)", 'done')

    # --- filter ---
    steps[3] = ("Filter boxes", f"conf ≥ {conf:.2f}", 'active')
    render_step(flow_box, steps)
    time.sleep(0.2)

    kept = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        ratio = w / h
        if not (1.2 <= ratio <= 7.0):      # plate jaisi shape nahi
            continue
        kept.append({
            'box': (x1, y1, x2, y2),
            'conf': float(box.conf[0]),
            'w': w, 'h': h,
            'ratio': round(ratio, 2)
        })

    kept.sort(key=lambda k: -k['conf'])
    dropped = len(boxes) - len(kept)

    steps[3] = ("Filter boxes", f"{len(kept)} kept · {dropped} dropped", 'done')

    if not kept:
        steps[4] = ("Crop plates", "nothing to crop", 'fail')
    else:
        steps[4] = ("Crop plates", f"{len(kept)} region(s)", 'done')
    render_step(flow_box, steps)

    stats_box.markdown(f"""
**Inference**

Time · `{elapsed*1000:.0f} ms`  
Raw detections · `{len(boxes)}`  
After filtering · `{len(kept)}`  
Image size · `{W} × {H}`
    """)

    # ---------------- Results ----------------
    with left:
        if not kept:
            st.warning("No number plate detected. Try lowering the confidence threshold.")
            st.image(image, use_container_width=True)
        else:
            drawn = image.copy()
            draw = ImageDraw.Draw(drawn)
            line_w = max(2, int(min(W, H) * 0.005))

            for i, k in enumerate(kept):
                x1, y1, x2, y2 = k['box']
                draw.rectangle([x1, y1, x2, y2], outline=(0, 230, 80), width=line_w)
                draw.text((x1 + 4, max(y1 - 16, 2)),
                          f"#{i+1}  {k['conf']:.2f}", fill=(0, 230, 80))

            st.image(drawn, caption="Detection result", use_container_width=True)

            st.subheader("Cropped plates")
            cols = st.columns(min(len(kept), 3))
            for i, k in enumerate(kept):
                x1, y1, x2, y2 = k['box']
                with cols[i % len(cols)]:
                    st.image(image.crop((x1, y1, x2, y2)),
                             caption=f"#{i+1} · conf {k['conf']:.2f} · "
                                     f"{k['w']}×{k['h']} · ratio {k['ratio']}",
                             use_container_width=True)

            # ---------------- Model output ----------------
            st.subheader("Model output")

            output = {
                "image": {"width": W, "height": H},
                "inference_ms": round(elapsed * 1000, 1),
                "confidence_threshold": conf,
                "detections": [
                    {
                        "id": i + 1,
                        "bbox_xyxy": list(k['box']),
                        "confidence": round(k['conf'], 4),
                        "width": k['w'],
                        "height": k['h'],
                        "aspect_ratio": k['ratio']
                    }
                    for i, k in enumerate(kept)
                ]
            }

            tab1, tab2 = st.tabs(["JSON", "Table"])

            with tab1:
                st.code(json.dumps(output, indent=2), language='json')

            with tab2:
                st.dataframe(
                    [
                        {
                            "#": i + 1,
                            "x1": k['box'][0], "y1": k['box'][1],
                            "x2": k['box'][2], "y2": k['box'][3],
                            "confidence": round(k['conf'], 3),
                            "w×h": f"{k['w']}×{k['h']}",
                            "ratio": k['ratio']
                        }
                        for i, k in enumerate(kept)
                    ],
                    use_container_width=True,
                    hide_index=True
                )
else:
    with left:
        st.info("Upload an image to get started.")

st.divider()
st.caption("Built with Mayank")