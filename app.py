import streamlit as st
from PIL import Image
import config
from gesture_utils import (
    load_local_teachable_machine_model,
    predict_gesture_from_image,
    normalise_label,
)

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="centered",
    initial_sidebar_state="collapsed",
)

POWER_MAP = {
    "Palm": {"power": "Shield Burst", "emoji": "🛡️", "effect": "Creates a glowing protective force field."},
    "Peace": {"power": "Healing Wave", "emoji": "✨", "effect": "Restores energy with a calm magical pulse."},
    "Pointer": {"power": "Thunder Shot", "emoji": "⚡", "effect": "Launches a sharp bolt of electric energy."},
    "Thumbs Up": {"power": "Phoenix Boost", "emoji": "🔥", "effect": "Triggers a fiery confidence boost and power-up."},
    "No Gesture": {"power": "Idle Mode", "emoji": "🌙", "effect": "No power triggered yet. Try another gesture."},
}

LABEL_MAP = {
    "palm": "Palm",
    "Palm": "Palm",
    "open palm": "Palm",
    "Open Palm": "Palm",
    "peace": "Peace",
    "Peace": "Peace",
    "pointer": "Pointer",
    "Pointer": "Pointer",
    "point": "Pointer",
    "Point": "Pointer",
    "thumbs up": "Thumbs Up",
    "Thumbs Up": "Thumbs Up",
    "thumbsup": "Thumbs Up",
    "Thumbsup": "Thumbs Up",
    "no gesture": "No Gesture",
    "No Gesture": "No Gesture",
}


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(124,58,237,.24), transparent 26%),
                radial-gradient(circle at top right, rgba(249,115,22,.16), transparent 24%),
                linear-gradient(180deg, #0b1020 0%, #10172d 100%);
        }
        .block-container {max-width: 860px; padding-top: 2rem; padding-bottom: 2rem;}
        .hero, .panel, .card, .small-card {
            border: 1px solid rgba(255,255,255,.10);
            background: rgba(16, 23, 45, .72);
            backdrop-filter: blur(10px);
            box-shadow: 0 14px 42px rgba(0,0,0,.22);
        }
        .hero {
            padding: 1.3rem 1.25rem;
            border-radius: 24px;
            margin-bottom: 1rem;
        }
        .hero h1 {font-size: 2rem; margin: 0 0 .35rem 0; line-height: 1.1;}
        .hero p {margin: 0; color: #d7dff7; font-size: 1rem;}
        .panel {
            padding: 1rem;
            border-radius: 24px;
            margin-top: .6rem;
        }
        .pill {
            display: inline-block;
            padding: .35rem .7rem;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(99,102,241,.25), rgba(236,72,153,.25));
            color: #eef2ff;
            font-size: .84rem;
            border: 1px solid rgba(255,255,255,.12);
            margin: .5rem 0 .75rem 0;
        }
        .card {
            border-radius: 22px;
            padding: 1rem;
            min-height: 138px;
        }
        .small-card {
            border-radius: 18px;
            padding: .9rem;
            margin-top: .7rem;
        }
        .meta {
            color: #b9c5ea;
            font-size: .88rem;
            margin-bottom: .35rem;
            text-transform: uppercase;
            letter-spacing: .05em;
        }
        .big {
            font-size: 1.25rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.2;
        }
        .sub {
            color: #dbe4ff;
            font-size: .96rem;
            line-height: 1.5;
            margin-top: .35rem;
        }
        .hint {
            color: #cbd5ff;
            font-size: .92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_model():
    return load_local_teachable_machine_model(
        str(config.MODEL_PATH),
        str(config.LABELS_PATH),
    )


def get_power_profile(label: str) -> dict:
    return POWER_MAP.get(label, {"power": "Arcane Pulse", "emoji": "🔮", "effect": "A mysterious fallback ability was triggered."})


def prediction_panel(image: Image.Image, source: str) -> None:
    st.image(image, caption="Gesture image", use_container_width=True)
    st.markdown(f"<div class='pill'>Input source: {source.title()}</div>", unsafe_allow_html=True)

    try:
        model, labels = get_model()
        pred = predict_gesture_from_image(model, labels, image, label_map=LABEL_MAP)
        profile = get_power_profile(pred["label"])

        st.success(f"Detected gesture: **{pred['label']}**")
        st.progress(float(pred["confidence"]), text=f"Confidence: {pred['confidence']:.1%}")

        left, right = st.columns(2)
        left.markdown(
            f"""
            <div class="card">
                <div class="meta">Recognised Gesture</div>
                <div class="big">{pred['label']}</div>
                <div class="sub">The model matched your image with this gesture class.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        right.markdown(
            f"""
            <div class="card">
                <div class="meta">Mapped Power</div>
                <div class="big">{profile['emoji']} {profile['power']}</div>
                <div class="sub">{profile['effect']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="small-card">
                <div class="meta">Game Action</div>
                <div class="sub"><b>{profile['power']}</b> is now active. Use this gesture in your game logic to trigger
                animations, powers, points, or scene changes.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("See all prediction scores"):
            for item in pred["top_predictions"]:
                st.progress(float(item["confidence"]), text=f"{item['label']} — {item['confidence']:.1%}")

    except Exception as e:
        st.error(
            f"Model error: {e}\n\n"
            "Check that your Teachable Machine files are inside the models folder and that you are using Python 3.10 or 3.11."
        )


def main() -> None:
    render_styles()

    st.markdown(
        f"""
        <div class="hero">
            <h1>{config.APP_ICON} {config.APP_TITLE}</h1>
            <p>{config.APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Supported gestures"):
        st.markdown(
            """
            <div class="hint">
            Train your model with four classes such as <b>palm</b>, <b>thumbs up</b>, <b>pointer</b>, and <b>peace</b>.
            You can change the power names later by editing the <code>POWER_MAP</code> dictionary in <code>app.py</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    tabs = st.tabs(["📷 Webcam Capture", "🖼️ Upload Image"])

    with tabs[0]:
        cam = st.camera_input("Capture a gesture", help=config.CAMERA_HELP)
        if cam:
            prediction_panel(Image.open(cam).convert("RGB"), "webcam")
        else:
            st.info("Take a photo with your webcam to detect a gesture and trigger a power.")

    with tabs[1]:
        up = st.file_uploader("Upload a gesture image", type=["png", "jpg", "jpeg"])
        if up:
            prediction_panel(Image.open(up).convert("RGB"), "upload")
        else:
            st.info("Upload a clear gesture image to test your power mapping app.")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()