"""Superbike telemetry and demonstration risk-indicator dashboard.

Author: Moe Htet Ar Kar (Phoe Cho)

Important: this learning project uses synthetic data and is not a real vehicle
safety system. The displayed risk indicator is not a calibrated crash probability.
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

st.set_page_config(
    page_title="Superbike Intelligence",
    page_icon = "🏍️",
    layout = "wide",
)
st.markdown("<style>" + """.stApp {background:linear-gradient(150deg,#08111d 0%,#101a2a 55%,#09131f 100%);color:#edf4ff;} 
[data-testid='stSidebar']{background:#101b2b;} [data-testid='stMetric']{background:#142439;border:1px solid 
#23445d;padding:14px;border-radius:12px;} .stButton>button{background:#00b8d9;color:#06131c;border:0;border-radius:9px;font-weight:700;} 
h1,h2,h3{color:#eaf8ff;}""" + "</style>", unsafe_allow_html = True)

st.title("🏍️ 1000cc Superbike Intelligence")
st.caption("Telemetry analysis · Random Forest + neural-network demonstration")
st.warning(
    "Portfolio demonstration only. This model uses synthetic data and does not "
    "estimate real-world crash probability or provide safety-critical advice."
)


@st.cache_resource
def train_models():

    """Train both models on reproducible synthetic telemetry."""
    rng = np.random.default_rng(4200)
    tf.keras.utils.set_random_seed(4200)
    sample_count = 1000
    speeds = rng.uniform(50, 300, sample_count)
    throttles = rng.uniform(10, 100, sample_count)
    lean_angles = rng.uniform(5, 55, sample_count)
    features = np.column_stack((speeds, throttles, lean_angles))

    # Create labels from an explicit demonstration rule, not real crash records.
    labels = (
        (features[:, 0] > 180)
        & (features[:, 1] > 75)
        & (features[:, 2] > 35)
    ).astype(int)
    noisy_rows = rng.choice(sample_count, int(sample_count * 0.05), replace = False)
    labels[noisy_rows] = 1 - labels[noisy_rows]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    forest = RandomForestClassifier(n_estimators = 100, random_state = 42)
    forest.fit(scaled_features, labels)

    neural_net = Sequential(
        [
            Input(shape = (3,)),
            Dense(32, activation = "relu"),
            Dropout(0.2),
            Dense(16, activation = "relu"),
            Dense(1, activation = "sigmoid"),
        ]
    )
    neural_net.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = ["accuracy"],
    )
    neural_net.fit(scaled_features, labels, epochs = 20, batch_size = 32, verbose = 0)
    return scaler, forest, neural_net


try:
    scaler, forest_model, neural_model = train_models()
except Exception as exc:
    st.error(f"Could not train the demo models: {exc}")
    st.stop()

st.sidebar.header("Telemetry Controls")
riding_mode = st.sidebar.selectbox("Riding profile", ["Street", "Track", "Rain"])
speed = st.sidebar.slider("Speed (km/h)", 30, 320, 120)
throttle = st.sidebar.slider("Throttle input (%)", 5, 100, 40)
lean_angle = st.sidebar.slider("Lean angle (degrees)", 0, 60, 15)
analyze = st.sidebar.button("Analyze telemetry", use_container_width = True)

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Current telemetry")
    st.metric("Speed", f"{speed} km/h")
    st.metric("Throttle", f"{throttle}%")
    st.metric("Lean angle", f"{lean_angle}°")
    st.caption(f"Selected profile: **{riding_mode}**")
    st.progress(min(speed / 320, 1.0), text = "Speed slider position (not a safety limit)")

with right:

    st.subheader("Ensemble analysis")
    if analyze:
        input_row = np.array([[speed, throttle, lean_angle]], dtype = float)
        scaled_row = scaler.transform(input_row)
        forest_score = float(forest_model.predict_proba(scaled_row)[0, 1])
        neural_score = float(neural_model.predict(scaled_row, verbose = 0)[0, 0])
        indicator = (forest_score + neural_score) / 2 * 100

        # These profile multipliers are simple display heuristics.
        if riding_mode == "Rain":
            indicator = min(100.0, indicator * 1.35)
        elif riding_mode == "Track":
            indicator = max(0.0, indicator * 0.85)

        st.metric("Ensemble risk indicator", f"{indicator:.2f}%")
        if indicator > 65:
            st.error("High indicator band — review the synthetic-model inputs.")
        elif indicator > 35:
            st.warning("Moderate indicator band — demonstration threshold reached.")
        else:
            st.success("Lower indicator band — this is not proof that riding is safe.")

        gear = 1 if speed < 40 else 2 if speed < 80 else 3 if speed < 120 else 4 if speed < 160 else 5 if speed < 200 else 6
        st.info(f"Rule-based gear suggestion: Gear {gear} (illustrative only)")

        if speed > 200 and throttle > 80:
            st.warning("Demo note: high speed and throttle entered together.")
        if lean_angle > 40 and speed > 120:
            st.warning("Demo note: high lean-angle and speed values entered together.")

        with st.expander("Model inspection"):
            st.write(f"Random Forest score: {forest_score * 100:.2f}%")
            st.write(f"Neural-network score: {neural_score * 100:.2f}%")
            st.write(f"Inputs: speed={speed}, throttle={throttle}, lean={lean_angle}")
            st.write("Training source: 1,000 synthetic rows with rule-generated labels.")
    else:
        st.info("Adjust the sidebar controls, then select **Analyze telemetry**.")

st.divider()
st.caption("Educational project · Synthetic data · Not validated for real-world vehicle safety")
