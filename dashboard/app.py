from __future__ import annotations

from pathlib import Path

import altair as alt
import streamlit as st
from data_loader import load_dashboard_data

_LEVEL_COLORS = {"LOW": "#28a745", "MODERATE": "#fd7e14", "HIGH": "#dc3545"}

st.set_page_config(page_title="Traffic Object Tracking", layout="wide")
st.title("Traffic Object Tracking")
st.caption("Vehicle count, pedestrian count, and traffic congestion level from a single video.")

output_dir = Path(st.sidebar.text_input("Output directory", value="outputs"))
data = load_dashboard_data(output_dir)

if data.tracks.empty:
    st.warning(
        f"No pipeline outputs found under `{output_dir}`. Run "
        "`python -m traffic_intelligence run --input data/raw/<video>.mp4` first."
    )
    st.stop()

summary = data.summary
traffic_level = summary.get("traffic_level", "N/A")
level_color = _LEVEL_COLORS.get(traffic_level, "#6c757d")
avg_speed = summary.get("average_vehicle_speed_kmh")

columns = st.columns(4)
columns[0].metric("Vehicles", summary.get("total_vehicles", 0))
columns[1].metric("Pedestrians", summary.get("total_pedestrians", 0))
with columns[2]:
    st.markdown("**Traffic level**")
    st.markdown(
        f"<span style='font-size:2rem;font-weight:700;color:{level_color}'>{traffic_level}</span>",
        unsafe_allow_html=True,
    )
columns[3].metric("Avg. vehicle speed", f"{avg_speed:.0f} km/h" if avg_speed is not None else "n/a")

st.subheader("Vehicles by class")
vehicles = data.tracks[data.tracks["class_name"] != "person"]
if not vehicles.empty:
    class_counts = vehicles["class_name"].value_counts().reset_index()
    class_counts.columns = ["class_name", "count"]
    st.altair_chart(
        alt.Chart(class_counts).mark_bar().encode(x="class_name", y="count", tooltip=["class_name", "count"]),
        use_container_width=True,
    )
else:
    st.info("No vehicles detected in this run.")

st.subheader("Vehicle speeds")
if summary.get("speed_estimated") and not vehicles.empty:
    st.caption(
        "Estimated from typical vehicle sizes (no camera calibration for this video), not a "
        "calibrated speed measurement -- treat as approximate."
    )
    speed_data = vehicles.dropna(subset=["avg_speed_kmh"])
    if not speed_data.empty:
        speed_data = speed_data.assign(
            vehicle=lambda df: "#" + df["track_id"].astype(str) + " " + df["class_name"]
        )
        st.altair_chart(
            alt.Chart(speed_data)
            .mark_bar()
            .encode(
                x="vehicle",
                y=alt.Y("avg_speed_kmh", title="avg speed (km/h)"),
                tooltip=["vehicle", "avg_speed_kmh", "max_speed_kmh"],
            ),
            use_container_width=True,
        )
    else:
        st.info("No per-vehicle speed available for this run.")
else:
    st.info("Not enough vehicles were seen to calibrate a speed estimate for this run.")

st.subheader("Annotated video")
if data.annotated_video_path is not None:
    st.video(str(data.annotated_video_path))
else:
    st.info("No annotated video found for this output directory.")

with st.expander("Raw track data"):
    st.dataframe(data.tracks, use_container_width=True)
