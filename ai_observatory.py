import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix

# Set page config
st.set_page_config(page_title="AI Observatory", layout="wide", page_icon="")

# Custom CSS for futuristic dark theme
st.markdown("""
   <style>
    body {
        background-color: #0e0e2c;
        color: #ffffff;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        color: #00ffe1;
    }

    .card {
        background-color: #1a1a3d;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        text-align: center;
    }

    .stButton>button {
        background-color: #1f2937; /* deep gray/blue */
        color: #e5e7eb; /* soft white */
        border: 1px solid #374151;
        border-radius: 6px;
        padding: 10px 20px;
        font-size: 15px;
        font-weight: 500;
        transition: background-color 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: inset 0 0 0 0 rgba(0,0,0,0);
    }

    .stButton>button:hover {
        background-color: #111827; /* darker on hover */
        color: #ffffff;
        border-color: #4b5563;
        box-shadow: 0 0 0 2px rgba(0,255,225,0.2);
    }

    .stButton>button:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(0,255,225,0.4);
    }
</style>


""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title(" AI Observatory")
start_training = st.sidebar.button(" Start Training Simulation")
reset = st.sidebar.button("Reset Dashboard")

if 'running' not in st.session_state:
    st.session_state.running = False
if start_training:
    st.session_state.running = True
if reset:
    st.session_state.running = False

# Main Title
st.title(" AI Observatory - Real-Time Neural Network Monitor")
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs([" Metrics", " Model Visualization"])

with tab1:
    placeholder = st.empty()
    loss_chart = st.empty()
    metric_chart = st.empty()
    gpu_chart = st.empty()

    if st.session_state.running:
        losses = []
        accuracies = []
        steps = 0

        with placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card"><h4> Loss</h4><p id="loss_val">--</p></div>', unsafe_allow_html=True)
                loss_progress = st.progress(0)
            with col2:
                st.markdown('<div class="card"><h4> Accuracy</h4><p id="acc_val">--</p></div>', unsafe_allow_html=True)
                acc_progress = st.progress(0)

        while st.session_state.running and steps < 100:
            loss = max(0.01, min(1.0, np.random.normal(loc=1 - steps/100, scale=0.1)))
            accuracy = min(0.99, np.random.normal(loc=steps/100, scale=0.05))

            losses.append(loss)
            accuracies.append(accuracy)

            # Update progress safely
            with placeholder.container():
                col1, col2 = st.columns(2)
                with col1:
                    loss_value = int(max(0, min(100, (1 - loss) * 100)))
                    loss_progress.progress(loss_value)
                with col2:
                    acc_value = int(max(0, min(100, accuracy * 100)))
                    acc_progress.progress(acc_value)

            # Loss Chart
            df_loss = pd.DataFrame({"Step": range(len(losses)), "Loss": losses})
            fig_loss = px.line(df_loss, x="Step", y="Loss", title="📉 Real-Time Training Loss", color_discrete_sequence=["#ff005e"])
            fig_loss.update_layout(paper_bgcolor="#0e0e2c", plot_bgcolor="#0e0e2c", font_color="white")
            loss_chart.plotly_chart(fig_loss, use_container_width=True)

            # Accuracy Chart
            df_acc = pd.DataFrame({"Step": range(len(accuracies)), "Accuracy": accuracies})
            fig_acc = px.line(df_acc, x="Step", y="Accuracy", title="📈 Real-Time Model Accuracy", color_discrete_sequence=["#00ffe1"])
            fig_acc.update_layout(paper_bgcolor="#0e0e2c", plot_bgcolor="#0e0e2c", font_color="white")
            metric_chart.plotly_chart(fig_acc, use_container_width=True)

            # GPU Gauge
            gpu_usage = np.random.uniform(30, 95)
            fig_gpu = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gpu_usage,
                title={'text': " GPU Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "#00ffe1"},
                       'steps': [
                           {'range': [0, 50], 'color': '#1f1f3a'},
                           {'range': [50, 80], 'color': '#1a1a3d'},
                           {'range': [80, 100], 'color': '#3a3aff'}],
                       }))
            fig_gpu.update_layout(paper_bgcolor="#0e0e2c", font_color="white")
            gpu_chart.plotly_chart(fig_gpu, use_container_width=True)

            time.sleep(0.2)
            steps += 1

        st.success(" Training Completed!")

        # Confusion Matrix Heatmap
        y_true = np.random.choice([0, 1, 2], size=100)
        y_pred = np.random.choice([0, 1, 2], size=100)
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Viridis', title="🧮 Confusion Matrix")
        st.plotly_chart(fig_cm)

    else:
        st.info(" Start the simulation from the sidebar to see live updates.")

with tab2:
    st.subheader(" Neural Network Architecture")

    # D3-inspired animated SVG
    st.components.v1.html("""
    <script src="https://d3js.org/d3.v7.min.js "></script>
    <svg width="100%" height="400" style="background:#0e0e2c;"></svg>
    <script>
        const svg = d3.select("svg");
        const width = +svg.attr("width");
        const height = +svg.attr("height");

        const layers = [4, 6, 5, 3];
        const nodes = [];
        let idx = 0;

        layers.forEach((size, layerIdx) => {
            for (let i = 0; i < size; i++) {
                nodes.push({x: layerIdx * 150 + 100, y: i * 50 + 50, id: idx++});
            }
        });

        // Draw connections
        for (let i = 0; i < nodes.length - 1; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                if (nodes[i].x < nodes[j].x) {
                    svg.append("line")
                        .attr("x1", nodes[i].x)
                        .attr("y1", nodes[i].y)
                        .attr("x2", nodes[j].x)
                        .attr("y2", nodes[j].y)
                        .attr("stroke", "#777")
                        .attr("stroke-width", 1);
                }
            }
        }

        // Draw neurons
        svg.selectAll("circle")
            .data(nodes).enter()
            .append("circle")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", 10)
            .attr("fill", "#00ffe1")
            .on("mouseover", function() { d3.select(this).attr("r", 15); })
            .on("mouseout", function() { d3.select(this).attr("r", 10); });
    </script>
    """, height=450)

# Sidebar Expanders
with st.sidebar.expander(" Training Settings"):
    learning_rate = st.slider("Learning Rate", 0.0001, 0.1, 0.001)
    epochs = st.number_input("Epochs", 1, 100, 10)

with st.sidebar.expander(" UI Preferences"):
    theme = st.radio("Theme", ["Dark", "Light"])
    if theme == "Dark":
        st.markdown("""
        <style>
            .main { background-color: #0e0e2c; color: white; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .main { background-color: white; color: black; }
        </style>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(" AI Observatory v2.0 — Powered by Streamlit, Plotly, and Deep Learning Magic!")