import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot

st.set_page_config(page_title="Process Capability App", layout="centered")
st.title("📊 Process Capability Analysis")
st.write("Upload a CSV or Excel file containing numeric data.")

uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

if uploaded_file is not None:

    # ---- Read file ----
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, header=None)
    else:
        df = pd.read_excel(uploaded_file, header=None)

    numeric = pd.to_numeric(df.values.flatten(), errors="coerce")
    data = numeric[~np.isnan(numeric)]

    if len(data) < 25:
        st.error("At least 25 data points are required.")
        st.stop()

    # ---- User inputs (dad-friendly sliders) ----
    st.sidebar.header("Specifications")
    USL = st.sidebar.number_input("USL", value=8.35)
    LSL = st.sidebar.number_input("LSL", value=6.92)

    subgroup_size = 5
    D2 = 2.33

    # ----- Subgroup calculations -----
    n_subgroups = len(data) // subgroup_size
    subgroups = data[: n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)

    xbar = subgroups.mean(axis=1)
    R = subgroups.max(axis=1) - subgroups.min(axis=1)

    R_bar = R.mean()
    sigma_within = R_bar / D2
    sigma_overall = data.std(ddof=0)
    mu = data.mean()

    # # ----- 1. Individual Plot -----
    # st.subheader("Individual Values (Last 25 Samples)")
    # fig, ax = plt.subplots()
    # ax.plot(range(1, 26), data[-25:], marker="o")
    # ax.axhline(mu, linestyle="--")
    # ax.set_xlabel("Sample")
    # ax.set_ylabel("Values")
    # st.pyplot(fig)

    # ----- Xbar Chart -----
    st.subheader("X̄ Chart")

    fig, ax = plt.subplots(figsize=(9, 5))

    sigma_xbar = sigma_within / np.sqrt(subgroup_size)
    center_line = xbar.mean()

    UCL_xbar = center_line + 3 * sigma_xbar
    LCL_xbar = center_line - 3 * sigma_xbar

    # Convert to numpy for masking
    xbar_vals = np.array(xbar)
    idx = np.arange(len(xbar_vals))

    # Mask for out-of-control points
    out_of_control = (xbar_vals > UCL_xbar) | (xbar_vals < LCL_xbar)

    # Plot all points (normal)
    ax.plot(idx, xbar_vals, marker="o", linewidth=2, label="X̄")

    # Highlight out-of-control points
    ax.scatter(
        idx[out_of_control],
        xbar_vals[out_of_control],
        color="red",
        marker="s",
        s=80,
        label="Out of Control",
    )

    # Control limits
    ax.axhline(center_line, linestyle="--", linewidth=1.5, label="CL")
    ax.axhline(UCL_xbar, linestyle="--", linewidth=1.5, label="UCL")
    ax.axhline(LCL_xbar, linestyle="--", linewidth=1.5, label="LCL")

    # Spec limits
    ax.axhline(USL, linestyle=":", linewidth=2, label="USL")
    ax.axhline(LSL, linestyle=":", linewidth=2, label="LSL")

    # ---- Tidy labels on right ----
    x_text = len(xbar_vals) + 0.5
    offset = 0.015

    ax.text(x_text, UCL_xbar + offset, f"UCL = {UCL_xbar:.3f}", va="bottom")
    ax.text(x_text, center_line + offset, f"CL = {center_line:.3f}", va="bottom")
    ax.text(x_text, LCL_xbar - offset, f"LCL = {LCL_xbar:.3f}", va="top")
    ax.text(x_text, USL + offset, f"USL = {USL:.3f}", va="bottom")
    ax.text(x_text, LSL - offset, f"LSL = {LSL:.3f}", va="top")

    # Layout
    ax.set_xlabel("Subgroup")
    ax.set_ylabel("Sample Mean")
    ax.set_xlim(-0.5, len(xbar_vals) + 3)
    ax.grid(alpha=0.3)
    # ax.legend(loc="lower left")

    st.pyplot(fig)

    # Test 1: One point beyond 3 sigma (UCL/LCL)
    xbar_failed_points = (idx[out_of_control] + 1).tolist()  # +1 for 1-based indexing

    # ----- 3. R Chart -----
    st.subheader("R Chart")

    D4 = 4.918
    D3 = 0.0

    UCL_R = R_bar * D4
    LCL_R = R_bar * D3

    R_vals = np.array(R)
    idx_R = np.arange(len(R_vals))

    out_of_control_R = (R_vals > UCL_R) | (R_vals < LCL_R)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot R values
    ax.plot(idx_R, R_vals, marker="o", linewidth=2)

    # Highlight out-of-control points
    ax.scatter(
        idx_R[out_of_control_R],
        R_vals[out_of_control_R],
        color="red",
        marker="s",
        s=80,
    )

    # Control limits
    ax.axhline(R_bar, linestyle="--", linewidth=1.5)
    ax.axhline(UCL_R, linestyle="--", linewidth=1.5)
    ax.axhline(LCL_R, linestyle="--", linewidth=1.5)

    # ---- Tidy labels on right (same style as X̄) ----
    x_text = len(R_vals) + 0.5
    offset = 0.015 * max(R_vals)  # scale offset to data

    ax.text(x_text, UCL_R + offset, f"UCL = {UCL_R:.3f}", va="bottom")
    ax.text(x_text, R_bar + offset, f"R̄ = {R_bar:.3f}", va="bottom")
    ax.text(x_text, LCL_R - offset, f"LCL = {LCL_R:.3f}", va="top")

    # Layout
    ax.set_xlabel("Subgroup")
    ax.set_ylabel("Sample Range")
    ax.set_xlim(-0.5, len(R_vals) + 3)
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # Store R-chart test failures (1-based indexing)
    r_failed_points = (idx_R[out_of_control_R] + 1).tolist()

    # ----- Last 25 Subgroups (X̄ vs Subgroup) -----
    st.subheader("Last 25 Subgroups (X̄)")

    n_show = min(25, len(xbar))
    last_xbar = xbar[-n_show:]
    subgroup_idx = np.arange(len(xbar) - n_show + 1, len(xbar) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(subgroup_idx, last_xbar, marker="o", linewidth=2)
    ax.axhline(center_line, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Subgroup")
    ax.set_ylabel("Subgroup Mean (X̄)")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # ----- 5. Normal Probability Plot -----
    st.subheader("Normal Probability Plot")
    fig = plt.figure()
    probplot(data, dist="norm", plot=plt)
    st.pyplot(fig)

    # ----- 6. Capability Indices -----
    Cp = (USL - LSL) / (6 * sigma_within)
    Cpk = min((USL - mu) / (3 * sigma_within), (mu - LSL) / (3 * sigma_within))
    Pp = (USL - LSL) / (6 * sigma_overall)
    Ppk = min((USL - mu) / (3 * sigma_overall), (mu - LSL) / (3 * sigma_overall))
    PPM = (
        1 - norm.cdf(USL, mu, sigma_overall) + norm.cdf(LSL, mu, sigma_overall)
    ) * 1e6

    st.subheader("Process Capability Indices")
    st.write(f"**Cp (Within):** {Cp:.3f}")
    st.write(f"**Cpk (Within):** {Cpk:.3f}")
    st.write(f"**Pp (Overall):** {Pp:.3f}")
    st.write(f"**Ppk (Overall):** {Ppk:.3f}")
    st.write(f"**PPM (Overall):** {PPM:.2f}")

    # Test Results

    # ----- Xbar Chart Test Results -----
    st.markdown("### Test Results for X̄ Chart of C1")
    if xbar_failed_points:
        st.info(
            f"""
            **TEST 1.** One point more than 3.00 standard deviations from center line.

            **Test Failed at points:** {', '.join(map(str, xbar_failed_points))}
            """
        )
    else:
        st.success(
            "**TEST 1 PASSED.** One point more than 3.00 standard deviations from center line."
        )

    # ----- R Chart Test Results -----
    st.subheader("Test Results for R Chart of C1")

    if r_failed_points:
        st.info(
            f"""
            **TEST 1.** One point more than 3.00 standard deviations from center line.

            **Test Failed at points:** {', '.join(map(str, r_failed_points))}
            """
        )
    else:
        st.success(
            "**TEST 1 PASSED.** One point more than 3.00 standard deviations from center line."
        )
