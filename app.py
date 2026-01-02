import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot
from statsmodels.stats.diagnostic import normal_ad
from matplotlib.backends.backend_pdf import PdfPages
import tempfile


def create_raw_data_figures(df, rows_per_page=25):
    figures = []

    display_df = df.fillna("")
    total_rows = len(display_df)
    num_pages = (total_rows // rows_per_page) + int(total_rows % rows_per_page != 0)

    for page in range(num_pages):
        start = page * rows_per_page
        end = start + rows_per_page
        page_df = display_df.iloc[start:end]

        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis("off")

        # ---- Title (tight spacing) ----
        title = f"Input Data (Uploaded File) — Page {page + 1} of {num_pages}"
        ax.text(
            0.5,
            0.94,
            title,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

        # ---- Pretty column headers for display ----
        display_columns = ["MH", "Batch No.", "Date"]

        table = ax.table(
            cellText=page_df.values,
            colLabels=display_columns,
            cellLoc="center",
            colLoc="center",
            loc="upper center",
            bbox=[0.03, 0.06, 0.94, 0.82],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.25)

        figures.append(fig)

    return figures


def create_summary_figure(
    data,
    mu,
    sigma_within,
    sigma_overall,
    Cp,
    Cpk,
    Pp,
    Ppk,
    PPM,
    LSL,
    USL,
    xbar_failed_points,
    r_failed_points,
):

    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
    ax.axis("off")

    # Title
    ax.text(
        0.5,
        0.96,
        "Process Capability Summary",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
    )

    # Capability verdict
    if Cpk < 1.00:
        verdict = "Process NOT capable"
    elif Cpk < 1.33:
        verdict = "Marginal capability"
    elif Cpk < 1.67:
        verdict = "Capable process"
    else:
        verdict = "Highly capable process"

    # ---- Control chart test results ----

    if xbar_failed_points:
        xbar_test = "X̄ Chart (Test 1):\n" "Failed at subgroups: " + ", ".join(
            map(str, xbar_failed_points)
        )
    else:
        xbar_test = "X̄ Chart (Test 1): PASSED"

    if r_failed_points:
        r_test = "R̄ Chart (Test 1):\n" "Failed at subgroups: " + ", ".join(
            map(str, r_failed_points)
        )
    else:
        r_test = "R̄ Chart (Test 1): PASSED"

    # Summary text
    summary_text = (
        f"Sample size (N)     : {len(data)}\n"
        f"Sample Mean         : {mu:.5f}\n\n"
        f"Specifications      : LSL = {LSL:.3f}   USL = {USL:.3f}\n\n"
        f"Within Std Dev      : {sigma_within:.5f}\n"
        f"Overall Std Dev     : {sigma_overall:.5f}\n\n"
        f"Cp                  : {Cp:.3f}\n"
        f"Cpk                 : {Cpk:.3f}\n"
        f"Pp                  : {Pp:.3f}\n"
        f"Ppk                 : {Ppk:.3f}\n"
        f"PPM (Overall)       : {PPM:.1f}\n\n"
        "CONTROL CHART TEST RESULTS\n"
        "---------------------------\n"
        f"{xbar_test}\n\n"
        f"{r_test}\n\n"
        f"CONCLUSION          : {verdict}"
    )

    ax.text(
        0.05,
        0.85,
        summary_text,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    return fig


# Theme and background

plt.rcParams.update(
    {
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "axes.edgecolor": "#9aa4af",
        "axes.linewidth": 1.1,
        "axes.labelcolor": "#1f2937",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "grid.color": "#d1d5db",
        "grid.alpha": 0.6,
    }
)


st.set_page_config(page_title="Process Capability App", layout="centered")
st.markdown(
    """
    <style>
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #e5e7eb;   /* slightly darker than main */
        border-right: 1px solid #cbd5e1;
    }

    /* Main app background (kept light) */
    .stApp {
        background-color: #eef2f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("📊 Process Capability Analysis")
st.caption("Version 1.0 — Process Capability Analysis")
st.write("Upload a CSV or Excel file containing numeric data.")

uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

if uploaded_file is not None:

    # ---- Read file (with headers) ----
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # ---- Normalize column names (robust to Excel weirdness) ----
    df.columns = df.columns.astype(str).str.strip().str.lower()

    # ---- Forward-fill Date column (for display only) ----
    if "date" in df.columns:
        df["date"] = df["date"].ffill()

    # ---- Fix Batch No. column (Excel float noise → integer IDs) ----
    if "batch no." in df.columns:
        df["batch no."] = (
            pd.to_numeric(df["batch no."], errors="coerce")
            .round()  # removes 8057.0000000001 type noise
            .astype("Int64")  # nullable integer (safe with NaN)
        )

    # ---- Validate required columns ----
    required_cols = {"mh", "batch no.", "date"}
    if not required_cols.issubset(df.columns):
        st.error("Uploaded file must contain columns: MH, Batch No., Date")
        st.stop()

    # ---- Use ONLY MH values for analysis ----
    mh_series = pd.to_numeric(df["mh"], errors="coerce")
    data = mh_series.dropna().values

    # ---- Minimum data check ----
    if len(data) < 25:
        st.error("At least 25 valid MH data points are required.")
        st.stop()

    # ---- User inputs ----
    st.sidebar.header("Specifications")
    USL = st.sidebar.number_input("USL", value=13.75)
    LSL = st.sidebar.number_input("LSL", value=11.25)

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

    if "figures" not in st.session_state:
        st.session_state.figures = []

    st.session_state.figures.clear()

    # ----- Xbar Chart -----
    st.subheader("X̄ Chart")

    fig, ax = plt.subplots(figsize=(9, 5))

    A2 = 0.577
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
    x_text = len(xbar_vals) - 1
    offset = 0.010

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
    ax.set_title("X_bar Chart", fontsize=13, fontweight="bold")

    st.pyplot(fig)
    st.session_state.figures.append(fig)

    # Test 1: One point beyond 3 sigma (UCL/LCL)
    xbar_failed_points = (idx[out_of_control] + 1).tolist()  # +1 for 1-based indexing

    # ----- 3. R Chart -----
    st.subheader("R̄ Chart")

    D4 = 2.114
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

    x_shift = 1.9
    ax.text(x_text - x_shift, UCL_R + offset, f"UCL = {UCL_R:.3f}", va="bottom")
    ax.text(x_text - x_shift, R_bar + offset, f"R_bar = {R_bar:.3f}", va="bottom")
    ax.text(x_text - x_shift, LCL_R + offset, f"LCL = {LCL_R:.3f}", va="bottom")

    # Layout
    ax.set_xlabel("Subgroup")
    ax.set_ylabel("Sample Range")
    ax.set_xlim(-0.5, len(R_vals) + 3)
    ax.grid(alpha=0.3)

    ax.set_title("R_bar Chart", fontsize=13, fontweight="bold")

    st.pyplot(fig)
    st.session_state.figures.append(fig)

    # Store R-chart test failures (1-based indexing)
    r_failed_points = (idx_R[out_of_control_R] + 1).tolist()

    # ----- Last 25 Subgroups (Individual Values) -----
    st.subheader("Last 25 Subgroups")

    n_show = min(25, n_subgroups)

    # Take last 25 subgroups
    last_subgroups = subgroups[-n_show:]  # shape: (n_show, subgroup_size)

    fig, ax = plt.subplots(figsize=(9, 4))

    # Plot each subgroup as a vertical stack of points
    for i in range(n_show):
        x_vals = np.full(subgroup_size, i + 1)  # subgroup index (1-based)
        y_vals = last_subgroups[i]
        ax.scatter(x_vals, y_vals, s=40)

    # Optional: overall mean line (Minitab-style)
    ax.axhline(mu, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Sample")
    ax.set_ylabel("Values")
    ax.set_title("Last 25 Subgroups", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    st.session_state.figures.append(fig)

    # ----- 4. Capability Histogram -----
    st.subheader("Capability Histogram")

    fig, ax = plt.subplots(figsize=(9, 5))

    # Histogram
    count, bins, _ = ax.hist(
        data,
        bins=12,
        density=True,
        color="#7da7d9",
        edgecolor="black",
        alpha=0.85,
    )

    # X range for curves
    x = np.linspace(min(bins), max(bins), 300)

    # Normal curves
    ax.plot(
        x,
        norm.pdf(x, mu, sigma_overall),
        color="brown",
        linewidth=2,
        label="Overall",
    )

    ax.plot(
        x,
        norm.pdf(x, mu, sigma_within),
        color="black",
        linestyle="--",
        linewidth=2,
        label="Within",
    )

    import matplotlib.transforms as transforms

    # --- Spec limit lines ---
    ax.axvline(LSL, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(USL, color="red", linestyle="--", linewidth=1.5)

    # --- Remove any existing LSL/USL labels (important for Streamlit reruns) ---
    for txt in ax.texts:
        if txt.get_text().startswith(("LSL", "USL")):
            txt.remove()

    # --- Spec limit labels (printed ONCE) ---
    y_text = ax.get_ylim()[1] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    dx = 3 / 72  # horizontal shift (points)
    left = transforms.ScaledTranslation(-dx, 0, ax.figure.dpi_scale_trans)
    right = transforms.ScaledTranslation(dx, 0, ax.figure.dpi_scale_trans)

    ax.text(
        LSL,
        y_text,
        f"LSL",
        transform=ax.transData + left,
        color="red",
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    ax.text(
        USL,
        y_text,
        f"USL",
        transform=ax.transData + right,
        color="red",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    # Axis labels & title
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Capability Histogram", fontsize=13, fontweight="bold")

    # ---- Right-side info box (Minitab-style) ----
    spec_text = (
        "Overall\n"
        "— Solid line\n\n"
        "Within\n"
        "-- Dashed line\n\n"
        "Specifications\n"
        f"LSL    {LSL:.2f}\n"
        f"USL    {USL:.2f}"
    )

    ax.text(
        1.03,
        0.5,
        spec_text,
        transform=ax.transAxes,
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray"),
    )

    ax.grid(alpha=0.3)

    st.pyplot(fig)
    st.session_state.figures.append(fig)

    # ----- 5. Normal Probability Plot -----
    st.subheader("Normal Probability Plot")

    # Anderson-Darling normality test
    ad_stat, p_value = normal_ad(data)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Probability plot
    (osm, osr), (slope, intercept, r) = probplot(data, dist="norm")

    # Correct axes:
    # X = theoretical normal quantiles
    # Y = ordered data
    ax.scatter(osm, osr, s=40)

    # Correct fitted line
    ax.plot(osm, slope * osm + intercept, color="brown", linewidth=2)

    ax.set_title(
        f"Normal Probability Plot\nAD: {ad_stat:.3f}, P: {p_value:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlabel("Normal Score")
    ax.set_ylabel("Data")
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    st.session_state.figures.append(fig)

    # ----- 6. Capability Indices -----
    Cp = (USL - LSL) / (6 * sigma_within)
    Cpk = min((USL - mu) / (3 * sigma_within), (mu - LSL) / (3 * sigma_within))
    Pp = (USL - LSL) / (6 * sigma_overall)
    Ppk = min((USL - mu) / (3 * sigma_overall), (mu - LSL) / (3 * sigma_overall))
    PPM = (
        1 - norm.cdf(USL, mu, sigma_overall) + norm.cdf(LSL, mu, sigma_overall)
    ) * 1e6

    # ----- Capability Plot -----
    st.subheader("Capability Plot")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Vertical positions (extra spacing for clarity)
    y_overall = 4
    y_within = 2.5
    y_specs = 1

    # ----- OVERALL -----
    ax.hlines(y_overall, mu - 3 * sigma_overall, mu + 3 * sigma_overall, linewidth=2)
    ax.vlines(
        [mu - 3 * sigma_overall, mu, mu + 3 * sigma_overall],
        y_overall - 0.15,
        y_overall + 0.15,
    )

    # ----- WITHIN -----
    ax.hlines(y_within, mu - 3 * sigma_within, mu + 3 * sigma_within, linewidth=2)
    ax.vlines(
        [mu - 3 * sigma_within, mu, mu + 3 * sigma_within],
        y_within - 0.15,
        y_within + 0.15,
    )

    # ----- SPECS (lighter style) -----
    ax.hlines(y_specs, LSL, USL, linewidth=1.5, color="gray")
    ax.vlines([LSL, USL], y_specs - 0.12, y_specs + 0.12, color="gray")

    # Y-axis labels
    ax.set_yticks([y_overall, y_within, y_specs])
    ax.set_yticklabels(["Overall", "Within", "Specs"])

    # Axis formatting
    ax.set_xlabel("Value")
    ax.set_title("Capability Plot", fontsize=13, fontweight="bold")

    xmin = min(LSL, mu - 3 * sigma_overall) - 0.1
    xmax = max(USL, mu + 3 * sigma_overall) + 0.1
    ax.set_xlim(xmin, xmax)

    ax.grid(axis="x", alpha=0.3)

    # Reserve margins for side text
    plt.subplots_adjust(left=0.22, right=0.78)

    # ----- Left-side stats (Within) -----
    left_text = (
        "Within\n"
        f"StDev   {sigma_within:>7.4f}\n"
        f"Cp      {Cp:>7.2f}\n"
        f"Cpk     {Cpk:>7.2f}\n"
        f"PPM     {PPM:>7.2f}"
    )

    # ---- Left-side stats (Within) - shifted UP to avoid overlap ----
    stats_y = y_within + 1.00  # 🔥 SAFE vertical clearance (key line)

    ax.text(
        xmin + 0.02 * (xmax - xmin),
        stats_y,
        left_text,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    # ----- Right-side stats (Overall) -----
    right_text = (
        "Overall\n"
        f"StDev   {sigma_overall:>7.4f}\n"
        f"Pp      {Pp:>7.2f}\n"
        f"Ppk     {Ppk:>7.2f}\n"
        f"Cpm     {'*':>7}\n"
        f"PPM     {PPM:>7.2f}"
    )

    # ---- Right-side stats (Overall) - shifted UP ----
    ax.text(
        xmax - 0.02 * (xmax - xmin),
        stats_y,
        right_text,
        va="top",
        ha="right",
        fontsize=9,
        family="monospace",
    )

    st.pyplot(fig)
    st.session_state.figures.append(fig)

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

    Cp = (USL - LSL) / (6 * sigma_within)

    Cpu = (USL - mu) / (3 * sigma_within)
    Cpl = (mu - LSL) / (3 * sigma_within)

    Cpk = min(Cpu, Cpl)

    Pp = (USL - LSL) / (6 * sigma_overall)
    Ppu = (USL - mu) / (3 * sigma_overall)
    Ppl = (mu - LSL) / (3 * sigma_overall)
    Ppk = min(Ppu, Ppl)

    st.subheader("Process Capability Indices")

    col1, col2, col3 = st.columns(3)

    col1.metric("Cp", f"{Cp:.2f}")
    col1.metric("Cpu", f"{Cpu:.2f}")
    col1.metric("Cpl", f"{Cpl:.2f}")

    col2.metric("Cpk", f"{Cpk:.2f}")

    col3.metric("Pp", f"{Pp:.2f}")
    col3.metric("Ppk", f"{Ppk:.2f}")

    # st.caption(
    #     f"""
    #     Cp measures potential capability assuming centering.
    #     Cpk accounts for mean shift.

    #     Current limiting side: {"Upper" if Cpu < Cpl else "Lower"}
    #     """
    # )

    if Cpk < 1.00:
        st.error(f"Cpk = {Cpk:.2f} → Process NOT capable")
    elif Cpk < 1.33:
        st.warning(f"Cpk = {Cpk:.2f} → Marginal capability")
    elif Cpk < 1.67:
        st.info(f"Cpk = {Cpk:.2f} → Capable process")
    else:
        st.success(f"Cpk = {Cpk:.2f} → Highly capable process")
    st.caption("Results assume a stable process and rational subgrouping.")

    if st.session_state.figures:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

        with PdfPages(tmp.name) as pdf:

            # ----- All plots -----
            for fig in st.session_state.figures:
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

            # ----- Summary page (ONCE) -----
            summary_fig = create_summary_figure(
                data=data,
                mu=mu,
                sigma_within=sigma_within,
                sigma_overall=sigma_overall,
                Cp=Cp,
                Cpk=Cpk,
                Pp=Pp,
                Ppk=Ppk,
                PPM=PPM,
                LSL=LSL,
                USL=USL,
                xbar_failed_points=xbar_failed_points,
                r_failed_points=r_failed_points,
            )
            pdf.savefig(summary_fig, dpi=300)
            plt.close(summary_fig)

            # ----- Raw input Excel pages (ONCE, LAST) -----
            raw_data_figures = create_raw_data_figures(df)

            for fig in raw_data_figures:
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

        with open(tmp.name, "rb") as f:
            st.download_button(
                label="📄 Download Analysis Report (PDF)",
                data=f,
                file_name="process_analysis_report.pdf",
                mime="application/pdf",
            )
