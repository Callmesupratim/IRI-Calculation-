import pandas as pd
import numpy as np
from scipy.signal import medfilt
from scipy import stats
from scipy.integrate import odeint
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import plotly.io as pio
from PIL import Image as PILImage
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import os
import warnings
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, TableStyle, Paragraph, SimpleDocTemplate, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Suppress warnings for integration
warnings.filterwarnings("ignore", category=UserWarning)

# === Helper Functions ===

def detect_outliers(data, threshold=3):
    z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
    return np.where(z_scores < threshold, data, np.nan)

def quarter_car_model(y, t, k, c, m, profile, spacing, speed):
    z, z_dot = y
    pos = t * speed
    z_r = np.interp(pos, np.arange(len(profile)) * spacing, profile)
    z_ddot = (k * (z_r - z) - c * z_dot) / m
    return [z_dot, z_ddot]

def calculate_segment_iri(segment, spacing, speed, k, c, m):
    dt = spacing / speed
    t = np.arange(0, len(segment) * dt, dt)
    try:
        solution = odeint(quarter_car_model, [0, 0], t,
                          args=(k, c, m, segment, spacing, speed))
        z_dot = solution[:, 1]
        return np.trapz(np.abs(z_dot), dx=dt) / (len(segment) * spacing) * 1000
    except Exception as e:
        print(f"ODE solver error: {e}")
        return np.nan

def classify_iri(iri):
    if iri < 2:
        return "Excellent/Good"
    elif iri < 3.5:
        return "Fair"
    elif iri < 5:
        return "Poor"
    return "Very Poor"

# === Main Execution ===
def main():
    # File selection
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Excel File", filetypes=[("Excel files", "*.xlsx")]
    )
    if not file_path:
        raise Exception("No file selected.")

    df = pd.read_excel(file_path)
    metadata_cols = ["Date", "Time", "Section Code", "NH No", "Direction", "Start Chainage"]
    for col in metadata_cols:
        if col not in df.columns:
            df[col] = "NA"

    metadata = df.iloc[0][metadata_cols].to_dict()

    required_cols = [f"Deflection {i}" for i in range(1, 8)] + ["Chainage", "Latitude", "Longitude", "Velocity (km/h)"]
    df = df.dropna(subset=["Chainage"]).sort_values("Chainage").reset_index(drop=True)

    deflections = df[[f"Deflection {i}" for i in range(1, 8)]].values
    deflection_avg = np.nanmean(deflections, axis=1)
    deflection_clean = pd.Series(detect_outliers(deflection_avg)).interpolate(method='spline', order=3).bfill().ffill().values

    deflection_m = deflection_clean / 1000
    offset = deflection_m[0]
    deflection_m -= offset
    deflection_filtered = medfilt(deflection_m, kernel_size=5)

    k, c, m = 65300, 6000, 240
    speed = np.nanmean(df["Velocity (km/h)"].values) / 3.6
    spacing = 5.87
    dt = spacing / speed
    critical_dt = np.sqrt(m / k)

    if dt > 0.1 * critical_dt:
        spacing = 0.1 * critical_dt * speed

    chainage = df["Chainage"].values
    lat = df["Latitude"].values
    lon = df["Longitude"].values

    chain_interp = np.arange(chainage[0], chainage[-1], spacing)
    profile_interp = np.interp(chain_interp, chainage, deflection_filtered)

    segment_len = 100
    samples_per_segment = int(segment_len / spacing)
    step = samples_per_segment // 2

    iri_vals, iri_chain, iri_lat, iri_lon = [], [], [], []

    for i in range(0, len(profile_interp) - samples_per_segment, step):
        segment = profile_interp[i:i + samples_per_segment]
        iri = calculate_segment_iri(segment, spacing, speed, k, c, m)
        if np.isnan(iri):
            continue
        iri_vals.append(iri)
        mid_chain = (chain_interp[i] + chain_interp[i + samples_per_segment - 1]) / 2
        iri_chain.append(mid_chain)
        idx = np.argmin(np.abs(chainage - mid_chain))
        iri_lat.append(lat[idx])
        iri_lon.append(lon[idx])

    iri_df = pd.DataFrame({
        "Chainage (m)": iri_chain,
        "Latitude": iri_lat,
        "Longitude": iri_lon,
        "IRI (m/km)": iri_vals
    })
    iri_df["IRI Quality"] = iri_df["IRI (m/km)"].apply(classify_iri)

    # Plotly chart export
    fig = px.line(iri_df, x="Chainage (m)", y="IRI (m/km)", title="IRI Profile")
    fig.add_hline(y=2, line_dash="dot", line_color="green", annotation_text="Good/Fair")
    fig.add_hline(y=3.5, line_dash="dot", line_color="orange", annotation_text="Fair/Poor")
    fig.add_hline(y=5, line_dash="dot", line_color="red", annotation_text="Poor/Very Poor")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(file_path)
    img_file = os.path.join(output_dir, f"iri_plot_{timestamp}.png")
    fig.write_image(img_file)

    # === Generate HTML Map ===
    iri_map = folium.Map(location=[np.mean(iri_lat), np.mean(iri_lon)], zoom_start=10)
    marker_cluster = MarkerCluster().add_to(iri_map)

    for _, row in iri_df.iterrows():
        color = {
            "Excellent/Good": "green",
            "Fair": "orange",
            "Poor": "red",
            "Very Poor": "darkred"
        }.get(row["IRI Quality"], "gray")
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"Chainage: {row['Chainage (m)']:.2f} m<br>IRI: {row['IRI (m/km)']:.2f} m/km<br>Quality: {row['IRI Quality']}"
        ).add_to(marker_cluster)

    map_file = os.path.join(output_dir, f"iri_map_{timestamp}.html")
    iri_map.save(map_file)
    print(f"✅ HTML map generated: {map_file}")

    # === Generate PDF Report ===
    output_pdf = os.path.join(output_dir, f"IRI_Report_{timestamp}.pdf")
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>IRI Report</b>", styles['Title']))
    elements.append(Spacer(1, 12))

    # Add metadata
    meta_table = Table([[k, str(v)] for k, v in metadata.items()])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica')
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 12))

    # Add chart image
    elements.append(Image(img_file, width=500, height=300))
    elements.append(Spacer(1, 12))

    # Summary stats
    max_iri_row = iri_df.loc[iri_df['IRI (m/km)'].idxmax()]
    min_iri_row = iri_df.loc[iri_df['IRI (m/km)'].idxmin()]
    avg_iri = iri_df['IRI (m/km)'].mean()

    summary_stats = [
        ["Max IRI", f"{max_iri_row['IRI (m/km)']:.2f} m/km at {max_iri_row['Chainage (m)']:.2f} m"],
        ["Min IRI", f"{min_iri_row['IRI (m/km)']:.2f} m/km at {min_iri_row['Chainage (m)']:.2f} m"],
        ["Average IRI", f"{avg_iri:.2f} m/km"]
    ]
    stats_table = Table(summary_stats)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 12))

    # Add page break before IRI summary table
    elements.append(PageBreak())

    # Add IRI summary table
    summary_data = [["Chainage (m)", "IRI (m/km)", "IRI Quality"]] + [
        [float(row[0]), float(row[1]), str(row[2])] for row in iri_df[["Chainage (m)", "IRI (m/km)", "IRI Quality"]].values
    ]
    iri_table = Table(summary_data, repeatRows=1)
    iri_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    ]))
    elements.append(iri_table)

    doc.build(elements)

    print(f"\n✅ PDF report generated: {output_pdf}")

if __name__ == "__main__":
    main()
