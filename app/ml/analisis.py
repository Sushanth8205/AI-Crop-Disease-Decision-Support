# ================================================================
# CROPCARE AI - MONGODB ANALYTICS & IN-MEMORY VISUALIZATION
# ================================================================
# Generates all diagnostic and analytical plots purely in-memory:
#   1. Diagnostic Image Feature Analysis (Segmentation, Edge, & Profile)
#   2. Month-wise Disease Ratio (Line Plot)
#   3. Image Processing Results (Normalized Feature Bar Chart)
#   4. Crop × Disease Heatmap (Frequency Heatmap)
#
# No temporary or permanent files are saved to disk.
# ================================================================

import os
import io
import sys
import base64
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
# Headless in-memory backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from bson.objectid import ObjectId
from pymongo import MongoClient

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")

# ================================================================
# 1. CONFIGURATION
# ================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "app", "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "cropcare"
COLLECTION_NAME = "detections"


# ================================================================
# 2. HELPER FUNCTIONS
# ================================================================

def determine_status(disease):
    disease_text = str(disease).lower().strip()
    healthy_words = ["healthy", "normal", "no disease", "no_disease"]
    for word in healthy_words:
        if word in disease_text:
            return "Healthy"
    return "Diseased"


def find_image(image_path, base_folder=UPLOAD_FOLDER):
    if image_path is None or pd.isna(image_path):
        return None
    image_path = str(image_path).replace("\\", "/").strip()
    filename = os.path.basename(image_path)
    if not filename:
        return None
    direct_path = os.path.join(base_folder, filename)
    if os.path.isfile(direct_path):
        return direct_path
    static_rel = os.path.join(PROJECT_ROOT, "app", "static", image_path)
    if os.path.isfile(static_rel):
        return static_rel
    root_rel = os.path.join(PROJECT_ROOT, image_path)
    if os.path.isfile(root_rel):
        return root_rel
    if os.path.isfile(image_path):
        return image_path
    for root, dirs, files in os.walk(base_folder):
        if filename in files:
            return os.path.join(root, filename)
    return None


def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (512, 512))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]
    mean_hue = float(np.mean(hue))
    mean_saturation = float(np.mean(saturation))
    mean_brightness = float(np.mean(brightness))

    # GREEN PIXEL RATIO
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([100, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixel_ratio = float((np.sum(green_mask > 0) / green_mask.size) * 100)

    # SHAPE FEATURES
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularity = 0.0
    aspect_ratio = 0.0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = float((4 * np.pi * area) / (perimeter ** 2))
        x, y, w, h = cv2.boundingRect(largest_contour)
        if h > 0:
            aspect_ratio = float(w / h)

    # EDGE DENSITY
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((np.sum(edges > 0) / edges.size) * 100)

    return {
        "Green Pixel Ratio": green_pixel_ratio,
        "Mean Hue": mean_hue,
        "Mean Saturation": mean_saturation,
        "Mean Brightness": mean_brightness,
        "Circularity": circularity,
        "Aspect Ratio": aspect_ratio,
        "Edge Density": edge_density
    }


def _get_user_dataframe(user_id=None):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    query = {}
    if user_id:
        try:
            query = {"user_id": ObjectId(user_id)}
        except Exception:
            query = {"user_id": user_id}
            
    records = list(collection.find(query))
    client.close()
    
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    for col in ["crop", "disease", "confidence", "image_path", "timestamp"]:
        if col not in df.columns:
            df[col] = np.nan

    df["crop"] = df["crop"].fillna("Unknown").astype(str).str.strip()
    df["disease"] = df["disease"].fillna("Unknown").astype(str).str.strip()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["status"] = df["disease"].apply(determine_status)
    df["month"] = df["timestamp"].dt.to_period("M").astype(str).replace("NaT", "Recent")
    return df


# ================================================================
# 3. INDIVIDUAL IN-MEMORY PLOT BUILDERS
# ================================================================

def build_diagnostic_analysis_plot(image_path, crop="Crop", disease="Detection", confidence=0.0):
    """
    DIAGNOSTIC PLOT: Leaf Segmentation, Edge Texture Map & Morphological Profile
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_resized = cv2.resize(image, (512, 512))
    rgb_img = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

    feats = extract_features(image_path)
    if feats is None:
        return None

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([100, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    edges = cv2.Canny(gray, 100, 200)

    mask_colored = np.zeros_like(rgb_img)
    mask_colored[:, :, 1] = green_mask
    segmented_overlay = cv2.addWeighted(rgb_img, 0.7, mask_colored, 0.3, 0)

    fig = plt.figure(figsize=(15, 4.6), facecolor='#ffffff')
    status_text = "Healthy" if determine_status(disease) == "Healthy" else "Diseased"
    fig.suptitle(
        f"Leaf Diagnostic Feature Analysis: {crop} - {disease} ({confidence}% Confidence)",
        fontsize=13,
        fontweight='bold',
        color='#15803d' if status_text == "Healthy" else '#b91c1c',
        y=0.98
    )

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(segmented_overlay)
    ax1.set_title("1. Leaf Segmentation & Tissue Mask", fontsize=10.5, fontweight='bold', color='#1e293b')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(edges, cmap='magma')
    ax2.set_title(f"2. Canny Edge & Lesion Texture ({feats['Edge Density']:.1f}%)", fontsize=10.5, fontweight='bold', color='#1e293b')
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 3)
    feature_labels = [
        "Green Ratio",
        "Mean Hue",
        "Saturation",
        "Brightness",
        "Circularity",
        "Aspect Ratio",
        "Edge Density"
    ]
    raw_vals = [
        feats["Green Pixel Ratio"],
        feats["Mean Hue"],
        feats["Mean Saturation"],
        feats["Mean Brightness"],
        feats["Circularity"] * 100,
        feats["Aspect Ratio"] * 100,
        feats["Edge Density"]
    ]

    norm_vals = [min(100.0, float(v)) for v in raw_vals]
    colors = ['#22c55e', '#10b981', '#06b6d4', '#3b82f6', '#8b5cf6', '#a855f7', '#ef4444']
    bars = ax3.barh(feature_labels, norm_vals, color=colors, edgecolor='#334155', height=0.55)

    for bar, raw_val in zip(bars, raw_vals):
        w = bar.get_width()
        ax3.text(
            w + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{raw_val:.1f}",
            va='center',
            ha='left',
            fontsize=8.5,
            fontweight='bold',
            color='#334155'
        )

    ax3.set_title("3. Extracted Morphological Profile", fontsize=10.5, fontweight='bold', color='#1e293b')
    ax3.set_xlim(0, 115)
    ax3.set_xlabel("Relative Metric Value", fontsize=9.5, fontweight='semibold')
    ax3.grid(axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    return fig


def build_plot1_monthly_ratio(df):
    """
    PLOT 1: Month-wise Disease Ratio Line Chart
    """
    fig = plt.figure(figsize=(10, 4.8), facecolor='#ffffff')
    if df.empty or "month" not in df.columns:
        plt.text(0.5, 0.5, "No monthly data available", ha='center', va='center', fontsize=12)
        plt.axis('off')
        return fig

    monthly = pd.crosstab(df["month"], df["status"])
    if "Healthy" not in monthly.columns:
        monthly["Healthy"] = 0
    if "Diseased" not in monthly.columns:
        monthly["Diseased"] = 0

    monthly["Total"] = monthly["Healthy"] + monthly["Diseased"]
    monthly["Disease_Ratio"] = (monthly["Diseased"] / monthly["Total"]) * 100
    monthly["Disease_Ratio"] = monthly["Disease_Ratio"].replace([np.inf, -np.inf], 0).fillna(0)
    monthly = monthly.sort_index()

    plt.plot(monthly.index, monthly["Disease_Ratio"], marker="o", color="#16a34a", linewidth=2.5, markersize=7)
    plt.title("Plot 1: Month-wise Disease Ratio", fontsize=13, fontweight="bold", pad=12, color="#1e293b")
    plt.xlabel("Month", fontsize=11, fontweight="semibold")
    plt.ylabel("Disease Ratio (%)", fontsize=11, fontweight="semibold")
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.4)

    for month, ratio in zip(monthly.index, monthly["Disease_Ratio"]):
        plt.annotate(
            f"{ratio:.1f}%",
            (month, ratio),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#15803d"
        )

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    return fig


def build_plot2_image_features(current_image_path=None, df=None):
    """
    PLOT 2: Image Processing Results Bar Chart
    """
    fig = plt.figure(figsize=(10, 4.8), facecolor='#ffffff')
    feats = None
    if current_image_path and os.path.isfile(current_image_path):
        feats = extract_features(current_image_path)

    if feats is None and df is not None and not df.empty:
        feature_rows = []
        for _, row in df.iterrows():
            img = find_image(row.get("image_path"))
            if img:
                f = extract_features(img)
                if f:
                    feature_rows.append(f)
        if feature_rows:
            feats = pd.DataFrame(feature_rows).mean().to_dict()

    if not feats:
        plt.text(0.5, 0.5, "No image feature data available", ha='center', va='center', fontsize=12)
        plt.axis('off')
        return fig

    series = pd.Series(feats)
    max_val = series.max()
    normalized = (series / max_val * 100) if max_val > 0 else series

    bars = plt.bar(normalized.index, normalized.values, color="#22c55e", edgecolor="#15803d", width=0.55)
    plt.title("Plot 2: Image Processing Results (Normalized)", fontsize=13, fontweight="bold", pad=12, color="#1e293b")
    plt.xlabel("Image Features", fontsize=11, fontweight="semibold")
    plt.ylabel("Normalized Feature Value (%)", fontsize=11, fontweight="semibold")
    plt.ylim(0, 115)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, (k, val) in zip(bars, series.items()):
        h = bar.get_height()
        plt.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#334155"
        )

    plt.tight_layout()
    return fig


def build_plot3_crop_heatmap(df):
    """
    PLOT 3: Crop × Disease Heatmap
    """
    fig = plt.figure(figsize=(11, 5.2), facecolor='#ffffff')
    if df.empty or "crop" not in df.columns:
        plt.text(0.5, 0.5, "No crop disease data available", ha='center', va='center', fontsize=12)
        plt.axis('off')
        return fig

    table = pd.crosstab(df["crop"], df["disease"])
    if table.empty:
        plt.text(0.5, 0.5, "No distribution data available", ha='center', va='center', fontsize=12)
        plt.axis('off')
        return fig

    sns.heatmap(table, annot=True, fmt="d", cmap="Greens", cbar=True, linewidths=0.6, linecolor="#e2e8f0")
    plt.title("Plot 3: Crop × Disease Heatmap", fontsize=13, fontweight="bold", pad=12, color="#1e293b")
    plt.xlabel("Disease", fontsize=11, fontweight="semibold")
    plt.ylabel("Crop", fontsize=11, fontweight="semibold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


# ================================================================
# 4. BASE64 & STREAM CONVERTERS
# ================================================================

def _fig_to_base64(fig):
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _fig_to_bytes(fig):
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def get_all_plots_base64(user_id=None, current_image_path=None, crop="Crop", disease="Detection", confidence=0.0):
    """
    Generates all in-memory base64 strings (Diagnostic Analysis + 3 Analytics Plots) for result page.
    """
    df = _get_user_dataframe(user_id)
    real_img = find_image(current_image_path) if current_image_path else None
    
    diag_fig = None
    if real_img:
        diag_fig = build_diagnostic_analysis_plot(real_img, crop, disease, confidence)
    diag_b64 = _fig_to_base64(diag_fig)

    fig1 = build_plot1_monthly_ratio(df)
    p1_b64 = _fig_to_base64(fig1)

    fig2 = build_plot2_image_features(real_img, df)
    p2_b64 = _fig_to_base64(fig2)

    fig3 = build_plot3_crop_heatmap(df)
    p3_b64 = _fig_to_base64(fig3)

    return {
        "diagnostic_b64": diag_b64,
        "plot1_b64": p1_b64,
        "plot2_b64": p2_b64,
        "plot3_b64": p3_b64
    }


def get_plot_bytes(plot_type, user_id=None, current_image_path=None, crop="Crop", disease="Detection", confidence=0.0):
    """
    Generates in-memory BytesIO for specific plot downloads.
    """
    df = _get_user_dataframe(user_id)
    real_img = find_image(current_image_path) if current_image_path else None
    
    if plot_type == "diagnostic":
        fig = build_diagnostic_analysis_plot(real_img, crop, disease, confidence) if real_img else None
    elif str(plot_type) == "1":
        fig = build_plot1_monthly_ratio(df)
    elif str(plot_type) == "2":
        fig = build_plot2_image_features(real_img, df)
    elif str(plot_type) == "3":
        fig = build_plot3_crop_heatmap(df)
    else:
        fig = build_plot1_monthly_ratio(df)
    return _fig_to_bytes(fig)


# ================================================================
# 5. CLI RUNNER
# ================================================================
def run_analytics(user_id=None, user_email=None):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    users_col = db["users"]

    query = {}
    target_user = None
    if user_id:
        try:
            target_user = users_col.find_one({"_id": ObjectId(user_id)})
            query = {"user_id": ObjectId(user_id)}
        except Exception:
            target_user = users_col.find_one({"_id": user_id})
            query = {"user_id": user_id}
    elif user_email:
        target_user = users_col.find_one({"email": user_email.strip()})
        if target_user:
            query = {"user_id": target_user["_id"]}

    if target_user:
        print(f"\nUser Name     : {target_user.get('name', 'User')}")
        print(f"User Email    : {target_user.get('email', 'N/A')}")

    records = list(collection.find(query))
    if not records:
        print(f"\nNo detection records found.")
        client.close()
        return

    df = pd.DataFrame(records)
    for column in ["crop", "disease", "confidence", "image_path", "timestamp"]:
        if column not in df.columns:
            df[column] = np.nan

    df["crop"] = df["crop"].fillna("Unknown").astype(str).str.strip()
    df["disease"] = df["disease"].fillna("Unknown").astype(str).str.strip()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["status"] = df["disease"].apply(determine_status)
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)

    monthly = pd.crosstab(df["month"], df["status"])
    if "Healthy" not in monthly.columns:
        monthly["Healthy"] = 0
    if "Diseased" not in monthly.columns:
        monthly["Diseased"] = 0
    monthly["Total"] = monthly["Healthy"] + monthly["Diseased"]
    monthly["Disease_Ratio"] = ((monthly["Diseased"] / monthly["Total"]) * 100).replace([np.inf, -np.inf], 0).fillna(0)
    monthly = monthly.sort_index()

    print("\n" + "=" * 70)
    print("PLOT 1: MONTH-WISE DISEASE RATIO")
    print("=" * 70)
    print(monthly[["Healthy", "Diseased", "Total", "Disease_Ratio"]].round(2).to_string())

    feature_rows = []
    for _, row in df.iterrows():
        img = find_image(row["image_path"])
        if img:
            f = extract_features(img)
            if f:
                feature_rows.append(f)
    features_df = pd.DataFrame(feature_rows)

    print("\n" + "=" * 70)
    print("IMAGE PROCESSING RESULTS")
    print("=" * 70)
    print(f"Total MongoDB records : {len(df)}")
    print(f"Images processed       : {len(feature_rows)}")
    if not features_df.empty:
        avg = features_df.mean()
        for k, v in avg.items():
            print(f"{k:<22}{v:.2f}")

    table = pd.crosstab(df["crop"], df["disease"])
    print("\n" + "=" * 70)
    print("PLOT 3: CROP x DISEASE FREQUENCY TABLE")
    print("=" * 70)
    print(table.to_string())

    client.close()
    print("\n[OK] Analytics completed in-memory.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", "--id", dest="user_id")
    parser.add_argument("--email", dest="user_email")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.user_id or args.user_email or args.all:
        target_id = None if args.all else args.user_id
        target_email = None if args.all else args.user_email
        run_analytics(user_id=target_id, user_email=target_email)
        return

    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        users = list(db["users"].find({}, {"password": 0}))
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        return

    if users:
        print("\n=== SELECT USER FOR ANALYTICS ===")
        for idx, u in enumerate(users, 1):
            print(f"  [{idx}] {u.get('name', 'User')} | {u.get('email', '')}")
        print(f"  [{len(users) + 1}] All Users")
        try:
            choice = input(f"Enter choice (1-{len(users) + 1}) [default: 1]: ").strip() or "1"
            idx = int(choice)
            if 1 <= idx <= len(users):
                run_analytics(user_id=users[idx - 1]["_id"])
            else:
                run_analytics()
        except Exception:
            run_analytics(user_id=users[0]["_id"])
    else:
        run_analytics()


if __name__ == "__main__":
    main()