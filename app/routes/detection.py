
import os
import io
import csv
import time

from flask import (
    Blueprint,
    request,
    render_template,
    redirect,
    url_for,
    flash,
    session,
    current_app,
    send_file,
    Response,
)
from werkzeug.utils import secure_filename

from app.models.detection import Detection

detection_bp = Blueprint("detection", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@detection_bp.route("/detect", methods=["GET", "POST"])
def detect():
    if "user_id" not in session:
        flash("Please login to detect crop diseases.", "error")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        if "image" not in request.files:
            flash("No image part", "error")
            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":
            flash("No selected image", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Please upload a PNG, JPG, or JPEG image.", "error")
            return redirect(url_for("detection.detect"))

        filename = secure_filename(file.filename)
        unique_filename = f"{int(time.time())}_{filename}"

        upload_folder = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(upload_folder, unique_filename)

        try:
            file.save(filepath)
        except Exception:
            current_app.logger.exception("Failed to save uploaded image")
            flash("Could not save the uploaded image.", "error")
            return redirect(url_for("detection.detect"))

        # Import and run prediction only when an image is submitted.
        try:
            current_app.logger.info("Starting crop disease prediction")

            from app.ml.inference import predict_disease

            current_app.logger.info("Prediction module imported")

            crop, disease, confidence, error_msg = predict_disease(filepath)

            current_app.logger.info("Prediction function completed")

        except Exception:
            current_app.logger.exception("Crop disease prediction failed")

            flash(
                "Prediction failed due to a server error. "
                "Please check the server logs.",
                "error",
            )
            return redirect(url_for("detection.detect"))

        if error_msg:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except OSError:
                current_app.logger.exception(
                    "Could not remove invalid uploaded image"
                )

            flash(f"⚠️ {error_msg}", "error")
            return redirect(url_for("detection.detect"))

        relative_path = f"uploads/{unique_filename}"

        try:
            detection_id = Detection.save_detection(
                session["user_id"],
                crop,
                disease,
                confidence,
                relative_path,
            )
        except Exception:
            current_app.logger.exception("Failed to save detection record")
            flash(
                "The prediction completed, but the result could not be saved.",
                "error",
            )
            return redirect(url_for("detection.detect"))

        return redirect(
            url_for("detection.result", detection_id=detection_id)
        )

    return render_template("detection.html")


@detection_bp.route("/result/<detection_id>")
def result(detection_id):
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    detection_record = Detection.get_detection(detection_id)

    if (
        not detection_record
        or str(detection_record.get("user_id")) != str(session["user_id"])
    ):
        flash("Result not found or unauthorized access.", "error")
        return redirect(url_for("main.dashboard"))

    plots = {}

    try:
        # Import analysis code only when the result page is requested.
        from app.ml.analisis import get_all_plots_base64

        img_full_path = os.path.join(
            current_app.root_path,
            "static",
            detection_record.get("image_path", ""),
        )

        plots = get_all_plots_base64(
            user_id=session["user_id"],
            current_image_path=img_full_path,
            crop=detection_record.get("crop", "Crop"),
            disease=detection_record.get("disease", "Detection"),
            confidence=detection_record.get("confidence", 0.0),
        )

    except Exception:
        current_app.logger.exception(
            "Error generating in-memory plots"
        )

    return render_template(
        "result.html",
        detection=detection_record,
        plots=plots,
    )


@detection_bp.route("/download-plot/<detection_id>/<plot_type>")
@detection_bp.route("/download-plot/<detection_id>")
def download_plot(detection_id, plot_type="diagnostic"):
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    detection_record = Detection.get_detection(detection_id)

    if (
        not detection_record
        or str(detection_record.get("user_id")) != str(session["user_id"])
    ):
        flash("Unauthorized access.", "error")
        return redirect(url_for("main.dashboard"))

    from app.ml.analisis import get_plot_bytes

    img_full_path = os.path.join(
        current_app.root_path,
        "static",
        detection_record.get("image_path", ""),
    )

    buf = get_plot_bytes(
        plot_type=plot_type,
        user_id=session["user_id"],
        current_image_path=img_full_path,
        crop=detection_record.get("crop", "Crop"),
        disease=detection_record.get("disease", "Detection"),
        confidence=detection_record.get("confidence", 0.0),
    )

    if buf is None:
        flash("Unable to generate plot for download.", "error")
        return redirect(
            url_for("detection.result", detection_id=detection_id)
        )

    plot_names = {
        "diagnostic": "Diagnostic_Feature_Analysis",
        "1": "Monthly_Disease_Ratio",
        "2": "Image_Processing_Results",
        "3": "Crop_Disease_Heatmap",
    }

    name = plot_names.get(str(plot_type), f"Plot_{plot_type}")

    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name=f"{name}.png",
    )


@detection_bp.route("/download-csv/<detection_id>")
def download_csv(detection_id):
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    detection_record = Detection.get_detection(detection_id)

    if (
        not detection_record
        or str(detection_record.get("user_id")) != str(session["user_id"])
    ):
        flash("Unauthorized access.", "error")
        return redirect(url_for("main.dashboard"))

    from app.ml.analisis import extract_features

    img_full_path = os.path.join(
        current_app.root_path,
        "static",
        detection_record.get("image_path", ""),
    )

    feats = extract_features(img_full_path) or {}

    si = io.StringIO()
    cw = csv.writer(si)

    headers = [
        "Detection_ID",
        "Timestamp",
        "Crop",
        "Disease",
        "Confidence_Score",
        "Green_Pixel_Ratio",
        "Mean_Hue",
        "Mean_Saturation",
        "Mean_Brightness",
        "Circularity",
        "Aspect_Ratio",
        "Edge_Density",
    ]
    cw.writerow(headers)

    row = [
        str(detection_record.get("_id")),
        str(detection_record.get("timestamp", "")),
        str(detection_record.get("crop", "")),
        str(detection_record.get("disease", "")),
        str(detection_record.get("confidence", "")),
        f"{feats.get('Green Pixel Ratio', 0.0):.2f}",
        f"{feats.get('Mean Hue', 0.0):.2f}",
        f"{feats.get('Mean Saturation', 0.0):.2f}",
        f"{feats.get('Mean Brightness', 0.0):.2f}",
        f"{feats.get('Circularity', 0.0):.2f}",
        f"{feats.get('Aspect Ratio', 0.0):.2f}",
        f"{feats.get('Edge Density', 0.0):.2f}",
    ]
    cw.writerow(row)

    safe_crop = str(
        detection_record.get("crop", "Crop")
    ).replace(" ", "_")

    safe_disease = str(
        detection_record.get("disease", "Disease")
    ).replace(" ", "_")

    output = si.getvalue()
    si.close()

    return Response(
        output,
        mimetype="text/csv",
        headers={
            "Content-Disposition": (
                f"attachment;filename="
                f"Feature_Data_{safe_crop}_{safe_disease}.csv"
            )
        },
    )
