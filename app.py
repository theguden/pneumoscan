import os
from io import BytesIO
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from utils.db import (
    init_db,
    save_analysis,
    save_brain_analysis,
    get_unified_history,
    clear_unified_history
)

from model.predict import analyze_image
from model.predict_brain import analyze_brain_image


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_db()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    image_url = None
    result = None
    message = None

    if request.method == "POST":
        if "image" not in request.files:
            message = "Файл не был выбран."
            return render_template("analyze.html", message=message)

        file = request.files["image"]

        if file.filename == "":
            message = "Пожалуйста, выберите изображение."
            return render_template("analyze.html", message=message)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            image_url = f"/static/uploads/{filename}"
            mode = request.form.get("mode", "xray")

            # выбор модели
            if mode == "brain":
                result = analyze_brain_image(save_path)
            else:
                result = analyze_image(save_path)

            if "error" in result:
                return render_template("analyze.html", image_url=image_url, result=result)

            # сохранение
            if mode == "brain":
                save_brain_analysis(
                    filename=filename,
                    result_class=result["class_name"],
                    glioma_probability=result["glioma"],
                    meningioma_probability=result["meningioma"],
                    pituitary_probability=result["pituitary"],
                    normal_probability=result["normal"]
                )
            else:
                save_analysis(
                    filename=filename,
                    result_class=result["class_name"],
                    normal_probability=result["normal"],
                    viral_probability=result["viral"],
                    bacterial_probability=result["bacterial"],
                    covid_probability=result["covid"]
                )

            message = "Изображение успешно обработано."

        else:
            message = "Разрешены только PNG, JPG и JPEG."

    return render_template("analyze.html", image_url=image_url, result=result, message=message)


@app.route("/history")
def history():
    selected_type = request.args.get("type", "all")
    history_list = get_unified_history()

    if selected_type == "xray":
        history_list = [item for item in history_list if item["analysis_type"] == "Рентген"]
    elif selected_type == "brain":
        history_list = [item for item in history_list if item["analysis_type"] == "МРТ мозга"]

    return render_template(
        "history.html",
        history_list=history_list,
        selected_type=selected_type
    )


@app.route("/history/clear_all", methods=["POST"])
def clear_all_history():
    clear_unified_history()
    return redirect(url_for("history"))


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/download_result_pdf", methods=["POST"])
def download_result_pdf():
    class_name = request.form.get("class_name", "")
    confidence = request.form.get("confidence", "")

    normal = request.form.get("normal", "")
    viral = request.form.get("viral", "")
    bacterial = request.form.get("bacterial", "")
    covid = request.form.get("covid", "")

    glioma = request.form.get("glioma", "")
    meningioma = request.form.get("meningioma", "")
    pituitary = request.form.get("pituitary", "")

    buffer = BytesIO()

    pdfmetrics.registerFont(TTFont("Arial", "static/fonts/arial.ttf"))

    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFont("Arial", 16)
    pdf.drawString(50, height - 60, "PneumoScan - отчет анализа")

    pdf.setFont("Arial", 11)
    pdf.drawString(50, height - 95, f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    pdf.drawString(50, height - 125, f"Класс: {class_name}")

    if confidence:
        pdf.drawString(50, height - 150, f"Уверенность: {confidence}")

    y = height - 190

    if glioma:
        pdf.drawString(70, y, f"Глиома: {glioma}")
        y -= 25
        pdf.drawString(70, y, f"Менингиома: {meningioma}")
        y -= 25
        pdf.drawString(70, y, f"Гипофиз: {pituitary}")
        y -= 25
        pdf.drawString(70, y, f"Норма: {normal}")
    else:
        pdf.drawString(70, y, f"Норма: {normal}")
        y -= 25
        pdf.drawString(70, y, f"Вирусная: {viral}")
        y -= 25
        pdf.drawString(70, y, f"Бактериальная: {bacterial}")
        y -= 25
        pdf.drawString(70, y, f"COVID-19: {covid}")

    pdf.save()
    buffer.seek(0)

    return Response(
        buffer,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment;filename=result.pdf"}
    )


if __name__ == "__main__":
    app.run(debug=True)