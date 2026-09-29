from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from io import BytesIO
import cv2
import numpy as np
import edge_stego as E

app = Flask(__name__)

ALLOWED = {"png", "bmp", "jpg", "jpeg"}

def read_image(data):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("The uploaded file is not a valid image.")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/embed", methods=["POST"])
def embed():
    try:
        image_file = request.files.get("cover")
        message = request.form.get("message", "")
        password = request.form.get("password", "")

        if not image_file:
            return render_template("index.html", error="Please upload a cover image.")
        if not message:
            return render_template("index.html", error="Please enter a secret message.")
        if not password:
            return render_template("index.html", error="Please enter a password.")

        img = read_image(image_file.read())

        bits = E.capacity_bits(img)
        usable = max(bits // 8 - E.HEADER_LEN, 0)

        if len(message.encode()) > usable:
            return render_template(
                "index.html",
                error=f"Message is too large. This image can hold about {usable} bytes."
            )

        stego = E.embed(img, message.encode(), password)

        ok, encoded = cv2.imencode(".png", stego)
        if not ok:
            raise ValueError("Could not create the output PNG.")

        return send_file(
            BytesIO(encoded.tobytes()),
            mimetype="image/png",
            as_attachment=True,
            download_name="stego.png"
        )

    except Exception as e:
        return render_template("index.html", error=f"Embedding failed: {e}")

@app.route("/extract", methods=["POST"])
def extract():
    try:
        image_file = request.files.get("stego")
        password = request.form.get("extract_password", "")

        if not image_file:
            return render_template("index.html", error="Please upload a stego image.")
        if not password:
            return render_template("index.html", error="Please enter the password.")

        img = read_image(image_file.read())
        message = E.extract(img, password).decode(errors="replace")

        return render_template("index.html", extracted=message)

    except ValueError:
        return render_template(
            "index.html",
            error="Extraction failed: wrong password, corrupted image, or invalid stego image."
        )
    except Exception as e:
        return render_template("index.html", error=f"Extraction failed: {e}")

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
