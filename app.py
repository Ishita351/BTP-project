from flask import Flask, request, render_template, send_file
from io import BytesIO
import base64
import cv2
import numpy as np
import edge_stego as E

app = Flask(__name__)


def image_from_upload(data):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("The uploaded file is not a valid image.")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def png_data_uri(img):
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode image.")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


def format_size(img):
    h, w = img.shape[:2]
    kind = "colour" if img.ndim == 3 else "grayscale"
    return w, h, kind


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

        cover = image_from_upload(image_file.read())

        # Step 1: cover image + edge map metadata
        H, W = cover.shape[:2]

        # Measure Step 1 edge-map computation time.
        import time
        edge_start = time.perf_counter()
        edges = E.edge_map(cover)
        edge_ms = (time.perf_counter() - edge_start) * 1000

        edge_pixels = int(edges.sum())
        total_pixels = H * W
        edge_percent = 100 * edges.mean()
        bits = E.capacity_bits(cover)
        usable = max(bits // 8 - E.HEADER_LEN, 0)

        if len(message.encode()) > usable:
            return render_template(
                "index.html",
                error=f"Message is {len(message.encode())} bytes but only {usable} bytes fit. Use a larger/more detailed image."
            )

        # Step 3: encrypt + embed
        start = time.perf_counter()
        stego = E.embed(cover, message.encode(), password)
        embed_ms = (time.perf_counter() - start) * 1000

        changed = int(np.count_nonzero(cover != stego))

        # Step 4: verify identical edge maps
        stego_read = stego
        same_edges = np.array_equal(E.edge_map(cover), E.edge_map(stego_read))

        # Step 5: montage + quality metrics
        diff = np.abs(cover.astype(int) - stego.astype(int)).astype(np.uint8)
        diff = (diff.max(axis=2) if diff.ndim == 3 else diff) * 255

        def to_bgr(g):
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        c3 = cover if cover.ndim == 3 else to_bgr(cover)
        s3 = stego if stego.ndim == 3 else to_bgr(stego)
        edge_img = (edges * 255).astype(np.uint8)
        montage = np.hstack([c3, to_bgr(edge_img), s3, to_bgr(diff)])

        mse_value = E.mse(cover, stego)
        psnr_value = E.psnr(cover, stego)
        ssim_value = E.ssim(cover, stego)

        # Return all demonstration information to the browser.
        result = {
            "W": W,
            "H": H,
            "kind": "colour" if cover.ndim == 3 else "grayscale",
            "edge_pixels": edge_pixels,
            "total_pixels": total_pixels,
            "edge_percent": edge_percent,
            "edge_ms": edge_ms,
            "capacity_bits": bits,
            "capacity_bytes": bits // 8,
            "usable": usable,
            "message_length": len(message.encode()),
            "embed_ms": embed_ms,
            "changed": changed,
            "same_edges": same_edges,
            "mse": mse_value,
            "psnr": psnr_value,
            "ssim": ssim_value,
            "edge_map": png_data_uri(edge_img),
            "montage": png_data_uri(montage),
            "stego": png_data_uri(stego),
        }

        return render_template("index.html", embed_result=result)

    except Exception as e:
        return render_template("index.html", error=f"Embedding failed: {e}")


@app.route("/download-stego", methods=["POST"])
def download_stego():
    try:
        image_b64 = request.form.get("stego_data", "")
        if not image_b64.startswith("data:image/png;base64,"):
            raise ValueError("Invalid stego image data.")

        raw = base64.b64decode(image_b64.split(",", 1)[1])
        return send_file(
            BytesIO(raw),
            mimetype="image/png",
            as_attachment=True,
            download_name="stego.png"
        )
    except Exception as e:
        return render_template("index.html", error=f"Download failed: {e}")


@app.route("/extract", methods=["POST"])
def extract():
    try:
        image_file = request.files.get("stego")
        password = request.form.get("extract_password", "")

        if not image_file:
            return render_template("index.html", error="Please upload a stego image.")
        if not password:
            return render_template("index.html", error="Please enter the password.")

        stego = image_from_upload(image_file.read())
        H, W = stego.shape[:2]

        # Extraction Step 1/2: recompute edge map and metadata
        import time
        start = time.perf_counter()
        edges = E.edge_map(stego)
        edge_ms = (time.perf_counter() - start) * 1000

        edge_pixels = int(edges.sum())
        total_pixels = H * W
        edge_percent = 100 * edges.mean()
        bits = E.capacity_bits(stego)

        # Password check + extraction
        start = time.perf_counter()
        message = E.extract(stego, password)
        extract_ms = (time.perf_counter() - start) * 1000

        result = {
            "W": W,
            "H": H,
            "kind": "colour" if stego.ndim == 3 else "grayscale",
            "edge_pixels": edge_pixels,
            "total_pixels": total_pixels,
            "edge_percent": edge_percent,
            "edge_ms": edge_ms,
            "bits": bits,
            "bits_per_pixel": bits / total_pixels,
            "edge_map": png_data_uri((edges * 255).astype("uint8")),
            "extract_ms": extract_ms,
            "message_length": len(message),
            "message": message.decode(errors="replace"),
        }

        return render_template("index.html", extract_result=result)

    except ValueError as e:
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
