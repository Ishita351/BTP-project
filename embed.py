"""
embed.py - SENDER side. Encrypts a message (AES-256-GCM) and hides it in an image.
Shows every step and saves demo_cover.png, <stego>, demo_montage.png.

Run:  python embed.py my_photo.png stego.png
      (it then asks for the message and the password)
Output must be a lossless format (.png or .bmp).
"""
import sys
import time
from getpass import getpass

import cv2
import numpy as np

import edge_stego as E


def step(n, title):
    print(f"\n=== Step {n}: {title} " + "=" * max(4, 50 - len(title)))


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python embed.py <cover_image> <output_stego.png>")
    cover_path, out_path = sys.argv[1], sys.argv[2]
    if not out_path.lower().endswith((".png", ".bmp")):
        raise SystemExit("Output must be lossless: use .png or .bmp (JPEG would destroy the hidden data).")

    #1
    step(1, "Cover image and edge map")
    cover = cv2.imread(cover_path, cv2.IMREAD_UNCHANGED)
    if cover is None:
        raise SystemExit(f"Cannot read image: {cover_path}")
    if cover.ndim == 3 and cover.shape[2] == 4:
        cover = cover[:, :, :3]
    cv2.imwrite("demo_cover.png", cover)
    H, W = cover.shape[:2]
    t = time.perf_counter()
    edges = E.edge_map(cover)
    edge_ms = (time.perf_counter() - t) * 1000
    bits = E.capacity_bits(cover)
    usable = max(bits // 8 - E.HEADER_LEN, 0)
    print(f"image file            : {cover_path}")
    print(f"image size            : {W} x {H}   ({'colour' if cover.ndim == 3 else 'grayscale'})")
    print(f"edge pixels           : {int(edges.sum())} of {H*W} ({100*edges.mean():.1f}% of the image)")
    print(f"edge detection time   : {edge_ms:.1f} ms")
    print(f"capacity              : {bits} bits = {bits//8} bytes  ({bits/(H*W):.3f} bits/pixel)")
    print(f"fixed crypto overhead : {E.HEADER_LEN} bytes (length + salt + nonce + GCM tag)")
    print(f"usable message size   : {usable} bytes")
    if usable == 0:
        raise SystemExit("This image has too few edge pixels to hide anything. Try a more detailed image.")

    #2
    step(2, "Enter message and password")
    message = input("Enter secret message : ").encode()
    if not message:
        raise SystemExit("Empty message.")
    if len(message) > usable:
        raise SystemExit(f"Message is {len(message)} bytes but only {usable} bytes fit. Shorten it or use another image.")
    pw = getpass("Enter password       : ")
    if not pw:
        raise SystemExit("Password cannot be empty.")
    if getpass("Confirm password     : ") != pw:
        raise SystemExit("Passwords do not match. Nothing was embedded.")
    print(f"message length        : {len(message)} bytes")

    #3
    step(3, "Encrypt with AES-256-GCM and embed in edge pixels")
    t = time.perf_counter()
    stego = E.embed(cover, message, pw)
    embed_ms = (time.perf_counter() - t) * 1000
    cv2.imwrite(out_path, stego)                       # PNG/BMP = lossless
    changed = int(np.count_nonzero(cover != stego))
    print(f"saved {out_path} (lossless)")
    print(f"total embed time      : {embed_ms:.1f} ms (includes key derivation + encryption)")
    print(f"pixel values changed  : {changed} (each by at most 1 grey level)")

    #4
    step(4, "Edge map is identical before and after embedding")
    stego_read = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
    same = np.array_equal(E.edge_map(cover), E.edge_map(stego_read))
    print(f"edge_map(cover) == edge_map(stego) : {same}")
    print("(this is why the receiver can extract without the original image)")

    #5
    step(5, "Visual comparison")
    diff = np.abs(cover.astype(int) - stego.astype(int)).astype(np.uint8)
    diff = (diff.max(axis=2) if diff.ndim == 3 else diff) * 255
    to_bgr = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    c3 = cover if cover.ndim == 3 else to_bgr(cover)
    s3 = stego if stego.ndim == 3 else to_bgr(stego)
    montage = np.hstack([c3, to_bgr((edges * 255).astype(np.uint8)), s3, to_bgr(diff.astype(np.uint8))])
    cv2.imwrite("demo_montage.png", montage)
    print("saved demo_cover.png   : copy of the cover image")
    print("saved demo_montage.png : cover | edge map | stego | changed pixels (white)")

    print("\nQuality of THIS one image (a demonstration, not an evaluation result):")
    print(f"  MSE={E.mse(cover, stego):.6f}   PSNR={E.psnr(cover, stego):.2f} dB   SSIM={E.ssim(cover, stego):.6f}")
    print(f"\nDone. Send {out_path} WITHOUT re-saving, resizing or sending as JPEG. Share the password separately.")


if __name__ == "__main__":
    main()