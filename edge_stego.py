"""
edge_stego.py - Lightweight edge-adaptive image steganography with AES-256-GCM.

Pipeline
  embed:   password -> scrypt -> (AES key, pixel-order seed)
           message  -> AES-256-GCM -> payload = len | salt | nonce | tag | ciphertext
           cover    -> clear LSB plane -> light Gaussian blur -> gradient magnitude
                    -> 3-class Otsu (Tl, Th) -> Canny/hysteresis -> edge map
           payload bits -> LSBs of edge pixels (key-shuffled order)
  extract: recompute the SAME edge map from the stego image with LSBs cleared,
           read bits in the same order, parse header, decrypt + verify tag.

Key idea: the edge map is computed only from the upper 7 bit-planes, so embedding
(which touches only the LSB plane) cannot change it -> sender and receiver agree.

Usage
  python edge_stego.py embed   cover.png stego.png "secret text" --password pw
  python edge_stego.py extract stego.png --password pw
  python edge_stego.py metrics cover.png stego.png
  python edge_stego.py capacity cover.png
Always use a LOSSLESS format (PNG/BMP) for the stego image.
"""
import argparse
import struct
import time

import cv2
import numpy as np
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from Crypto.Random import get_random_bytes

SALT_LEN, NONCE_LEN, TAG_LEN, LEN_FIELD = 16, 12, 16, 4
HEADER_LEN = LEN_FIELD + SALT_LEN + NONCE_LEN + TAG_LEN  # 48 bytes overhead


#crypto
def derive(password: str, salt: bytes):
    """scrypt -> 32-byte AES key + 8-byte seed for the pixel-order shuffle."""
    material = scrypt(password.encode(), salt, key_len=40, N=2**14, r=8, p=1)
    return material[:32], int.from_bytes(material[32:], "big")


def encrypt(message: bytes, password: str) -> bytes:
    salt, nonce = get_random_bytes(SALT_LEN), get_random_bytes(NONCE_LEN)
    key, _ = derive(password, salt)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(message)
    return struct.pack(">I", len(ct)) + salt + nonce + tag + ct


#edge map
def otsu3(hist: np.ndarray):
    """3-class Otsu on a 256-bin histogram (two thresholds t1<t2), prefix-sum form."""
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    i = np.arange(256, dtype=np.float64)
    w = np.cumsum(p)            # class probability prefix
    m = np.cumsum(p * i)        # first-moment prefix
    mt = m[-1]
    best, bt = -1.0, (85, 170)
    for t1 in range(1, 254):
        w1, m1 = w[t1], m[t1]
        for t2 in range(t1 + 1, 255):
            w2 = w[t2] - w1
            w3 = 1.0 - w[t2]
            if w1 < 1e-9 or w2 < 1e-9 or w3 < 1e-9:
                continue
            mu1 = m1 / w1
            mu2 = (m[t2] - m1) / w2
            mu3 = (mt - m[t2]) / w3
            var = w1 * (mu1 - mt) ** 2 + w2 * (mu2 - mt) ** 2 + w3 * (mu3 - mt) ** 2
            if var > best:
                best, bt = var, (t1, t2)
    return bt


def edge_map(img: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Boolean HxW edge map computed ONLY from the upper 7 bit-planes."""
    msb = img & 0xFE                                   # clear LSB plane
    gray = msb if msb.ndim == 2 else cv2.cvtColor(msb, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    scale = max(float(mag.max()), 1e-6) / 255.0
    hist = np.bincount(np.clip(mag / scale, 0, 255).astype(np.uint8).ravel(), minlength=256)
    t_low, t_high = otsu3(hist)
    edges = cv2.Canny(smooth, t_low * scale, t_high * scale, L2gradient=True)
    return edges > 0


def capacity_bits(img: np.ndarray) -> int:
    ch = 1 if img.ndim == 2 else img.shape[2]
    return int(edge_map(img).sum()) * ch


#embed/extract
def all_pixels(img: np.ndarray) -> np.ndarray:
    """Baseline site selector: every pixel is an embedding site."""
    return np.ones(img.shape[:2], dtype=bool)


def sobel_fixed(img: np.ndarray, thresh: float = 100.0) -> np.ndarray:
    """Baseline site selector: Sobel magnitude > fixed threshold (LSB plane cleared)."""
    msb = img & 0xFE
    gray = msb if msb.ndim == 2 else cv2.cvtColor(msb, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy) > thresh


def embed(cover: np.ndarray, message: bytes, password: str,
          edge_fn=None, shuffle: bool = True) -> np.ndarray:
    payload = encrypt(message, password)
    salt = payload[LEN_FIELD:LEN_FIELD + SALT_LEN]
    _, seed = derive(password, salt)
    # NOTE: the shuffle seed depends on the salt, which lives in the payload.
    # To let the receiver find the salt before knowing the order, the first
    # HEADER bits (len+salt) are stored in UNSHUFFLED order, the rest shuffled.
    em = (edge_fn or edge_map)(cover)
    ch = 1 if cover.ndim == 2 else cover.shape[2]
    idx = np.flatnonzero(em.ravel())
    slots = (idx[:, None] * ch + np.arange(ch)[None, :]).ravel()
    head_bits = (LEN_FIELD + SALT_LEN) * 8
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    if bits.size > slots.size:
        raise ValueError(f"Payload needs {bits.size} bits but only {slots.size} edge slots available.")
    order = np.concatenate([slots[:head_bits],
                            np.random.default_rng(seed).permutation(slots[head_bits:])]) if shuffle else slots
    stego = cover.copy().reshape(-1)
    pos = order[:bits.size]
    stego[pos] = (stego[pos] & 0xFE) | bits
    return stego.reshape(cover.shape)


def extract(stego: np.ndarray, password: str, edge_fn=None, shuffle: bool = True) -> bytes:
    em = (edge_fn or edge_map)(stego)
    ch = 1 if stego.ndim == 2 else stego.shape[2]
    idx = np.flatnonzero(em.ravel())
    slots = (idx[:, None] * ch + np.arange(ch)[None, :]).ravel()
    flat = stego.reshape(-1)
    head_bits = (LEN_FIELD + SALT_LEN) * 8
    head = np.packbits(flat[slots[:head_bits]] & 1).tobytes()
    ct_len = struct.unpack(">I", head[:LEN_FIELD])[0]
    total_bits = (HEADER_LEN + ct_len) * 8
    if ct_len > slots.size // 8 or total_bits > slots.size:
        raise ValueError("No valid payload found (wrong image or corrupted).")
    salt = head[LEN_FIELD:]
    key, seed = derive(password, salt)
    order = np.concatenate([slots[:head_bits],
                            np.random.default_rng(seed).permutation(slots[head_bits:])]) if shuffle else slots
    raw = np.packbits(flat[order[:total_bits]] & 1).tobytes()
    nonce = raw[LEN_FIELD + SALT_LEN: LEN_FIELD + SALT_LEN + NONCE_LEN]
    tag = raw[LEN_FIELD + SALT_LEN + NONCE_LEN: HEADER_LEN]
    ct = raw[HEADER_LEN:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ct, tag)   # raises ValueError if wrong key / tampered


#metrics
def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=None if a.ndim == 2 else 2, data_range=255))


def sequential_lsb(cover, message, password):
    """Baseline: same encrypted payload, plain sequential LSB from pixel 0."""
    bits = np.unpackbits(np.frombuffer(encrypt(message, password), dtype=np.uint8))
    s = cover.copy().reshape(-1)
    s[:bits.size] = (s[:bits.size] & 0xFE) | bits
    return s.reshape(cover.shape)


#CLI
def _read(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Cannot read {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed"); e.add_argument("cover"); e.add_argument("stego"); e.add_argument("message")
    e.add_argument("--password", required=True)
    x = sub.add_parser("extract"); x.add_argument("stego"); x.add_argument("--password", required=True)
    m = sub.add_parser("metrics"); m.add_argument("cover"); m.add_argument("stego")
    c = sub.add_parser("capacity"); c.add_argument("cover")
    a = ap.parse_args()

    if a.cmd == "embed":
        cover = _read(a.cover)
        t = time.perf_counter()
        stego = embed(cover, a.message.encode(), a.password)
        dt = time.perf_counter() - t
        if not a.stego.lower().endswith((".png", ".bmp")):
            raise SystemExit("Use a lossless output format (.png or .bmp).")
        cv2.imwrite(a.stego, stego)
        print(f"embedded in {dt*1000:.1f} ms -> {a.stego}")
    elif a.cmd == "extract":
        t = time.perf_counter()
        try:
            msg = extract(_read(a.stego), a.password)
        except ValueError as err:
            raise SystemExit(f"Extraction failed: {err}")
        print(msg.decode(errors="replace"))
        print(f"[extracted in {(time.perf_counter()-t)*1000:.1f} ms]")
    elif a.cmd == "metrics":
        A, B = _read(a.cover), _read(a.stego)
        print(f"MSE={mse(A,B):.6f}  PSNR={psnr(A,B):.3f} dB  SSIM={ssim(A,B):.6f}")
    elif a.cmd == "capacity":
        img = _read(a.cover)
        bits = capacity_bits(img)
        print(f"edge slots: {bits} bits = {bits//8} bytes total, "
              f"{max(bits//8 - HEADER_LEN, 0)} bytes of usable message, "
              f"{bits / (img.shape[0]*img.shape[1]):.3f} bits/pixel")


if __name__ == "__main__":
    main()
