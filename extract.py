"""
extract.py - RECEIVER side. Asks for the password; the message is revealed only if it is correct.

Run:  python extract.py stego.png
AES-GCM verifies the password/integrity through its authentication tag, so a wrong
password (or a tampered image) never produces any plaintext - it is simply rejected.
"""
import sys
import time
from getpass import getpass

import cv2

import edge_stego as E

MAX_ATTEMPTS = 3


def step(n, title):
    print(f"\n=== Step {n}: {title} " + "=" * max(4, 50 - len(title)))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python extract.py <stego_image.png>")

    #1
    step(1, "Load stego image")
    stego = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
    if stego is None:
        raise SystemExit(f"Cannot read image: {sys.argv[1]}")
    if stego.ndim == 3 and stego.shape[2] == 4:
        stego = stego[:, :, :3]
    H, W = stego.shape[:2]
    print(f"image file            : {sys.argv[1]}")
    print(f"image size            : {W} x {H}   ({'colour' if stego.ndim == 3 else 'grayscale'})")
    print("(only the stego image is needed - the original cover is NOT required)")

    #2
    step(2, "Recompute the edge map from the stego image")
    t = time.perf_counter()
    edges = E.edge_map(stego)                # uses only the upper 7 bit-planes
    edge_ms = (time.perf_counter() - t) * 1000
    bits = E.capacity_bits(stego)
    cv2.imwrite("extract_edge_map.png", (edges * 255).astype("uint8"))
    print(f"edge pixels           : {int(edges.sum())} of {H*W} ({100*edges.mean():.1f}% of the image)")
    print(f"edge detection time   : {edge_ms:.1f} ms")
    print(f"embedding sites       : {bits} bits ({bits/(H*W):.3f} bits/pixel)")
    print("saved extract_edge_map.png (same map the sender used)")

    #3
    step(3, "Password check and decryption")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        pw = getpass(f"Enter password (attempt {attempt}/{MAX_ATTEMPTS}): ")
        t = time.perf_counter()
        try:
            message = E.extract(stego, pw)
        except ValueError as err:
            if "No valid payload" in str(err):
                raise SystemExit("\nNo hidden message found: this is not a stego image made by embed.py, or it was altered/re-saved.")
            print("Access denied: wrong password, or the image was modified.\n")
            continue
        ms = (time.perf_counter() - t) * 1000
        #4
        step(4, "Password correct - hidden message recovered")
        print(f"authentication tag verified (AES-256-GCM), extraction + decryption: {ms:.1f} ms")
        print(f"message length        : {len(message)} bytes")
        print(f"hidden message        : {message.decode(errors='replace')}")
        return
    raise SystemExit(f"\nToo many failed attempts ({MAX_ATTEMPTS}). Exiting.")


if __name__ == "__main__":
    main()