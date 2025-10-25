# utils/frame_extractor.py
import cv2
import numpy as np
import logging
import time
from typing import List
logger = logging.getLogger(__name__)
def extract_frames_from_path(video_path: str) -> List[np.ndarray]:
    """
    Extracts frames using OpenCV. It first attempts a GPU-accelerated backend
    and falls back to the standard CPU backend if the first attempt fails.
    """
    start_time = time.perf_counter()
    frames = []
    cap = None
    method_used = "OpenCV GPU (Direct Attempt)"
    logger.info(f"Attempting extraction with: {method_used}")
    # --- ROBUSTNESS FIX: ATTEMPT GPU-HINTED, FALLBACK TO CPU ---
    # First, try using the FFMPEG backend which can leverage hardware.
    try:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.warning("OpenCV (GPU hint) could not open video. Falling back to CPU.")
            cap = None # Force fallback
    except Exception as e:
        logger.warning(f"OpenCV (GPU hint) raised an exception: {e}. Falling back to CPU.")
        cap = None # Force fallback
    # If the first attempt failed (cap is None), use the default CPU backend.
    if cap is None:
        method_used = "OpenCV CPU (Fallback)"
        logger.info(f"Attempting extraction with: {method_used}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("OpenCV could not open video with any method.")
            raise ValueError("OpenCV could not open video file.")
    # --- END FIX ---
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("Frame extraction returned no frames.")
    duration = time.perf_counter() - start_time
    fps_processed = len(frames) / duration if duration > 0 else 0
    summary = (
        f"\n--- Frame Extraction Summary ---\n"
        f"Method: {method_used}\n"
        f"Time: {duration:.4f}s | Frames: {len(frames)} | Speed: {fps_processed:.1f} FPS\n"
        f"--------------------------------\n"
    )
    print(summary)
    return frames
