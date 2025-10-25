import os
import logging
from yolo_detector import YoloPersonDetector
from retinaface import RetinaFace
from arcface import ArcFaceONNX

# --- Basic Setup ---
logger = logging.getLogger(__name__)
SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVICES_DIR)

print("Triton-based models loading starting")
try:
    logger.info("Loading Triton-based Face Recognition models...")

    # ---------------- Triton YOLO Detector ----------------
    triton_yolo_url = "provider.rtx4090.wyo.eg.akash.pub:30609/"
    yolo_model_name = "yolo_person_detection"
    detector = YoloPersonDetector(triton_url=triton_yolo_url, model_name=yolo_model_name)

    # ---------------- Triton RetinaFace + ArcFace ----------------
    triton_buffalo_url = "provider.rtx4090.wyo.eg.akash.pub:30609/"
    buffalo_face_model = "buffalo_face_detection"
    buffalo_embed_model = "buffalo_face_embedding"

    face_detector = RetinaFace(triton_url=triton_buffalo_url, model_name=buffalo_face_model)
    face_detector.prepare(ctx_id=0, input_size=(640, 640))

    face_embedder = ArcFaceONNX(url=triton_buffalo_url, model_name=buffalo_embed_model)
    face_embedder.prepare(ctx_id=0, input_size=(112, 112))

    logger.info("✅ Triton-based YOLO, RetinaFace, and ArcFace models loaded successfully.")
except Exception as e:
    logger.critical(f"❌ FAILED to load Triton-based Face Recognition models: {e}", exc_info=True)
    raise e

print("Triton-based models loading ending")
