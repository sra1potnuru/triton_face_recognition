import cv2
import numpy as np
import pickle
import json
import pytz
import os
import time
import base64
import asyncio
import torch
from collections import defaultdict, Counter
from face import Face
from triton_model_loading import detector, face_detector, face_embedder
import faiss

class FaceReIDPipeline:
    def __init__(self, sim_threshold=0.40):
        # Config
        self.ist = pytz.timezone("Asia/Kolkata")
        self.sim_threshold = sim_threshold

        # ---------------- Use Triton-loaded models ----------------
        self.detector = detector
        self.face_detector = face_detector
        self.face_embedder = face_embedder

        # ---------------- Load FAISS index ----------------
        self.db_path = os.path.join(os.path.dirname(__file__), "face_database_gudivada_triton.pkl")
        self.names, self.index = self._load_faiss_index(self.db_path)

        # ---------------- State holders ----------------
        self.id_name_map = {}
        self.track_best_crop = {}
        self.timing = defaultdict(float)

    def _load_faiss_index(self, db_path):
        with open(db_path, "rb") as f:
            face_db_list = pickle.load(f)
        names, embeddings = zip(*face_db_list)
        names = np.array(names)
        embeddings = np.stack(embeddings).astype("float32")

        N, dim = len(embeddings), embeddings.shape[1]
        nlist = max(1, min(N, int(4 * np.sqrt(N))))
        nprobe = max(1, nlist // 4)

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print("Training FAISS index...")
        index.train(embeddings)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        index.nprobe = nprobe

        print(f"--- FAISS Config: N={N}, dim={dim}, nlist={nlist}, nprobe={nprobe} ---")
        return names, index

    def _process_crop_sync(self, tid, crop):
        """Run face detection + embedding + FAISS search synchronously."""
        try:
            t0_buffalo = time.perf_counter()
            bboxes, kpss = self.face_detector.detect(crop, input_size=(640, 640))
            self.timing["buffalo_processing"] += time.perf_counter() - t0_buffalo

            if bboxes.shape[0] == 0:
                return tid, {"status": "No face"}, None

            best_score = -1.0
            best_name = "Unknown"
            crop_b64 = None

            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, :4]
                det_score = float(bboxes[i, 4])
                if det_score < 0.5:
                    continue

                kps = kpss[i] if kpss is not None else None
                face_obj = Face(bbox=bbox, kps=kps, det_score=det_score)

                emb = self.face_embedder.get(crop, face_obj)
                emb = emb.astype("float32").reshape(1, -1)
                faiss.normalize_L2(emb)

                t0_faiss = time.perf_counter()
                scores, indices = self.index.search(emb, 1)
                self.timing["faiss_search"] += time.perf_counter() - t0_faiss

                score, idx = float(scores[0][0]), int(indices[0][0])

                if score > best_score:
                    best_score = score
                    best_name = self.names[idx] if score >= self.sim_threshold else "Unknown"
                    if crop.size > 0:
                        try:
                            _, buffer = cv2.imencode(".jpg", crop)
                            crop_b64 = base64.b64encode(buffer).decode("utf-8")
                        except Exception:
                            crop_b64 = None

            return tid, {"status": best_name}, crop_b64

        except Exception as e:
            print(f"[Error in _process_crop_sync] {e}")
            return tid, {"status": "Error"}, None

    async def process_frames(self, frames: list[np.ndarray]):
        """Main async frame processing (process every 5th frame only)."""
        start_time = time.perf_counter()
        self.track_best_crop.clear()
        self.id_name_map.clear()
        for k in ["yolo_inference", "tracker_inference", "buffalo_processing", "faiss_search"]:
            self.timing.setdefault(k, 0.0)

        for frame_idx, frame in enumerate(frames):
            # ✅ Process only every 5th frame
            if frame_idx % 5 != 0:
                continue

            t0 = time.perf_counter()
            detections = self.detector.infer(frame, conf_threshold=0.4, iou_threshold=0.7, max_detections=30)
            self.timing["yolo_inference"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            tracks = self.detector.track(frame, detections)
            self.timing["tracker_inference"] += time.perf_counter() - t0

            if not tracks:
                continue

            for tr in tracks:
                x1, y1, x2, y2 = map(int, tr["bbox"])
                tid = tr["track_id"]
                h = y2 - y1
                if h <= 0:
                    continue

                face_crop = frame[y1:y1 + int(h * 0.5), x1:x2]
                if face_crop.size:
                    tid, result_obj, crop_b64 = self._process_crop_sync(tid, face_crop)

                    if tid not in self.id_name_map:
                        self.id_name_map[tid] = []
                    self.id_name_map[tid].append(result_obj)

                    if crop_b64:
                        self.track_best_crop[tid] = crop_b64

        total_exec_time = time.perf_counter() - start_time
        finalized = self._finalize_results()

        all_face_crops_to_upload = {
            str(tid): self.track_best_crop[tid]
            for tid, result_obj in finalized.items()
            if result_obj["status"] != "No face" and tid in self.track_best_crop
        }

        return {
            "status": "success",
            "execution_time": round(total_exec_time, 2),
            "raw_id_name_map": self.id_name_map,
            "finalized_id_name_map": finalized,
            "all_face_crops_to_upload": all_face_crops_to_upload
        }

    def _finalize_results(self):
        finalized = {}
        for tid, result_obj_list in self.id_name_map.items():
            valid = [obj for obj in result_obj_list if obj["status"] not in ["Unknown", "No face", "Error"]]
            if valid:
                names_count = Counter([obj["status"] for obj in valid])
                final_name = names_count.most_common(1)[0][0]
            else:
                if any(obj["status"] == "Unknown" for obj in result_obj_list):
                    final_name = "Unknown"
                else:
                    final_name = "No face"
            finalized[tid] = {"status": final_name}
        return finalized


# ---------------- Async wrapper ----------------
async def run_face_recognition(frames: list[np.ndarray]):
    pipeline = FaceReIDPipeline()
    return await pipeline.process_frames(frames)
