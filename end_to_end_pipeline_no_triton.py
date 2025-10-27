# face_reid_pipeline.py
import os
import cv2
import numpy as np
import pickle
import faiss
import torch
import time
import asyncio
import logging
import base64
import uuid
import pytz

from collections import defaultdict, Counter
from types import SimpleNamespace

from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import psycopg2

# Import global models from models.py (make sure models.py sets FACE_YOLO_MODEL and FACE_APP)
from models import FACE_YOLO_MODEL as YOLO_MODEL, FACE_APP as FACE_RECOGNITION_MODEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FaceReIDPipeline:
    def __init__(self, sim_threshold=0.40):
        self.ist = pytz.timezone("Asia/Kolkata")
        self.sim_threshold = sim_threshold

        # file paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        # where pickle face DB lives (same as your original code path)
        self.db_path = os.path.join(self.script_dir, "face_database_gudivada_pipeline.pkl")

        # device & models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO_MODEL  # from models.py (yolov8l.pt)
        self.face_app = FACE_RECOGNITION_MODEL  # insightface FaceAnalysis (buffalo_l)

        # tracker (BotSORT)
        tracker_args = {
            "tracker_type": "botsort", "track_high_thresh": 0.5, "track_low_thresh": 0.3,
            "new_track_thresh": 0.5, "track_buffer": 300, "match_thresh": 0.95,
            "fuse_score": True, "gmc_method": None, "proximity_thresh": 0.5,
            "appearance_thresh": 0.8, "with_reid": False, "model": "auto"
        }
        args = SimpleNamespace(**tracker_args)
        self.tracker = BOTSORT(args=args, frame_rate=25)

        # load faiss index + name map
        self.idx_to_person_map, self.index = self._load_faiss_index()

        # runtime maps
        self.id_name_map = {}
        self.track_best_crop = {}     # tid -> base64 crop for UI
        self.best_face_per_id = {}    # tid -> (crop, name, conf, embedding)
        self.timing = defaultdict(float)

    # ------------------ FAISS loader (kept from original code1) ------------------
    def _load_faiss_index(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Face database not found at {self.db_path}. Run utils/face_builder.py first.")
        
        with open(self.db_path, "rb") as f:
            face_db_list = pickle.load(f)
            
        if not face_db_list:
            raise ValueError(f"Empty face database at {self.db_path}.")

        idx_map = []
        embeddings_list = []
        for item_tuple in face_db_list:
            # support both formats (4-tuple or extended)
            # expecting: (uid, person_name, role, embedding)
            try:
                uid, person_name, role, embedding = item_tuple
            except Exception:
                # fallback if saved differently
                uid = str(uuid.uuid4().hex)
                person_name = item_tuple[0]
                role = ""
                embedding = item_tuple[1]
            idx_map.append({
                "hash_id": uid, "name": person_name, "role": role
            })
            embeddings_list.append(embedding)

        embeddings = np.stack(embeddings_list).astype("float32")
        N, dim = len(embeddings), embeddings.shape[1]
        nlist = max(1, min(N, int(4 * np.sqrt(N))))
        nprobe = max(1, nlist // 4)

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        index.nprobe = nprobe
        
        return idx_map, index

    # ------------------ Helpers to update best face per track ------------------
    def _update_best_face(self, tid, crop, name, conf, embedding=None):
        """
        Maintain best_face_per_id[tid] = (crop, name, conf, embedding)
        logic:
        - ignore "No face"
        - "Unknown" stored only if we don't have a known person or if its conf is better
        - known names always preferred over Unknown
        """
        if name == "No face":
            return

        # defensive copies
        try:
            crop_copy = crop.copy()
        except Exception:
            crop_copy = crop

        emb_copy = None
        if embedding is not None:
            emb_copy = np.array(embedding, dtype="float32").copy()

        if name == "Unknown":
            # store unknown only if no record yet or better conf than existing unknown
            if tid not in self.best_face_per_id or self.best_face_per_id[tid][1] == "Unknown":
                if tid not in self.best_face_per_id or conf > self.best_face_per_id[tid][2]:
                    self.best_face_per_id[tid] = (crop_copy, name, conf, emb_copy)
            return

        # named person
        if tid not in self.best_face_per_id:
            self.best_face_per_id[tid] = (crop_copy, name, conf, emb_copy)
        else:
            # If currently unknown stored, or new conf higher, replace
            current_name = self.best_face_per_id[tid][1]
            current_conf = self.best_face_per_id[tid][2]
            if current_name == "Unknown" or conf > current_conf:
                self.best_face_per_id[tid] = (crop_copy, name, conf, emb_copy)

    # ------------------ Postgres helpers (insert / fetch) ------------------
    # NOTE: change PG credentials to your secure config or use env variables
    def _get_pg_conn(self):
        PG_HOST = "145.190.8.4"
        PG_PORT = "5432"
        PG_USER = "spectra"
        PG_PASSWORD = "SpectraParabola9"
        PG_DATABASE = "p9warehouse"
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DATABASE
        )
        return conn

    def _insert_batch_to_pg(self, img_names, crops, embeddings, camera_id):
        """
        Inserts one batch of images into Postgres table warehouse.face_clustering_batch_replica
        Expects:
            img_names: list[str]
            crops: list[np.ndarray] (BGR)
            embeddings: list[np.ndarray] (1D float32)
            camera_id: str
        """
        try:
            conn = self._get_pg_conn()
            cur = conn.cursor()

            crops_bytes = []
            for crop in crops:
                _, buffer = cv2.imencode(".jpg", crop)
                crops_bytes.append(buffer.tobytes())

            embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else list(emb) for emb in embeddings]

            insert_query = """
                INSERT INTO warehouse.face_clustering_batch_replica (img_names, crops, embeddings, camera_id)
                VALUES (%s, %s, %s, %s)
            """
            cur.execute(insert_query, (img_names, crops_bytes, embeddings_list, camera_id))

            conn.commit()
            cur.close()
            conn.close()
            logger.info(f"Inserted batch of {len(img_names)} faces into PostgreSQL")
        except Exception as e:
            logger.exception(f"[Error inserting batch to PostgreSQL] {e}")

    def _fetch_face_clustering_batches(self, camera_id):
        """
        Fetch historical batches for camera_id from Postgres and decode crops & embeddings.
        Returns list of dicts {batch_id, img_names, crops, embeddings, created_at}
        """
        try:
            conn = self._get_pg_conn()
            cur = conn.cursor()

            query = """
                SELECT batch_id, img_names, crops, embeddings, created_at
                FROM warehouse.face_clustering_batch_replica
                WHERE camera_id = %s
                ORDER BY created_at DESC;
            """
            cur.execute(query, (camera_id,))
            rows = cur.fetchall()
            cur.close()
            conn.close()

            results = []
            for batch_id, img_names, crops_bytes, embeddings_list, created_at in rows:
                # crops_bytes is list of bytea, embeddings_list is list of arrays
                crops = [cv2.imdecode(np.frombuffer(crop_bytes, np.uint8), cv2.IMREAD_COLOR)
                         for crop_bytes in crops_bytes]
                embeddings = [np.array(emb, dtype=np.float32) for emb in embeddings_list]

                results.append({
                    "batch_id": batch_id,
                    "img_names": img_names,
                    "crops": crops,
                    "embeddings": embeddings,
                    "created_at": created_at
                })
            return results
        except Exception as e:
            logger.exception(f"[Error fetching batches from PostgreSQL] {e}")
            return []

    # ------------------ Clustering & representative selection ------------------
    def _save_all_best_faces(self, camera_id):
        """
        Build dataset from:
        - best_face_per_id for this run (authorized & unknown)
        - appended historical embeddings from Postgres for same camera
        Then run AgglomerativeClustering, choose representatives, insert current-video batch into PG,
        and return (representative_faces: dict idx->b64, finalized_id_map: dict idx->{"status": name})
        """
        authorized_best = {}
        unauthorized_best = {}

        # Collect best faces from current run
        for tid, (crop, name, conf, embedding) in self.best_face_per_id.items():
            if name not in ["Unknown", "No face"]:
                if name not in authorized_best or conf > authorized_best[name][2]:
                    authorized_best[name] = (crop.copy(), name, conf, embedding.copy() if embedding is not None else None)
            elif name == "Unknown":
                if tid not in unauthorized_best or conf > unauthorized_best[tid][2]:
                    unauthorized_best[tid] = (crop.copy(), name, conf, embedding.copy() if embedding is not None else None)

        # Prepare lists
        all_embeddings = []
        all_img_names = []
        all_crops = []

        # Add authorized (named)
        for name, (crop, _, conf, embedding) in authorized_best.items():
            if embedding is None:
                continue
            all_embeddings.append(embedding)
            filename = f"{name}_{uuid.uuid4().hex[:8]}.jpg"
            all_img_names.append(filename)
            all_crops.append(crop.copy())

        # Add unauthorized
        for tid, (crop, name, conf, embedding) in unauthorized_best.items():
            if embedding is None:
                continue
            all_embeddings.append(embedding)
            filename = f"{name}_ID{tid}_{uuid.uuid4().hex[:8]}.jpg"
            all_img_names.append(filename)
            all_crops.append(crop.copy())

        # Fetch historical embeddings from Postgres and append
        pg_batches = self._fetch_face_clustering_batches(camera_id)
        for batch in pg_batches:
            all_embeddings.extend(batch["embeddings"])
            all_img_names.extend(batch["img_names"])
            all_crops.extend(batch["crops"])

        # Insert only current-video batch (authorized + unauthorized from this run) into Postgres
        try:
            current_count = len(authorized_best) + len(unauthorized_best)
            if current_count > 0:
                # we only insert the first current_count items; historical batches appended after those
                self._insert_batch_to_pg(all_img_names[:current_count], all_crops[:current_count], all_embeddings[:current_count], camera_id)
                logger.info("Inserted current video batch into PostgreSQL")
        except Exception as e:
            logger.exception(f"[Error while inserting current batch] {e}")

        if len(all_embeddings) == 0:
            logger.warning("No embeddings available for clustering")
            return {}, {}

        all_embeddings = np.array(all_embeddings, dtype="float32")
        # If embeddings are row vectors or 2D ensure shape (N, dim)
        if all_embeddings.ndim == 1:
            all_embeddings = all_embeddings.reshape(1, -1)

        # choose clustering threshold with silhouette heuristic
        if len(all_embeddings) > 1:
            best_t, best_score = 0.5, -1.0
            for t in np.arange(0.1, 1.0, 0.05):
                try:
                    model = AgglomerativeClustering(n_clusters=None, distance_threshold=t, metric="cosine", linkage="average")
                    labels = model.fit_predict(all_embeddings)
                except Exception:
                    continue
                n_labels = len(set(labels))
                if n_labels < 2 or n_labels >= len(all_embeddings):
                    continue
                try:
                    score = silhouette_score(all_embeddings, labels, metric="cosine")
                except Exception:
                    continue
                if score > best_score:
                    best_score, best_t = score, t
            logger.info(f"Agglomerative: Chosen distance_threshold = {best_t:.3f}")
            model = AgglomerativeClustering(n_clusters=None, distance_threshold=best_t, metric="cosine", linkage="average")
            labels = model.fit_predict(all_embeddings)
        else:
            labels = np.zeros(len(all_embeddings), dtype=int)

        # Build clusters and choose representatives
        cluster_map = defaultdict(list)
        for lbl, img_name, crop, emb in zip(labels, all_img_names, all_crops, all_embeddings):
            cluster_map[lbl].append((img_name, crop, emb))

        representative_faces = {}
        finalized_id_map = {}
        counter = 0

        for lbl, members in cluster_map.items():
            chosen_img = None
            # prefer named (authorized) images first (i.e., filenames that don't include "Unknown")
            for img_name, crop, emb in members:
                if "Unknown" not in img_name:
                    chosen_img = (img_name, crop, emb)
                    break
            if chosen_img is None:
                chosen_img = members[np.random.randint(len(members))]
            img_name, crop, emb = chosen_img
            _, buffer = cv2.imencode(".jpg", crop)
            b64_crop = base64.b64encode(buffer).decode("utf-8")
            name = os.path.splitext(os.path.basename(img_name))[0]
            representative_faces[counter] = b64_crop
            finalized_id_map[counter] = {"status": name}
            counter += 1

        logger.info(f"Saved {len(representative_faces)} representative faces.")
        return representative_faces, finalized_id_map

    # ------------------ Face crop processing (using insightface FaceAnalysis.get) ------------------
    async def _process_crop(self, tid, face_crop):
        """
        Run InsightFace get() (in thread) for a crop, choose best face (by det_score),
        perform FAISS search and return:
            tid, result_obj, crop_b64, embedding_vector (1D float32) or None, det_score
        """
        try:
            faces = await asyncio.to_thread(self.face_app.get, face_crop)
            if not faces:
                return tid, {"status": "No face"}, None, None, 0.0

            best_score = -1.0
            best_match_object = {"status": "Unknown"}
            best_emb_vec = None
            best_det_score = 0.0

            for face in faces:
                det_score = float(face.get('det_score', 0.0))
                if det_score < 0.5:
                    continue

                emb = face['embedding'].astype('float32')
                query_emb = emb.reshape(1, -1)
                faiss.normalize_L2(query_emb)
                scores, indices = await asyncio.to_thread(self.index.search, query_emb, 1)
                score, idx = float(scores[0][0]), int(indices[0][0])

                if score > best_score:
                    best_score = score
                    best_det_score = det_score
                    if score >= self.sim_threshold:
                        person_data = self.idx_to_person_map[idx]
                        best_match_object = {
                            "status": "Recognized",
                            "hash_id": person_data["hash_id"],
                            "name": person_data["name"],
                            "role": person_data.get("role", "")
                        }
                    else:
                        best_match_object = {"status": "Unknown"}
                    best_emb_vec = emb.copy()

            crop_b64 = None
            if best_match_object["status"] != "No face" and face_crop is not None and face_crop.size > 0:
                try:
                    _, buffer = cv2.imencode('.jpg', face_crop)
                    crop_b64 = base64.b64encode(buffer).decode('utf-8')
                except Exception as e:
                    logger.warning(f"Failed to encode crop for track {tid}: {e}")

            return tid, best_match_object, crop_b64, best_emb_vec, best_det_score

        except Exception as e:
            logger.exception(f"[Error in _process_crop] {e}")
            return tid, {"status": "Error"}, None, None, 0.0

    # ------------------ Main frame processing (modified to use clustering helpers) ------------------
    async def process_frames(self, frames: list[np.ndarray], camera_id: str):
        """
        Process frames (list of numpy arrays). Uses YOLO local model and InsightFace.
        Returns final maps and representative faces (b64).
        """
        start_time = time.perf_counter()
        self.tracker.reset()
        self.track_best_crop.clear()
        self.id_name_map.clear()
        self.best_face_per_id.clear()

        for frame_idx, frame in enumerate(frames):
            # run YOLO (in thread to avoid blocking event loop)
            results = await asyncio.to_thread(self.model, frame, classes=[0], conf=0.4, iou=0.7, verbose=False)
            if not results:
                continue
            results = results[0]
            if results.boxes is None or len(results.boxes) == 0:
                continue

            dets_tensor = torch.hstack([
                results.boxes.xyxy.cpu(),
                results.boxes.conf.cpu().unsqueeze(1),
                results.boxes.cls.cpu().unsqueeze(1)
            ])
            dets = Boxes(dets_tensor, frame.shape[:2])

            if len(dets) == 1:
                # skip single detection frame (BotSORT needs >1 to associate reliably)
                continue

            tracks = await asyncio.to_thread(self.tracker.update, dets, frame.copy())

            for t in tracks:
                x1_f, y1_f, x2_f, y2_f, track_id_float, cls_id, conf = t[:7]
                track_id = int(track_id_float)
                x1, y1, x2, y2 = map(int, [x1_f, y1_f, x2_f, y2_f])
                h = y2 - y1
                if h <= 0:
                    continue

                # crop roughly top half of bbox (face region)
                face_crop = frame[y1: y1 + int(h * 0.5), x1:x2]
                if not face_crop.size:
                    continue

                # process crop: face detection + embedding + faiss search
                tid, result_obj, crop_b64, embedding_vec, det_conf = await self._process_crop(track_id, face_crop)

                # maintain id_name_map (raw sequence)
                if tid not in self.id_name_map:
                    self.id_name_map[tid] = []
                self.id_name_map[tid].append(result_obj)

                # maintain track_best_crop for UI (prefer recognized or higher det_conf)
                if crop_b64:
                    prev = self.track_best_crop.get(tid)
                    # if not present, set; if present, keep existing (we may also choose to update based on heuristic)
                    self.track_best_crop[tid] = crop_b64

                # update best_face_per_id using embedding (if we have embedding)
                if embedding_vec is not None:
                    # convert crop_b64 to image for storage in best_face_per_id (if crop_b64 exists)
                    crop_img = None
                    if crop_b64:
                        try:
                            crop_img = cv2.imdecode(np.frombuffer(base64.b64decode(crop_b64), np.uint8), cv2.IMREAD_COLOR)
                        except Exception:
                            crop_img = face_crop.copy()
                    else:
                        crop_img = face_crop.copy()

                    status_name = result_obj.get("status", "Unknown")
                    self._update_best_face(tid, crop_img, status_name, float(det_conf), embedding=np.array(embedding_vec, dtype="float32"))

        total_exec_time = time.perf_counter() - start_time
        # finalize decisions per track (use most common recognized name or Unknown/No face)
        finalized = self._finalize_results()

        # Run clustering & save best faces to DB, get representative faces
        save_start = time.perf_counter()
        rep_faces, finalized_map_from_clustering = self._save_all_best_faces(camera_id)
        self.timing["save_face"] += time.perf_counter() - save_start

        # Build all_face_crops_to_upload (tid->b64)
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
            "cluster_finalized_map": finalized_map_from_clustering,
            "all_face_crops_to_upload": all_face_crops_to_upload,
            "representative_faces": rep_faces,
            "timing": dict(self.timing)
        }

    def _finalize_results(self):
        """
        From id_name_map (list of result objs per tid), determine final per tid:
          - If any Recognized found -> choose most common recognized hash_id
          - Else if any Unknown -> Unknown
          - Else -> No face
        """
        finalized = {}
        for tid, result_obj_list in self.id_name_map.items():
            valid_recognitions = [obj for obj in result_obj_list if obj.get("status") == "Recognized"]
            if valid_recognitions:
                hash_id_counts = Counter([person["hash_id"] for person in valid_recognitions])
                most_common_hash_id = hash_id_counts.most_common(1)[0][0]
                final_result_obj = next(obj for obj in valid_recognitions if obj["hash_id"] == most_common_hash_id)
            else:
                if any(obj.get("status") == "Unknown" for obj in result_obj_list):
                    final_result_obj = {"status": "Unknown"}
                else:
                    final_result_obj = {"status": "No face"}
            finalized[tid] = final_result_obj
        return finalized


# -------------- Async wrapper --------------
async def run_face_recognition(frames: list[np.ndarray], camera_id: str):
    pipeline = FaceReIDPipeline()
    return await pipeline.process_frames(frames, camera_id)
