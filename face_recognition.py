import cv2
import numpy as np
import pickle
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from types import SimpleNamespace
import pytz
import os
from collections import defaultdict, Counter
import faiss
import torch
import time
import asyncio
import logging
import base64

# Import global models from models.py
from models import FACE_YOLO_MODEL as YOLO_MODEL, FACE_APP as FACE_RECOGNITION_MODEL

logger = logging.getLogger(__name__)

class FaceReIDPipeline:
    def __init__(self, sim_threshold=0.40):
        self.ist = pytz.timezone("Asia/Kolkata")
        self.sim_threshold = sim_threshold
        # print("started")
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        # print(self.script_dir)
        # print(self.project_root)
        self.db_path = os.path.join(self.script_dir, "face_database_gudivada_pipeline.pkl")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO_MODEL  # Loaded from global variable

        tracker_args = {
            "tracker_type": "botsort", "track_high_thresh": 0.5, "track_low_thresh": 0.3,
            "new_track_thresh": 0.5, "track_buffer": 300, "match_thresh": 0.95,
            "fuse_score": True, "gmc_method": None, "proximity_thresh": 0.5,
            "appearance_thresh": 0.8, "with_reid": False, "model": "auto"
        }
        args = SimpleNamespace(**tracker_args)
        self.tracker = BOTSORT(args=args, frame_rate=25)

        self.face_app = FACE_RECOGNITION_MODEL  # Loaded from global variable

        self.idx_to_person_map, self.index = self._load_faiss_index()
        self.id_name_map = {}
        self.timing = defaultdict(float)
        self.track_best_crop = {}

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
            uid, person_name, role, embedding = item_tuple
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

    async def _process_crop(self, tid, face_crop):
        faces = await asyncio.to_thread(self.face_app.get, face_crop)
        if not faces:
            return tid, {"status": "No face"}, None

        best_score = -1.0
        best_match_object = {"status": "Unknown"}

        for face in faces:
            if face['det_score'] > 0.5:
                emb = face['embedding'].astype('float32')
                query_emb = emb.reshape(1, -1)
                faiss.normalize_L2(query_emb)
                scores, indices = await asyncio.to_thread(self.index.search, query_emb, 1)
                score, idx = scores[0][0], indices[0][0]
                
                if score > best_score:
                    best_score = score
                    if score >= self.sim_threshold:
                        person_data = self.idx_to_person_map[idx]
                        best_match_object = {
                            "status": "Recognized",
                            "hash_id": person_data["hash_id"],
                            "name": person_data["name"],
                            "role": person_data["role"]
                        }
                        print("tid: ",tid, best_match_object)
                    else:
                        best_match_object = {"status": "Unknown"}
        
        crop_b64 = None
        if best_match_object["status"] != "No face" and face_crop.size > 0:
            try:
                _, buffer = cv2.imencode('.jpg', face_crop)
                crop_b64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to encode crop for track {tid}: {e}")

        return tid, best_match_object, crop_b64

    async def process_frames(self, frames: list[np.ndarray]):
        start_time = time.perf_counter()
        self.tracker.reset()
        self.track_best_crop.clear()
        self.id_name_map.clear()

        for frame_idx, frame in enumerate(frames):
            results = await asyncio.to_thread(self.model, frame, classes=[0], conf=0.4, iou=0.7, verbose=False)
            if not results: continue
            
            results = results[0]
            if results.boxes is None or len(results.boxes) == 0: continue

            dets_tensor = torch.hstack([
                results.boxes.xyxy.cpu(), 
                results.boxes.conf.cpu().unsqueeze(1), 
                results.boxes.cls.cpu().unsqueeze(1)
            ])
            dets = Boxes(dets_tensor, frame.shape[:2])
            
            dets_to_track = dets
            if len(dets) == 1:
                # dummy_class = dets_tensor[0, 5]
                # dummy_det = torch.tensor([[0, 0, 1, 1, 0.01, dummy_class]], device=dets_tensor.device, dtype=dets_tensor.dtype)
                # padded_dets_tensor = torch.vstack([dets_tensor, dummy_det])
                # dets_to_track = Boxes(padded_dets_tensor, frame.shape[:2])
                continue
            
            tracks = await asyncio.to_thread(self.tracker.update, dets_to_track, frame.copy())

            # if frame_idx % 5 == 1 and len(tracks) > 0:
            for t in tracks:
                x1_f, y1_f, x2_f, y2_f, track_id_float, cls_id, conf = t[:7]
                # if x1_f == 0 and y1_f == 0 and x2_f == 1 and y2_f == 1: continue
                
                track_id = int(track_id_float)
                x1, y1, x2, y2 = map(int, [x1_f, y1_f, x2_f, y2_f])
                h = y2 - y1
                if h <= 0: continue
                
                face_crop = frame[y1:y1 + int(h * 0.5), x1:x2]
                
                if face_crop.size:
                    tid, result_obj, crop_b64 = await self._process_crop(track_id, face_crop)
                    
                    if tid not in self.id_name_map:
                        self.id_name_map[tid] = []
                    self.id_name_map[tid].append(result_obj)
                    
                    if crop_b64:
                        self.track_best_crop[tid] = crop_b64

        total_exec_time = time.perf_counter() - start_time
        print(total_exec_time)
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
            
            valid_recognitions = [obj for obj in result_obj_list if obj["status"] == "Recognized"]
            
            if valid_recognitions:
                hash_id_counts = Counter([person["hash_id"] for person in valid_recognitions])
                most_common_hash_id = hash_id_counts.most_common(1)[0][0]
                final_result_obj = next(obj for obj in valid_recognitions if obj["hash_id"] == most_common_hash_id)
            else:
                if any(obj["status"] == "Unknown" for obj in result_obj_list):
                    final_result_obj = {"status": "Unknown"}
                else:
                    final_result_obj = {"status": "No face"}
                
            finalized[tid] = final_result_obj
        return finalized

async def run_face_recognition(frames: list[np.ndarray]):
    pipeline = FaceReIDPipeline()
    return await pipeline.process_frames(frames)



# import cv2
# import numpy as np
# import pickle
# from ultralytics.engine.results import Boxes
# from ultralytics.trackers.bot_sort import BOTSORT
# from types import SimpleNamespace
# import pytz
# import os
# from collections import defaultdict, Counter
# import faiss
# import torch
# import time
# import asyncio
# import logging
# import base64

# # Import global models from models.py
# from models import FACE_YOLO_MODEL as YOLO_MODEL, FACE_APP as FACE_RECOGNITION_MODEL

# logger = logging.getLogger(__name__)

# class FaceReIDPipeline:
#     def __init__(self, sim_threshold=0.40):
#         self.ist = pytz.timezone("Asia/Kolkata")
#         self.sim_threshold = sim_threshold
#         # print("started")
#         self.script_dir = os.path.dirname(os.path.abspath(__file__))
#         self.project_root = os.path.dirname(self.script_dir)
#         # print(self.script_dir)
#         # print(self.project_root)
#         self.db_path = os.path.join(self.script_dir, "face_database_gudivada_pipeline.pkl")

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = YOLO_MODEL  # Loaded from global variable

#         tracker_args = {
#             "tracker_type": "botsort", "track_high_thresh": 0.5, "track_low_thresh": 0.3,
#             "new_track_thresh": 0.5, "track_buffer": 500, "match_thresh": 0.95,
#             "fuse_score": True, "gmc_method": None, "proximity_thresh": 0.5,
#             "appearance_thresh": 0.8, "with_reid": False, "model": "auto"
#         }
#         args = SimpleNamespace(**tracker_args)
#         self.tracker = BOTSORT(args=args, frame_rate=25)

#         self.face_app = FACE_RECOGNITION_MODEL  # Loaded from global variable

#         self.idx_to_person_map, self.index = self._load_faiss_index()
#         self.id_name_map = {}
#         self.timing = defaultdict(float)
#         self.track_best_crop = {}

#     def _load_faiss_index(self):
#         if not os.path.exists(self.db_path):
#             raise FileNotFoundError(f"Face database not found at {self.db_path}. Run utils/face_builder.py first.")
        
#         with open(self.db_path, "rb") as f:
#             face_db_list = pickle.load(f)
            
#         if not face_db_list:
#             raise ValueError(f"Empty face database at {self.db_path}.")

#         idx_map = []
#         embeddings_list = []
#         for item_tuple in face_db_list:
#             uid, person_name, role, embedding = item_tuple
#             idx_map.append({
#                 "hash_id": uid, "name": person_name, "role": role
#             })
#             embeddings_list.append(embedding)

#         embeddings = np.stack(embeddings_list).astype("float32")
#         N, dim = len(embeddings), embeddings.shape[1]
#         nlist = max(1, min(N, int(4 * np.sqrt(N))))
#         nprobe = max(1, nlist // 4)

#         quantizer = faiss.IndexFlatIP(dim)
#         index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
#         index.train(embeddings)
#         faiss.normalize_L2(embeddings)
#         index.add(embeddings)
#         index.nprobe = nprobe
        
#         return idx_map, index

#     async def _process_crop(self, tid, face_crop):
#         faces = await asyncio.to_thread(self.face_app.get, face_crop)
#         if not faces:
#             return tid, {"status": "No face"}, None

#         best_score = -1.0
#         best_match_object = {"status": "Unknown"}

#         for face in faces:
#             if face['det_score'] > 0.55:
#                 emb = face['embedding'].astype('float32')
#                 query_emb = emb.reshape(1, -1)
#                 faiss.normalize_L2(query_emb)
#                 scores, indices = await asyncio.to_thread(self.index.search, query_emb, 1)
#                 score, idx = scores[0][0], indices[0][0]
                
#                 if score > best_score:
#                     best_score = score
#                     if score >= self.sim_threshold:
#                         person_data = self.idx_to_person_map[idx]
#                         best_match_object = {
#                             "status": "Recognized",
#                             "hash_id": person_data["hash_id"],
#                             "name": person_data["name"],
#                             "role": person_data["role"]
#                         }
#                     else:
#                         best_match_object = {"status": "Unknown"}
        
#         crop_b64 = None
#         if best_match_object["status"] != "No face" and face_crop.size > 0:
#             try:
#                 _, buffer = cv2.imencode('.jpg', face_crop)
#                 crop_b64 = base64.b64encode(buffer).decode('utf-8')
#             except Exception as e:
#                 logger.warning(f"Failed to encode crop for track {tid}: {e}")

#         return tid, best_match_object, crop_b64

    

#     async def process_frames(self, frames: list[np.ndarray], output_path="output_video.mp4", fps=25):
#         start_time = time.perf_counter()
#         self.tracker.reset()
#         self.track_best_crop.clear()
#         self.id_name_map.clear()

#         # ---- Video Writer Setup ----
#         h, w = frames[0].shape[:2]
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

#         for frame_idx, frame in enumerate(frames):
#             results = await asyncio.to_thread(self.model, frame, classes=[0], conf=0.4, iou=0.7, verbose=False)
#             if results[0].boxes is None or len(results[0].boxes) == 0:
#                 # out.write(frame)  # still write unmodified frame
#                 continue
            
#             results = results[0]
#             # if results.boxes is None or len(results.boxes) == 0:
#             #     out.write(frame)
#             #     continue

#               # Convert YOLO boxes to tracker input
#             bboxes = results.boxes.xyxy.detach().cpu()
#             confs = results.boxes.conf.detach().cpu()
#             cls = results.boxes.cls.detach().cpu()
#             dets_tensor = torch.hstack([bboxes, confs.unsqueeze(1), cls.unsqueeze(1)])
#             dets = Boxes(dets_tensor, frame.shape[:2])
#             # if len(dets) == 1:
#             #     # out.write(frame)
#             #     continue

#             tracks = await asyncio.to_thread(self.tracker.update, dets, frame)

#             # if len(tracks) > 0:
#             for t in tracks:
#                 x1_f, y1_f, x2_f, y2_f, track_id_float, cls_id, conf = t[:7]
#                 track_id = int(track_id_float)
#                 x1, y1, x2, y2 = map(int, [x1_f, y1_f, x2_f, y2_f])
#                 h_box = y2 - y1
#                 # if h_box <= 0: 
#                 #     continue
                
#                 # ---- Crop face ----
#                 face_crop = frame[y1:y1 + int(h_box * 0.5), x1:x2]
#                 display_name = "Unknown"

#                 # if frame_idx % 5 == 1 and face_crop.size:
#                 if face_crop.size:
#                     tid, result_obj, crop_b64 = await self._process_crop(track_id, face_crop)
                
#                     if tid not in self.id_name_map:
#                         self.id_name_map[tid] = []
#                     self.id_name_map[tid].append(result_obj)

#                     if crop_b64:
#                         self.track_best_crop[tid] = crop_b64

#                     if result_obj["status"] == "Recognized":
#                         display_name = f"{result_obj['name']} ({result_obj['role']})"
#                     elif result_obj["status"] == "Unknown":
#                         display_name = "Unknown"
#                     else:
#                         display_name = "No Face"

#                     # ---- Draw bounding box ----
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"ID:{track_id} {display_name}", 
#                                 (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                                 0.6, (0, 255, 255), 2)

#             # ---- Write annotated frame ----
#             out.write(frame)

#         out.release()  # save video
#         total_exec_time = time.perf_counter() - start_time
#         print(f"Video saved at {output_path}, execution time: {total_exec_time:.2f}s")

#         finalized = self._finalize_results()
        
#         all_face_crops_to_upload = {
#             str(tid): self.track_best_crop[tid]
#             for tid, result_obj in finalized.items()
#             if result_obj["status"] != "No face" and tid in self.track_best_crop
#         }

#         return {
#             "status": "success",
#             "execution_time": round(total_exec_time, 2),
#             "raw_id_name_map": self.id_name_map,
#             "finalized_id_name_map": finalized, 
#             "all_face_crops_to_upload": all_face_crops_to_upload,
#             "output_video": output_path
#         }

#     def _finalize_results(self):
#         finalized = {}
#         for tid, result_obj_list in self.id_name_map.items():
            
#             valid_recognitions = [obj for obj in result_obj_list if obj["status"] == "Recognized"]
            
#             if valid_recognitions:
#                 hash_id_counts = Counter([person["hash_id"] for person in valid_recognitions])
#                 most_common_hash_id = hash_id_counts.most_common(1)[0][0]
#                 final_result_obj = next(obj for obj in valid_recognitions if obj["hash_id"] == most_common_hash_id)
#             else:
#                 if any(obj["status"] == "Unknown" for obj in result_obj_list):
#                     final_result_obj = {"status": "Unknown"}
#                 else:
#                     final_result_obj = {"status": "No face"}
                
#             finalized[tid] = final_result_obj
#         return finalized

# async def run_face_recognition(frames: list[np.ndarray]):
#     pipeline = FaceReIDPipeline()
#     return await pipeline.process_frames(frames)




# import cv2
# import numpy as np
# import pickle
# from ultralytics.engine.results import Boxes
# import pytz
# import os
# from collections import defaultdict, Counter
# import faiss
# import torch
# import time
# import asyncio
# import logging
# import base64

# # Import global models from models.py
# from models import FACE_YOLO_MODEL as YOLO_MODEL, FACE_APP as FACE_RECOGNITION_MODEL

# logger = logging.getLogger(__name__)

# class FaceReIDPipeline:
#     def __init__(self, sim_threshold=0.40):
#         self.ist = pytz.timezone("Asia/Kolkata")
#         self.sim_threshold = sim_threshold
#         self.script_dir = os.path.dirname(os.path.abspath(__file__))
#         self.project_root = os.path.dirname(self.script_dir)
#         self.db_path = os.path.join(self.script_dir, "face_database_gudivada_pipeline.pkl")

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = YOLO_MODEL  # Loaded from global variable
#         self.face_app = FACE_RECOGNITION_MODEL  # Loaded from global variable

#         self.idx_to_person_map, self.index = self._load_faiss_index()
#         self.id_name_map = {}
#         self.timing = defaultdict(float)
#         self.track_best_crop = {}

#     def _load_faiss_index(self):
#         if not os.path.exists(self.db_path):
#             raise FileNotFoundError(f"Face database not found at {self.db_path}. Run utils/face_builder.py first.")
        
#         with open(self.db_path, "rb") as f:
#             face_db_list = pickle.load(f)
            
#         if not face_db_list:
#             raise ValueError(f"Empty face database at {self.db_path}.")

#         idx_map = []
#         embeddings_list = []
#         for item_tuple in face_db_list:
#             uid, person_name, role, embedding = item_tuple
#             idx_map.append({
#                 "hash_id": uid, "name": person_name, "role": role
#             })
#             embeddings_list.append(embedding)

#         embeddings = np.stack(embeddings_list).astype("float32")
#         N, dim = len(embeddings), embeddings.shape[1]
#         nlist = max(1, min(N, int(4 * np.sqrt(N))))
#         nprobe = max(1, nlist // 4)

#         quantizer = faiss.IndexFlatIP(dim)
#         index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
#         index.train(embeddings)
#         faiss.normalize_L2(embeddings)
#         index.add(embeddings)
#         index.nprobe = nprobe
        
#         return idx_map, index

#     async def _process_crop(self, face_crop):
#         faces = await asyncio.to_thread(self.face_app.get, face_crop)
#         if not faces:
#             return {"status": "No face"}, None

#         best_score = -1.0
#         best_match_object = {"status": "Unknown"}

#         for face in faces:
#             if face['det_score'] > 0.55:
#                 emb = face['embedding'].astype('float32')
#                 query_emb = emb.reshape(1, -1)
#                 faiss.normalize_L2(query_emb)
#                 scores, indices = await asyncio.to_thread(self.index.search, query_emb, 1)
#                 score, idx = scores[0][0], indices[0][0]
                
#                 if score > best_score:
#                     best_score = score
#                     if score >= self.sim_threshold:
#                         person_data = self.idx_to_person_map[idx]
#                         best_match_object = {
#                             "status": "Recognized",
#                             "hash_id": person_data["hash_id"],
#                             "name": person_data["name"],
#                             "role": person_data["role"]
#                         }
#                     else:
#                         best_match_object = {"status": "Unknown"}
        
#         crop_b64 = None
#         if best_match_object["status"] != "No face" and face_crop.size > 0:
#             try:
#                 _, buffer = cv2.imencode('.jpg', face_crop)
#                 crop_b64 = base64.b64encode(buffer).decode('utf-8')
#             except Exception as e:
#                 logger.warning(f"Failed to encode crop: {e}")

#         return best_match_object, crop_b64

#     async def process_frames(self, frames: list[np.ndarray]):
#         start_time = time.perf_counter()
#         self.track_best_crop.clear()
#         self.id_name_map.clear()
    
#         max_people_count = 0
#         authorized_set = set()
#         saved_hash_ids = set()  # Track which authorized persons already have crops
    
#         # Folder to save authorized face crops
#         save_dir = os.path.join(self.script_dir, "authorized_faces")
#         os.makedirs(save_dir, exist_ok=True)
    
#         for frame_idx, frame in enumerate(frames):
#             results = await asyncio.to_thread(self.model, frame, classes=[0], conf=0.4, iou=0.7, verbose=True)
#             if not results:
#                 continue
            
#             results = results[0]
#             if results.boxes is None or len(results.boxes) == 0:
#                 continue
    
#             # --- Track max number of people detected ---
#             max_people_count = max(max_people_count, len(results.boxes))
    
#             dets_tensor = torch.hstack([
#                 results.boxes.xyxy.cpu(),
#                 results.boxes.conf.cpu().unsqueeze(1),
#                 results.boxes.cls.cpu().unsqueeze(1)
#             ])
#             dets = Boxes(dets_tensor, frame.shape[:2])
    
#             if frame_idx % 5 == 1:  # sample crops
#                 for i, box in enumerate(dets):
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     h = y2 - y1
#                     if h <= 0:
#                         continue
                    
#                     face_crop = frame[y1:y1 + int(h * 0.5), x1:x2]
#                     if face_crop.size:
#                         result_obj, crop_b64 = await self._process_crop(face_crop)
    
#                         if i not in self.id_name_map:
#                             self.id_name_map[i] = []
#                         self.id_name_map[i].append(result_obj)
    
#                         if result_obj.get("status") == "Recognized":
#                             hash_id = result_obj["hash_id"]
#                             name = result_obj["name"].replace(" ", "_")  # clean filename
#                             authorized_set.add(hash_id)
    
#                             # --- Save only one crop per authorized person ---
#                             if hash_id not in saved_hash_ids and crop_b64:
#                                 self.track_best_crop[hash_id] = crop_b64
#                                 saved_hash_ids.add(hash_id)
    
#                                 # Decode and save crop to folder
#                                 crop_img = base64.b64decode(crop_b64)
#                                 nparr = np.frombuffer(crop_img, np.uint8)
#                                 img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#                                 save_path = os.path.join(save_dir, f"{hash_id}_{name}.jpg")
#                                 cv2.imwrite(save_path, img_np)
    
#         total_exec_time = time.perf_counter() - start_time
#         finalized = self._finalize_results()
        
#         # all_face_crops_to_upload now keyed by hash_id (unique person), not index
#         all_face_crops_to_upload = {
#             str(hash_id): crop
#             for hash_id, crop in self.track_best_crop.items()
#         }
    
#         # --- Unauthorized count ---
#         authorized_count = len(authorized_set)
#         print("max count",max_people_count)
#         possible_unauthorized_count = max(0, max_people_count - authorized_count)
    
#         return {
#             "status": "success",
#             "execution_time": round(total_exec_time, 2),
#             "raw_id_name_map": self.id_name_map,
#             "finalized_id_name_map": finalized,
#             "all_face_crops_to_upload": all_face_crops_to_upload,
#             "possible_unauthorized_count": possible_unauthorized_count
#         }

#     def _finalize_results(self):
#         finalized = {}
#         for idx, result_obj_list in self.id_name_map.items():
            
#             valid_recognitions = [obj for obj in result_obj_list if obj["status"] == "Recognized"]
            
#             if valid_recognitions:
#                 hash_id_counts = Counter([person["hash_id"] for person in valid_recognitions])
#                 most_common_hash_id = hash_id_counts.most_common(1)[0][0]
#                 final_result_obj = next(obj for obj in valid_recognitions if obj["hash_id"] == most_common_hash_id)
#             else:
#                 if any(obj["status"] == "Unknown" for obj in result_obj_list):
#                     final_result_obj = {"status": "Unknown"}
#                 else:
#                     final_result_obj = {"status": "No face"}
                
#             finalized[idx] = final_result_obj
#         return finalized

# async def run_face_recognition(frames: list[np.ndarray]):
#     pipeline = FaceReIDPipeline()
#     return await pipeline.process_frames(frames)





# import cv2
# import numpy as np
# import pickle
# from ultralytics.engine.results import Boxes
# from ultralytics.trackers.bot_sort import BOTSORT
# from types import SimpleNamespace
# import pytz
# import os
# from collections import defaultdict, Counter
# import faiss
# import torch
# import time
# import asyncio
# import logging
# import base64
# from insightface.app import FaceAnalysis
# # Import global models from models.py
# from models import FACE_YOLO_MODEL as YOLO_MODEL, FACE_APP as FACE_RECOGNITION_MODEL
# from ultralytics import YOLO

# logger = logging.getLogger(__name__)

# class FaceReIDPipeline:
#     def __init__(self, sim_threshold=0.40):
#         self.ist = pytz.timezone("Asia/Kolkata")
#         self.sim_threshold = sim_threshold
#         # print("started")
#         self.script_dir = os.path.dirname(os.path.abspath(__file__))
#         self.project_root = os.path.dirname(self.script_dir)
#         # print(self.script_dir)
#         # print(self.project_root)
#         self.db_path = os.path.join(self.script_dir, "face_database_gudivada_pipeline.pkl")

#         # self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         # self.model = YOLO_MODEL  # Loaded from global variable
#                # ---------------- YOLO Detector ----------------
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = YOLO("yolov8l.pt").to(self.device)

        
#         tracker_args = {
#             "tracker_type": "botsort", "track_high_thresh": 0.5, "track_low_thresh": 0.3,
#             "new_track_thresh": 0.5, "track_buffer": 500, "match_thresh": 0.95,
#             "fuse_score": True, "gmc_method": None, "proximity_thresh": 0.5,
#             "appearance_thresh": 0.8, "with_reid": False, "model": "auto"
#         }
#         args = SimpleNamespace(**tracker_args)
#         self.tracker = BOTSORT(args=args)

  
#         # self.face_app = FACE_RECOGNITION_MODEL  # Loaded from global variable
#          # ---------------- Face Recognition ----------------
#         self.face_app = FaceAnalysis(name="buffalo_l", root="face_recognition/insightface", providers=['CUDAExecutionProvider'])
#         self.face_app.prepare(ctx_id=0)

#         self.idx_to_person_map, self.index = self._load_faiss_index()
#         self.id_name_map = {}
#         self.timing = defaultdict(float)
#         self.track_best_crop = {}

#     def _load_faiss_index(self):
#         if not os.path.exists(self.db_path):
#             raise FileNotFoundError(f"Face database not found at {self.db_path}. Run utils/face_builder.py first.")
        
#         with open(self.db_path, "rb") as f:
#             face_db_list = pickle.load(f)
            
#         if not face_db_list:
#             raise ValueError(f"Empty face database at {self.db_path}.")

#         idx_map = []
#         embeddings_list = []
#         for item_tuple in face_db_list:
#             uid, person_name, role, embedding = item_tuple
#             idx_map.append({
#                 "hash_id": uid, "name": person_name, "role": role
#             })
#             embeddings_list.append(embedding)

#         embeddings = np.stack(embeddings_list).astype("float32")
#         N, dim = len(embeddings), embeddings.shape[1]
#         nlist = max(1, min(N, int(4 * np.sqrt(N))))
#         nprobe = max(1, nlist // 4)

#         quantizer = faiss.IndexFlatIP(dim)
#         index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
#         index.train(embeddings)
#         faiss.normalize_L2(embeddings)
#         index.add(embeddings)
#         index.nprobe = nprobe
        
#         return idx_map, index

#     async def _process_crop(self, tid, face_crop):
#         # faces = await asyncio.to_thread(self.face_app.get, face_crop)
#         faces = self.face_app.get(face_crop)
#         if not faces:
#             return tid, {"status": "No face"}, None

#         best_score = -1.0
#         best_match_object = {"status": "Unknown"}

#         for face in faces:
#             if face['det_score'] > 0.55:
#                 emb = face['embedding'].astype('float32')
#                 query_emb = emb.reshape(1, -1)
#                 faiss.normalize_L2(query_emb)
#                 scores, indices = await asyncio.to_thread(self.index.search, query_emb, 1)
#                 score, idx = scores[0][0], indices[0][0]
                
#                 if score > best_score:
#                     best_score = score
#                     if score >= self.sim_threshold:
#                         person_data = self.idx_to_person_map[idx]
                        
#                         best_match_object = {
#                             "status": "Recognized",
#                             "hash_id": person_data["hash_id"],
#                             "name": person_data["name"],
#                             "role": person_data["role"]
#                         }
#                         # if tid==14 or tid==12:
#                         print("tid:",tid, best_match_object)
#                     else:
#                         best_match_object = {"status": "Unknown"}
        
#         crop_b64 = None
#         if best_match_object["status"] != "No face" and face_crop.size > 0:
#             try:
#                 _, buffer = cv2.imencode('.jpg', face_crop)
#                 crop_b64 = base64.b64encode(buffer).decode('utf-8')
#             except Exception as e:
#                 logger.warning(f"Failed to encode crop for track {tid}: {e}")

#         return tid, best_match_object, crop_b64

    

#     async def process_frames(self, video_path: str, output_path="output_video.mp4"):
#         start_time = time.perf_counter()
#         self.tracker.reset()
#         self.track_best_crop.clear()
#         self.id_name_map.clear()

#         # ---- Video Capture Setup ----
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise RuntimeError(f"Error: Could not open video file {video_path}")

#         width, height = int(cap.get(3)), int(cap.get(4))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         output_fps = fps / 1

#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

#         frame_idx = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_idx += 1
#             # results_list = await asyncio.to_thread(
#             #     self.model, frame, classes=[0], conf=0.4, iou=0.7, verbose=False
#             # )
#             results_list = self.model(frame, classes=[0], conf=0.4, iou=0.7, verbose=False)[0]
#             results = results_list

#             if results.boxes is None or len(results.boxes) == 0:
#                 # out.write(frame)  # still write unmodified frame
#                 continue

#             # ---- Convert YOLO boxes to tracker input ----
#             bboxes = results.boxes.xyxy.detach().cpu()
#             confs = results.boxes.conf.detach().cpu()
#             cls = results.boxes.cls.detach().cpu()
#             dets_tensor = torch.hstack([bboxes, confs.unsqueeze(1), cls.unsqueeze(1)])
#             dets = Boxes(dets_tensor, frame.shape[:2])

#             tracks = await asyncio.to_thread(self.tracker.update, dets, frame)

#             for t in tracks:
#                 x1, y1, x2, y2, track_id_float, cls_id, conf = t[:7]
#                 track_id = int(track_id_float)
#                 x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#                 h_box = y2 - y1

#                 # if h_box <= 0:
#                 #     continue

#                 # ---- Crop face ----
#                 face_crop = frame[y1:y1 + int(h_box * 0.5), x1:x2]
#                 display_name = ""

#                 if face_crop.size:
#                     tid, result_obj, crop_b64 = await self._process_crop(track_id, face_crop)

#                     if tid not in self.id_name_map:
#                         self.id_name_map[tid] = []
#                     self.id_name_map[tid].append(result_obj)

#                     if crop_b64:
#                         self.track_best_crop[tid] = crop_b64

#                     if result_obj["status"] == "Recognized":
#                         display_name = f"{result_obj['name']} ({result_obj['role']})"
#                     elif result_obj["status"] == "Unknown":
#                         display_name = "Unknown"
#                     else:
#                         display_name = "No Face"

#                     # ---- Draw bounding box ----
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(
#                         frame,
#                         f"ID:{track_id} {display_name}",
#                         (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (0, 255, 255),
#                         2,
#                     )

#             # ---- Write annotated frame ----
#             out.write(frame)

#         cap.release()
#         out.release()

#         total_exec_time = time.perf_counter() - start_time
#         print(f"Video saved at {output_path}, execution time: {total_exec_time:.2f}s")

#         finalized = self._finalize_results()

#         all_face_crops_to_upload = {
#             str(tid): self.track_best_crop[tid]
#             for tid, result_obj in finalized.items()
#             if result_obj["status"] != "No face" and tid in self.track_best_crop
#         }

#         return {
#             "status": "success",
#             "execution_time": round(total_exec_time, 2),
#             "raw_id_name_map": self.id_name_map,
#             "finalized_id_name_map": finalized,
#             "all_face_crops_to_upload": all_face_crops_to_upload,
#             "output_video": output_path,
#         }


#     def _finalize_results(self):
#         finalized = {}
#         for tid, result_obj_list in self.id_name_map.items():
            
#             valid_recognitions = [obj for obj in result_obj_list if obj["status"] == "Recognized"]
            
#             if valid_recognitions:
#                 hash_id_counts = Counter([person["hash_id"] for person in valid_recognitions])
#                 most_common_hash_id = hash_id_counts.most_common(1)[0][0]
#                 final_result_obj = next(obj for obj in valid_recognitions if obj["hash_id"] == most_common_hash_id)
#             else:
#                 if any(obj["status"] == "Unknown" for obj in result_obj_list):
#                     final_result_obj = {"status": "Unknown"}
#                 else:
#                     final_result_obj = {"status": "No face"}
                
#             finalized[tid] = final_result_obj
#         return finalized

# async def run_face_recognition(video_path: str, output_path="output_video.mp4"):
#     pipeline = FaceReIDPipeline()
    
#     return await pipeline.process_frames(video_path, output_path=output_path)


