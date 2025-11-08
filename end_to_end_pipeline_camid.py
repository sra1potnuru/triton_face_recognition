from __future__ import division
import os
import os.path as osp
import cv2
import numpy as np
import pickle
import json
import pytz
import time
import base64
import asyncio
import torch
from collections import defaultdict, Counter
from types import SimpleNamespace
from skimage import transform as trans
import faiss
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from numpy.linalg import norm as l2norm
from tracker import FaceTracker
import uuid
import datetime
# ------------------- Person Detector -------------------
class YoloPersonDetector:
    def __init__(self, triton_url="provider.rtx4090.wyo.eg.akash.pub:30247", model_name="yolo_person_detection"):
        self.model_name = model_name
        self.client = grpcclient.InferenceServerClient(url=triton_url)
        self.input_name = "images"
        self.output_name = "output0"
        self.input_size = 640
        self.target_class = 0  # person

        # Initialize face tracker with configuration values
        self.tracker = FaceTracker(
            tracker_type="botsort",
            track_high_thresh=0.5,
            track_low_thresh=0.3,
            new_track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.9,
            fuse_score=True,
            gmc_method=None,
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model="auto"
        )

    async def infer_single_frame(self, image: np.ndarray, conf_threshold: float, iou_threshold: float, max_detections: int, frame_idx: int):
        """Asynchronously run inference for a single frame."""
        # Preprocess (letterbox)
        letterboxed, scale, pad_left, pad_top = letterbox(image, (self.input_size, self.input_size))
        tensor = letterboxed.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
 
        # Define the synchronous blocking call
        def _blocking_infer():
            inputs = [grpcclient.InferInput(self.input_name, tensor.shape, np_to_triton_dtype(tensor.dtype))]
            inputs[0].set_data_from_numpy(tensor)
            outputs = [grpcclient.InferRequestedOutput(self.output_name)]
            return self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
 
        # Run the blocking call in a separate thread
        response = await asyncio.to_thread(_blocking_infer)
        detections = response.as_numpy(self.output_name)[0]
 
        # Post-processing
        filtered = detections[detections[:, 4] >= conf_threshold]
        filtered = apply_nms(filtered, iou_threshold)
 
        results = []
        for row in filtered:
            cls_id = int(round(row[5]))
            if cls_id != self.target_class:
                continue
            x1, y1, x2, y2 = row[:4]
            score = float(row[4])
 
            # Undo letterbox
            x1 = (x1 - pad_left) / scale
            y1 = (y1 - pad_top) / scale
            x2 = (x2 - pad_left) / scale
            y2 = (y2 - pad_top) / scale
 
            x1 = max(min(x1, image.shape[1]), 0.0)
            y1 = max(min(y1, image.shape[0]), 0.0)
            x2 = max(min(x2, image.shape[1]), 0.0)
            y2 = max(min(y2, image.shape[0]), 0.0)
 
            results.append([x1, y1, x2, y2, score, cls_id])
            if len(results) == max_detections:
                break
 
        return frame_idx, np.array(results, dtype=np.float32)

    async def infer_batch(self, frames: list, conf_threshold: float, iou_threshold: float, max_detections: int):
        """Run inference on all frames in parallel."""
        tasks = [
            self.infer_single_frame(frame, conf_threshold, iou_threshold, max_detections, idx)
            for idx, frame in enumerate(frames)
        ]
        results = await asyncio.gather(*tasks)
        # Sort by frame index to maintain order
        results.sort(key=lambda x: x[0])
        return results
 
    def track(self, image: np.ndarray, detections: np.ndarray):
        """Track faces using the FaceTracker"""
        if len(detections) == 0:
            return []
   
        # Convert to torch Boxes format
        dets_tensor = torch.tensor(detections, dtype=torch.float32)
        dets = Boxes(dets_tensor, image.shape[:2])
        dets_to_track = dets
   
        # Add a dummy detection if only one detection exists
        if len(dets) == 1:
            dummy_class = dets_tensor[0, 5]
            dummy_det = torch.tensor([[0, 0, 1, 1, 0.01, dummy_class]],
                                     device=dets_tensor.device,
                                     dtype=dets_tensor.dtype)
            padded_dets_tensor = torch.vstack([dets_tensor, dummy_det])
            dets_to_track = Boxes(padded_dets_tensor, image.shape[:2])
   
        # Use the FaceTracker to update tracks
        return self.tracker.update(dets_to_track, image)


# Keep all utility functions
def apply_nms(detections: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(detections) == 0:
        return detections
    boxes = detections[:, :4]
    scores = detections[:, 4]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return detections[keep]
 
def letterbox(image: np.ndarray, size: tuple[int, int]):
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = target_w - new_w, target_h - new_h
    pad_left, pad_top = pad_w / 2.0, pad_h / 2.0
    left, right = int(np.floor(pad_left)), int(np.ceil(pad_w - np.floor(pad_left)))
    top, bottom = int(np.floor(pad_top)), int(np.ceil(pad_h - np.floor(pad_top)))
    bordered = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return bordered, scale, float(left), float(top)
 
def draw_tracks(image: np.ndarray, tracks: list[dict]) -> np.ndarray:
    canvas = image.copy()
    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr["bbox"])
        tid = tr["track_id"]
        score = tr["score"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"ID:{tid} {score:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas


# arcface_utils
arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
 
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M
 
def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
 
def norm_crop2(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M


# arcface
class ArcFaceONNX:
    def __init__(self, model_name=None, url="provider.rtx4090.wyo.eg.akash.pub:30247"):
        assert model_name is not None, "Provide Triton model name"
        self.model_name = model_name
        self.taskname = 'recognition'
        self.url = url
        self.client = grpcclient.InferenceServerClient(url=self.url)
        self.input_mean = 127.5
        self.input_std = 127.5
 
        model_metadata = self.client.get_model_metadata(model_name=self.model_name)
        model_config = self.client.get_model_config(model_name=self.model_name)
       
        self.input_name = model_metadata.inputs[0].name
        self.output_names = [o.name for o in model_metadata.outputs]
        self.input_shape = list(model_metadata.inputs[0].shape)
        self.input_size = (self.input_shape[2], self.input_shape[3])
        self.output_shape = list(model_metadata.outputs[0].shape)

    def prepare(self, ctx_id, **kwargs):
        pass

    async def get_async(self, img, face):
        """Async version of get()"""
        aimg = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        face.embedding = await self.get_feat_async(aimg)
        return face.embedding.flatten()

    async def get_feat_async(self, imgs):
        """Async wrapper for get_feat"""
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(
            imgs,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )

        def _blocking_infer():
            inputs = [grpcclient.InferInput(self.input_name, blob.shape, "FP32")]
            inputs[0].set_data_from_numpy(blob)
            outputs = [grpcclient.InferRequestedOutput(name) for name in self.output_names]
            result = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
            return result.as_numpy(self.output_names[0])

        net_out = await asyncio.to_thread(_blocking_infer)
        return net_out


# retinaface
def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div
 
def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clip(0, max_shape[1])
        y1 = y1.clip(0, max_shape[0])
        x2 = x2.clip(0, max_shape[1])
        y2 = y2.clip(0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)
 
def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, (i % 2) + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clip(0, max_shape[1])
            py = py.clip(0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class RetinaFace:
    def __init__(self, model_file=None, session=None,
                 triton_url="provider.rtx4090.wyo.eg.akash.pub:30247",
                 model_name="buffalo_face_detection"):
        self.model_file = model_file
        self.taskname = 'detection'
        self.triton_client = grpcclient.InferenceServerClient(url=triton_url)
        self.model_name = model_name
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        self.input_name = "input.1"
        self.output_names = [
            "448", "471", "494",
            "451", "474", "497",
            "454", "477", "500"
        ]
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = True
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.input_size = None

    def prepare(self, ctx_id, **kwargs):
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size already set, ignore')
            else:
                self.input_size = input_size

    async def forward_async(self, img, threshold):
        """Async version of forward"""
        scores_list, bboxes_list, kpss_list = [], [], []
        input_size = tuple(img.shape[0:2][::-1])

        blob = cv2.dnn.blobFromImage(img, 1.0 / self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean),
                                     swapRB=True)

        def _blocking_infer():
            inputs = [grpcclient.InferInput(self.input_name, blob.shape, np_to_triton_dtype(blob.dtype))]
            inputs[0].set_data_from_numpy(blob)
            outputs = [grpcclient.InferRequestedOutput(name) for name in self.output_names]
            results = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
            return [results.as_numpy(name) for name in self.output_names]

        net_outs = await asyncio.to_thread(_blocking_infer)

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc] * stride
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    async def detect_async(self, img, input_size=None, max_num=0, metric='default'):
        """Async version of detect"""
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = await self.forward_async(det_img, self.det_thresh)

        scores = np.vstack(scores_list) if scores_list else np.array([])
        if len(scores) == 0:
            return np.zeros((0, 5)), None
            
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area - offset_dist_squared * 2.0 if metric != 'max' else area
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep


# face
class Face(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                    if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'


from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import psycopg2

# Import your existing YOLO, RetinaFace, ArcFace classes and Face dataclass
# from some_module import YoloPersonDetector, RetinaFace, ArcFaceONNX, Face

class FaceReIDPipeline:
    # In face_recognitation.py

    def __init__(self, sim_threshold=0.40):
        self.ist = pytz.timezone("Asia/Kolkata")
        self.sim_threshold = sim_threshold

        # Triton YOLO Detector
        self.triton_yolo_url = "provider.rtx4090.wyo.eg.akash.pub:30247"
        self.yolo_model_name = "yolo_person_detection"
        self.detector = YoloPersonDetector(triton_url=self.triton_yolo_url, model_name=self.yolo_model_name)

        # Triton RetinaFace + ArcFace
        self.triton_buffalo_url = "provider.rtx4090.wyo.eg.akash.pub:30247"
        self.buffalo_face_model = "buffalo_face_detection"
        self.buffalo_embed_model = "buffalo_face_embedding"

        self.face_detector = RetinaFace(triton_url=self.triton_buffalo_url, model_name=self.buffalo_face_model)
        self.face_detector.prepare(ctx_id=0, input_size=(640, 640))

        self.face_embedder = ArcFaceONNX(url=self.triton_buffalo_url, model_name=self.buffalo_embed_model)
        self.face_embedder.prepare(ctx_id=0, input_size=(112, 112))

        # FAISS index load
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        # self.db_path = os.path.join(project_root, "data", "face_database_gudivada_format.pkl")
        self.db_path = os.path.join(current_dir, "", "face_database_gudivada_format.pkl")
        
        # --- ADD THIS LINE ---
        self.name_to_id_map = {}  # This will store {"Venkateshwar": "H123"}
        
        # self.db_path = os.path.join(current_dir, "", "face_database_gudivada_format.pkl")
        self.names, self.index = self._load_faiss_index(self.db_path)

        # State holders
        self.id_name_map = {}
        self.track_best_crop = {}
        self.best_face_per_id = {}  # tid -> (crop, name, conf, embedding)
        self.timing = defaultdict(float)
        for k in ["yolo_inference", "tracker_inference", "buffalo_detection", "buffalo_embedding", "faiss_search", "update_face", "save_face", "clustering"]:
            self.timing.setdefault(k, 0.0)
    # In face_recognitation.py

    def _load_faiss_index(self, db_path):
        with open(db_path, "rb") as f:
            face_db_list = pickle.load(f)

        # support both (ids,names,roles,embeddings) or (names,embeddings)
        try:
            ids, names, roles, embeddings = zip(*face_db_list)
            names = np.array(names)
            embeddings = np.stack(embeddings).astype("float32")
            # --- ADD THIS LINE ---
            # Populate the name-to-ID map
            self.name_to_id_map = {name: wid for name, wid in zip(names, ids)}
            
        except Exception:
            names, embeddings = zip(*face_db_list)
            names = np.array(names)
            embeddings = np.stack(embeddings).astype("float32")
            # --- ADD THIS LINE ---
            self.name_to_id_map = {} # No IDs found, create empty map
            
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
    # ------------------ from code2: update/save/fetch/insert ------------------
    def _update_best_face(self, tid, crop, name, conf, embedding=None):
        """Maintain best face per track_id. embedding expected as 1D array or None"""
        if name == "No face":
            return
        if name == "Unknown":
            if tid not in self.best_face_per_id or self.best_face_per_id[tid][1] == "Unknown":
                if tid not in self.best_face_per_id or conf > self.best_face_per_id[tid][2]:
                    self.best_face_per_id[tid] = (crop.copy(), name, conf, embedding.copy() if embedding is not None else None)
            return
        if tid not in self.best_face_per_id:
            self.best_face_per_id[tid] = (crop.copy(), name, conf, embedding.copy() if embedding is not None else None)
        else:
            if self.best_face_per_id[tid][1] == "Unknown":
                self.best_face_per_id[tid] = (crop.copy(), name, conf, embedding.copy() if embedding is not None else None)

    def _insert_batch_to_pg(self, img_names, crops, embeddings, cam_id: str | None = None,
                        batch_date: datetime.date | None = None, warehouse_id: str | None = None):
        """
        Insert a batch into wh_face_clustering_batch.
    
        Tries an "array-friendly" insert first (img_names as text[], crops as bytea[], embeddings as jsonb),
        and if that fails, falls back to a "single-value" insert (img_names joined as comma string,
        crops -> first image bytea, embeddings -> json string).
        """
        try:
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
            cur = conn.cursor()
    
            # Ensure a date value
            if batch_date is None:
                batch_date = datetime.datetime.now(self.ist).date()
    
            # created_at timestamp in IST
            created_at = datetime.datetime.now(self.ist)
    
            # Prepare crops as bytes (list of byte strings)
            crops_bytes = []
            for crop in crops:
                try:
                    _, buffer = cv2.imencode(".jpg", crop)
                    crops_bytes.append(buffer.tobytes())
                except Exception:
                    # skip invalid crop
                    continue
    
            # Prepare embeddings as list-of-lists and JSON
            embeddings_list = []
            for emb in embeddings:
                try:
                    embeddings_list.append(np.array(emb, dtype=np.float32).tolist())
                except Exception:
                    # fallback: try to coerce
                    try:
                        embeddings_list.append(list(emb))
                    except Exception:
                        continue
    
            # Prepare img_names as list and comma-joined str (fallback)
            try:
                if isinstance(img_names, (list, tuple)):
                    img_names_list = list(img_names)
                elif isinstance(img_names, str):
                    # If comma-separated string, split; otherwise single
                    if "," in img_names:
                        img_names_list = [s.strip() for s in img_names.split(",") if s.strip()]
                    elif img_names.strip() == "":
                        img_names_list = []
                    else:
                        img_names_list = [img_names.strip()]
                else:
                    img_names_list = []
            except Exception:
                img_names_list = []
    
            img_names_joined = ",".join(img_names_list) if img_names_list else ""
    
            # First attempt: insert as array/json (works if your columns are text[], bytea[], jsonb/double precision[])
            try:
                insert_query = """
                    INSERT INTO wh_face_clustering_batch
                    (warehouse_id, cam_id, img_names, crops, embeddings, date, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                # psycopg2 will convert Python lists -> SQL arrays for compatible column types.
                # For bytea[] we wrap each bytes in psycopg2.Binary
                crops_param = [psycopg2.Binary(b) for b in crops_bytes] if crops_bytes else None
    
                # embeddings -> store as JSON (jsonb) for maximum compatibility
                embeddings_param = psycopg2.extras.Json(embeddings_list) if embeddings_list else psycopg2.extras.Json([])
    
                cur.execute(insert_query, (
                    warehouse_id,
                    cam_id,
                    img_names_list if img_names_list else None,
                    crops_param,
                    embeddings_param,
                    batch_date,
                    created_at
                ))
                conn.commit()
                cur.close()
                conn.close()
                print(f"💾 Inserted batch (array/json attempt) of {len(img_names_list)} faces into PostgreSQL "
                      f"(warehouse_id={warehouse_id}, cam_id={cam_id}, date={batch_date})")
                return True
    
            except Exception as e_array:
                # rollback and try fallback
                conn.rollback()
                # print a debug message, but continue to fallback
                print(f"[Array insert failed, falling back] {e_array}")
    
            # Fallback attempt: single-value insert (img_names joined string, first crop as bytea, embeddings as JSON text)
            try:
                insert_query_fallback = """
                    INSERT INTO wh_face_clustering_batch
                    (warehouse_id, cam_id, img_names, crops, embeddings, date, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                first_crop = crops_bytes[0] if crops_bytes else None
                # store embeddings as JSON string
                embeddings_json_text = json.dumps(embeddings_list, default=lambda x: list(x)) if embeddings_list else "[]"
    
                cur.execute(insert_query_fallback, (
                    warehouse_id,
                    cam_id,
                    img_names_joined if img_names_joined else None,
                    psycopg2.Binary(first_crop) if first_crop is not None else None,
                    embeddings_json_text,
                    batch_date,
                    created_at
                ))
                conn.commit()
                cur.close()
                conn.close()
                print(f"💾 Inserted batch (fallback) of {len(img_names_list)} faces into PostgreSQL "
                      f"(warehouse_id={warehouse_id}, cam_id={cam_id}, date={batch_date})")
                return True
    
            except Exception as e_fallback:
                conn.rollback()
                cur.close()
                conn.close()
                print(f"[Error inserting batch to PostgreSQL - fallback failed] {e_fallback}")
                return False
    
        except Exception as e:
            print(f"[Error inserting batch to PostgreSQL] {e}")
            return False

    def _fetch_face_clustering_batches(self, cam_id: str | None = None, batch_date: datetime.date | None = None, warehouse_id: str | None = None):
        try:
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
            cur = conn.cursor()

            # default to today's date in IST if not provided
            if batch_date is None:
                batch_date = datetime.datetime.now(self.ist).date()
    
            # Build query dynamically
            query = """
                SELECT batch_id, warehouse_id, cam_id, img_names, crops, embeddings, date, created_at
                FROM wh_face_clustering_batch
                WHERE date = %s
            """
            params: list = [batch_date]
    
            if cam_id is not None:
                query += " AND cam_id = %s"
                params.append(cam_id)
            if warehouse_id is not None:
                query += " AND warehouse_id = %s"
                params.append(warehouse_id)
    
            query += " ORDER BY created_at DESC;"
    
            # DEBUG: print(query, params)  # uncomment for debugging
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
    
            results = []
            for row in rows:
                # Unpack row safely (should match the SELECT order)
                batch_id, warehouse_row, cam_row, img_names_raw, crops_raw, embeddings_raw, date_row, created_at = row
    
                # --- parse img_names ---
                img_names = []
                try:
                    # If returned as list/tuple
                    if isinstance(img_names_raw, (list, tuple)):
                        img_names = list(img_names_raw)
                    elif isinstance(img_names_raw, str):
                        # assume comma-separated or single name
                        if "," in img_names_raw:
                            img_names = [s.strip() for s in img_names_raw.split(",") if s.strip()]
                        elif img_names_raw.strip() == "":
                            img_names = []
                        else:
                            img_names = [img_names_raw.strip()]
                    else:
                        img_names = []  # fallback
                except Exception:
                    img_names = []
    
                # --- parse crops (try multiple formats) ---
                crops = []
                try:
                    # If crops_raw is a list/tuple of bytea items
                    if isinstance(crops_raw, (list, tuple)):
                        for item in crops_raw:
                            try:
                                buf = np.frombuffer(item, dtype=np.uint8)
                                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                                if img is None:
                                    # try via memoryview
                                    img = cv2.imdecode(np.frombuffer(bytes(item), np.uint8), cv2.IMREAD_COLOR)
                                crops.append(img)
                            except Exception:
                                continue
                    elif isinstance(crops_raw, (bytes, memoryview)):
                        # Could be a single image stored as bytea
                        try:
                            buf = np.frombuffer(bytes(crops_raw), dtype=np.uint8)
                            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                            if img is not None:
                                crops.append(img)
                            else:
                                # maybe crops_raw contains multiple concatenated images — best-effort: return empty
                                pass
                        except Exception:
                            pass
                    else:
                        # unknown format, try str -> base64?
                        if isinstance(crops_raw, str):
                            try:
                                b = base64.b64decode(crops_raw)
                                buf = np.frombuffer(b, dtype=np.uint8)
                                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                                if img is not None:
                                    crops.append(img)
                            except Exception:
                                pass
                except Exception:
                    crops = []
    
                # --- parse embeddings (try JSON, list, bytes/pickle) ---
                embeddings = []
                try:
                    if isinstance(embeddings_raw, (list, tuple)):
                        # list of lists or list of numbers
                        for emb in embeddings_raw:
                            embeddings.append(np.array(emb, dtype=np.float32))
                    elif isinstance(embeddings_raw, str):
                        # maybe stored as JSON string
                        try:
                            parsed = json.loads(embeddings_raw)
                            if isinstance(parsed, (list, tuple)):
                                for emb in parsed:
                                    embeddings.append(np.array(emb, dtype=np.float32))
                        except Exception:
                            # maybe comma separated numbers
                            try:
                                nums = [float(x) for x in embeddings_raw.split(",") if x.strip()]
                                embeddings.append(np.array(nums, dtype=np.float32))
                            except Exception:
                                pass
                    elif isinstance(embeddings_raw, (bytes, memoryview)):
                        # try JSON decode
                        try:
                            parsed = json.loads(bytes(embeddings_raw).decode("utf-8"))
                            if isinstance(parsed, (list, tuple)):
                                for emb in parsed:
                                    embeddings.append(np.array(emb, dtype=np.float32))
                        except Exception:
                            # try pickle
                            try:
                                parsed = pickle.loads(bytes(embeddings_raw))
                                if isinstance(parsed, (list, tuple)):
                                    for emb in parsed:
                                        embeddings.append(np.array(emb, dtype=np.float32))
                            except Exception:
                                pass
                    else:
                        # unknown
                        pass
                except Exception:
                    embeddings = []
    
                results.append({
                    "batch_id": batch_id,
                    "warehouse_id": warehouse_row,
                    "cam_id": cam_row,
                    "img_names": img_names,
                    "crops": crops,
                    "embeddings": embeddings,
                    "date": date_row,
                    "created_at": created_at
                })
    
            cur.close()
            conn.close()
            return results

        except Exception as e:
            print(f"[Error fetching batches from PostgreSQL] {e}")
            return []

    # In face_recognitation.py

    def _save_all_best_faces(self,cam_id: str | None = None, batch_date: datetime.date | None = None, warehouse_id: str | None = None):
        os.makedirs("pipeline_images", exist_ok=True)
        os.makedirs("best_crops", exist_ok=True)
        """
        Use the best_face_per_id collected for this video, append historical batches from Postgres,
        run Agglomerative clustering, pick representative faces per cluster, and insert current video batch into PG.
        Returns: representative_faces (dict idx->b64), finalized_id_map (dict idx->name)
        """
        authorized_best = {}
        unauthorized_best = {}
    
        # Collect best faces from current video
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
    
        # Add authorized
        for name, (crop, _, conf, embedding) in authorized_best.items():
            if embedding is None:
                continue
            all_embeddings.append(embedding)
            filename = f"{name}_{uuid.uuid4().hex[:8]}.jpg"
            # filepath = os.path.join("best_crops", filename)
            # cv2.imwrite(filepath, crop)
            all_img_names.append(filename)
            all_crops.append(crop.copy())
    
        # Add unauthorized
        for tid, (crop, name, conf, embedding) in unauthorized_best.items():
            if embedding is None:
                continue
            all_embeddings.append(embedding)
            filename = f"{name}ID{tid}{uuid.uuid4().hex[:8]}.jpg"
            # filepath = os.path.join("best_crops", filename)
            # cv2.imwrite(filepath, crop)
            all_img_names.append(filename)
            all_crops.append(crop.copy())
    
        # --- NEW ---
        # Store how many embeddings belong to the current (new) video before adding historical ones
        current_count = len(all_embeddings)
    
        # Fetch historical embeddings from Postgres
        
        historical_img_names = []
        
        pg_batches = self._fetch_face_clustering_batches(cam_id, batch_date,warehouse_id)
        for batch in pg_batches:
            all_embeddings.extend(batch["embeddings"])
            all_img_names.extend(batch["img_names"])
            all_crops.extend(batch["crops"])
            
            historical_img_names.extend(batch["img_names"])
            
    
        historical_count = len(all_embeddings) - current_count  # --- NEW ---
    
        # Insert only current video batch into Postgres (first chunk lengths)
        try:
            if current_count > 0:
                self._insert_batch_to_pg(
                    all_img_names[:current_count],
                    all_crops[:current_count],
                    all_embeddings[:current_count],
                    cam_id, batch_date,warehouse_id
                )
                print("💾 Inserted current video batch into PostgreSQL")
        except Exception as e:
            print(f"[Error while inserting current batch] {e}")
    
        if len(all_embeddings) == 0:
            print("⚠ No embeddings available for clustering")
            return {}, {}
    
        all_embeddings = np.array(all_embeddings, dtype="float32")
    
        # Clustering: choose distance_threshold using silhouette_score heuristic
        if len(all_embeddings) > 1:
            best_t, best_score = 0.5, -1
            for t in np.arange(0.1, 1.0, 0.05):
                model = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=t, metric="cosine", linkage="average"
                )
                try:
                    labels = model.fit_predict(all_embeddings)
                except Exception:
                    continue
                n_labels = len(set(labels))
                if n_labels < 2 or n_labels >= len(all_embeddings):
                    continue
                try:
                    score = silhouette_score(all_embeddings, labels)
                except Exception:
                    continue
                if score > best_score:
                    best_score, best_t = score, t
            print(f"➡ Agglomerative: Chosen distance_threshold = {best_t:.3f}")
            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=best_t, metric="cosine", linkage="average"
            )
            labels = model.fit_predict(all_embeddings)
        else:
            labels = np.zeros(len(all_embeddings), dtype=int)
    
        # Build clusters
        cluster_map = defaultdict(list)
        for lbl, img_name, crop, emb in zip(labels, all_img_names, all_crops, all_embeddings):
            cluster_map[lbl].append((img_name, crop, emb))
        # Collect all authorized names from historical embeddings (before current_count)
        # print("historiucal names:",historical_img_names)
        historical_authorized_names = set()
        for i, name in enumerate(historical_img_names):
                base = os.path.splitext(os.path.basename(name))[0].split('_')[0]
                # print("base",base)
                if base not in ["Unknown", "No", "face"]:
                    historical_authorized_names.add(base)
        # print("historical names: ",historical_authorized_names)
        print("\n🔹 Agglomerative Clusters:")
        for lbl, members in cluster_map.items():
            item_names = [name for name, _, _ in members]
            print(f"  Cluster {lbl}: {item_names}")

        # import matplotlib.pyplot as plt
        
        # # --- Visualization ---
        # num_clusters = len(cluster_map)
        # fig, axes = plt.subplots(num_clusters, 1, figsize=(15, 3 * num_clusters))
        
        # if num_clusters == 1:
        #     axes = [axes]  # Make iterable if only one cluster
        
        # for ax, (lbl, members) in zip(axes, cluster_map.items()):
        #     # Get crops for each cluster
        #     imgs = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for _, crop, _ in members]
        #     item_names = [os.path.basename(name) for name, _, _ in members]
        
        #     # Concatenate all images in a row
        #     try:
        #         combined = cv2.hconcat(imgs)
        #     except Exception:
        #         # Resize all to same height if mismatch
        #         heights = [img.shape[0] for img in imgs]
        #         min_h = min(heights)
        #         resized = [cv2.resize(img, (int(img.shape[1] * min_h / img.shape[0]), min_h)) for img in imgs]
        #         combined = cv2.hconcat(resized)
        
        #     ax.imshow(combined)
        #     ax.set_title(f"Cluster {lbl}: {', '.join(item_names)}", fontsize=10)
        #     ax.axis("off")
        
        # plt.tight_layout()
        # plt.show()

    
        # --- NEW ---
        # Identify "truly new" clusters (contain only new embeddings, no historical)
        # --- Identify new clusters ---
        # Collect all authorized names from historical embeddings (before current_count)
  

        pure_new_clusters = {}
        
        for lbl, members in cluster_map.items():
            indices = [all_img_names.index(name) for name, _, _ in members]
            has_historical = any(idx >= current_count for idx in indices)
            has_new = any(idx < current_count for idx in indices)
        
            # # Check for new authorized face
            # new_authorized_found = any(
            #     not os.path.splitext(os.path.basename(n))[0].startswith(("Unknown", "No", "face"))
            #     and all_img_names.index(n) < current_count
            #     for n, _, _ in members
            # )
            # Check for unseen authorized face
            new_authorized_found = any(
                not os.path.splitext(os.path.basename(n))[0].startswith(("Unknown", "No", "face"))
                and all_img_names.index(n) < current_count
                and os.path.splitext(os.path.basename(n))[0].split('_')[0] not in historical_authorized_names
                for n, _, _ in members
            )

        
            # Apply the rule
            if (has_new and not has_historical) or new_authorized_found:
                pure_new_clusters[lbl] = members
        
        # ---- Logging ----
        print(f"\n✨ Found {len(pure_new_clusters)} new or newly-updated clusters!")
        if pure_new_clusters:
            print("🆕 New / Updated Clusters:")
            for lbl, members in pure_new_clusters.items():
                item_names = [name for name, _, _ in members]
        
                # Determine reason
                has_new_authorized = any(
                    not os.path.splitext(os.path.basename(n))[0].startswith(("Unknown", "No", "face"))
                    and all_img_names.index(n) < current_count
                    for n, _, _ in members
                )
                has_historical = any(all_img_names.index(n) >= current_count for n, _, _ in members)
        
                if has_new_authorized:
                    reason = "new authorized added"
                elif not has_historical:
                    reason = "completely new cluster"
                else:
                    reason = "—"
        
                print(f"  🔸 Cluster {lbl}: {item_names}  ({reason})")
        else:
            print("No new or updated clusters found.")

        # --- Replace cluster_map with only new clusters for saving ---
        cluster_map = pure_new_clusters
    
        # Choose representative faces
        representative_faces = {}
        finalized_id_map = {}
        counter = 0
    
        for lbl, members in cluster_map.items():
            chosen_img = None
            for img_name, crop, emb in members:
                if "Unknown" not in img_name:
                    chosen_img = (img_name, crop, emb)
                    break
            if chosen_img is None:
                chosen_img = members[np.random.randint(len(members))]
    
            img_name, crop, emb = chosen_img
            _, buffer = cv2.imencode(".jpg", crop)
            b64_crop = base64.b64encode(buffer).decode("utf-8")
            # file_path=os.path.join("pipeline_images", img_name)
            # cv2.imwrite(file_path, crop)
            name_from_file = os.path.splitext(os.path.basename(img_name))[0]
            base_name = name_from_file.split('_')[0]
            final_status = "Unknown"
    
            if base_name != "Unknown":
                final_status = self.name_to_id_map.get(base_name, base_name)
    
            representative_faces[counter] = b64_crop
            finalized_id_map[counter] = {"status": final_status}
            counter += 1
    
        print(f"💾 Saved {len(representative_faces)} representative faces from new clusters.")
        return representative_faces, finalized_id_map

    # -------------- Async helpers borrowed from code2 (but extended) --------------
    async def _process_single_face(self, crop, face_obj):
        """Process a single face: embedding + FAISS search.
        Returns: (score(float), name(str), embedding_flat(np.ndarray), det_score(float))
        """
        try:
            emb = await self.face_embedder.get_async(crop, face_obj)
            emb = emb.astype("float32").reshape(1, -1)
            faiss.normalize_L2(emb)

            # FAISS search in thread pool (CPU-bound operation)
            def _faiss_search():
                return self.index.search(emb, 1)

            scores, indices = await asyncio.to_thread(_faiss_search)
            score, idx = float(scores[0][0]), int(indices[0][0])

            name = self.names[idx] if score >= self.sim_threshold else "Unknown"
            return score, name, emb.flatten().copy(), float(face_obj.det_score if hasattr(face_obj, 'det_score') else 0.0)

        except Exception as e:
            print(f"[Error in _process_single_face] {e}")
            return -1.0, "Unknown", None, 0.0

    async def _process_single_crop_async(self, tid, crop):
        """Process a single crop: face detection + per-face async embedding + FAISS search.
        Returns: tid, result_obj ({"status": ...}), crop_b64 or None, best_embedding (1D np array or None), best_conf(float)
        """
        try:
            bboxes, kpss = await self.face_detector.detect_async(crop, input_size=(640, 640))

            if bboxes.shape[0] == 0:
                return tid, {"status": "No face"}, None, None, 0.0

            best_score = -1.0
            best_name = "Unknown"
            crop_b64 = None
            best_emb = None
            best_conf = 0.0

            # Process all faces in parallel
            face_tasks = []
            face_objs = []
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, :4]
                det_score = float(bboxes[i, 4])
                if det_score < 0.5:
                    continue

                kps = kpss[i] if kpss is not None else None
                face_obj = Face(bbox=bbox, kps=kps, det_score=det_score)
                face_objs.append(face_obj)
                face_tasks.append(self._process_single_face(crop, face_obj))

            if not face_tasks:
                return tid, {"status": "No face"}, None, None, 0.0

            face_results = await asyncio.gather(*face_tasks)

            for (score, name, emb_flat, det_score) in face_results:
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_emb = emb_flat.copy() if emb_flat is not None else None
                    best_conf = det_score

            if crop.size > 0:
                try:
                    _, buffer = cv2.imencode(".jpg", crop)
                    crop_b64 = base64.b64encode(buffer).decode("utf-8")
                except Exception:
                    crop_b64 = None

            return tid, {"status": best_name}, crop_b64, best_emb, best_conf

        except Exception as e:
            print(f"[Error in _process_single_crop_async] {e}")
            return tid, {"status": "Error"}, None, None, 0.0

    # --------------------- Main async pipeline (merged) ---------------------
    async def process_frames(self, frames: list[np.ndarray],camera_id: str | None = None,warehouse_id: str | None = None):
        """Main async frame processing with TRUE parallel Triton calls and clustering integration."""
        start_time = time.perf_counter()
        self.track_best_crop.clear()
        self.id_name_map.clear()
        self.best_face_per_id.clear()

        timing_detailed = {
            "yolo_inference": 0.0,
            "tracker_inference": 0.0,
            "buffalo_detection": 0.0,
            "buffalo_embedding": 0.0,
            "faiss_search": 0.0
        }

        # STEP 1: PARALLEL YOLO INFERENCE (subset frames)
        t0_yolo = time.perf_counter()
        frames_to_process = [(idx, frame) for idx, frame in enumerate(frames) if idx % 5 == 0]
        frames_subset = [frame for _, frame in frames_to_process]
        yolo_results = await self.detector.infer_batch(
            frames_subset,
            conf_threshold=0.4,
            iou_threshold=0.7,
            max_detections=30
        )
        timing_detailed["yolo_inference"] = time.perf_counter() - t0_yolo

        # STEP 2: SEQUENTIAL TRACKING
        t0_tracker = time.perf_counter()
        tracked_frames = []
        for (frame_idx, detections), original_frame in zip(yolo_results, frames_subset):
            tracks = await asyncio.to_thread(self.detector.track, original_frame, detections)
            if tracks:
                tracked_frames.append((frame_idx, original_frame, tracks))
        timing_detailed["tracker_inference"] = time.perf_counter() - t0_tracker

        if not tracked_frames:
                total_exec_time = time.perf_counter() - start_time
                return {
                "status": "success",
                "execution_time": round(total_exec_time, 2),
                "raw_id_name_map": {},
                "finalized_id_name_map": {},
                "all_face_crops_to_upload": {},
                "timing": timing_detailed
            }

        # STEP 3: COLLECT CROPS
        all_crops_data = []  # List of (tid, crop, frame_idx)
        for frame_idx, frame, tracks in tracked_frames:
            for tr in tracks:
                x1, y1, x2, y2 = map(int, tr["bbox"])
                tid = tr["track_id"]
                h = y2 - y1
                if h <= 0:
                    continue
                face_crop = frame[y1:y1 + int(h * 0.5), x1:x2]
                if face_crop.size:
                    all_crops_data.append((tid, face_crop, frame_idx))

        if not all_crops_data:
             return {
                "status": "success",
                "execution_time": round(time.perf_counter() - start_time, 2),
                "raw_id_name_map": {},
                "finalized_id_name_map": {},
                "all_face_crops_to_upload": {},
                "timing": timing_detailed
            }         

        # STEP 4: PARALLEL BUFFALO FACE DETECTION + RECOGNITION per crop (from code2 async style)
        t0_detection = time.perf_counter()
        detection_tasks = [self._process_single_crop_async(tid, crop) for tid, crop, frame_idx in all_crops_data]
        detection_results = await asyncio.gather(*detection_tasks)
        timing_detailed["buffalo_detection"] = time.perf_counter() - t0_detection

        # STEP 5: PROCESS RESULTS, UPDATE BEST FACE PER TID
        track_best_scores = defaultdict(lambda: {"score": -1.0, "name": "Unknown", "crop_b64": None})

        # We also accumulate embeddings for clustering (current + historical)
        for tid, result_obj, crop_b64, embedding_vec, det_conf in detection_results:
            # update id_name_map
            if tid not in self.id_name_map:
                self.id_name_map[tid] = []
            self.id_name_map[tid].append(result_obj)

            # update track_best_scores for quick crop upload
            status_name = result_obj.get("status", "Unknown")
            # we had crop_b64 if present
            if embedding_vec is not None:
                score_for_compare = float( self.sim_threshold if status_name == "Unknown" else 1.0 )  # fallback
                # We stored FAISS score in best selection earlier; but here embedding_vec present -> use det_conf as conf
                if det_conf > track_best_scores[tid]["score"]:
                    track_best_scores[tid]["score"] = det_conf
                    track_best_scores[tid]["name"] = status_name
                    if crop_b64:
                        track_best_scores[tid]["crop_b64"] = crop_b64

            # update best_face_per_id using embedding (if available)
            if embedding_vec is not None:
                self._update_best_face(tid, cv2.imdecode(np.frombuffer(base64.b64decode(crop_b64), np.uint8), cv2.IMREAD_COLOR) if crop_b64 else np.zeros((1,1,3), np.uint8),
                                       status_name, float(det_conf), embedding=np.array(embedding_vec, dtype="float32"))

            # also keep track_best_crop for quick upload UI
            if crop_b64:
                self.track_best_crop[tid] = crop_b64

        # STEP: Save best faces, clustering (follows code1 logic)
        t0_save = time.perf_counter()
        save_best_faces,finalized_id_map = self._save_all_best_faces(camera_id=camera_id,warehouse_id=warehouse_id)
        self.timing["save_face"] += time.perf_counter() - t0_save

        total_exec_time = time.perf_counter() - start_time
        # finalized = self._finalize_results()

        # Performance prints
        total_buffalo_time = (timing_detailed["buffalo_detection"] +
                              timing_detailed["buffalo_embedding"] +
                              timing_detailed["faiss_search"])

        print("\n" + "="*60)
        print("⚡ PERFORMANCE SUMMARY")
        print("="*60)
        print(f"📊 YOLO Inference:        {timing_detailed['yolo_inference']:.4f}s")
        print(f"🎯 Tracker:               {timing_detailed['tracker_inference']:.4f}s")
        print(f"🔍 Buffalo Detection:     {timing_detailed['buffalo_detection']:.4f}s")
        print(f"🧠 Buffalo Embedding:     {timing_detailed['buffalo_embedding']:.4f}s")
        print(f"🔎 FAISS Search:          {timing_detailed['faiss_search']:.4f}s")
        print(f"📦 Total Buffalo Time:    {total_buffalo_time:.4f}s")
        print("-" * 60)
        print(f"⏱️  Total Execution:       {total_exec_time:.4f}s")
        print(f"🚀 Parallel Speedup:      {(sum(timing_detailed.values()) / total_exec_time):.2f}x")
        print("="*60)
        print(f"✅ Processed {len(all_crops_data)} crops → Found {len([r for r in detection_results if r[1]['status'] not in ['No face','Error']])} faces")
        print(f"✅ Identified {len([f for f in finalized_id_map.values() if f['status'] not in ['Unknown', 'No face']])} known persons")
        print("="*60 + "\n")

        # Persist maps to disk for debug / downstream use
        # try:
        #     with open("id_name_map.json", "w") as f:
        #         json.dump(self.id_name_map, f, indent=2)
        #     with open("finalized_id_name_map.json", "w") as f:
        #         json.dump(finalized_id_map, f, indent=2)
        # except Exception:
        #     pass

        return {
            "status": "success",
            "execution_time": round(total_exec_time, 2),
            "raw_id_name_map": self.id_name_map,
            "finalized_id_name_map": finalized_id_map,
            "all_face_crops_to_upload": save_best_faces,
            "timing": timing_detailed,
             "stats": {
                "total_crops": len(all_crops_data),
                "total_faces": len([r for r in detection_results if r[1]['status'] not in ['No face','Error']]),
                "identified_persons": len([f for f in finalized_id_map.values() if f['status'] not in ['Unknown', 'No face']])
            }
        }

    # def _finalize_results(self):
    #     finalized = {}
    #     for tid, result_obj_list in self.id_name_map.items():
    #         valid = [obj for obj in result_obj_list if obj["status"] not in ["Unknown", "No face", "Error"]]
    #         if valid:
    #             names_count = Counter([obj["status"] for obj in valid])
    #             final_name = names_count.most_common(1)[0][0]
    #         else:
    #             if any(obj["status"] == "Unknown" for obj in result_obj_list):
    #                 final_name = "Unknown"
    #             else:
    #                 final_name = "No face"
    #         finalized[tid] = {"status": final_name}
    #     return finalized

# ---------------- Async wrapper ----------------
async def run_face_recognition(frames: list[np.ndarray],cam_id: str | None = None,warehouse_id: str | None=None):
    pipeline = FaceReIDPipeline()
    return await pipeline.process_frames(frames,camera_id=cam_id,warehouse_id=warehouse_id)