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
# ------------------- Person Detector -------------------
class YoloPersonDetector:
    # def __init__(self, triton_url="provider.rtx4090.wyo.eg.akash.pub:30609/", model_name="yolo_person_detection"):
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

    async def infer(self, image: np.ndarray, conf_threshold: float, iou_threshold: float, max_detections: int):
        """Asynchronously run inference using a thread to avoid blocking."""
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
 
        # Post-processing (this is fast, so it can stay here)
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
 
        return np.array(results, dtype=np.float32)
 
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

# Keep all utility functions in face_recognition.py
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
 
 
 
#arcface_utils
 
 
 
arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
 
def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
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
 
def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im, scale
 
 
def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M
 
 
def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]
 
    return new_pts
 
 
def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    #print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale
 
    return new_pts
 
 
def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)
 
 
#arcface
 
 
#from arcface_utils import norm_crop
_all_ = [
    'ArcFaceONNX',
]
 
class ArcFaceONNX:
    def __init__(self, model_name=None, url="provider.rtx4090.wyo.eg.akash.pub:30247"):
        assert model_name is not None, "Provide Triton model name"
        self.model_name = model_name
        self.taskname = 'recognition'
        self.url = url
 
        # Triton client
        # self.client = httpclient.InferenceServerClient(url=self.url)
        self.client = grpcclient.InferenceServerClient(url=self.url)
 
 
        # ArcFace default preprocessing (mxnet vs normal)
        # Adjust as needed for your model
        self.input_mean = 127.5
        self.input_std = 127.5
 
        # # Get model metadata from Triton
        # model_metadata = self.client.get_model_metadata(model_name=self.model_name)
        # model_config = self.client.get_model_config(model_name=self.model_name)
 
        # # Extract input name, output name, and input shape
        # self.input_name = model_metadata['inputs'][0]['name']
        # self.output_names = [o['name'] for o in model_metadata['outputs']]
        # self.input_shape = model_metadata['inputs'][0]['shape']
        # self.input_size = (self.input_shape[2], self.input_shape[3])  # (W, H)
        # self.output_shape = model_metadata['outputs'][0]['shape']
        # Get model metadata from Triton
        model_metadata = self.client.get_model_metadata(model_name=self.model_name)
        model_config = self.client.get_model_config(model_name=self.model_name)
        # Extract input name, output name, and input shape
       
        self.input_name = model_metadata.inputs[0].name
        self.output_names = [o.name for o in model_metadata.outputs]
        self.input_shape = list(model_metadata.inputs[0].shape)
        self.input_size = (self.input_shape[2], self.input_shape[3])
        self.output_shape = list(model_metadata.outputs[0].shape)
 
 
 
    def prepare(self, ctx_id, **kwargs):
        pass  # Triton handles CPU/GPU automatically on the server
 
    def get(self, img, face):
        aimg = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        face.embedding = self.get_feat(aimg).flatten()
        return face.embedding
 
    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim
 
    def get_feat(self, imgs):
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
 
        # Send to Triton
        # inputs = []
        # inputs.append(httpclient.InferInput(self.input_name, blob.shape, "FP32"))
        # inputs[0].set_data_from_numpy(blob)
 
        # outputs = []
        # for out_name in self.output_names:
        #     outputs.append(httpclient.InferRequestedOutput(out_name))
        inputs = []
        inputs.append(grpcclient.InferInput(self.input_name, blob.shape, "FP32"))
        inputs[0].set_data_from_numpy(blob)
        outputs = []
        for out_name in self.output_names:
            outputs.append(grpcclient.InferRequestedOutput(out_name))
 
 
        result = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
 
        net_out = result.as_numpy(self.output_names[0])
        return net_out
 
    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
 
        inputs = [httpclient.InferInput(self.input_name, blob.shape, "FP32")]
        inputs[0].set_data_from_numpy(blob)
 
        outputs = [httpclient.InferRequestedOutput(self.output_names[0])]
        result = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
 
        net_out = result.as_numpy(self.output_names[0])
        return net_out
 
#retinaface
 
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
 
        # Triton gRPC client setup
        self.triton_client = grpcclient.InferenceServerClient(url=triton_url)
        self.model_name = model_name
 
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
 
        self._init_vars()
 
    def _init_vars(self):
        self.input_name = "input.1"
        self.output_names = [
            "448", "471", "494",  # scores
            "451", "474", "497",  # bboxes
            "454", "477", "500"   # kps
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
 
    def forward(self, img, threshold):
        scores_list, bboxes_list, kpss_list = [], [], []
        input_size = tuple(img.shape[0:2][::-1])
 
        blob = cv2.dnn.blobFromImage(img, 1.0 / self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean),
                                     swapRB=True)
 
        # Triton gRPC input
        inputs = [grpcclient.InferInput(self.input_name, blob.shape, np_to_triton_dtype(blob.dtype))]
        inputs[0].set_data_from_numpy(blob)
 
        # Triton gRPC outputs
        outputs = [grpcclient.InferRequestedOutput(name) for name in self.output_names]
 
        # Run inference
        results = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
 
        net_outs = [results.as_numpy(name) for name in self.output_names]
 
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
 
    def detect(self, img, input_size=None, max_num=0, metric='default'):
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
 
        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)
 
        scores = np.vstack(scores_list)
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
 
def get_retinaface(name, download=False, root='~/.insightface/models', **kwargs):
    if not download:
        assert os.path.exists(name)
        return RetinaFace(name)
    else:
        from .model_store import get_model_file
        _file = get_model_file("retinaface_%s" % name, root=root)
        return RetinaFace(_file)
 
 
#face
class Face(dict):
 
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        #for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))
 
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
        return 'M' if self.gender==1 else 'F'
 
 
from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Import your existing YOLO, RetinaFace, ArcFace classes
# from some_module import YoloPersonDetector, RetinaFace, ArcFaceONNX, Face

class FaceReIDPipeline:
    def __init__(self, sim_threshold=0.40):
        # Config
        self.ist = pytz.timezone("Asia/Kolkata")
        self.sim_threshold = sim_threshold

        # ---------------- Triton YOLO Detector ----------------
        self.triton_yolo_url = "provider.rtx4090.wyo.eg.akash.pub:30247"
        self.yolo_model_name = "yolo_person_detection"
        self.detector = YoloPersonDetector(triton_url=self.triton_yolo_url, model_name=self.yolo_model_name)

        # ---------------- Triton RetinaFace + ArcFace ----------------
        self.triton_buffalo_url = "provider.rtx4090.wyo.eg.akash.pub:30247"
        self.buffalo_face_model = "buffalo_face_detection"
        self.buffalo_embed_model = "buffalo_face_embedding"

        self.face_detector = RetinaFace(triton_url=self.triton_buffalo_url, model_name=self.buffalo_face_model)
        self.face_detector.prepare(ctx_id=0, input_size=(640, 640))

        self.face_embedder = ArcFaceONNX(url=self.triton_buffalo_url, model_name=self.buffalo_embed_model)
        self.face_embedder.prepare(ctx_id=0, input_size=(112, 112))

        # ---------------- Load FAISS index ----------------
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(current_dir, "face_database_gudivada_format.pkl")
        # self.db_path = os.path.join(project_root, "data", "face_database_gudivada_format.pkl")
        self.names, self.index = self._load_faiss_index(self.db_path)

        # ---------------- State holders ----------------
        self.id_name_map = {}
        self.best_face_per_id = {}  # tid -> (crop, name, conf, embedding)
        self.track_best_embeddings = {}
        self.timing = defaultdict(float)
        for k in ["yolo_inference", "tracker_inference", "buffalo_processing", "faiss_search", "update_face", "save_face", "clustering"]:
            self.timing.setdefault(k, 0.0)

    def _load_faiss_index(self, db_path):
        with open(db_path, "rb") as f:
            face_db_list = pickle.load(f)
        try:
            ids, names, roles, embeddings = zip(*face_db_list)
            names = np.array(names)
            embeddings = np.stack(embeddings).astype("float32")
        except Exception:
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

    def _update_best_face(self, tid, crop, name, conf, embedding=None):
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

    def _save_all_best_faces(self):
        # os.makedirs("pipeline_images", exist_ok=True)

        authorized_best = {}
        unauthorized_best = {}

        for tid, (crop, name, conf, embedding) in self.best_face_per_id.items():
            if name not in ["Unknown", "No face"]:
                if name not in authorized_best or conf > authorized_best[name][2]:
                    authorized_best[name] = (crop.copy(), name, conf, embedding.copy() if embedding is not None else None)
            elif name == "Unknown":
                if tid not in unauthorized_best or conf > unauthorized_best[tid][2]:
                    unauthorized_best[tid] = (crop.copy(), name, conf, embedding.copy() if embedding is not None else None)

        embeddings_list = []
        save_best_faces = {}

        # Combine all authorized and unauthorized for clustering
        all_embeddings = []
        all_img_names = []
        all_crops = []
        for name, (crop, _, conf, embedding) in authorized_best.items():
            all_embeddings.append(embedding)
            all_img_names.append(f"{name}_{uuid.uuid4().hex[:8]}.jpg")
            all_crops.append(crop.copy())
        for tid, (crop, name, conf, embedding) in unauthorized_best.items():
            all_embeddings.append(embedding)
            all_img_names.append(f"{name}_ID{tid}_{uuid.uuid4().hex[:8]}.jpg")
            all_crops.append(crop.copy())
        # print("all embeddings:"all_embeddings)
        # os.makedirs("best_crops", exist_ok=True)
        # Save authorized crops
        for name, (crop, _, conf, embedding) in authorized_best.items():
            filename = f"{name}_{uuid.uuid4().hex[:8]}.jpg"
            # filepath = os.path.join("best_crops", filename)
            # cv2.imwrite(filepath, crop)
            print(f"✅ Saved best authorized crop for {name} | conf={conf:.2f}")
            embeddings_list.append((filename, embedding))

        # Save unauthorized crops (Unknown/No face)
        for tid, (crop, name, conf, embedding) in unauthorized_best.items():
            filename = f"{name}_ID{tid}_{uuid.uuid4().hex[:8]}.jpg"
            # filepath = os.path.join("best_crops", filename)
            # cv2.imwrite(filepath, crop)
            print(f"⚠️  Saved best unauthorized crop for ID:{tid} ({name}) | conf={conf:.2f}")
            embeddings_list.append((filename, embedding))

        # print("saved to best_crops folder")
        all_embeddings = np.array(all_embeddings, dtype="float32")

        # =====================================================
        #           Agglomerative Clustering
        # =====================================================
        if len(all_embeddings) > 1:
            # Find best threshold using silhouette score
            best_t, best_score = 0.5, -1
            for t in np.arange(0.1, 1.0, 0.05):
                model = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=t,
                    metric="cosine",
                    linkage="average"
                )
                labels = model.fit_predict(all_embeddings)
                n_labels = len(set(labels))
                if n_labels < 2 or n_labels >= len(all_embeddings):
                    continue
                score = silhouette_score(all_embeddings, labels)
                if score > best_score:
                    best_score, best_t = score, t
            print(f"➡ Agglomerative: Chosen distance_threshold = {best_t:.3f}")
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=best_t,
                metric="cosine",
                linkage="average"
            )
            labels = model.fit_predict(all_embeddings)
            self.timing["clustering"] += 0.0
        else:
            labels = np.zeros(len(all_embeddings), dtype=int)

        # Pick representative from each cluster
        cluster_map = defaultdict(list)
        for lbl, img_name, crop, embedding in zip(labels, all_img_names, all_crops, all_embeddings):
            cluster_map[lbl].append((img_name, crop, embedding))
        # Print clusters with items
        print("\n🔹 Agglomerative Clusters:")
        for lbl, members in cluster_map.items():
            item_names = [name for name, _, _ in members]
            print(f"  Cluster {lbl}: {item_names}")

        representative_faces = {}
        finalized_id_map={}
        counter=0
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
            file_path=img_name
            name = os.path.splitext(os.path.basename(file_path))[0]
            # cv2.imwrite(filepath, crop)
            # representative_faces[img_name] = {"type": "cluster_representative", "conf": 1.0, "path": filepath, "embedding": emb}
            
            representative_faces[counter]=b64_crop
            finalized_id_map[counter]=name
            counter+=1

        # # Save embeddings of representatives
        # with open("pipeline_images_embeddings.pkl", "wb") as f:
        #     pickle.dump([(name, data["embedding"]) for name, data in representative_faces.items()], f)

        print(f"💾 Saved {len(representative_faces)} representative faces.")
        return representative_faces,finalized_id_map

    def _process_crop_sync(self, tid, crop):
        try:
            t0_buffalo = time.perf_counter()
            bboxes, kpss = self.face_detector.detect(crop, input_size=(640, 640))
            self.timing["buffalo_processing"] += time.perf_counter() - t0_buffalo

            if bboxes.shape[0] == 0:
                return tid, {"status": "No face"}, None, None, 0.0

            best_score = -1.0
            best_name = "Unknown"
            best_emb = None
            best_conf = 0.0

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
                    best_emb = emb.flatten().copy()
                    best_conf = det_score
            
            return tid, {"status": best_name}, None, best_emb, best_conf

        except Exception as e:
            print(f"[Error in _process_crop_sync] {e}")
            return tid, {"status": "Error"}, None, None, 0.0

    async def process_frames(self, frames: list[np.ndarray]):
        start_time = time.perf_counter()
        self.best_face_per_id.clear()
        self.id_name_map.clear()
        for k in ["yolo_inference", "tracker_inference", "buffalo_processing", "faiss_search", "update_face", "save_face", "clustering"]:
            self.timing.setdefault(k, 0.0)

        for frame_idx, frame in enumerate(frames):
            if frame_idx % 5 != 0:
                continue

            t0 = time.perf_counter()
            detections = await self.detector.infer(frame, conf_threshold=0.4, iou_threshold=0.7, max_detections=30)
            self.timing["yolo_inference"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            tracks = await asyncio.to_thread(self.detector.track, frame, detections)
            self.timing["tracker_inference"] += time.perf_counter() - t0

            if not tracks:
                continue

            crop_processing_tasks = []
            for tr in tracks:
                x1, y1, x2, y2 = map(int, tr["bbox"])
                tid = tr["track_id"]
                h = y2 - y1
                if h <= 0:
                    continue
                face_crop = frame[y1:y1 + int(h * 0.5), x1:x2]
                if face_crop.size:
                    task = asyncio.to_thread(self._process_crop_sync, tid, face_crop)
                    crop_processing_tasks.append(task)

            if crop_processing_tasks:
                crop_results = await asyncio.gather(*crop_processing_tasks)
                for tid, result_obj, _, emb, conf in crop_results:
                    name = result_obj.get("status", "No face")
                    if tid not in self.id_name_map:
                        self.id_name_map[tid] = []
                    self.id_name_map[tid].append(name)

                    matching_bbox = next((tr["bbox"] for tr in tracks if tr["track_id"] == tid), None)
                    if matching_bbox:
                        x1, y1, x2, y2 = map(int, matching_bbox)
                        h = y2 - y1
                        face_crop = frame[y1:y1 + int(h * 0.5), x1:x2] if h > 0 else None
                    else:
                        face_crop = None

                    t1 = time.perf_counter()
                    if face_crop is not None and face_crop.size:
                        self._update_best_face(tid, face_crop, name, conf, embedding=(emb.flatten() if emb is not None else None))
                    else:
                        if emb is not None:
                            placeholder = np.zeros((112, 112, 3), dtype=np.uint8)
                            self.best_face_per_id[tid] = (placeholder, name, conf, emb.flatten().copy())
                    self.timing["update_face"] += time.perf_counter() - t1

        total_exec_time = time.perf_counter() - start_time
        finalized = self._finalize_results()

        t0 = time.perf_counter()
        save_best_faces,finalized_id_map = self._save_all_best_faces()
        self.timing["save_face"] += time.perf_counter() - t0

        with open("id_name_map.json", "w") as f:
            json.dump(self.id_name_map, f, indent=2)
        with open("finalized_id_name_map.json", "w") as f:
            json.dump(finalized, f, indent=2)

        return {
            "status": "success",
            "execution_time": round(total_exec_time, 2),
            "raw_id_name_map": self.id_name_map,
            "finalized_id_name_map": finalized_id_map,
            "all_face_crops_to_upload": save_best_faces
        }

    def _finalize_results(self):
        finalized = {}
        for tid, result_obj_list in self.id_name_map.items():
            valid = [obj for obj in result_obj_list if obj not in ["Unknown", "No face", "Error"]]
            if valid:
                names_count = Counter(valid)
                final_name = names_count.most_common(1)[0][0]
            else:
                if any(obj == "Unknown" for obj in result_obj_list):
                    final_name = "Unknown"
                else:
                    final_name = "No face"
            finalized[tid] = final_name
        return finalized


# ---------------- Async wrapper ----------------
async def run_face_recognition(frames: list[np.ndarray]):
    pipeline = FaceReIDPipeline()
    return await pipeline.process_frames(frames)

