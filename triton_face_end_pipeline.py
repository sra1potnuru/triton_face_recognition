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
 
# ------------------- Person Detector -------------------
class YoloPersonDetector:
    # def __init__(self, triton_url="provider.rtx4090.wyo.eg.akash.pub:30609/", model_name="yolo_person_detection"):
    def __init__(self, triton_url="provider.rtx4090.wyo.eg.akash.pub:30247", model_name="yolo_person_detection"):
        self.model_name = model_name
        # self.client = httpclient.InferenceServerClient(url=triton_url, ssl=False)
        # NEW
        self.client = grpcclient.InferenceServerClient(url=triton_url)
        self.input_name = "images"
        self.output_name = "output0"
        self.input_size = 640
        self.target_class = 0  # person
 
        # Init BoT-SORT tracker
        tracker_args = {
            "tracker_type": "botsort",
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.3,
            "new_track_thresh": 0.5,
            "track_buffer": 30,
            "match_thresh": 0.9,
            "fuse_score": True,
            "gmc_method": None,
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.8,
            "with_reid": False,
            "model": "auto"
        }
        args = SimpleNamespace(**tracker_args)
        self.tracker = BOTSORT(args=args)
 
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
   
        # Perform tracking
        tracks = self.tracker.update(dets_to_track, image)
   
        tracked = []
        for t in tracks:
            x1, y1, x2, y2, track_id, cls_id, conf = t[:7]
            tracked.append({
                "track_id": int(track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(conf)
            })
   
        return tracked
 
 
 
# ------------------- Utils -------------------
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
 
 
class FaceReIDPipeline:
    def __init__(self, sim_threshold=0.40):
        # Config
        self.ist = pytz.timezone("Asia/Kolkata")
        self.sim_threshold = sim_threshold
       
        # # ---------------- Use Triton-loaded models ----------------
        # self.detector = detector
        # self.face_detector = face_detector
        # self.face_embedder = face_embedder
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
        project_root = os.path.dirname(current_dir)
        self.db_path = os.path.join(current_dir, "", "face_database_nandhyala_format.pkl")
 
        self.names, self.index = self._load_faiss_index(self.db_path)
 
        # ---------------- State holders ----------------
        self.id_name_map = {}
        self.track_best_crop = {}
        self.track_best_embeddings = {}
        self.timing = defaultdict(float)
 
    # def _load_faiss_index(self, db_path):
    #     with open(db_path, "rb") as f:
    #         face_db_list = pickle.load(f)
    #     names, embeddings = zip(*face_db_list)
    #     names = np.array(names)
    #     embeddings = np.stack(embeddings).astype("float32")
 
    #     N, dim = len(embeddings), embeddings.shape[1]
    #     nlist = max(1, min(N, int(4 * np.sqrt(N))))
    #     nprobe = max(1, nlist // 4)
 
    #     quantizer = faiss.IndexFlatIP(dim)
    #     index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    #     print("Training FAISS index...")
    #     index.train(embeddings)
    #     faiss.normalize_L2(embeddings)
    #     index.add(embeddings)
    #     index.nprobe = nprobe
 
    #     print(f"--- FAISS Config: N={N}, dim={dim}, nlist={nlist}, nprobe={nprobe} ---")
    #     return names, index
    def _load_faiss_index(self, db_path):
        with open(db_path, "rb") as f:
            face_db_list = pickle.load(f)
   
        # New format: (id, name, role, embedding)
        ids, names, roles, embeddings = zip(*face_db_list)
        ids = np.array(ids)          # optional, if you want to track ID separately
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
                # return tid, {"status": "No face"}, None
                return tid, {"status": "No face"}, None, None
 
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
 
            # return tid, {"status": best_name}, crop_b64
            return tid, {"status": best_name}, crop_b64, emb

 
        except Exception as e:
            print(f"[Error in _process_crop_sync] {e}")
            # return tid, {"status": "Error"}, None
            return tid, {"status": "Error"}, None, None

 
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
            # FIX: Await the async infer method
            detections = await self.detector.infer(frame, conf_threshold=0.4, iou_threshold=0.7, max_detections=30)
            self.timing["yolo_inference"] += time.perf_counter() - t0
 
            t0 = time.perf_counter()
            # Tracker is CPU-bound, run it in a thread
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
                    # FIX: Run the synchronous crop processing in a thread
                    task = asyncio.to_thread(self._process_crop_sync, tid, face_crop)
                    crop_processing_tasks.append(task)
           
            if crop_processing_tasks:
                # Gather results from all crop processing tasks for this frame
                crop_results = await asyncio.gather(*crop_processing_tasks)
                # for tid, result_obj, crop_b64 in crop_results:
                for tid, result_obj, crop_b64, emb in crop_results:
                    if tid not in self.id_name_map:
                        self.id_name_map[tid] = []
                    self.id_name_map[tid].append(result_obj)
 
                    if crop_b64:
                        self.track_best_crop[tid] = crop_b64
                    if emb is not None:
                        self.track_best_embeddings[tid] = emb.flatten()
        
 
        total_exec_time = time.perf_counter() - start_time
        finalized = self._finalize_results()
        
        # # Folder where images will be saved
        save_folder = "all_face_crops_to_upload"
        os.makedirs(save_folder, exist_ok=True)

        all_face_crops_to_upload = {
            str(tid): self.track_best_crop[tid]
            for tid, result_obj in finalized.items()
            if result_obj["status"] not in ["No face", "Error"] and tid in self.track_best_crop
        }
        # # Save each image
        # for tid, b64_string in all_face_crops_to_upload.items():
        #     # Decode the base64 string
        #     image_data = base64.b64decode(b64_string)
        #     # # Create a filename
        #     # filename = os.path.join(save_folder, f"{tid}.jpg")
            
        #     # Save it as a file
        #     with open(f"{tid}.jpg", "wb") as f:
        #         f.write(image_data)
        embeddings_dict = {}
        print("Crops to upload:", all_face_crops_to_upload.keys())

        for tid, b64_string in all_face_crops_to_upload.items():
            image_data = base64.b64decode(b64_string)
            filename = os.path.join(save_folder, f"{tid}.jpg")
        
            # Save face image
            with open(filename, "wb") as f:
                f.write(image_data)
        
            # Store embedding for this image
             # FIX: Convert tid to int before lookup
            tid_int = int(tid)
            if tid_int in self.track_best_embeddings:
                embeddings_dict[f"{tid_int}.jpg"] = self.track_best_embeddings[tid_int]
        print("Finalized:", finalized)
        print("Track crops:", self.track_best_crop.keys())
        print("Embeddings:", self.track_best_embeddings.keys())
        print("Embeddings dict before saving:", embeddings_dict.keys())
        # Save all embeddings as a pickle file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(current_dir, "face_crops_embeddings.pkl")
        print("pickle path: ",pkl_path)

        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(embeddings_dict, pkl_file)
            print(f"✅ Saved {len(embeddings_dict)} embeddings to {pkl_path}")

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

# from __future__ import division
# import os
# import os.path as osp
# import cv2
# import numpy as np
# import pickle
# import json
# import pytz
# import time
# import base64
# import asyncio
# import torch
# from collections import defaultdict, Counter
# from types import SimpleNamespace
# from skimage import transform as trans
# import faiss
# import tritonclient.grpc as grpcclient
# from tritonclient.utils import np_to_triton_dtype
# from ultralytics.engine.results import Boxes
# from ultralytics.trackers.bot_sort import BOTSORT
# from numpy.linalg import norm as l2norm
 
# # ------------------- Person Detector -------------------
# class YoloPersonDetector:
#     # def __init__(self, triton_url="provider.rtx4090.wyo.eg.akash.pub:30609/", model_name="yolo_person_detection"):
#     def __init__(self, triton_url="provider.rtx4090.wyo.eg.akash.pub:30247", model_name="yolo_person_detection"):
#         self.model_name = model_name
#         # self.client = httpclient.InferenceServerClient(url=triton_url, ssl=False)
#         # NEW
#         self.client = grpcclient.InferenceServerClient(url=triton_url)
#         self.input_name = "images"
#         self.output_name = "output0"
#         self.input_size = 640
#         self.target_class = 0  # person
 
#         # Init BoT-SORT tracker
#         tracker_args = {
#             "tracker_type": "botsort",
#             "track_high_thresh": 0.5,
#             "track_low_thresh": 0.3,
#             "new_track_thresh": 0.5,
#             "track_buffer": 30,
#             "match_thresh": 0.9,
#             "fuse_score": True,
#             "gmc_method": None,
#             "proximity_thresh": 0.5,
#             "appearance_thresh": 0.8,
#             "with_reid": False,
#             "model": "auto"
#         }
#         args = SimpleNamespace(**tracker_args)
#         self.tracker = BOTSORT(args=args)
 
#     async def infer(self, image: np.ndarray, conf_threshold: float, iou_threshold: float, max_detections: int):
#         """Asynchronously run inference using a thread to avoid blocking."""
#         # Preprocess (letterbox)
#         letterboxed, scale, pad_left, pad_top = letterbox(image, (self.input_size, self.input_size))
#         tensor = letterboxed.astype(np.float32) / 255.0
#         tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
 
#         # Define the synchronous blocking call
#         def _blocking_infer():
#             inputs = [grpcclient.InferInput(self.input_name, tensor.shape, np_to_triton_dtype(tensor.dtype))]
#             inputs[0].set_data_from_numpy(tensor)
#             outputs = [grpcclient.InferRequestedOutput(self.output_name)]
#             return self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
 
#         # Run the blocking call in a separate thread
#         response = await asyncio.to_thread(_blocking_infer)
#         detections = response.as_numpy(self.output_name)[0]
 
#         # Post-processing (this is fast, so it can stay here)
#         filtered = detections[detections[:, 4] >= conf_threshold]
#         filtered = apply_nms(filtered, iou_threshold)
 
#         results = []
#         for row in filtered:
#             cls_id = int(round(row[5]))
#             if cls_id != self.target_class:
#                 continue
#             x1, y1, x2, y2 = row[:4]
#             score = float(row[4])
 
#             # Undo letterbox
#             x1 = (x1 - pad_left) / scale
#             y1 = (y1 - pad_top) / scale
#             x2 = (x2 - pad_left) / scale
#             y2 = (y2 - pad_top) / scale
 
#             x1 = max(min(x1, image.shape[1]), 0.0)
#             y1 = max(min(y1, image.shape[0]), 0.0)
#             x2 = max(min(x2, image.shape[1]), 0.0)
#             y2 = max(min(y2, image.shape[0]), 0.0)
 
#             results.append([x1, y1, x2, y2, score, cls_id])
#             if len(results) == max_detections:
#                 break
 
#         return np.array(results, dtype=np.float32)
 
#     def track(self, image: np.ndarray, detections: np.ndarray):
#         if len(detections) == 0:
#             return []
   
#         # Convert to torch Boxes format
#         dets_tensor = torch.tensor(detections, dtype=torch.float32)
#         dets = Boxes(dets_tensor, image.shape[:2])
#         dets_to_track = dets
   
#         # Add a dummy detection if only one detection exists
#         if len(dets) == 1:
#             dummy_class = dets_tensor[0, 5]
#             dummy_det = torch.tensor([[0, 0, 1, 1, 0.01, dummy_class]],
#                                      device=dets_tensor.device,
#                                      dtype=dets_tensor.dtype)
#             padded_dets_tensor = torch.vstack([dets_tensor, dummy_det])
#             dets_to_track = Boxes(padded_dets_tensor, image.shape[:2])
   
#         # Perform tracking
#         tracks = self.tracker.update(dets_to_track, image)
   
#         tracked = []
#         for t in tracks:
#             x1, y1, x2, y2, track_id, cls_id, conf = t[:7]
#             tracked.append({
#                 "track_id": int(track_id),
#                 "bbox": [float(x1), float(y1), float(x2), float(y2)],
#                 "score": float(conf)
#             })
   
#         return tracked
 
 
 
# # ------------------- Utils -------------------
# def apply_nms(detections: np.ndarray, iou_threshold: float) -> np.ndarray:
#     if len(detections) == 0:
#         return detections
#     boxes = detections[:, :4]
#     scores = detections[:, 4]
#     x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = scores.argsort()[::-1]
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         iou = inter / (areas[i] + areas[order[1:]] - inter)
#         inds = np.where(iou <= iou_threshold)[0]
#         order = order[inds + 1]
#     return detections[keep]
 
# def letterbox(image: np.ndarray, size: tuple[int, int]):
#     target_w, target_h = size
#     h, w = image.shape[:2]
#     scale = min(target_w / w, target_h / h)
#     new_w, new_h = int(round(w * scale)), int(round(h * scale))
#     resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#     pad_w, pad_h = target_w - new_w, target_h - new_h
#     pad_left, pad_top = pad_w / 2.0, pad_h / 2.0
#     left, right = int(np.floor(pad_left)), int(np.ceil(pad_w - np.floor(pad_left)))
#     top, bottom = int(np.floor(pad_top)), int(np.ceil(pad_h - np.floor(pad_top)))
#     bordered = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
#     return bordered, scale, float(left), float(top)
 
# def draw_tracks(image: np.ndarray, tracks: list[dict]) -> np.ndarray:
#     canvas = image.copy()
#     for tr in tracks:
#         x1, y1, x2, y2 = map(int, tr["bbox"])
#         tid = tr["track_id"]
#         score = tr["score"]
#         cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(canvas, f"ID:{tid} {score:.2f}", (x1, max(y1 - 5, 0)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     return canvas
 
 
 
# #arcface_utils
 
 
 
# arcface_dst = np.array(
#     [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
#      [41.5493, 92.3655], [70.7299, 92.2041]],
#     dtype=np.float32)
 
# def estimate_norm(lmk, image_size=112,mode='arcface'):
#     assert lmk.shape == (5, 2)
#     assert image_size%112==0 or image_size%128==0
#     if image_size%112==0:
#         ratio = float(image_size)/112.0
#         diff_x = 0
#     else:
#         ratio = float(image_size)/128.0
#         diff_x = 8.0*ratio
#     dst = arcface_dst * ratio
#     dst[:,0] += diff_x
#     tform = trans.SimilarityTransform()
#     tform.estimate(lmk, dst)
#     M = tform.params[0:2, :]
#     return M
 
# def norm_crop(img, landmark, image_size=112, mode='arcface'):
#     M = estimate_norm(landmark, image_size, mode)
#     warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
#     return warped
 
# def norm_crop2(img, landmark, image_size=112, mode='arcface'):
#     M = estimate_norm(landmark, image_size, mode)
#     warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
#     return warped, M
 
# def square_crop(im, S):
#     if im.shape[0] > im.shape[1]:
#         height = S
#         width = int(float(im.shape[1]) / im.shape[0] * S)
#         scale = float(S) / im.shape[0]
#     else:
#         width = S
#         height = int(float(im.shape[0]) / im.shape[1] * S)
#         scale = float(S) / im.shape[1]
#     resized_im = cv2.resize(im, (width, height))
#     det_im = np.zeros((S, S, 3), dtype=np.uint8)
#     det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
#     return det_im, scale
 
 
# def transform(data, center, output_size, scale, rotation):
#     scale_ratio = scale
#     rot = float(rotation) * np.pi / 180.0
#     #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
#     t1 = trans.SimilarityTransform(scale=scale_ratio)
#     cx = center[0] * scale_ratio
#     cy = center[1] * scale_ratio
#     t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
#     t3 = trans.SimilarityTransform(rotation=rot)
#     t4 = trans.SimilarityTransform(translation=(output_size / 2,
#                                                 output_size / 2))
#     t = t1 + t2 + t3 + t4
#     M = t.params[0:2]
#     cropped = cv2.warpAffine(data,
#                              M, (output_size, output_size),
#                              borderValue=0.0)
#     return cropped, M
 
 
# def trans_points2d(pts, M):
#     new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
#     for i in range(pts.shape[0]):
#         pt = pts[i]
#         new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
#         new_pt = np.dot(M, new_pt)
#         #print('new_pt', new_pt.shape, new_pt)
#         new_pts[i] = new_pt[0:2]
 
#     return new_pts
 
 
# def trans_points3d(pts, M):
#     scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
#     #print(scale)
#     new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
#     for i in range(pts.shape[0]):
#         pt = pts[i]
#         new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
#         new_pt = np.dot(M, new_pt)
#         #print('new_pt', new_pt.shape, new_pt)
#         new_pts[i][0:2] = new_pt[0:2]
#         new_pts[i][2] = pts[i][2] * scale
 
#     return new_pts
 
 
# def trans_points(pts, M):
#     if pts.shape[1] == 2:
#         return trans_points2d(pts, M)
#     else:
#         return trans_points3d(pts, M)
 
 
# #arcface
 
 
# #from arcface_utils import norm_crop
# _all_ = [
#     'ArcFaceONNX',
# ]
 
# class ArcFaceONNX:
#     def __init__(self, model_name=None, url="provider.rtx4090.wyo.eg.akash.pub:30247"):
#         assert model_name is not None, "Provide Triton model name"
#         self.model_name = model_name
#         self.taskname = 'recognition'
#         self.url = url
 
#         # Triton client
#         # self.client = httpclient.InferenceServerClient(url=self.url)
#         self.client = grpcclient.InferenceServerClient(url=self.url)
 
 
#         # ArcFace default preprocessing (mxnet vs normal)
#         # Adjust as needed for your model
#         self.input_mean = 127.5
#         self.input_std = 127.5
 
#         # # Get model metadata from Triton
#         # model_metadata = self.client.get_model_metadata(model_name=self.model_name)
#         # model_config = self.client.get_model_config(model_name=self.model_name)
 
#         # # Extract input name, output name, and input shape
#         # self.input_name = model_metadata['inputs'][0]['name']
#         # self.output_names = [o['name'] for o in model_metadata['outputs']]
#         # self.input_shape = model_metadata['inputs'][0]['shape']
#         # self.input_size = (self.input_shape[2], self.input_shape[3])  # (W, H)
#         # self.output_shape = model_metadata['outputs'][0]['shape']
#         # Get model metadata from Triton
#         model_metadata = self.client.get_model_metadata(model_name=self.model_name)
#         model_config = self.client.get_model_config(model_name=self.model_name)
#         # Extract input name, output name, and input shape
       
#         self.input_name = model_metadata.inputs[0].name
#         self.output_names = [o.name for o in model_metadata.outputs]
#         self.input_shape = list(model_metadata.inputs[0].shape)
#         self.input_size = (self.input_shape[2], self.input_shape[3])
#         self.output_shape = list(model_metadata.outputs[0].shape)
 
 
 
#     def prepare(self, ctx_id, **kwargs):
#         pass  # Triton handles CPU/GPU automatically on the server
 
#     def get(self, img, face):
#         aimg = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
#         face.embedding = self.get_feat(aimg).flatten()
#         return face.embedding
 
#     def compute_sim(self, feat1, feat2):
#         from numpy.linalg import norm
#         feat1 = feat1.ravel()
#         feat2 = feat2.ravel()
#         sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
#         return sim
 
#     def get_feat(self, imgs):
#         if not isinstance(imgs, list):
#             imgs = [imgs]
#         input_size = self.input_size
 
#         blob = cv2.dnn.blobFromImages(
#             imgs,
#             1.0 / self.input_std,
#             input_size,
#             (self.input_mean, self.input_mean, self.input_mean),
#             swapRB=True
#         )
 
#         # Send to Triton
#         # inputs = []
#         # inputs.append(httpclient.InferInput(self.input_name, blob.shape, "FP32"))
#         # inputs[0].set_data_from_numpy(blob)
 
#         # outputs = []
#         # for out_name in self.output_names:
#         #     outputs.append(httpclient.InferRequestedOutput(out_name))
#         inputs = []
#         inputs.append(grpcclient.InferInput(self.input_name, blob.shape, "FP32"))
#         inputs[0].set_data_from_numpy(blob)
#         outputs = []
#         for out_name in self.output_names:
#             outputs.append(grpcclient.InferRequestedOutput(out_name))
 
 
#         result = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
 
#         net_out = result.as_numpy(self.output_names[0])
#         return net_out
 
#     def forward(self, batch_data):
#         blob = (batch_data - self.input_mean) / self.input_std
 
#         inputs = [httpclient.InferInput(self.input_name, blob.shape, "FP32")]
#         inputs[0].set_data_from_numpy(blob)
 
#         outputs = [httpclient.InferRequestedOutput(self.output_names[0])]
#         result = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
 
#         net_out = result.as_numpy(self.output_names[0])
#         return net_out
 
# #retinaface
 
# def softmax(z):
#     assert len(z.shape) == 2
#     s = np.max(z, axis=1)
#     s = s[:, np.newaxis]
#     e_x = np.exp(z - s)
#     div = np.sum(e_x, axis=1)
#     div = div[:, np.newaxis]
#     return e_x / div
 
# def distance2bbox(points, distance, max_shape=None):
#     x1 = points[:, 0] - distance[:, 0]
#     y1 = points[:, 1] - distance[:, 1]
#     x2 = points[:, 0] + distance[:, 2]
#     y2 = points[:, 1] + distance[:, 3]
#     if max_shape is not None:
#         x1 = x1.clip(0, max_shape[1])
#         y1 = y1.clip(0, max_shape[0])
#         x2 = x2.clip(0, max_shape[1])
#         y2 = y2.clip(0, max_shape[0])
#     return np.stack([x1, y1, x2, y2], axis=-1)
 
# def distance2kps(points, distance, max_shape=None):
#     preds = []
#     for i in range(0, distance.shape[1], 2):
#         px = points[:, i % 2] + distance[:, i]
#         py = points[:, (i % 2) + 1] + distance[:, i + 1]
#         if max_shape is not None:
#             px = px.clip(0, max_shape[1])
#             py = py.clip(0, max_shape[0])
#         preds.append(px)
#         preds.append(py)
#     return np.stack(preds, axis=-1)
 
# class RetinaFace:
#     def __init__(self, model_file=None, session=None,
#                  triton_url="provider.rtx4090.wyo.eg.akash.pub:30247",
#                  model_name="buffalo_face_detection"):
#         self.model_file = model_file
#         self.taskname = 'detection'
 
#         # Triton gRPC client setup
#         self.triton_client = grpcclient.InferenceServerClient(url=triton_url)
#         self.model_name = model_name
 
#         self.center_cache = {}
#         self.nms_thresh = 0.4
#         self.det_thresh = 0.5
 
#         self._init_vars()
 
#     def _init_vars(self):
#         self.input_name = "input.1"
#         self.output_names = [
#             "448", "471", "494",  # scores
#             "451", "474", "497",  # bboxes
#             "454", "477", "500"   # kps
#         ]
#         self.input_mean = 127.5
#         self.input_std = 128.0
#         self.use_kps = True
#         self.fmc = 3
#         self._feat_stride_fpn = [8, 16, 32]
#         self._num_anchors = 2
#         self.input_size = None
 
#     def prepare(self, ctx_id, **kwargs):
#         nms_thresh = kwargs.get('nms_thresh', None)
#         if nms_thresh is not None:
#             self.nms_thresh = nms_thresh
#         det_thresh = kwargs.get('det_thresh', None)
#         if det_thresh is not None:
#             self.det_thresh = det_thresh
#         input_size = kwargs.get('input_size', None)
#         if input_size is not None:
#             if self.input_size is not None:
#                 print('warning: det_size already set, ignore')
#             else:
#                 self.input_size = input_size
 
#     def forward(self, img, threshold):
#         scores_list, bboxes_list, kpss_list = [], [], []
#         input_size = tuple(img.shape[0:2][::-1])
 
#         blob = cv2.dnn.blobFromImage(img, 1.0 / self.input_std, input_size,
#                                      (self.input_mean, self.input_mean, self.input_mean),
#                                      swapRB=True)
 
#         # Triton gRPC input
#         inputs = [grpcclient.InferInput(self.input_name, blob.shape, np_to_triton_dtype(blob.dtype))]
#         inputs[0].set_data_from_numpy(blob)
 
#         # Triton gRPC outputs
#         outputs = [grpcclient.InferRequestedOutput(name) for name in self.output_names]
 
#         # Run inference
#         results = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
 
#         net_outs = [results.as_numpy(name) for name in self.output_names]
 
#         input_height = blob.shape[2]
#         input_width = blob.shape[3]
#         fmc = self.fmc
#         for idx, stride in enumerate(self._feat_stride_fpn):
#             scores = net_outs[idx]
#             bbox_preds = net_outs[idx + fmc] * stride
#             if self.use_kps:
#                 kps_preds = net_outs[idx + fmc * 2] * stride
 
#             height = input_height // stride
#             width = input_width // stride
#             K = height * width
#             key = (height, width, stride)
#             if key in self.center_cache:
#                 anchor_centers = self.center_cache[key]
#             else:
#                 anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
#                 anchor_centers = (anchor_centers * stride).reshape((-1, 2))
#                 if self._num_anchors > 1:
#                     anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
#                 if len(self.center_cache) < 100:
#                     self.center_cache[key] = anchor_centers
 
#             pos_inds = np.where(scores >= threshold)[0]
#             bboxes = distance2bbox(anchor_centers, bbox_preds)
#             pos_scores = scores[pos_inds]
#             pos_bboxes = bboxes[pos_inds]
#             scores_list.append(pos_scores)
#             bboxes_list.append(pos_bboxes)
#             if self.use_kps:
#                 kpss = distance2kps(anchor_centers, kps_preds)
#                 kpss = kpss.reshape((kpss.shape[0], -1, 2))
#                 pos_kpss = kpss[pos_inds]
#                 kpss_list.append(pos_kpss)
#         return scores_list, bboxes_list, kpss_list
 
#     def detect(self, img, input_size=None, max_num=0, metric='default'):
#         assert input_size is not None or self.input_size is not None
#         input_size = self.input_size if input_size is None else input_size
 
#         im_ratio = float(img.shape[0]) / img.shape[1]
#         model_ratio = float(input_size[1]) / input_size[0]
#         if im_ratio > model_ratio:
#             new_height = input_size[1]
#             new_width = int(new_height / im_ratio)
#         else:
#             new_width = input_size[0]
#             new_height = int(new_width * im_ratio)
#         det_scale = float(new_height) / img.shape[0]
#         resized_img = cv2.resize(img, (new_width, new_height))
#         det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
#         det_img[:new_height, :new_width, :] = resized_img
 
#         scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)
 
#         scores = np.vstack(scores_list)
#         scores_ravel = scores.ravel()
#         order = scores_ravel.argsort()[::-1]
#         bboxes = np.vstack(bboxes_list) / det_scale
#         if self.use_kps:
#             kpss = np.vstack(kpss_list) / det_scale
#         pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
#         pre_det = pre_det[order, :]
#         keep = self.nms(pre_det)
#         det = pre_det[keep, :]
#         if self.use_kps:
#             kpss = kpss[order, :, :]
#             kpss = kpss[keep, :, :]
#         else:
#             kpss = None
#         if max_num > 0 and det.shape[0] > max_num:
#             area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
#             img_center = img.shape[0] // 2, img.shape[1] // 2
#             offsets = np.vstack([
#                 (det[:, 0] + det[:, 2]) / 2 - img_center[1],
#                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]
#             ])
#             offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
#             values = area - offset_dist_squared * 2.0 if metric != 'max' else area
#             bindex = np.argsort(values)[::-1][:max_num]
#             det = det[bindex, :]
#             if kpss is not None:
#                 kpss = kpss[bindex, :]
#         return det, kpss
 
#     def nms(self, dets):
#         thresh = self.nms_thresh
#         x1 = dets[:, 0]
#         y1 = dets[:, 1]
#         x2 = dets[:, 2]
#         y2 = dets[:, 3]
#         scores = dets[:, 4]
 
#         areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#         order = scores.argsort()[::-1]
#         keep = []
#         while order.size > 0:
#             i = order[0]
#             keep.append(i)
#             xx1 = np.maximum(x1[i], x1[order[1:]])
#             yy1 = np.maximum(y1[i], y1[order[1:]])
#             xx2 = np.minimum(x2[i], x2[order[1:]])
#             yy2 = np.minimum(y2[i], y2[order[1:]])
 
#             w = np.maximum(0.0, xx2 - xx1 + 1)
#             h = np.maximum(0.0, yy2 - yy1 + 1)
#             inter = w * h
#             ovr = inter / (areas[i] + areas[order[1:]] - inter)
 
#             inds = np.where(ovr <= thresh)[0]
#             order = order[inds + 1]
#         return keep
 
# def get_retinaface(name, download=False, root='~/.insightface/models', **kwargs):
#     if not download:
#         assert os.path.exists(name)
#         return RetinaFace(name)
#     else:
#         from .model_store import get_model_file
#         _file = get_model_file("retinaface_%s" % name, root=root)
#         return RetinaFace(_file)
 
 
# #face
# class Face(dict):
 
#     def __init__(self, d=None, **kwargs):
#         if d is None:
#             d = {}
#         if kwargs:
#             d.update(**kwargs)
#         for k, v in d.items():
#             setattr(self, k, v)
#         # Class attributes
#         #for k in self.__class__.__dict__.keys():
#         #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
#         #        setattr(self, k, getattr(self, k))
 
#     def __setattr__(self, name, value):
#         if isinstance(value, (list, tuple)):
#             value = [self.__class__(x)
#                     if isinstance(x, dict) else x for x in value]
#         elif isinstance(value, dict) and not isinstance(value, self.__class__):
#             value = self.__class__(value)
#         super(Face, self).__setattr__(name, value)
#         super(Face, self).__setitem__(name, value)
 
#     __setitem__ = __setattr__
 
#     def __getattr__(self, name):
#         return None
 
#     @property
#     def embedding_norm(self):
#         if self.embedding is None:
#             return None
#         return l2norm(self.embedding)
 
#     @property
#     def normed_embedding(self):
#         if self.embedding is None:
#             return None
#         return self.embedding / self.embedding_norm
 
#     @property
#     def sex(self):
#         if self.gender is None:
#             return None
#         return 'M' if self.gender==1 else 'F'
 
 
# class FaceReIDPipeline:
#     def __init__(self, sim_threshold=0.40):
#         # Config
#         self.ist = pytz.timezone("Asia/Kolkata")
#         self.sim_threshold = sim_threshold
       
#         # # ---------------- Use Triton-loaded models ----------------
#         # self.detector = detector
#         # self.face_detector = face_detector
#         # self.face_embedder = face_embedder
#          # ---------------- Triton YOLO Detector ----------------
#         self.triton_yolo_url = "provider.rtx4090.wyo.eg.akash.pub:30247"
#         self.yolo_model_name = "yolo_person_detection"
#         self.detector = YoloPersonDetector(triton_url=self.triton_yolo_url, model_name=self.yolo_model_name)
   
#         # ---------------- Triton RetinaFace + ArcFace ----------------
#         self.triton_buffalo_url = "provider.rtx4090.wyo.eg.akash.pub:30247"
#         self.buffalo_face_model = "buffalo_face_detection"
#         self.buffalo_embed_model = "buffalo_face_embedding"
   
#         self.face_detector = RetinaFace(triton_url=self.triton_buffalo_url, model_name=self.buffalo_face_model)
#         self.face_detector.prepare(ctx_id=0, input_size=(640, 640))
   
#         self.face_embedder = ArcFaceONNX(url=self.triton_buffalo_url, model_name=self.buffalo_embed_model)
#         self.face_embedder.prepare(ctx_id=0, input_size=(112, 112))
 
#         # ---------------- Load FAISS index ----------------
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.dirname(current_dir)
#         # self.db_path = os.path.join(current_dir, "", "face_database_nandhyala_format.pkl")
#         self.db_path = os.path.join(project_root, "data", "face_database_gudivada_format.pkl")
 
#         self.names, self.index = self._load_faiss_index(self.db_path)
 
#         # ---------------- State holders ----------------
#         self.id_name_map = {}
#         self.track_best_crop = {}
#         self.timing = defaultdict(float)

#         # self.timing.update({
#         #     "yolo_inference": 0.0,
#         #     "tracker_update": 0.0,
#         #     "sequential_face": 0.0,
#         #     "draw_and_write": 0.0,
#         #     "no_det": 0.0,
#         #     "buffalo_processing": 0.0,
#         #     "faiss_search": 0.0,
#         #     "skipped_frames": 0.0,
#         #     "update_face":0.0,
#         #     "save_face":0.0
#         # })
 
 
#     # def _load_faiss_index(self, db_path):
#     #     with open(db_path, "rb") as f:
#     #         face_db_list = pickle.load(f)
#     #     names, embeddings = zip(*face_db_list)
#     #     names = np.array(names)
#     #     embeddings = np.stack(embeddings).astype("float32")
 
#     #     N, dim = len(embeddings), embeddings.shape[1]
#     #     nlist = max(1, min(N, int(4 * np.sqrt(N))))
#     #     nprobe = max(1, nlist // 4)
 
#     #     quantizer = faiss.IndexFlatIP(dim)
#     #     index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
#     #     print("Training FAISS index...")
#     #     index.train(embeddings)
#     #     faiss.normalize_L2(embeddings)
#     #     index.add(embeddings)
#     #     index.nprobe = nprobe
 
#     #     print(f"--- FAISS Config: N={N}, dim={dim}, nlist={nlist}, nprobe={nprobe} ---")
#     #     return names, index
#     def _load_faiss_index(self, db_path):
#         with open(db_path, "rb") as f:
#             face_db_list = pickle.load(f)
   
#         # New format: (id, name, role, embedding)
#         ids, names, roles, embeddings = zip(*face_db_list)
#         ids = np.array(ids)          # optional, if you want to track ID separately
#         names = np.array(names)
#         embeddings = np.stack(embeddings).astype("float32")
   
#         N, dim = len(embeddings), embeddings.shape[1]
#         nlist = max(1, min(N, int(4 * np.sqrt(N))))
#         nprobe = max(1, nlist // 4)
   
#         quantizer = faiss.IndexFlatIP(dim)
#         index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
#         print("Training FAISS index...")
#         index.train(embeddings)
#         faiss.normalize_L2(embeddings)
#         index.add(embeddings)
#         index.nprobe = nprobe
   
#         print(f"--- FAISS Config: N={N}, dim={dim}, nlist={nlist}, nprobe={nprobe} ---")
#         return names, index
 
 
#     def _process_crop_sync(self, tid, crop):
#         """Run face detection + embedding + FAISS search synchronously."""
#         try:
#             # t0_buffalo = time.perf_counter()
#             bboxes, kpss = self.face_detector.detect(crop, input_size=(640, 640))
            
 
#             if bboxes.shape[0] == 0:
#                 return tid, {"status": "No face"}, None
 
#             best_score = -1.0
#             best_name = "Unknown"
#             crop_b64 = None
 
#             for i in range(bboxes.shape[0]):
#                 bbox = bboxes[i, :4]
#                 det_score = float(bboxes[i, 4])
#                 if det_score < 0.5:
#                     continue
 
#                 kps = kpss[i] if kpss is not None else None
#                 face_obj = Face(bbox=bbox, kps=kps, det_score=det_score)
 
#                 emb = self.face_embedder.get(crop, face_obj)
#                 # self.timing["buffalo_processing"] += time.perf_counter() - t0_buffalo
#                 emb = emb.astype("float32").reshape(1, -1)
#                 faiss.normalize_L2(emb)
 
#                 # t0_faiss = time.perf_counter()
#                 scores, indices = self.index.search(emb, 1)
#                 # self.timing["faiss_search"] += time.perf_counter() - t0_faiss
 
#                 score, idx = float(scores[0][0]), int(indices[0][0])
 
#                 if score > best_score:
#                     best_score = score
#                     best_name = self.names[idx] if score >= self.sim_threshold else "Unknown"
#                     if crop.size > 0:
#                         try:
#                             _, buffer = cv2.imencode(".jpg", crop)
#                             crop_b64 = base64.b64encode(buffer).decode("utf-8")
#                         except Exception:
#                             crop_b64 = None
 
#             return tid, {"status": best_name}, crop_b64
 
#         except Exception as e:
#             print(f"[Error in _process_crop_sync] {e}")
#             return tid, {"status": "Error"}, None
 
#     async def process_frames(self, frames: list[np.ndarray]):
#         """Main async frame processing (process every 5th frame only)."""
#         start_time = time.perf_counter()
#         self.track_best_crop.clear()
#         self.id_name_map.clear()
#         for k in ["yolo_inference", "tracker_inference", "buffalo_processing"]:
#             self.timing.setdefault(k, 0.0)
 
#         for frame_idx, frame in enumerate(frames):
#             # ✅ Process only every 5th frame
#             if frame_idx % 5 != 0:
#                 continue
 
#             t0 = time.perf_counter()
#             # FIX: Await the async infer method
#             detections = await self.detector.infer(frame, conf_threshold=0.4, iou_threshold=0.7, max_detections=30)
#             self.timing["yolo_inference"] += time.perf_counter() - t0
 
#             t0 = time.perf_counter()
#             # Tracker is CPU-bound, run it in a thread
#             tracks = await asyncio.to_thread(self.detector.track, frame, detections)
#             self.timing["tracker_inference"] += time.perf_counter() - t0
 
#             if not tracks:
#                 continue
            
#             crop_processing_tasks = []
#             for tr in tracks:
#                 x1, y1, x2, y2 = map(int, tr["bbox"])
#                 tid = tr["track_id"]
#                 h = y2 - y1
#                 if h <= 0:
#                     continue
 
#                 face_crop = frame[y1:y1 + int(h * 0.5), x1:x2]
#                 if face_crop.size:
#                     # FIX: Run the synchronous crop processing in a thread
                    
#                     task = asyncio.to_thread(self._process_crop_sync, tid, face_crop)
                    
#                     crop_processing_tasks.append(task)
            
#             if crop_processing_tasks:
#                 # Gather results from all crop processing tasks for this frame
#                 t0_buffalo = time.perf_counter()
#                 crop_results = await asyncio.gather(*crop_processing_tasks)
#                 self.timing["buffalo_processing"] += time.perf_counter() - t0_buffalo
#                 for tid, result_obj, crop_b64 in crop_results:
#                     if tid not in self.id_name_map:
#                         self.id_name_map[tid] = []
#                     self.id_name_map[tid].append(result_obj)
 
#                     if crop_b64:
#                         self.track_best_crop[tid] = crop_b64
 
#         total_exec_time = time.perf_counter() - start_time
#         finalized = self._finalize_results()
 
#         all_face_crops_to_upload = {
#             str(tid): self.track_best_crop[tid]
#             for tid, result_obj in finalized.items()
#             if result_obj["status"] != "No face" and tid in self.track_best_crop
#         }

#         print("===== Timing Summary =====")
#         for k, v in self.timing.items():
#             print(f"{k}: {v:.4f}s")
#         print("Total Execution Time: ",round(total_exec_time, 4))

#         return {
#             "status": "success",
#             "execution_time": round(total_exec_time, 2),
#             "raw_id_name_map": self.id_name_map,
#             "finalized_id_name_map": finalized,
#             "all_face_crops_to_upload": all_face_crops_to_upload
#         }
 
#     def _finalize_results(self):
#         finalized = {}
#         for tid, result_obj_list in self.id_name_map.items():
#             valid = [obj for obj in result_obj_list if obj["status"] not in ["Unknown", "No face", "Error"]]
#             if valid:
#                 names_count = Counter([obj["status"] for obj in valid])
#                 final_name = names_count.most_common(1)[0][0]
#             else:
#                 if any(obj["status"] == "Unknown" for obj in result_obj_list):
#                     final_name = "Unknown"
#                 else:
#                     final_name = "No face"
#             finalized[tid] = {"status": final_name}
#         return finalized
 
 
# # ---------------- Async wrapper ----------------
# async def run_face_recognition(frames: list[np.ndarray]):
#     pipeline = FaceReIDPipeline()
#     return await pipeline.process_frames(frames)
