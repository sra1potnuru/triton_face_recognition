# ------------------- Person Detector -------------------
import cv2
import numpy as np
# import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from types import SimpleNamespace

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

    def infer(self, image: np.ndarray, conf_threshold: float, iou_threshold: float, max_detections: int):
        # Preprocess (letterbox)
        letterboxed, scale, pad_left, pad_top = letterbox(image, (self.input_size, self.input_size))
        tensor = letterboxed.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

        # Triton inference
        # inputs = [httpclient.InferInput(self.input_name, tensor.shape, np_to_triton_dtype(tensor.dtype))]
        # inputs[0].set_data_from_numpy(tensor)
        # outputs = [httpclient.InferRequestedOutput(self.output_name)]
        # response = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        inputs = [grpcclient.InferInput(self.input_name, tensor.shape, np_to_triton_dtype(tensor.dtype))]
        inputs[0].set_data_from_numpy(tensor)
        outputs = [grpcclient.InferRequestedOutput(self.output_name)]
        response = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)


        detections = response.as_numpy(self.output_name)[0]

        # Filter by confidence
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

        tracks = self.tracker.update(dets, image)

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

