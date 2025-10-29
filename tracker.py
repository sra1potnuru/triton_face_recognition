import asyncio
import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
import logging
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from types import SimpleNamespace

logger = logging.getLogger(__name__)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1, inter_y1 = max(x1, x3), max(y1, y3)
    inter_x2, inter_y2 = min(x2, x4), min(y2, y4)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_distance(box1, box2):
    """Calculate Euclidean distance between centers of two bounding boxes"""
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


class Track:
    """Enhanced Track class with 8-state Kalman Filter for robust object tracking"""
    
    def __init__(self, track_id, bbox, frame_id, line_x, facing_direction, min_hits_to_count, width, height, frame_skip, fps=30):
        self.id = track_id
        self.frame_created = frame_id
        self.line_x = line_x
        self.facing_direction = facing_direction
        self.min_hits_to_count = min_hits_to_count
        self.frame_skip = frame_skip
        
        # Enhanced 8-state Kalman Filter
        self.kalman = cv2.KalmanFilter(8, 4)
        dt = 1.0 / max(fps, 1)
        
        # State: [cx, cy, w, h, vx, vy, ax, ay]
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0,  dt*dt/2, 0],
            [0, 1, 0, 0, 0,  dt, 0,      dt*dt/2],
            [0, 0, 1, 0, 0,  0,  0,      0],
            [0, 0, 0, 1, 0,  0,  0,      0],
            [0, 0, 0, 0, 0.9, 0, dt,     0],
            [0, 0, 0, 0, 0,  0.9, 0,     dt],
            [0, 0, 0, 0, 0,  0,  0.7,    0],
            [0, 0, 0, 0, 0,  0,  0,      0.7]
        ], np.float32)
        
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        
        self.kalman.processNoiseCov = np.diag([10, 10, 2, 2, 50, 50, 25, 25]).astype(np.float32)
        self.kalman.measurementNoiseCov = np.diag([5, 5, 3, 3]).astype(np.float32)
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32) * 100
        
        x1, y1, x2, y2 = bbox
        cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
        self.kalman.statePre = np.array([cx, cy, w, h, 0, 0, 0, 0], np.float32)
        self.kalman.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], np.float32)
        
        self.bbox = bbox
        self.predicted_bbox = bbox
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.consecutive_misses = 0
        
        self.state = "tentative"
        
        self.counted_loading = False
        self.counted_unloading = False
        self.loading_frame = -1
        self.unloading_frame = -1
        
        self.positions = [(cx, cy, frame_id)]
        self.max_history = 50
        
        self.initial_side = self._get_side(cx)
        self.current_side = self.initial_side
        self.side_history = [self.initial_side]
        
        self.velocity_x = 0
        self.velocity_y = 0
        self.acceleration_x = 0
        self.acceleration_y = 0
        
        self.velocity_history = []
        self.raw_velocity_history = []
        self.smooth_factor = 0.6

    def _get_side(self, cx):
        """Determine which side of the line the object is on"""
        return -1 if cx < self.line_x else 1

    def _smooth_velocity(self, new_vx, new_vy):
        """Apply smoothing to velocity measurements"""
        self.raw_velocity_history.append((new_vx, new_vy))
        if len(self.raw_velocity_history) > 10:
            self.raw_velocity_history.pop(0)
        
        if len(self.velocity_history) == 0:
            return new_vx, new_vy
        
        last_vx, last_vy = self.velocity_history[-1]
        velocity_change = np.sqrt((new_vx - last_vx)**2 + (new_vy - last_vy)**2)
        
        smooth_factor = 0.3 if velocity_change > 50 else self.smooth_factor
        
        smooth_vx = smooth_factor * last_vx + (1 - smooth_factor) * new_vx
        smooth_vy = smooth_factor * last_vy + (1 - smooth_factor) * new_vy
        
        return smooth_vx, smooth_vy

    def predict(self, width, height):
        """Predict next state using Kalman filter"""
        self.age += 1
        if self.time_since_update > 0:
            self.time_since_update += 1
            self.consecutive_misses += 1
        
        # Dynamic process noise
        if len(self.raw_velocity_history) >= 3:
            recent_velocities = self.raw_velocity_history[-3:]
            velocity_variance = np.var([v[1] for v in recent_velocities])
            
            noise_multiplier = 2.0 if velocity_variance > 100 else 1.0
            base_noise = np.array([10, 10, 2, 2, 50, 50, 25, 25], dtype=np.float32)
            self.kalman.processNoiseCov = np.diag(base_noise * noise_multiplier)
        
        pred = self.kalman.predict()
        cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
        self.velocity_x, self.velocity_y = pred[4], pred[5]
        self.acceleration_x, self.acceleration_y = pred[6], pred[7]
        
        # Multi-step prediction for missed detections
        if self.consecutive_misses > 0:
            dt = 1.0 / max(30, 1)
            steps = min(self.consecutive_misses, 8)
            
            future_cx = cx + self.velocity_x * steps * dt + 0.5 * self.acceleration_x * (steps * dt) ** 2
            future_cy = cy + self.velocity_y * steps * dt + 0.5 * self.acceleration_y * (steps * dt) ** 2
            
            if self.velocity_y > 20:
                future_cy += 0.5 * 300 * (steps * dt) ** 2
            
            cx, cy = future_cx, future_cy
        
        w = max(15, min(w, width/2))
        h = max(15, min(h, height/2))
        cx = max(w/2, min(cx, width - w/2))
        cy = max(h/2, min(cy, height - h/2))
        
        self.predicted_bbox = (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))
        return self.predicted_bbox

    def update(self, bbox, frame_id):
        """Update track with new detection"""
        self.time_since_update = 0
        self.consecutive_misses = 0
        self.hits += 1
        
        if self.state == "tentative" and self.hits >= self.min_hits_to_count:
            self.state = "confirmed"
        
        x1, y1, x2, y2 = bbox
        cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
        
        if len(self.positions) > 0:
            last_cx, last_cy, last_frame = self.positions[-1]
            dt = max(1, frame_id - last_frame) / max(30, 1)
            
            raw_vx = (cx - last_cx) / dt
            raw_vy = (cy - last_cy) / dt
            
            self.velocity_x, self.velocity_y = self._smooth_velocity(raw_vx, raw_vy)
            
            if len(self.velocity_history) > 0:
                last_vx, last_vy = self.velocity_history[-1]
                self.acceleration_x = np.clip((self.velocity_x - last_vx) / dt, -500, 500)
                self.acceleration_y = np.clip((self.velocity_y - last_vy) / dt, -500, 500)
            
            self.velocity_history.append((self.velocity_x, self.velocity_y))
            if len(self.velocity_history) > 8:
                self.velocity_history.pop(0)
        
        measurement = np.array([cx, cy, w, h], np.float32)
        self.kalman.correct(measurement)
        
        state = self.kalman.statePost.copy()
        state[4:8] = [self.velocity_x, self.velocity_y, self.acceleration_x, self.acceleration_y]
        self.kalman.statePost = state
        
        self.bbox = bbox
        
        self.positions.append((cx, cy, frame_id))
        if len(self.positions) > self.max_history:
            self.positions.pop(0)
        
        new_side = self._get_side(cx)
        self.current_side = new_side
        self.side_history.append(new_side)
        if len(self.side_history) > 20:
            self.side_history.pop(0)
        
        return self.check_crossings(frame_id)

    def mark_missed(self):
        """Mark track as missed in current frame"""
        if self.time_since_update == 0:
            self.time_since_update = 1
            self.consecutive_misses = 1
        else:
            self.time_since_update += 1
            self.consecutive_misses += 1

    def check_crossings(self, frame_id):
        """Check if object crossed the counting line"""
        if self.state != "confirmed" or len(self.positions) < 5:
            return False, False
        
        min_positions = min(15, len(self.positions))
        recent_positions = self.positions[-min_positions:]
        
        if len(recent_positions) < 5:
            return False, False
        
        crossed_right = False
        crossed_left = False
        
        for i in range(len(recent_positions) - 3):
            segment = recent_positions[i:i+4]
            
            for j in range(len(segment) - 1):
                x1, _, _ = segment[j]
                x2, _, _ = segment[j + 1]
                
                if abs(x2 - x1) > 3:
                    if x1 < self.line_x <= x2:
                        crossed_right = True
                    elif x1 > self.line_x >= x2:
                        crossed_left = True
        
        start_x = recent_positions[0][0]
        end_x = recent_positions[-1][0]
        total_x_movement = abs(end_x - start_x)
        
        if total_x_movement < 15:
            return False, False
        
        middle_idx = len(recent_positions) // 2
        middle_x = recent_positions[middle_idx][0]
        
        direction_consistent = ((crossed_right and start_x < middle_x < end_x) or 
                               (crossed_left and start_x > middle_x > end_x))
        
        if not direction_consistent:
            return False, False
        
        new_loading = False
        new_unloading = False
        
        if self.facing_direction == 1:
            if crossed_right and not self.counted_loading and start_x < self.line_x < end_x:
                self.counted_loading = True
                self.loading_frame = frame_id
                new_loading = True
                logger.info(f"✅ Loading: Track {self.id}, Frame {frame_id}, Path: {start_x:.0f} -> {end_x:.0f}")
            
            if crossed_left and not self.counted_unloading and start_x > self.line_x > end_x:
                self.counted_unloading = True
                self.unloading_frame = frame_id
                new_unloading = True
                logger.info(f"✅ Unloading: Track {self.id}, Frame {frame_id}, Path: {start_x:.0f} -> {end_x:.0f}")
        else:
            if crossed_left and not self.counted_loading and start_x > self.line_x > end_x:
                self.counted_loading = True
                self.loading_frame = frame_id
                new_loading = True
                logger.info(f"✅ Loading: Track {self.id}, Frame {frame_id}, Path: {start_x:.0f} -> {end_x:.0f}")
            
            if crossed_right and not self.counted_unloading and start_x < self.line_x < end_x:
                self.counted_unloading = True
                self.unloading_frame = frame_id
                new_unloading = True
                logger.info(f"✅ Unloading: Track {self.id}, Frame {frame_id}, Path: {start_x:.0f} -> {end_x:.0f}")
        
        return new_loading, new_unloading

    def get_state(self):
        """Get current or predicted bounding box"""
        return self.bbox if self.time_since_update == 0 else self.predicted_bbox

    def is_deleted(self, max_disappeared, max_age):
        """Check if track should be deleted"""
        if self.state == "tentative" and self.consecutive_misses > 15:
            return True
        if self.state == "confirmed" and self.consecutive_misses > max_disappeared:
            return True
        return self.age > max_age


class UnifiedTracker:
    """
    Asynchronous tracker that can handle multiple tracking sessions concurrently.
    Each session maintains its own set of tracks and counting logic.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._sessions = {}
        self._next_session_id = 0
    
    async def create_session(self, width: int, height: int, line_x: int, 
                            facing_direction: int, fps: int = 30,
                            min_hits_to_count: int = 3, frame_skip: int = 0) -> str:
        """Create a new tracking session"""
        async with self._lock:
            session_id = f"session_{self._next_session_id}"
            self._next_session_id += 1
            
            self._sessions[session_id] = {
                'tracks': [],
                'next_id': 0,
                'loading_count': 0,
                'unloading_count': 0,
                'width': width,
                'height': height,
                'line_x': line_x,
                'facing_direction': facing_direction,
                'fps': fps,
                'min_hits_to_count': min_hits_to_count,
                'frame_skip': frame_skip
            }
            
            logger.info(f"Created tracking session: {session_id}")
            return session_id
    
    async def destroy_session(self, session_id: str):
        """Destroy a tracking session and free resources"""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Destroyed tracking session: {session_id}")
    
    async def process_frame(self, session_id: str, detections: List[Tuple], 
                           frame_id: int, iou_threshold: float = 0.25,
                           distance_threshold: float = 200,
                           max_disappeared: int = 120, 
                           max_age: int = 400) -> Tuple[int, int]:
        """
        Process a single frame for a given session.
        Returns: (loading_count, unloading_count) for this frame
        """
        async with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self._sessions[session_id]
            tracks = session['tracks']
            
            # Predict all tracks
            for track in tracks:
                track.predict(session['width'], session['height'])
            
            # Associate detections with tracks
            matches, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
                detections, tracks, iou_threshold, distance_threshold, session['fps']
            )
            
            frame_loading = 0
            frame_unloading = 0
            
            # Update matched tracks
            for track_idx, det_idx in matches:
                loading, unloading = tracks[track_idx].update(detections[det_idx], frame_id)
                if loading:
                    session['loading_count'] += 1
                    frame_loading += 1
                if unloading:
                    session['unloading_count'] += 1
                    frame_unloading += 1
            
            # Mark unmatched tracks as missed
            for track_idx in unmatched_trks:
                tracks[track_idx].mark_missed()
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                new_track = Track(
                    session['next_id'], 
                    detections[det_idx], 
                    frame_id,
                    session['line_x'], 
                    session['facing_direction'], 
                    session['min_hits_to_count'],
                    session['width'], 
                    session['height'], 
                    session['frame_skip'], 
                    session['fps']
                )
                tracks.append(new_track)
                session['next_id'] += 1
            
            # Remove dead tracks
            session['tracks'] = [t for t in tracks if not t.is_deleted(max_disappeared, max_age)]
            
            return frame_loading, frame_unloading
    
    async def get_session_stats(self, session_id: str) -> dict:
        """Get current statistics for a session"""
        async with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self._sessions[session_id]
            tracks = session['tracks']
            
            confirmed_tracks = sum(1 for t in tracks if t.state == "confirmed")
            fast_tracks = sum(1 for t in tracks if np.sqrt(t.velocity_x**2 + t.velocity_y**2) > 20)
            predicted_tracks = sum(1 for t in tracks if t.consecutive_misses > 0)
            
            return {
                'total_tracks': len(tracks),
                'confirmed_tracks': confirmed_tracks,
                'fast_tracks': fast_tracks,
                'predicted_tracks': predicted_tracks,
                'loading_count': session['loading_count'],
                'unloading_count': session['unloading_count']
            }
    
    def _associate_detections_to_tracks(self, detections, tracks, iou_threshold, distance_threshold, fps):
        """Associate detections to existing tracks using Hungarian algorithm"""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for t, track in enumerate(tracks):
            for d, detection in enumerate(detections):
                pred_bbox = track.get_state()
                
                iou = calculate_iou(pred_bbox, detection)
                distance = calculate_distance(pred_bbox, detection)
                
                speed = np.sqrt(track.velocity_x**2 + track.velocity_y**2)
                
                speed_bonus = min(speed * 2, 200) if abs(track.velocity_y) > 30 else min(speed * 1.5, 100)
                adaptive_threshold = distance_threshold + speed_bonus
                
                if distance > adaptive_threshold:
                    cost = 1.0
                else:
                    iou_cost = 1.0 - iou
                    distance_cost = distance / adaptive_threshold
                    
                    if track.consecutive_misses > 0:
                        cx_track = (pred_bbox[0] + pred_bbox[2]) / 2
                        cy_track = (pred_bbox[1] + pred_bbox[3]) / 2
                        
                        dt = 1.0 / max(fps, 1)
                        steps = track.consecutive_misses
                        
                        predicted_cx = cx_track + track.velocity_x * steps * dt
                        predicted_cy = cy_track + track.velocity_y * steps * dt
                        
                        if track.velocity_y > 20:
                            predicted_cy += 0.5 * 200 * (steps * dt) ** 2
                        
                        cx_det = (detection[0] + detection[2]) / 2
                        cy_det = (detection[1] + detection[3]) / 2
                        
                        predicted_distance = np.sqrt((predicted_cx - cx_det)**2 + (predicted_cy - cy_det)**2)
                        
                        if predicted_distance < distance:
                            distance_cost *= 0.4
                    
                    cost = (0.6 * iou_cost + 0.4 * distance_cost) if track.state == "confirmed" else (0.5 * iou_cost + 0.5 * distance_cost)
                
                cost_matrix[t, d] = cost
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))
        
        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] <= 0.9:
                matches.append((r, c))
                unmatched_dets.remove(c)
                unmatched_trks.remove(r)
        
        return matches, unmatched_dets, unmatched_trks


class FaceTracker:
    """Face tracking specialization using BoT-SORT"""
    
    def __init__(self, tracker_type, track_high_thresh, track_low_thresh, 
                 new_track_thresh, track_buffer, match_thresh, fuse_score,
                 gmc_method, proximity_thresh, appearance_thresh, with_reid, model):
        """Initialize tracker with configurable settings passed from face_recognition.py"""
        tracker_args = {
            "tracker_type": tracker_type,
            "track_high_thresh": track_high_thresh,
            "track_low_thresh": track_low_thresh,
            "new_track_thresh": new_track_thresh,
            "track_buffer": track_buffer,
            "match_thresh": match_thresh,
            "fuse_score": fuse_score,
            "gmc_method": gmc_method,
            "proximity_thresh": proximity_thresh,
            "appearance_thresh": appearance_thresh,
            "with_reid": with_reid,
            "model": model
        }
        args = SimpleNamespace(**tracker_args)
        self.tracker = BOTSORT(args=args)

    def update(self, dets_to_track: Boxes, image: np.ndarray):
        """Track faces using BoT-SORT. This is the main tracking interface."""
        if len(dets_to_track) == 0:
            return []

        # Perform tracking
        tracks = self.tracker.update(dets_to_track, image)

        # Convert to common format
        tracked = []
        for t in tracks:
            x1, y1, x2, y2, track_id, cls_id, conf = t[:7]
            tracked.append({
                "track_id": int(track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(conf)
            })

        return tracked