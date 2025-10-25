import asyncio
import logging
from triton_frame_extractor import extract_frames_from_path
from triton_face_recognition import run_face_recognition

logger = logging.getLogger(__name__)

async def run_services(video_path, service_ids, camera_id, facing_direction=1):
    
    # --- Extract frames from video ---
    print("frames extraction started")
    frames = extract_frames_from_path(video_path)
    print("frames extraction ended")
    if not frames:
        raise RuntimeError("No frames extracted from video")
    print(f"Extracted {len(frames)} frames for camera: {camera_id}")

    results = {}

    # Gunnybag Counting (example service)
    if 0 in service_ids:
        print("🔹 Running Gunnybag Counting...")
        res = await gunnybag_process(frames, facing_direction)
        results["gunnybag"] = res

    # Face Recognition
    if 1 in service_ids:
        print("🔹 Running Face Recognition...")
        results["face_recognition"] = await run_face_recognition(frames)

    # Number Plate Detection (example service)
    if 2 in service_ids:
        print("🔹 Running Number Plate Detection...")
        res = await plate_process(frames)
        results["number_plate"] = res

    print("All requested services completed for camera:", camera_id)
    return results
