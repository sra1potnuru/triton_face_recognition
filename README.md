## 📁 Codebase Overview

This repository contains multiple implementations of the face recognition pipeline, each evolving toward a more modular and production-ready architecture. Below is a breakdown of each file and its purpose.

| File Name | Description | Pipeline Ready | Key Features |
|----------|-------------|----------------|--------------|
| `face_recognition.py` | Initial pipeline implementation. Models are loaded inside the same file. | ✅ Yes | RetinaFace + ArcFace, simple pipeline structure |
| `triton_face_end_pipeline.py` | Single-file version containing RetinaFace, ArcFace, utils, face class, and YOLO person detector all embedded together. | ✅ Yes | Monolithic structure with all modules bundled |
| `triton_2layer.py` | Contains Triton code + `update_best_face` and `save_best_faces` for **chunk-level clustering**, along with model classes, but **not yet integrated** into pipeline form. | ❌ No | Experimental clustering logic (chunk-level), pre-pipeline |
| `face_recognition_triton.py` | Modular pipeline with Triton inference + chunk-level clustering + tracker separated into dedicated module. | ✅ Yes | Clean modular structure, tracker separated |
| `face_recognition_triton_global_cluster.py` | Full pipeline with Triton inference + **global-level clustering** and PostgreSQL (pgAdmin) access for storing metadata. Tracker is also modularized. | ✅ Yes | Global clustering + DB integration + production pipeline |

---

### ✅ Summary

| Capability | File |
|-----------|------|
| Basic (local) face recognition | `face_recognition.py` |
| Single-file Triton pipeline | `triton_face_end_pipeline.py` |
| Chunk-level clustering (experimental) | `triton_2layer.py` |
| Chunk-level clustering (pipeline ready) | `face_recognition_triton.py` |
| Global-level clustering + DB | `face_recognition_triton_global_cluster.py` |

---

Each version is an incremental improvement over the previous one, moving from a fully bundled model setup → modular pipeline → tracking → clustering → database-backed global clustering.

