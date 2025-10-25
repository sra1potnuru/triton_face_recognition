face_recognition.py (pipeline ready code): this file is the initial pipeline code, where the models will be loaded from the model file.
triton_face_end_pipeline.py: this file is the initial pipeline code, where all the files such as and retinaface, arcface,arcface_utils, face class and yoloperson detector classes in the same file.
triton_2layer.py: this file contains the triton code with the update_best_face and save_best_faces for the clustering (chunk level) and also retinaface, arcface,arcface_utils, face class and yoloperson detector classes in the same file but not ready to integrate with the pipeline.
face_recognition_triton.py (pipeline ready code): this file contains the triton code with the update_best_face and save_best_faces for the clustering (chunk level) with tracker as a separate module.
face_recognition_triton_global_cluster.py (pipeline ready code): this file contains the triton code with the update_best_face and save_best_faces for the clustering and the pgadmin db access for the global level clustering with tracker as a separate module.

