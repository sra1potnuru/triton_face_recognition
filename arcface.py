#arcface
from __future__ import division

import numpy as np
import cv2
# import tritonclient.http as httpclient  # Triton client
import tritonclient.grpc as grpcclient  # replace httpclient import

from arcface_utils import norm_crop
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
