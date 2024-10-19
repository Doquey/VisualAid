import cv2
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
import numpy as np


class FaceRecognitionService():
    def __init__(self, models_path: str = "./models/"):
        self.det_model = model_zoo.get_model(
            f"{models_path}/det_500m.onnx", providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
        self.rec_model = model_zoo.get_model(
            f"{models_path}/rec_model.onnx", providers=['AzureExecutionProvider', 'CPUExecutionProvider'])

        self.det_model.prepare(ctx_id=0, input_size=(640, 640), det_thres=0.5)

    def generate_embedding(self, img: np.array):
        bboxes, kpss = self.det_model.detect(img, max_num=0, metric="default")

        if len(bboxes) == 0:
            return None, "No faces detected"

        face = Face(bbox=bboxes[0], kps=kpss[0], det_score=bboxes[0][4])
        self.rec_model.get(img, face)
        face_embedding = face.normed_embedding

        return face_embedding, None


cap = cv2.VideoCapture()
