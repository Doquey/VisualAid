from insightface.app.common import Face
from insightface.model_zoo import model_zoo
import numpy as np
import whisper


class VoiceToText():
    def __init__(self, model_type: str):
        self.model = whisper.load_model(model_type)

    def inference(self, audio_file: str):
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        _, probs = self.model.detect_language(mel)
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)

        return result.text


class FaceRecognitionService():
    def __init__(self, models_path: str = "./models/"):
        self.det_model = model_zoo.get_model(
            f"{models_path}/det_500m.onnx", providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
        self.rec_model = model_zoo.get_model(
            f"{models_path}/rec_model.onnx", providers=['AzureExecutionProvider', 'CPUExecutionProvider'])

        self.det_model.prepare(ctx_id=0, input_size=(640, 640), det_thres=0.5)

    def generate_embedding(self, img: np.array):
        bboxes, kpss = self.det_model.detect(img, max_num=0, metric="default")
        embeddings = []
        if len(bboxes) == 0:
            return None, "No faces detected"
        for box, kps in zip(bboxes, kpss):
            face = Face(bbox=box, kps=kps, det_score=box[4])
            self.rec_model.get(img, face)
            face_embedding = face.normed_embedding
            embeddings.append(face_embedding)

        return embeddings, None, bboxes
