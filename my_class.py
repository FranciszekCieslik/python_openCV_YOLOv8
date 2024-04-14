from ultralytics import YOLO
import numpy as np

class YOLOsegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect(self, img, conf:float=0.5):
        height, width = img.shape[:2]
        result = self.model.predict(source = img.copy(), conf = conf, save = False, save_txt = False)[0]
        segmentation_contours_idx = []
        if hasattr(result.masks, "segments"):
            for seg in result.masks.segments:
                seg[:, 0] *= width
                seg[:, 1] *= height
                segment = np.array(seg, dtype=np.int32)
                segmentation_contours_idx.append(segment)

            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
            scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
            return bboxes, class_ids, segmentation_contours_idx, scores

class YOLOdetection:
    def __init__(self, model_path, version):
        self.model = YOLO(model_path, version)
       
    def detect(self, img, conf:float=0.5):
        result = self.model.predict(source= img.copy(), conf=conf, save=False)
        boxes = result[0].boxes
        class_list = result[0].names
        return boxes, class_list
