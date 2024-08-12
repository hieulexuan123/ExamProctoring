from ultralytics import YOLO
import cv2

class ItemDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect(self, frame):
        prediction = (self.model.predict(frame, imgsz=640, iou=0.5))[0]
        frame_h, frame_w = frame.shape[:2]

        cls_names = prediction.names
        boxes = prediction.boxes
        coords, cls_ids, confs = boxes.xyxyn, boxes.cls, boxes.conf
        for coord, cls_id, conf in zip(coords, cls_ids, confs):
            x1 = int(coord[0].item() * frame_w)
            y1 = int(coord[1].item() * frame_h)
            x2 = int(coord[2].item() * frame_w)
            y2 = int(coord[3].item() * frame_h)
            cls = cls_names[cls_id.item()]
            conf = conf.item()

            self.annotate(frame, (x1, y1, x2, y2), cls, conf)
    
    def annotate(self, frame, coord, cls, conf):
        x1, y1, x2, y2 = coord
        print(x1, y1, x2, y2)
        #draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        #annotate
        label = f"{cls}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(frame, (label_x, label_y - label_size[1] - 2), 
                      (label_x + label_size[0], label_y + 2), (0, 255, 0), -1)
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)

    