import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from time import time

def process_video(input_path, output_path):
    # 选择模型
    if torch.cuda.is_available():
        model = YOLO('weights/yolov8s.pt')
        print('[INFO] Using GPU')
    else:
        model = YOLO('weights/yolov8s.onnx')
        print('[INFO] Using CPU')

    kamera = cv2.VideoCapture(input_path)
    width = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = kamera.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    label_dict = dict()
    count_set = set()
    trk_history = defaultdict(list)
    dist_data = {}
    trk_idslist = []
    spdl_dist_thresh = 10
    trk_previous_times = {}
    trk_previous_points = {}
    reg_pts = [(0, int(height / 2)), (width, int(height / 2))]

    while True:
        ret, frame = kamera.read()
        if not ret:
            break

        results = model.track(frame, conf=0.4, classes=[2,5,7], persist=True, verbose=False)
        names = results[0].names

        if len(label_dict) == 0:
            for value in names.values():
                label_dict[value] = 0

        boxes = results[0].boxes.xyxy.cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()
        clses = results[0].boxes.cls.cpu().tolist()
        ids_tensor = results[0].boxes.id
        ids = ids_tensor.int().cpu().tolist() if ids_tensor is not None else [-1] * len(boxes)

        for box, cls_id, id in zip(boxes, clses, ids):
            if id != -1 and id not in count_set:
                label = names[int(cls_id)]
                label_dict[label] += 1
                count_set.add(id)

        for r in results[0]:
            if r.boxes.id is None:
                continue
            boxes_wh = r.boxes.xywh.cpu().tolist()
            track_ids = r.boxes.id.int().cpu().tolist()
            for box, trk_id in zip(boxes_wh, track_ids):
                x, y, w, h = box
                track = trk_history[trk_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)

                if trk_id not in trk_previous_times:
                    trk_previous_times[trk_id] = 0

                if reg_pts[0][0] < track[-1][0] < reg_pts[1][0]:
                    if reg_pts[1][1] - spdl_dist_thresh < track[-1][1] < reg_pts[1][1] + spdl_dist_thresh:
                        direction = "known"
                    elif reg_pts[0][1] - spdl_dist_thresh < track[-1][1] < reg_pts[0][1] + spdl_dist_thresh:
                        direction = "known"
                    else:
                        direction = "unknown"

                if trk_previous_times.get(trk_id) != 0 and direction != "unknown" and trk_id not in trk_idslist:
                    trk_idslist.append(trk_id)
                    time_difference = time() - trk_previous_times[trk_id]
                    if time_difference > 0:
                        dist_difference = np.abs(track[-1][1] - trk_previous_points[trk_id][1])
                        speed = dist_difference / time_difference
                        dist_data[trk_id] = speed

                trk_previous_times[trk_id] = time()
                trk_previous_points[trk_id] = track[-1]

        # 使用自带的检测框绘图方法
        annotation_frame = results[0].plot()

        cv2.line(annotation_frame, reg_pts[0], reg_pts[1], (0, 255, 0), 2)
        for index, (key, value) in enumerate(label_dict.items()):
            if value > 0:
                label_str = f"{key} : {value}"
                cv2.putText(annotation_frame, label_str, (10, 20 * (1 + index)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        out.write(annotation_frame)

    kamera.release()
    out.release()
    print('[INFO] Process completed. Output saved to:', output_path)