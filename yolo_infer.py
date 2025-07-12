import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from time import time
import subprocess
import os 

'''
  使用opencv处理后的mp4的编码与浏览器不兼容，需要使用FFmpeg转码为H.264 
'''

def convert(path):
    tmp_path=path+'.tmp.mp4'
    # 没有直接的包可以进行转换，需要下载FFmpeg再用命令行的方式处理
    subprocess.run(['ffmpeg','-y','-i',path,'-vcodec','libx264','-acodec','aac',tmp_path],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    # 替换源文件
    os.replace(tmp_path,path)


def process_video(input_path, output_path, classes):
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

    trk_history = defaultdict(list)
    dist_data = {}
    trk_idslist = []
    spdl_dist_thresh = 10
    trk_previous_times = {}
    trk_previous_points = {}
    reg_pts = [(0, int(height / 2)), (width, int(height / 2))]

    reg_line_y = int(height / 2)
    counted_ids = set()
    object_count = 0
    previous_y = {}

    while True:
        ret, frame = kamera.read()
        if not ret:
            break

        results = model.track(frame, conf=0.4, classes=classes, persist=True, verbose=False)

        for r in results[0]:
            if r.boxes.id is None:
                continue
            boxes_wh = r.boxes.xywh.cpu().tolist()
            track_ids = r.boxes.id.int().cpu().tolist()

            for box, trk_id in zip(boxes_wh, track_ids):
                x, y, w, h = box
                y = float(y)

                track = trk_history[trk_id]
                track.append((x, y))
                if len(track) > 30:
                    track.pop(0)

                if trk_id in previous_y:
                    if previous_y[trk_id] < reg_line_y <= y and trk_id not in counted_ids:
                        object_count += 1
                        counted_ids.add(trk_id)
                        print(f"[COUNT] ID {trk_id} crossed the line → Total: {object_count}")

                previous_y[trk_id] = y

                if trk_id not in trk_previous_times:
                    trk_previous_times[trk_id] = 0

                if reg_pts[0][0] < x < reg_pts[1][0]:
                    if reg_pts[1][1] - spdl_dist_thresh < y < reg_pts[1][1] + spdl_dist_thresh:
                        direction = "known"
                    elif reg_pts[0][1] - spdl_dist_thresh < y < reg_pts[0][1] + spdl_dist_thresh:
                        direction = "known"
                    else:
                        direction = "unknown"

                if trk_previous_times.get(trk_id) != 0 and direction != "unknown" and trk_id not in trk_idslist:
                    trk_idslist.append(trk_id)
                    time_difference = time() - trk_previous_times[trk_id]
                    if time_difference > 0:
                        dist_difference = np.abs(y - trk_previous_points[trk_id][1])
                        speed = dist_difference / time_difference
                        dist_data[trk_id] = speed

                trk_previous_times[trk_id] = time()
                trk_previous_points[trk_id] = (x, y)

        annotation_frame = results[0].plot()
        cv2.line(annotation_frame, reg_pts[0], reg_pts[1], (0, 255, 0), 2)
        cv2.putText(annotation_frame, f"Count: {object_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(annotation_frame)

    kamera.release()
    out.release()
    convert(output_path)
    print('[INFO] Process completed. Output saved to:', output_path)


def process_image(input_path,output_path,classes):
    if torch.cuda.is_available():
        model=YOLO('weights/yolov8s.pt')
    else:
        model=YOLO('weights/yolov8s.onnx')
    
    img=cv2.imread(input_path)
    results=model(img,classes=classes,conf=0.4)
    annotated_img=results[0].plot()
    cv2.imwrite(output_path,annotated_img)

