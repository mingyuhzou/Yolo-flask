import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from time import time
import subprocess
import os 
from draw import plot

# 使用opencv处理后的mp4的编码与浏览器不兼容，需要使用FFmpeg转码为H.264 
def convert(path):
    tmp_path=path+'.tmp.mp4'
    # 没有直接的包可以进行转换，需要下载FFmpeg再用命令行的方式处理
    subprocess.run(['ffmpeg','-y','-i',path,'-vcodec','libx264','-acodec','aac',tmp_path],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    # 替换源文件
    os.replace(tmp_path,path)

# 视频处理逻辑
def process_video(input_path, output_path, classes):
    # 判断宿主机环境，选择合适的权重文件
    if torch.cuda.is_available():
        model = YOLO('weights/yolov8s.pt')
        print('[INFO] Using GPU')
    else:
        model = YOLO('weights/yolov8s.onnx')
        print('[INFO] Using CPU')
    
    # 将视频文件转换为opcv对象
    kamera = cv2.VideoCapture(input_path)

    # 获取一帧的宽高
    width = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 获取帧率
    fps = kamera.get(cv2.CAP_PROP_FPS)
    
    # 初始化写入视频文件的对象，把图像帧保存为mp4格式的文件
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    trk_history = defaultdict(list)

    # 中线坐标
    reg_pts = [(0, int(height / 2)), (width, int(height / 2))]

    reg_line_y = int(height / 2)
    counted_ids = set()
    object_count = 0
    # 更新每个对象上一时刻的纵坐标
    previous_y = {}
    
    thred=10
    prev_time={}
    speed_d={}

    while True:
        # 获取每一帧，ret指示是否获取成功
        ret, frame = kamera.read()
        if not ret:
            break
        
        # classes设置要追踪那些对象，persist是否保留追踪状态
        results = model.track(frame, conf=0.4, classes=classes, persist=True, verbose=False)

        # 存储了所有识别出的置信框
        for r in results[0]:
            if r.boxes.id is None:
                continue
            # 在GPU环境下返回张量，转换回数组类型
            boxes_wh = r.boxes.xywh.cpu().tolist()
            track_ids = r.boxes.id.int().cpu().tolist()

            for box, trk_id in zip(boxes_wh, track_ids):
                # 中心点坐标
                x, y, w, h = box
                y = float(y)

                track = trk_history[trk_id]
                track.append((x, y))
                if len(track) > 30:
                    track.pop(0)

                # 检测是否过线
                if trk_id in previous_y:
                    # 如果上一时刻在线上，这一时刻在线下，并且这个对象没有记录过
                    if previous_y[trk_id] < reg_line_y <= y and trk_id not in counted_ids:
                        object_count += 1
                        counted_ids.add(trk_id)
                        print(f"[COUNT] ID {trk_id} crossed the line → Total: {object_count}")
                # 更新

                if reg_line_y-thred<=y<=reg_line_y+thred:
                    if trk_id not in prev_time:
                        prev_time[trk_id]=time()
                    else:
                        diff_time=time()-prev_time[trk_id]
                        dis=abs(y-previous_y[trk_id])
                        speed_d[trk_id]=dis/diff_time

                previous_y[trk_id] = y
                prev_time[trk_id]=time()

        annotation_frame = plot(frame.copy(),results,speed_d)
        
        # 绘制中线
        cv2.line(annotation_frame, reg_pts[0], reg_pts[1], (0, 255, 0), 2)
        cv2.putText(annotation_frame, f"Count: {object_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(annotation_frame)
    # 释放资源
    kamera.release()
    out.release()
    # 转换视频编码
    convert(output_path)
    print('[INFO] Process completed. Output saved to:', output_path)

# 图像处理逻辑
def process_image(input_path,output_path,classes):
    # 选择环境
    if torch.cuda.is_available():
        model=YOLO('weights/yolov8s.pt')
    else:
        model=YOLO('weights/yolov8s.onnx')
    
    # 直接识别
    img=cv2.imread(input_path)
    results=model(img,classes=classes,conf=0.4)
    annotated_img=results[0].plot()
    cv2.imwrite(output_path,annotated_img)