from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import random

def Color(cls_id):
    random.seed(cls_id)
    return (random.randint(0,255),random.randint(0,255), random.randint(0,255)) 

def plot(frame,results,speed_d):
    if not results or results[0].boxes is None:return frame
    boxes=results[0].boxes

    boxes_xyxy=boxes.xyxy.cpu().tolist()
    ids=boxes.id.int().cpu().tolist()
    classes=boxes.cls.int().cpu().tolist()

    for (x1,y1,x2,y2),trk_id,cls_id in zip(boxes_xyxy,ids,classes):
        color=Color(cls_id)
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
        label=f'ID: {trk_id}'
        cv2.putText(frame,label,(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if trk_id in speed_d:
            speed=speed_d[trk_id]
            speed_label=f'{speed:.2f} px/s'
            cv2.putText(frame, speed_label, (int(x1), int(y2) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

