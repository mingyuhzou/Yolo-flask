from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.array): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
colors = Colors() 
def plot(im,boxes,scores,class_ids,ids,names,speed_info):
    lw = max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
    im = im if isinstance(im, Image.Image) else Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    try:
        size = max(round(sum(im.size) / 2 * 0.025), 12)
        font = ImageFont.truetype('./font/SimHei.ttf', size)
        # font = ImageFont.truetype('Arial.ttf', size)
    except Exception:
        font = ImageFont.load_default()
        # print('默认font')
    for box,score,cls_id,id in zip(boxes,scores,class_ids,ids):
        color = colors(int(cls_id),True) 
        _name = ('' if id is None else f'id:{int(id)} ') + names[cls_id]
        draw_text = (f'{_name} {score:.2f}' if score else _name)
        if id is not None:
            speed=speed_info.get(id,None)
            if speed is not None:
                draw_text=(f'{draw_text} speed:{int(speed_info[id])} km/h')
        draw.rectangle(box, width=lw, outline=color)  # box
        #pillow版本 大于等于10.0 小于11.0
        x0, y0, x1, y1=font.getbbox(draw_text)
        w, h = x1-x0, y1-y0
        outside = box[1] - h >= 0  # label fits outside box
        draw.rectangle(
            (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
            box[1] + 1 if outside else box[1] + h + 1),
            fill=color,
        )
        # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
        draw.text((box[0], box[1] - h if outside else box[1]), draw_text, fill=(255, 255, 255), font=font)
    
    # return cv2.cvtColor(np.array(im),cv2.COLOR_BGR2RGB)
    return np.array(im)