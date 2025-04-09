import cv2
import numpy as np

def draw_boxes(image, boxes, labels=None, scores=None, color=(0,255,0), thickness=2):
    img = image.copy()
    for i,b in enumerate(boxes):
        x1,y1,x2,y2 = b.astype(int)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,thickness)
        if labels is not None and scores is not None:
            txt = f"{labels[i]}:{scores[i]:.2f}"
            cv2.putText(img, txt, (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
    return img

def draw_mask(image, mask, color=(0,255,0), alpha=0.5):
    img = image.copy()
    m = (mask>0.5).astype(np.uint8)
    colored = np.zeros_like(img)
    colored[:,:,1] = m*color[1]
    return cv2.addWeighted(img,1-alpha,colored,alpha,0)
