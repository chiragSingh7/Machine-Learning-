import cv2
import numpy as np

def draw_boxes(image, boxes, labels=None, scores=None,
               color=(0,255,0), thickness=2):
    vis = image.copy()
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = box.astype(int)
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,thickness)
        text = ""
        if labels is not None:
            text += str(labels[i])
        if scores is not None:
            text += f" {scores[i]:.2f}"
        if text:
            cv2.putText(vis,text,(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
    return vis

def draw_mask(image, mask, color=(0,255,0), alpha=0.5):
    vis = image.copy()
    colored = np.zeros_like(vis)
    for c in range(3):
        colored[:,:,c] = mask*color[c]
    return cv2.addWeighted(vis,1-alpha,colored,alpha,0)
