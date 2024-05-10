from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, extract_face
import os

img = Image.open(os.path.join("test.jpg"))



mtcnn = MTCNN(keep_all=True)
boxes, probs, points = mtcnn.detect(img, landmarks=True)
 # Draw boxes and save faces
boxes, probs, points = mtcnn.detect(img, landmarks=True)
# Draw boxes and save faces
img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)
for i, (box, point) in enumerate(zip(boxes, points)):
    draw.rectangle(box.tolist(), width=5)
    
    for p in point:
        draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        extract_face(img, box, save_path='detected_face_{}.png'.format(i))
        img_draw.save('annotated_faces.png')