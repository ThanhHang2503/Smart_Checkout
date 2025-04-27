import cv2
from facenet_pytorch import MTCNN
import torch

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

cap = cv2.VideoCapture(0)  # Mở camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển từ BGR (OpenCV) sang RGB (PIL/Image)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect khuôn mặt
    boxes, probs = mtcnn.detect(img_rgb)

    # Vẽ khung quanh các khuôn mặt
    if boxes is not None:
        for box in boxes:
            (x1, y1, x2, y2) = [int(b) for b in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Face Detection (MTCNN)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
