import cv2
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import numpy as np

# 1. Khởi tạo InsightFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)  # 0 = GPU, -1 = CPU

# 2. Khởi tạo webcam
cap = cv2.VideoCapture(0)  # 0 là webcam mặc định

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# 3. Dùng để lưu khuôn mặt đầu tiên làm mẫu (nếu muốn so sánh)
reference_embedding = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        box = face.bbox.astype(int)
        emb = face.embedding
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)

        if reference_embedding is not None:
            sim = cosine_similarity(reference_embedding, emb)
            text = f"Sim: {sim:.2f}"
            if sim > 0.6:
                text += " (Match)"
            cv2.putText(frame, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            cv2.putText(frame, "Press 's' to save face", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Face Recognition - Press 'q' to quit", frame)
    key = cv2.waitKey(1)

    if key == ord('s') and faces:
        reference_embedding = faces[0].embedding
        print("Đã lưu khuôn mặt tham chiếu!")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
