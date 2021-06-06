import dlib
import cv2
import imutils


# 選擇第一隻攝影機
camera_port = 0
cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)

#調整預設影像大小，預設值很大，很吃效能
cap.set(cv2. CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 480)

# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()

# 以迴圈從影片檔案讀取影格，並顯示出來
while(cap.isOpened()):
    ret, frame = cap.read()

    # 偵測人臉
    face_rects, scores, idx = detector.run(frame, 0)

    # 取出所有偵測的結果
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        text = "%2.2f(%d)" % (scores[i], idx[i])

        # 以方框標示偵測的人臉
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

        # 標示分數
        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # 顯示結果
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()