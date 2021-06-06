#Kasperskey 要關到

# 讀取Webcam frame 寫成png檔

import cv2
import numpy as np
import time


# 設定照片儲存資料夾路徑
path = "pics"

# 選擇第一隻攝影機
camera_port = 0
cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)

#調整預設影像大小，預設值很大，很吃效能
cap.set(cv2. CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 480)

i=0
while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    i=i+1
    print("%d: %s", i, type(frame))


    # 使用日期以及時間替每張照片命名
    date=time.strftime("%Y-%m-%d_%H-%M-%S")
    # 儲存到特定資料夾中，避免放在同資料夾下造成雜亂
    cv2.imwrite("pics/"+date+".png", frame)

    # 顯示圖片
    cv2.imshow('frame', frame)
    #cv2.imshow('frame', np.array(frame, dtype = np.uint8 ))
    
    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()