import mysql.connector
import sys, os, dlib, glob, numpy
import cv2
from skimage import io


#由資料庫讀出姓名 與 68 face landmark 資訊

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="PUCCR"
)

mycursor = mydb.cursor()

mycursor.execute("SELECT name, SN, 68_face_landmark FROM user")
myresult = mycursor.fetchall()


#比對人臉名稱子列表
candidates = []
#比對人臉描述子列表
descriptors = []

for x , y, z  in myresult:  # name, SN, 68_face_landmark
    candidates.append(x+" "+y) #姓名 SN
    
    arr = [float(i.strip()) for i in z.split(" ")]
    descriptors.append(numpy.array(arr)) #face_landmark

#####################################################



#選擇第一隻攝影機
camera_port = 0
cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)

#調整預設影像大小，預設值很大，很吃效能
cap.set(cv2. CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 480)

#取得預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
# 人臉68特徵點模型路徑
predictor_path = "shape_predictor_68_face_landmarks.dat"
# 人臉辨識模型路徑
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"



#載入人臉68特徵點檢測器
sp = dlib.shape_predictor(predictor_path)

#載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


#當攝影機打開時，對每個frame進行偵測
while(cap.isOpened()):
    #讀出frame資訊
    ret, frame = cap.read()

    # 偵測人臉
    face_rects, scores, idx = detector.run(frame, 0)
    
    #取出偵測的結果
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

        
        #給68特徵點辨識取得一個轉換顏色的frame
        landmarks_frame = cv2.cvtColor(frame, cv2. COLOR_BGR2RGB)

        #找出特徵點位置
        shape = sp(landmarks_frame, d)
 
        #繪製68個特徵點
        #for i in range(68):
        #    cv2.circle(frame,(shape.part(i).x,shape.part(i).y), 3,( 0, 0, 255), 2)
        #    cv2.putText(frame, str(i),(shape.part(i).x,shape.part(i).y),cv2. FONT_HERSHEY_COMPLEX, 0.5,( 255, 0, 0), 1)
        #輸出到畫面

        #3.取得描述子，68維特徵量
        face_descriptor = facerec.compute_face_descriptor(landmarks_frame, shape)
    
        #轉換numpy array格式
        test = numpy.array(face_descriptor)


        dist = []
        # 計算歐式距離 
        for i in descriptors:
            dist_ = numpy.linalg.norm(i - test)
            dist.append(dist_)

        # 將比對人名和比對出來的歐式距離組成一個dictionary
        cd = dict(zip(candidates,dist))
        # 依距離排序
        cd_sorted = sorted(cd.items(), key = lambda d:d[1])

        print("cd_sorted")
        print(cd_sorted)
        
        # 標示身分
        # 取得第一個(距離最小)的人名
        
        if cd_sorted[0][1]<0.42 : # 辨識出身份
            #繪製出偵測人臉的矩形範圍
            cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)
            
            text = 'id: ' + ''.join(cd_sorted[0][0]) 
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
                0.7, (255, 0, 0), 1, cv2.LINE_AA)
        else: # 未能辨識出身份，畫紅框

            cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 0, 255), 4, cv2. LINE_AA) 
            text = '***UNKNOWN***'
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
                0.7, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("Identity Detection ESC to escape", frame)

    #如果按下ESC键，則離開迴圈
    if cv2.waitKey(10) == 27:
        break

#釋放記憶體
cap.release()
#關閉所有視窗
cv2.destroyAllWindows()