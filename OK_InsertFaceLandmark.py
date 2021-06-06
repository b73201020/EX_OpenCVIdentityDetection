import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="PUCCR"
)

mycursor = mydb.cursor()

#mycursor.execute("SELECT * FROM users")
#myresult = mycursor.fetchall()
#for x in myresult:
#  print(x)


import sys, os, dlib, glob, numpy
import cv2
from skimage import io


#取得預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
# 人臉68特徵點模型路徑
predictor_path = "shape_predictor_68_face_landmarks.dat"
# 人臉辨識模型路徑
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
# 比對人臉圖片資料夾
faces_folder_path = "./memberPic"


#載入人臉68特徵點檢測器
sp = dlib.shape_predictor(predictor_path)

#載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)



### 處理資料夾裡每張圖片
### 建議
### descriptors, candidate 存入資料庫，後續直接讀取使用

#比對人臉描述子列表
descriptors = []
#比對人臉名稱列表
candidate = []

#針對比對資料夾裡每張圖片做比對：
#1.人臉偵測
#2.特徵點偵測
#3.取得描述子
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    base = os.path.basename(f).split('.')[0]  # 檔名 
    #依序取得圖片檔案人名
    #candidate.append(base)
    img = io.imread(f)
    
    #1.人臉偵測
    dets = detector(img, 1)

    #2.特徵點偵測
    for k, d in enumerate(dets):
        shape = sp(img, d)
        
    #3.取得描述子，68維特徵量
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    
    #轉換numpy array格式
    v = numpy.array(face_descriptor)
    #將值放入descriptors
    print(face_descriptor)

    #descriptors.append(v)


    sql = "INSERT INTO user (name, 68_face_landmark) VALUES (%s, %s)"
    
    str1 = ' '.join(str(e) for e in face_descriptor)
    
    val = (base, str1)
    print(val)
    mycursor.execute(sql, val)

mydb.commit()




