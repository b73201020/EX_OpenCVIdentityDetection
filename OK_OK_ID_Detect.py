# 1. 由資料庫讀出姓名 與 68 face landmark 資訊
# 2. 建立視窗
# 3. 開啟webcam 進行身份辨識
# 4. 寫入資料庫

import tkinter as tk
import tkinter.font as tkFont

import mysql.connector
import sys, os, dlib, glob, numpy
import cv2
from skimage import io
import time


from PIL import ImageFont, ImageDraw, Image

# cv2顯示中文字
# 使用 PIL
# paint_chinese_opencv(影像, 文字, 座標, 字型路徑, 大小, 顏色)
def paint_chinese_opencv(im,chinese,pos,fontpath='C:/windows/fonts/mingliu.ttc', size=20, color=(255,0,0)):
    
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    #font = ImageFont.truetype('C:/windows/fonts/mingliu.ttc',20)
    font = ImageFont.truetype(fontpath,size)
    fillColor = color #(255,0,0)
    position = pos #(100,100)
    if not isinstance(chinese,str):
        chinese = chinese.decode('utf-8')

    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,font=font,fill=fillColor)
 
    img = cv2.cvtColor(numpy.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img



################################################
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



# 送出座位資訊 目前僅輸出到 result_label
def submit_seat():
    name = name_entry.get()
    SN = SN_entry.get()
    #seat = seat_entry.get() + seat_select.get()
    seat = seat_select.get()
    result = "姓名："+name +", SN:"+ SN +", 座位："+ seat
    result_label.configure(text=result)
    print(name)
    print(SN)


# 啟動攝影機，進行身辨識
def activate_cam():
    #選擇第一隻攝影機
    camera_port = 0
    cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)

    #調整預設影像大小，預設值很大，很吃效能
    cap.set(cv2. CAP_PROP_FRAME_WIDTH, 400)
    cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 400)

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
            
            if cd_sorted[0][1]<0.45 : # 辨識出身份
                #繪製出偵測人臉的矩形範圍
                cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)
                
                text = 'Identity Detected'
                cv2.putText(frame, text, (x1, y1-40), cv2.FONT_HERSHEY_DUPLEX,
                    0.7, (0, 255, 255), 1, cv2.LINE_AA)
                text =  'ESC to escape'
                cv2.putText(frame, text, (x1, y1-20), cv2.FONT_HERSHEY_DUPLEX,
                    0.7, (0, 255, 255), 1, cv2.LINE_AA)
                

                text = 'id: ' + ''.join(cd_sorted[0][0]) 
                
                # 20201/6/6
                #cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
                #    0.7, (255, 0, 0), 1, cv2.LINE_AA)

                ## 顯示中文姓名：
                fontpath='C:/windows/fonts/mingliu.ttc'
                size=20
                frame = paint_chinese_opencv(frame,text, (x1, y1-20), fontpath, size, (255, 0, 0))    


                A= cd_sorted[0][0].split(" ")
                name_text.set(A[0]) #存回
                SN_text.set(A[1]) #存回
                
                # 設定照片儲存資料夾路徑
                path = "pics"
                # 使用日期以及時間替每張照片命名
                date=time.strftime("%Y-%m-%d_%H-%M-%S")
                # 儲存到特定資料夾中，避免放在同資料夾下造成雜亂
                cv2.imwrite("pics/"+date+".png", frame)
                photo = tk.PhotoImage(file="pics/"+date+".png")
                photo_label.configure(image=photo)
                photo_label.image = photo

                print(A[0])
                print(A[1])
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


# 清除內容
def clear_seat():
    name_text.set("")
    SN_text.set("")
    #seat_text.set("")
    result = ""
    result_label.configure(text=result)
    photo_label.configure(image=logo)
    photo_label.image = logo

########################################

# 建立視窗內元件
window = tk.Tk()
window.title('PU電腦教室座位管理系統')
window.geometry()
window.configure(background='white')

###
name_label = tk.Label(window, text='姓名(name)')
name_label.grid(row=0,column=0, sticky="w"+"e")
name_text=tk.StringVar()
name_entry = tk.Entry(window, textvariable = name_text)
name_entry.grid(row=0,column=1, sticky="w"+"e")


###
SN_label = tk.Label(window, text='學號(Studnet ID)')
SN_label.grid(row=1,column=0, sticky="w"+"e")
SN_text=tk.StringVar()
SN_entry = tk.Entry(window, textvariable = SN_text)
SN_entry.grid(row=1, column=1,sticky="w"+"e")


###
seat_label = tk.Label(window, text='座位(Seat)')
seat_label.grid(row=2,column=0, sticky="w"+"e")
#seat_text=tk.StringVar()
#seat_entry = tk.Entry(window, textvariable = seat_text)
#seat_entry.grid(row=2, column=1,sticky="w")

OptionList = [
"Room316 - 1",
"Room316 - 2",
"Room316 - 3",
"Room316 - 4"
] 

seat_select = tk.StringVar()
seat_select.set(OptionList[0])

opt = tk.OptionMenu(window, seat_select, *OptionList)
opt.config(font=('Helvetica', 12))
opt.grid(row=2, column=1,sticky="e")



###
logo = tk.PhotoImage(file='oits100.png')
logo_label = tk.Label(window, image=logo, width=100, height=100)

logo_label.grid(row=0, column=2, rowspan=3,
               sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)




###
fontStyle = tkFont.Font(family="Lucida Grande", size=14)
result_label = tk.Label(window, bg="#CCFFCC", height=2, font=fontStyle, fg='#FF0000', pady=10, padx=10)
result_label.grid(row=3,column=0, columnspan=3, sticky="w"+"e")



###
cam_btn = tk.Button(window, text='開啟鏡頭(Activate Cam)', command=activate_cam)
cam_btn.grid(row=4,column=0,  sticky="w"+"e")
submit_btn = tk.Button(window, text='送出(Submit)', command=submit_seat)
submit_btn.grid(row=4,column=1)
clear_btn = tk.Button(window, text='清除(Clear)', command=clear_seat)
clear_btn.grid(row=4,column=2, sticky="w"+"e")

###
photo_label = tk.Label(window, image=logo, width=400, height=400)
photo_label.grid(row=5, column=0, columnspan=3,
               sticky=tk.W+tk.E+tk.N+tk.S)




window.mainloop()