from tkinter import *
import tkinter.font as tkFont
from tkinter import messagebox

import mysql.connector
import sys, os, dlib, glob, numpy
import cv2
from skimage import io
import time

from PIL import ImageFont, ImageDraw, Image

from functools import partial

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

### END OF GLOBAL FUNCTIONS ###


class Application(Frame):

    ################################################
    #由資料庫讀出姓名 與 68 face landmark 資訊
    mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="PUCCR"
    )

    mycursor = mydb.cursor()

    #比對人臉名稱子列表
    candidates = []
    #比對人臉描述子列表
    descriptors = []



    def loadUserList(self):
        self.mycursor.execute("SELECT name, SN, 68_face_landmark FROM user")
        myresult = self.mycursor.fetchall()
        
        for x , y, z  in myresult:  # name, SN, 68_face_landmark
            self.candidates.append(x+" "+y) #姓名 SN
            
            arr = [float(i.strip()) for i in z.split(" ")]
            self.descriptors.append(numpy.array(arr)) #face_landmark



    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.loadUserList()
        self.configure(background='#FFFFFF')
        self.pack()
        self.create_widgets()
        #print(self.descriptors)
       

    # 選座位
    def take_seat(self,x):
        self.seat_text.set(x)
        for i in range(50):
            self.seat_btn[i].configure(bg="white")
        self.seat_btn[x-1].configure(bg="yellow")

    # 啟動攝影機，進行身辨識
    def activate_cam(self):
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
                for i in self.descriptors:
                    dist_ = numpy.linalg.norm(i - test)
                    dist.append(dist_)

                # 將比對人名和比對出來的歐式距離組成一個dictionary
                cd = dict(zip(self.candidates,dist))
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
                    self.name_text.set(A[0]) #存回
                    self.SN_text.set(A[1]) #存回
                    
                    # 設定照片儲存資料夾路徑
                    path = "pics"
                    # 使用日期以及時間替每張照片命名
                    date=time.strftime("%Y-%m-%d_%H-%M-%S")
                    # 儲存到特定資料夾中，避免放在同資料夾下造成雜亂
                    cv2.imwrite("pics/"+date+".png", frame)
                    photo = PhotoImage(file="pics/"+date+".png")
                    self.photo_label.configure(image=photo)
                    self.photo_label.image = photo

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
    
    
    # 送出座位資訊 目前僅輸出到 result_label
    def submit_seat(self):
        class_code = self.class_entry.get()
        room_code = self.room_entry.get()
        name = self.name_entry.get()
        SN = self.SN_entry.get()
        seat_id = self.seat_entry.get() 

        result = "姓名："+name +", 教室："+ room_code +", SN："+ SN +", 座位："+ seat_id
        self.result_label.configure(text=result)

        if messagebox.askyesno("確定送出","確定送出座位?"):

            sql = "INSERT INTO rollcall (u_id, class_code, room_code, seat_id) VALUES (%s, %s,%s, %s)"
            u_id = int(SN)
            
            val = (u_id, class_code, room_code, int(seat_id))
            print(val)
            
            self.mycursor.execute(sql, val)
            self.mydb.commit()
            self.result_label.configure(text="已儲存")
        else:
            self.result_label.configure(text="不儲存，請重新操作")
        

 
    # 清除內容
    def clear_seat(self):   
        self.name_text.set("")
        self.SN_text.set("")
        self.seat_text.set("")
        for i in range(50):
            self.seat_btn[i].configure(bg="white")

        result = ""
        self.result_label.configure(text=result)
        self.photo_label.configure(image=self.logo)
        self.photo_label.image = self.logo

    def create_widgets(self):
        # create all of the main containers
        self.top_frame = Frame(self, bg='#99FF99', height=40)
        self.top_frame2 = Frame(self, bg='#CCFFCC', width=450, height=40)
        self.center = Frame(self, bg='gray2', width=450, height=40)
        self.btm_frame = Frame(self, bg='white', width=450, height=45)
        self.btm_frame2 = Frame(self, bg='lavender', width=450, height=60)

        # layout all of the main containers
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.top_frame.grid(row=0, sticky="ew")
        self.top_frame2.grid(row=1, sticky="ew")
        self.center.grid(row=2, sticky="nsew")
        self.btm_frame.grid(row=3, sticky="ew")
        self.btm_frame2.grid(row=4, sticky="ew")

        # create the center widgets
        self.center.grid_rowconfigure(0, weight=1)
        self.center.grid_columnconfigure(1, weight=1)

        self.ctr_left = Frame(self.center, bg='#99FF99', width=100, height=190)
        self.ctr_mid = Frame(self.center, bg='#DDFFDD', width=250, height=190)
        self.ctr_right = Frame(self.center, bg='#99FF99', width=100, height=190)

        self.ctr_left.grid(row=0, column=0, sticky="ns")
        self.ctr_mid.grid(row=0, column=1, sticky="nsew")
        self.ctr_right.grid(row=0, column=2, sticky="ns")

        #####################################################

        ###OITS logo
        self.logo = PhotoImage(file='oits100.png')
        self.logo_label = Label(self.top_frame, image=self.logo, width=100, height=100)
        self.logo_label.grid(row=0, column=0, rowspan=5,sticky="wens")

        ###班級
        Label(self.top_frame, text='課程班級(Course Class)').grid(row=0,column=1, sticky="we")
        class_text=StringVar()
        self.class_entry = Entry(self.top_frame, textvariable = class_text)
        class_text.set('資料庫管理 資管二A')
        self.class_entry.grid(row=0, column=2,sticky="we")

        ###教室
        Label(self.top_frame, text='教室(Room)').grid(row=1,column=1, sticky="we")
        self.room_text=StringVar()
        self.room_entry = Entry(self.top_frame, textvariable = self.room_text)
        self.room_text.set('PH316')
        self.room_entry.grid(row=1, column=2,sticky="we")

        ###姓名
        Label(self.top_frame, text='姓名(name)').grid(row=2,column=1, sticky="we")
        self.name_text=StringVar()
        self.name_entry = Entry(self.top_frame, textvariable = self.name_text)
        self.name_entry.grid(row=2,column=2, sticky="we")

        ###學號
        Label(self.top_frame, text='學號(Studnet ID)').grid(row=3,column=1, sticky="w"+"e")
        self.SN_text=StringVar()
        self.SN_entry = Entry(self.top_frame, textvariable = self.SN_text)
        self.SN_entry.grid(row=3, column=2,sticky="we")

        ###座位
        Label(self.top_frame, text='座位(Seat)').grid(row=4,column=1, sticky="w"+"e")
        self.seat_text= StringVar()
        self.seat_entry = Entry(self.top_frame, textvariable = self.seat_text)
        self.seat_entry.grid(row=4, column=2,sticky="we")

        ###
        cam_btn = Button(self.top_frame, text='開啟鏡頭(Activate Cam)', command=self.activate_cam)
        cam_btn.grid(row=1,column=4,  sticky="we")
        submit_btn = Button(self.top_frame, text='送出(Submit)', command=self.submit_seat)
        submit_btn.grid(row=2,column=4, sticky="we")
        clear_btn = Button(self.top_frame, text='清除(Clear)', command=self.clear_seat)
        clear_btn.grid(row=3,column=4, sticky="we")

        ### 訊息
        fontStyle = tkFont.Font(family="Lucida Grande", size=20)
        self.result_label = Label(self.top_frame2, font=fontStyle, fg='#FF0000', bg='#CCFFCC', pady=10, padx=10)
        self.result_label.grid(row=0,column=0, sticky="wesn")


        ###照片欄
        self.photo_label = Label(self.ctr_left, image=self.logo, width=300, height=300)
        self.photo_label.grid(row=0, column=0,sticky="wesn")

        ########座位
        
        
        # podium講台
        fontStyle = tkFont.Font(family="Lucida Grande", size=20)
        Label(self.ctr_mid, text="講台", bg="#DDDDFF", font=fontStyle, padx=5, pady=5) \
            .grid(row=0, column=0,columnspan=10,sticky="wesn")

        self.seat_btn = []
        for i in range(50):
            #使用 functools 模組中的 partial 物件來傳遞引數
            self.seat_btn.append(Button(self.ctr_mid, text=(i+1),width=5, command=partial(self.take_seat,i+1)))
            #seat_btn.append(Button(ctr_mid, text=(i+1),width=10, command=lambda: take_seat(i+1)))
            
            self.seat_btn[i].grid(row=int(i/10)+1,column=i%10,  sticky="we", padx=5, pady=5)   



        #self.hi_there = Button(self)
        #self.hi_there["text"] = "Hello World\n(click me)"
        #self.hi_there["command"] = self.say_hi
        #self.hi_there.pack(side="top")


        # QUIT button
        self.quit = Button(self.btm_frame, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="top")


### END OF CLASSES ###

def main():
    global root
    # create Tk widget
    root = Tk()
    # set program title
    root.title('PU電腦教室座位管理系統')
    # create game instance
    app = Application(master=root)
    # run event loop
    app.mainloop()

if __name__ == "__main__":
    main()


