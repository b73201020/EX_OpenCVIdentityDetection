#使用grid
#

import tkinter as tk

import dlib
import cv2
import imutils


window = tk.Tk()
window.title('PU電腦教室座位管理系統')
window.geometry('800x600')
window.configure(background='white')

def submit_seat():
    name = name_entry.get()
    SID = SID_entry.get()
    seat = seat_entry.get()
    result = name +" "+ SID +" "+ seat
    result_label.configure(text=result)
    print(name)
    print(SID)

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

          #如果按下ESC键，則離開迴圈
        if cv2.waitKey(10) == 27:
            seat_text.set(text) #存回
            break

    cap.release()
    cv2.destroyAllWindows()



def clear_seat():
    seat_text.set("")
    result = ""
    result_label.configure(text=result)


###
name_label = tk.Label(window, text='姓名(name)')
name_label.grid(row=0,column=0, sticky="w"+"e")

name_entry = tk.Entry(window)
name_entry.grid(row=0,column=1, sticky="w"+"e")


###
SID_label = tk.Label(window, text='學號(Studnet ID)')
SID_label.grid(row=1,column=0, sticky="w"+"e")

SID_entry = tk.Entry(window)
SID_entry.grid(row=1, column=1,sticky="w"+"e")


###
seat_label = tk.Label(window, text='座位(Seat)')
seat_label.grid(row=2,column=0, sticky="w"+"e")
seat_text=tk.StringVar()
seat_entry = tk.Entry(window, textvariable = seat_text)
seat_entry.grid(row=2, column=1,sticky="w"+"e")

###
result_label = tk.Label(window)
result_label.grid(row=3,column=0, columnspan=2, sticky="w"+"e")



###
submit_btn = tk.Button(window, text='送出(Submit)', command=submit_seat)
submit_btn.grid(row=4,column=0, sticky="w"+"e")
clear_btn = tk.Button(window, text='清除(Clear)', command=clear_seat)
clear_btn.grid(row=4,column=1, sticky="w"+"e")



window.mainloop()