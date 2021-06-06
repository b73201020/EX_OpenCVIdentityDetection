# test
# test2
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image



def paint_chinese_opencv(im,chinese,pos,color):
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('C:/windows/fonts/mingliu.ttc',25)
    fillColor = color #(255,0,0)
    position = pos #(100,100)
    if not isinstance(chinese,str):
        chinese = chinese.decode('utf-8')

    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,font=font,fill=fillColor)
 
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img



img = np.zeros((200, 200, 3), np.uint8)

# 將背景設定為大紅色
img[:] = (123, 123, 255)

# 文字
text = '招財進寶 123 ABC'

# 指定 TTF 字體檔
#fontPath = "./康熙字典體.ttf"

#fontPath = "./mingliu.ttc"
fontPath = "C:/windows/fonts/mingliu.ttc"


# 載入字體
font = ImageFont.truetype(fontPath, 20)

# 將 NumPy 陣列轉為 PIL 影像
imgPil = Image.fromarray(img)

# 在圖片上加入文字
draw = ImageDraw.Draw(imgPil)
draw.text((10, 10),  text, font = font, fill = (0, 0, 0))

# 將 PIL 影像轉回 NumPy 陣列
img = np.array(imgPil)

#使用方法：
img = paint_chinese_opencv(img,"中文", (30,30),(0,0,255))
#Mat 图片，string "中文"，Point (30,30),color (0,0,255)


cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()