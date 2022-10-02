import cv2
import math
import numpy as np
# /home/pi/Yahboom_Project/3.Al视觉课程/CTSM

def trace_fun(self, imageFrame):
    '''寻迹函数:保持'''
    height = len(imageFrame)
    width = len(imageFrame[0])
    widthCenter = math.floor(width / 2)
    # 3204#真实图
#     cv2.imshow("real_img", imageFrame)
#     cv2.waitKey(0)
    # 转化为灰度图
    gray = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)  # cv2.imshow(" grax. img", gray)
    # 大津法二值化
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # cv2.imshow(" dst. jimg,"; dst)
    # 单单看图片的中间高度上一行的像素
    color = dst[math.floor(height / 2)]  # 400
    #找到白色的像素点个数
    white_count = np.sum(color == 255)  # 找到白色的像素点索引

    white_index = np.where(color == 255)  # 防止white, count=0的报错

    if white_count == 0:
        white_count = 1

    # 找到白色像素的中心点位置

    center = (white_index[0][white_count - 1] + white_index[0][0]) / 2

    # 计算出center与标准中心点的偏移量,图片宽度的一半

    direction = center - widthCenter  # 320

    # 画线图像，起点坐标，终点坐标，线的颜色，线的大小

    cv2.line(frame, (widthCenter, 120), (widthCenter, 350), (0, 255, 0), 1, 4)  # 320

    cv2.line(frame, (widthCenter - 20, 200), (widthCenter - 20, 280), (0, 255, 0), 1, 4)

    cv2.line(frame, (widthCenter + 20, 200), (widthCenter + 20, 280), (0, 255, 0), 1, 4)

    print(direction)

    cv2.line(frame, (int(center), 200), (int(center), 280), (0, 0, 255), 1, 4)

    # 添加文字图像，文字内容，坐标，字体，大小，颜色，字体厚度+

    cv2.putText(frame, 'distance:%s' % direction, (int(center) - 40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
               (255, 255, 255), 2)

#     cv2.imshow("frame", frame)
#     cv2.waitKey(0)
    return direction
    

# frame=cv2.imread("1.png")
# trace_fun(frame,frame)


import RPi.GPIO as GPIO
import time

#小车电机引脚定义
IN1 = 20
IN2 = 21
IN3 = 19
IN4 = 26
ENA = 16
ENB = 13

#设置GPIO口为BCM编码方式
GPIO.setmode(GPIO.BCM)

#忽略警告信息
GPIO.setwarnings(False)

#电机引脚初始化操作
def motor_init():
    global pwm_ENA
    global pwm_ENB
    global delaytime
    GPIO.setup(ENA,GPIO.OUT,initial=GPIO.HIGH)
    GPIO.setup(IN1,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN2,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(ENB,GPIO.OUT,initial=GPIO.HIGH)
    GPIO.setup(IN3,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN4,GPIO.OUT,initial=GPIO.LOW)
    #设置pwm引脚和频率为2000hz
    pwm_ENA = GPIO.PWM(ENA, 2000)
    pwm_ENB = GPIO.PWM(ENB, 2000)
    pwm_ENA.start(0)
    pwm_ENB.start(0)

#小车前进	
def run(delaytime):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(5)
    pwm_ENB.ChangeDutyCycle(5)
    time.sleep(delaytime)

#小车后退
def back(delaytime):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(20)
    pwm_ENB.ChangeDutyCycle(20)
    time.sleep(delaytime)

#小车左转	
def left(delaytime):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(10)
    pwm_ENB.ChangeDutyCycle(10)
    time.sleep(delaytime)

#小车右转
def right(delaytime):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(10)
    pwm_ENB.ChangeDutyCycle(10)
    time.sleep(delaytime)

#小车原地左转
def spin_left(delaytime):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(80)
    pwm_ENB.ChangeDutyCycle(80)
    time.sleep(delaytime)

#小车原地右转
def spin_right(delaytime):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(80)
    pwm_ENB.ChangeDutyCycle(80)
    time.sleep(delaytime)

#小车停止	
def brake(delaytime):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(80)
    pwm_ENB.ChangeDutyCycle(80)
    time.sleep(delaytime)

import cv2
import time

motor_init()
rat,frame = cv2.VideoCapture(0).read()
# print(frame)
direction=trace_fun(frame,frame)
run(0.05)
while(direction!=0):
    if direction>20:
        time.sleep(2.00)
        print("偏左大了，往右大转")
        right(0.4)
    elif direction<10 and direction>1.5:
        time.sleep(2.00)
        print("稍微偏左了，往右小转一点")
        right(0.2)
    elif direction<20 and direction>10:
        time.sleep(2.00)
        print("稍微偏左了，往右小转一点")
        right(0.4)
    elif direction<-10 and direction>-20:
        time.sleep(2.00)
        print("稍微偏右了，往左小转一点")
        right(0.6)
    elif direction<-1.5 and direction>-20:
        time.sleep(2.00)
        print("稍微偏右了，往左小转一点")
        right(0.2)
    elif direction<-20:
        print("偏右大了，往左大转")
        time.sleep(2.00)
        left(0.4)
    run(0.014)
    rat,frame = cv2.VideoCapture(0).read()
    # print(frame)
    direction=trace_fun(frame,frame)
    print(direction)

    
        
        

