# 1、不能两个class同时调用摄像头
import RPi.GPIO as GPIO
import cv2
import time
import numpy as np
import argparse
import imutils
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# 小车电机引脚定义
IN1 = 20
IN2 = 21
IN3 = 19
IN4 = 26
ENA = 16
ENB = 13

# 设置GPIO口为BCM编码方式
GPIO.setmode(GPIO.BCM)

# 忽略警告信息
GPIO.setwarnings(False)


# 电机引脚初始化操作
def motor_init():
    global pwm_ENA
    global pwm_ENB
    global delaytime
    GPIO.setup(ENA, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ENB, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)
    # 设置pwm引脚和频率为2000hz
    pwm_ENA = GPIO.PWM(ENA, 2000)
    pwm_ENB = GPIO.PWM(ENB, 2000)
    pwm_ENA.start(0)
    pwm_ENB.start(0)


# 小车前进
def run(delaytime):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(20)
    pwm_ENB.ChangeDutyCycle(20)
    time.sleep(delaytime)


# 小车后退
def back(delaytime):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(20)
    pwm_ENB.ChangeDutyCycle(20)
    time.sleep(delaytime)


# 小车左转
def left(delaytime):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(20)
    pwm_ENB.ChangeDutyCycle(20)
    time.sleep(delaytime)


# 小车右转
def right(delaytime):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(10)
    pwm_ENB.ChangeDutyCycle(10)
    time.sleep(delaytime)


# 小车原地左转
def spin_left(delaytime):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(20)
    pwm_ENB.ChangeDutyCycle(20)
    time.sleep(delaytime)


# 小车原地右转
def spin_right(delaytime):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(20)
    pwm_ENB.ChangeDutyCycle(20)
    time.sleep(delaytime)


# 小车停止
def brake(delaytime):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(80)
    pwm_ENB.ChangeDutyCycle(80)
    time.sleep(delaytime)


class CarUltrasound(object):
    def __init__(self):

        self.GPIO_TRIGGER = 1  # GPIO setting (BCM coding)
        self.GPIO_ECHO = 0

        GPIO.setup(self.GPIO_TRIGGER, GPIO.OUT)  # GPIO input/output definiation
        GPIO.setup(self.GPIO_ECHO, GPIO.IN)

        self.dist_mov_ave = 0

    def distMeasure(self):  # distance measuing

        GPIO.output(self.GPIO_TRIGGER, GPIO.HIGH)
        time.sleep(0.000015)
        GPIO.output(self.GPIO_TRIGGER, GPIO.LOW)
        while not GPIO.input(self.GPIO_ECHO):
            pass
        t1 = time.time()
        while GPIO.input(self.GPIO_ECHO):
            pass
        t2 = time.time()
        print
        "distance is %d " % (((t2 - t1) * 340 / 2) * 100)
        time.sleep(0.01)
        return ((t2 - t1) * 340 / 2) * 100

        # GPIO.output(self.GPIO_TRIGGER, False)
        # time.sleep(0.000002)
        # GPIO.output(self.GPIO_TRIGGER, True)  # emit ultrasonic pulse
        # time.sleep(0.00001)                   # last 10us
        # GPIO.output(self.GPIO_TRIGGER, False) # end the pulse

        # ii = 0
        # while GPIO.input(self.GPIO_ECHO) == 0:  # when receiving the echo, ECHO will become 1
        #     ii = ii + 1
        #     if ii > 10000:
        #         print('Ultrasound error: the sensor missed the echo')
        #         return 0
        #     pass
        # start_time = time.time()
        #
        # while GPIO.input(self.GPIO_ECHO) == 1:  # the duration of high level of ECHO is the time between the emitting the pulse and receiving the echo
        #         pass
        # stop_time = time.time()
        #
        # time_elapsed = stop_time - start_time
        # distance = (time_elapsed * 34300) / 2
        #
        # return distance

    def distMeasureMovingAverage(self):
        dist_current = self.distMeasure()
        if dist_current == 0:  # if the sensor missed the echo, the output dis_mov_ave will equal the last dis_mov_ave
            return self.dist_mov_ave
        else:
            self.dist_mov_ave = 0.8 * dist_current + 0.2 * self.dist_mov_ave  # using the moving average of distance measured by sensor to reduce the error
            return self.dist_mov_ave


class ColorRecognition(object):
    def __init__(self):
        self.red_lower = np.array([0, 43, 46])
        self.red_upper = np.array([10, 255, 255])
        self.green_lower = np.array([35, 43, 46])
        self.green_upper = np.array([77, 255, 255])
        self.yellow_lower = np.array([26, 43, 46])
        self.yellow_upper = np.array([34, 255, 255])
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 320)
        self.cap.set(4, 240)

    def ChestRed(self, frame):
        isRed = False
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.red_lower, self.red_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        # res = cv2.bitwise_and(frame, frame, mask=mask)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if 5 < len(cnts) < 30:
            # print("Red!")
            isRed = True
        return isRed

    def ChestGreen(self, frame):
        isGreen = False
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if 20 < len(cnts) < 30:
            # print("Green!")
            isGreen = True
        return isGreen

    def ChestYellow(self, frame):
        isYellow = False
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if 20 < len(cnts) < 30:
            print("Yellow!")
            isYellow = True
        return isYellow

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()


class SighRecongination(object):
    def __init__(self):
        # 根据使用的模型，确定图像需要resize的尺寸
        self.norm_size = 64
        # 加载训练好的卷积神经网络
        print("[INFO] loading network...")
        self.model = load_model(args["model"])

    # 预测函数，
    # 输入： 包含配置参数的字典

    def predict(self, args, frame):
        # 加载图像
        # image = cv2.imread(args["image"])
        # 因为对图像需要进行写入标签，影响较大所以复制一个图像
        orig = frame.copy()

        # 预处理图像进行分类
        # 图像的尺寸重载
        frame = cv2.resize(frame, (self.norm_size, self.norm_size))
        # 图像的序列的归一化处理
        frame = frame.astype("float") / 255.0
        # 将图像进行序列化
        frame = img_to_array(frame)
        # 展开数组的形状.
        # 插入一个新的轴，该轴将出现在扩展阵列形状的轴位置
        frame = np.expand_dims(frame, axis=0)

        # 对输入的图像进行分类
        result = self.model.predict(frame)[0]
        # print (result.shape)
        proba = np.max(result)
        label0 = str(np.where(result == proba)[0])
        label = "{}: {:.2f}%".format(label0, proba * 100)
        #         print(label)

        # 在需要加载图像的情况下
        #         if args['show']:
        #             output = imutils.resize(orig, width=400)
        #         # 在图像上绘制标签字符串
        #             cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        #                         0.7, (0, 255, 0), 2)
        #         # 显示带标签的图像
        #         cv2.imshow("Output", frame)
        #         cv2.waitKey(0)
        return label0


if __name__ == '__main__':
    args = {}
    print("1")
    # 模型的输入路径
    args['model'] = './MODE/traffic_sign.model'
    print("1")
    # 图像的输入路径
    # args['image'] = './predict/00022_00000.png'
    args['show'] = 'true'
    try:
        motor_init()
        car = CarUltrasound()
        print("1")
        colorRecog = ColorRecognition()
        print("1")
        SighRecong = SighRecongination()
        print("1")
        '''
        avoid->改造成函数，代替之前的run()
        '''


        def avoid():
            print("______________________智能避障______________________")
            for i in range(10):
                dist = car.distMeasure()
                print("Measured Distance = {:.2f} cm".format(dist))
                if (dist <= 40):
                    print("快撞了！！！")
                    brake(0.3)  # 小车停止2s
                    back(0.3)
                    left(0.6)
                else:
                    print("问题不大。。。")
                    run(0.3)  # 小车前进5s
                    time.sleep(0.5)


        while True:
            '''
            color
            '''
            print("______________________红绿路灯______________________")
            time.sleep(0.2)
            ret, frame = colorRecog.cap.read()
            # frame = cv2.imread("yellow-red.png")
            isRed = colorRecog.ChestRed(frame)
            if not isRed:
                print("not red!!")
                avoid()
            else:
                print("Red!")
                brake(5)

            '''
            signal
            '''
            print("______________________交通信号______________________")
            time.sleep(1)
            #             ret,frame =  SighRecong.cap.read()
            label0 = SighRecong.predict(args, frame)
            print(label0)

            if label0 == '[0]':
                print("stop!")
                brake(5)
            elif label0 == '[1]':
                print("wait!")
                brake(3)
            elif label0 == '[2]':
                print("stop!")
                brake(5)
            elif label0 == '[3]':
                print("stop!")
                brake(5)
            elif label0 == '[4]':
                print("straight!")
                avoid()
            elif label0 == '[5]':
                print("left!")
                spin_left(1)
                avoid()
            elif label0 == '[6]':
                print("right!")
                spin_right(1)
                avoid()

        # Reset by pressing CTRL + C
    except KeyboardInterrupt:
        print("Measurement stopped by User")
        GPIO.cleanup()
        car.allStop()