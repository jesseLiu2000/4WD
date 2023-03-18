# 第三步 模型应用，识别交通标志图片
# 加载工程中必要的库
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
 
# 根据使用的模型，确定图像需要resize的尺寸
norm_size = 64
 
# 预测函数，
# 输入： 包含配置参数的字典
def predict(args):
    
    # 加载训练好的卷积神经网络
    print("[INFO] loading network...")
    model = load_model(args["model"])
 
    # 加载图像
    image = cv2.imread(args["image"])
    # 因为对图像需要进行写入标签，影响较大所以复制一个图像
    orig = image.copy()
 
    # 预处理图像进行分类
    # 图像的尺寸重载
    image = cv2.resize(image, (norm_size, norm_size))
    # 图像的序列的归一化处理
    image = image.astype("float") / 255.0
    # 将图像进行序列化
    image = img_to_array(image)
    # 展开数组的形状.
    # 插入一个新的轴，该轴将出现在扩展阵列形状的轴位置
    image = np.expand_dims(image, axis=0)
 
    # 对输入的图像进行分类
    result = model.predict(image)[0]
    # print (result.shape)
    proba = np.max(result)
    label0 = str(np.where(result == proba)[0])
    print(label0)
    if label0 == '[0]':
        print("stop!")
    elif label0 == '[1]':
        print("wait!")
    elif label0 == '[2]':
        print("stop!")
    elif label0 == '[3]':
        print("stop!")
    elif label0 == '[4]':
        print("straight!")
    elif label0 == '[5]':
        print("left!")
    elif label0 == '[6]':
        print("right!")
    label = "{}: {:.2f}%".format(label0, proba * 100)
    print(label)
    # 在需要加载图像的情况下
    if args['show']:
        output = imutils.resize(orig, width=400)
        # 在图像上绘制标签字符串
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        # 显示带标签的图像
        cv2.imshow("Output", output)
        cv2.waitKey(0)
 
 
# python predict.py --model traffic_sign.model -i ../2.png -s
if __name__ == '__main__':
    args = {}
    # 模型的输入路径
    args['model'] = './MODE/traffic_sign.model'
    # 图像的输入路径
    args['image'] = './predict/00375_00000.png'
    args['show'] = 'true'
    # 执行预测
    predict(args)
