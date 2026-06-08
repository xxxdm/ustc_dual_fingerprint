import numpy as np
##本项目用于数据处理
##原则：1、一份数据两种角度表示：1、序列 2、图像   2、让不同分支学习不同类型特征：1、序列分支看顺序2、图像分支看局部二维结构
##packet_to_fixed_bytes（）函数用于把一个原始网络包的字节流转换成固定长度的一维数组。
##原因：神经网络训练需要根据固定长度的数据进行训练
##思路：截长补短
def packet_to_fixed_bytes(pkt_bytes, seq_len=784):  #这一步前需要把每个网站流量文件的每一个数据包先转换成原始数据六
    arr = np.frombuffer(pkt_bytes, dtype=np.uint8)##把数据包的原始 bytes 转成 uint8 数组，uint8数组元素范围为0~255
    ##判断数组长度是否大于既定len长度，长的按既定长度截断，短的补0至既定长度
    if len(arr) >= seq_len:
        arr = arr[:seq_len]
    else:
        pad = np.zeros(seq_len - len(arr), dtype=np.uint8)
        arr = np.concatenate([arr, pad], axis=0)
    return arr
##bytes_to_image（）函数将一维字节序列转换成二维图像，用于多头注意力机制
##原因：CNN 擅长从二维局部结构中提取模式
def bytes_to_image(arr, image_size=28):
    arr = arr[:image_size * image_size]  ##784的尺寸，前28x28的数reshape 成(1, 28, 28)的灰度图
    return arr.reshape(1, image_size, image_size).astype(np.float32) / 255.0