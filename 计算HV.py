import sys
from collections import Counter
import pathlib
import math
import os

import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import cv2 as cv
import tqdm
from PIL import Image

"""

这里的代码通过模型输出的分割结果，进行血肿体积计算。


"""

import numpy as np

def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Square Error between two sequences
    :param y_true: array-like of true values
    :param y_pred: array-like of predicted values
    :return: float, the RMSE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def sd(values):
    """
    Calculate the Standard Deviation of a sequence
    :param values: array-like of numerical data
    :return: float, the Standard Deviation
    """
    values = np.array(values)
    mean = np.mean(values)
    return np.sqrt(np.mean((values - mean) ** 2))

def mae(y_true, y_pred):
    """
    Calculate the Mean Absolute Error between two sequences
    :param y_true: array-like of true values
    :param y_pred: array-like of predicted values
    :return: float, the MAE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))



def calculate_hemorrhage_volume_from_fullCT(label_dir, pixel_spacing=0.75, slice_thickness=5):
    """
    计算基于CT标签图像的血肿体积。

    :param image_dir: 存放CT图像的目录路径
    :param label_dir: 存放CT标签图像的目录路径
    :param pixel_spacing: CT图像中每个像素代表的实际距离（毫米）
    :param slice_thickness: 每张CT切片的厚度（毫米）
    :return: 返回一个字典，键为病人ID，值为相应的预测血肿体积（毫升）
    """
    # 每个体素的体积
    voxel_volume = pixel_spacing * pixel_spacing * slice_thickness

    # 用来存储每个病人血肿体积的字典
    hemorrhage_volumes = {}

    # 读取所有标签文件名
    label_files = os.listdir(label_dir)

    for label_file in label_files:
        # 构建完整的文件路径
        label_path = os.path.join(label_dir, label_file)

        # 加载标注图像
        label_image = Image.open(label_path)
        label_array = np.array(label_image)

        # 计算白色像素的数量（即肿块的体积）
        hemorrhage_volume = np.sum(label_array >= 254) * voxel_volume

        # 解析病人ID和切片编号
        patient_id, _ = label_file.split('.')[0].split('_')

        # 将体积加到该病人的总体积中
        hemorrhage_volumes[patient_id] = hemorrhage_volumes.get(patient_id, 0) + hemorrhage_volume

    # 返回每个病人的血肿总体积（转换为毫升）
    return {patient_id: volume / 1000 for patient_id, volume in hemorrhage_volumes.items()}


if __name__ == '__main__':

    ##################--下面的是通过完整的CT标注图计算血肿体积--##################

    # 使用示例
    label_dir = r"D:\Pycharm_Projects\UNet\计算HV与统计分析\test_label (4)\kaggle\working\ddrhv\DataV1\CV0\test\fullCT\label"
    # 使用完整的CT图进行计算血肿体积
    label_volumes = calculate_hemorrhage_volume_from_fullCT(label_dir)

    # 打印每个病人的血肿总体积

    segm_dir = r"D:\Pycharm_Projects\UNet\计算HV与统计分析\Pred_Segament (4)\results_trial1\fullCT_original\CV0"

    seg_volumes = calculate_hemorrhage_volume_from_fullCT(segm_dir)

    y_true_li = []
    y_pred_li = []

    for pid in (set(seg_volumes.keys()) & set(label_volumes.keys())):
        y_true = label_volumes[pid]
        y_pred = seg_volumes[pid]
        y_true_li.append(y_true)
        y_pred_li.append(y_pred)

        print('## 病人', pid)
        print(f'真实血肿体积：{y_true}')
        print(f'预测血肿体积：{y_pred}')
        print('------\n')


    print(f'MAE:      {mae(y_true_li, y_pred_li)}')
    print(f'RMSE:     {rmse(y_true_li, y_pred_li)}')
    print(f'SD(True): {sd(y_true_li)}')
    print(f'SD(Predict): {sd(y_pred_li)}')

    print('单位: 毫升(ml)')