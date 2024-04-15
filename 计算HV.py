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
        hemorrhage_volume = np.sum(label_array > 0) * voxel_volume

        # 解析病人ID和切片编号
        patient_id, _ = label_file.split('.')[0].split('_')

        # 将体积加到该病人的总体积中
        hemorrhage_volumes[patient_id] = hemorrhage_volumes.get(patient_id, 0) + hemorrhage_volume

    # 返回每个病人的血肿总体积（转换为毫升）
    return {patient_id: volume / 1000 for patient_id, volume in hemorrhage_volumes.items()}


def calculate_hemorrhage_volume_from_subCT(path, pixel_spacing=0.75, slice_thickness=5.0):
    # 用于存储每个病人血肿体积的字典
    hemorrhage_volumes = {}

    for filename in os.listdir(path):
        if filename.endswith(".png"):
            # 解析文件名以获取病人ID
            patient_id = filename.split('_')[0]
            # 读取图像文件
            img = Image.open(os.path.join(path, filename))
            # 将图像转换为灰度数组
            img_array = np.array(img)
            # 计算标记为血肿的白色像素点的数量
            white_pixels = np.sum(img_array > 0)
            # 调整像素点数量以匹配原始CT图像的尺寸，计算血肿体积
            vol = white_pixels * slice_thickness * pixel_spacing
            # 将计算得出的体积加到对应病人ID的总体积中
            hemorrhage_volumes[patient_id] = hemorrhage_volumes.get(patient_id, 0) + vol

    # 返回每个病人的预测血肿体积结果
    return {patient_id: volume / 1000 for patient_id, volume in hemorrhage_volumes.items()}


if __name__ == '__main__':

    # 使用示例
    label_dir = r"D:\Pycharm_Projects\UNet\DataV1\CV0\test\fullCT\label"
    volumes = calculate_hemorrhage_volume_from_fullCT(label_dir)

    # 打印每个病人的血肿总体积

    print('Patient : hemorrhage volume(ml)')
    print(sorted(volumes.items(), key=lambda x: x[0]))

    print('-' * 100)

    # 函数调用示例
    pixel_spacing = 0.75
    slice_thickness = 1.0
    original_dim = 512
    crop_dim = 128
    path = r"D:\Pycharm_Projects\UNet\DataV1\CV0\test\crops\label"

    # 计算体积
    volumes = calculate_hemorrhage_volume_from_subCT(path, pixel_spacing, slice_thickness)

    # 打印结果
    print('Patient : hemorrhage volume(ml)')
    print(sorted(volumes.items(), key=lambda x: x[0]))
