import warnings
warnings.filterwarnings('ignore')
import numpy as np, os, pickle, cv2, glob
from imageio.v2 import imread
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics
import imageio.v2 as imageio
from pathlib import Path

from prepare_data import *
from data_process import *
from model import *

"""

如果能用github，直接在kaggle连接github把工程文件下载下来，更新和修改都很方便
可惜这里要手动上传，要等好几分钟

这个文件将会在kaggle上面运行，运行结果可以下载下来，再进行血肿体积预测

"""


def Sens(y_true, y_pred):
    cm1 = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])  # labels =[1,0] [positive [Hemorrhage], negative]
    SensI = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    return SensI  # TPR is also known as sensitivity

def Precision(y_true, y_pred):
    return tf.keras.metrics.Precision(name='precision')(y_true, y_pred)


def Speci(y_true, y_pred):
    cm1 = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])  # labels =[1,0] [positive [Hemorrhage], negative]
    SpeciI = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    return SpeciI  # FPR is one minus the specificity or true negative rate


def Jaccard_img(y_true, y_pred):  # https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    iou_score = 0
    counter = 0
    for i in range(y_true.shape[0]):
        if np.sum(y_true[
                      i]) > 0:  # Considering only the slices that have hemorrhage regions, if y_true is all zeros -> iou_score=nan.
            im1 = np.asarray(y_true[i]).astype(np.bool_)
            im2 = np.asarray(y_pred[i]).astype(np.bool_)
            intersection = np.logical_and(im1, im2)
            union = np.logical_or(im1, im2)
            iou_score += np.sum(intersection) / np.sum(union)
            counter += 1
    if counter > 0:
        return iou_score / counter
    else:
        return np.nan


def dice_img(y_true, y_pred):
    dice = 0
    counter = 0
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:  # Considering only the slices that have hemorrhage regions,
            dice += dice_fun(y_true[i], y_pred[i])
            counter += 1
    if counter > 0:
        return dice / counter
    else:
        return np.nan


def dice_fun(im1, im2):
    im1 = np.asarray(im1).astype(np.bool_)
    im2 = np.asarray(im2).astype(np.bool_)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


# def testModel(model_path, test_path, save_path):
#     """
#     这个函数，用来测试模型在测试集上面的分割效果
#
#     模型在测试集上面的分割结果，将会出现在result_trial[次数]的文件夹中
#     届时，可用“计算HV.py”这个代码文件，读取其中的分割结果，计算出对各个病人预测的血肿体积大小。
#
#     :param model_path:
#     :param test_path:
#     :param save_path:
#     :return:
#     """
#     batch_size = 16
#     modelUnet = dr_unet(pretrained_weights=model_path, input_size=(windowLen, windowLen, 1))
#     testGener = testGenerator(test_path, target_size=(windowLen, windowLen, 1))
#     testPredictions = modelUnet.predict(x=testGener, verbose=1,
#                                         steps=1)
#     saveResult(test_path, save_path,
#                testPredictions)  # sending the test image path so same name will be used for saving masks

def testModel(model_path, test_path, save_path):
    modelUnet = dr_unet(pretrained_weights=model_path, input_size=(windowLen, windowLen, 1))
    testGener = testGenerator(test_path, target_size=(windowLen, windowLen, 1))
    testGener = enumerate(testGener)

    total = len(glob.glob(os.path.join(test_path, "*.png")))
    flag = True

    for batch_start_idx in range(0, total, batch_size):
        cur_batch_id = batch_start_idx // batch_size + 1
        print('Ready to test in Batch: ', cur_batch_id)
        test_images = []
        for _ in range(batch_size):
            try:
                i, img = next(testGener)
                test_images.append(img)
            except Exception as e:
                print(e)
                flag = False
                break

        if not test_images:
            break

        test_images_np = np.array(test_images)
        print('Test data shape: ', test_images_np.shape)

        res = modelUnet.predict(test_images_np, batch_size=1, verbose=1)
        saveResult(test_path, save_path, res, start_idx=batch_start_idx, batch_size=batch_size)


data_gen_args = dict(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    # brightness_range=(0,1.5),
    fill_mode="nearest"
)

num_CV = 1  # 这里是交叉验证的折数
NumEpochs = 1  # 这里控制训练的epoch数量
NumEpochEval = 1  # validated the model each NumEpochEval epochs
batch_size = 100  # batch_size的设置
learning_rateI = 1e-5

if __name__ == '__main__':
    #############################################Training Parameters#######################################################
    import argparse

    parser = argparse.ArgumentParser(description='Process the pretrained weights file path.')

    parser.add_argument('--w', dest='pretrained_weights', type=str, default=None,
                        help='Path to the pretrained weights file (default: None)')

    args = parser.parse_args()

    if NumEpochs != 0:
        decayI = learning_rateI / NumEpochs
    else:
        decayI = 0

    detectionSen = 20 * 20  # labeling each slice as ICH if hemorrhage is detected in detectionSen pixels
    thresholdI = 0.5
    detectionThreshold = thresholdI * 256  # threshold on detection probability
    numSubj = 75
    num_WindowsCT = 49
    imageLen = 512
    windowLen = 128
    strideLen = 64
    num_Moves = int(imageLen / strideLen) - 1
    window_specs = [40, 120]  # Brain window
    kernel_closing = np.ones((10, 10), np.uint8)
    kernel_opening = np.ones((5, 5), np.uint8)  # 5*5 in order not to delete thin hemorrhage
    counterI = 1

    pretrained_weights = args.pretrained_weights

    # 重复训练的时候，这里会自动新建一个results_trial[次数]的文件夹，代表你第几次训练的结果
    SaveDir = Path('results_trial' + str(counterI))
    while (os.path.isdir(str(SaveDir))):
        counterI += 1
        SaveDir = Path('results_trial' + str(counterI))
    os.mkdir(str(SaveDir))
    os.mkdir(str(Path(SaveDir, 'crops')))
    os.mkdir(str(Path(SaveDir, 'fullCT_original')))  # Testing without morphological operations
    os.mkdir(str(Path(SaveDir, 'fullCT_morph')))  # Testing with morphological operations
    print('The results of the training, validation and testing will be saved to:' + str(SaveDir))

    #############################################Downloading and unzipping the dataset######################################
    dataset_zip_dir = 'computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1.zip'
    crossvalid_dir = 'DataV1'
    prepare_data(dataset_zip_dir, crossvalid_dir, numSubj, imageLen, windowLen, strideLen,
                 num_Moves, window_specs)  # preparing the data and saving it to ICH_DataSegmentV1.pkl

    # Loading full image mask from the crops predictions
    with open(str(Path(crossvalid_dir, 'ICH_DataSegmentV1.pkl')), 'rb') as Dataset1:
        [hemorrhageDiagnosisArray, AllCTscans, testMasks, subject_nums_shaffled] = pickle.load(Dataset1)
    del AllCTscans

    testMasks = np.uint8(testMasks)
    testMasksAvg = np.where(np.sum(np.sum(testMasks, axis=1), axis=1) > detectionSen, 1, 0)  #
    testPredictions = np.zeros((testMasks.shape[0], imageLen, imageLen), dtype=np.uint8)  # predicted segmentation

    ############################################Cross-validation############################################################
    print('Starting the cross-validation!!')
    # num_CV = 1时，只会使用CV0中的数据进行训练
    for cvI in range(0, num_CV):
        save_model_path = str(Path(SaveDir, 'DR_UNet_CV' + str(cvI) + '.keras'))

        print("Working on fold #" + str(cvI) + ", starting training U-Net")
        SaveDir_crops_cv = Path(SaveDir, 'crops', 'CV' + str(cvI))
        if os.path.isdir(str(SaveDir_crops_cv)) == False:
            os.mkdir(str(SaveDir_crops_cv))
        SaveDir_full_cv = Path(SaveDir, 'fullCT_original', 'CV' + str(cvI))
        if os.path.isdir(str(SaveDir_full_cv)) == False:
            os.mkdir(str(SaveDir_full_cv))
        SaveDir_cv = Path(SaveDir, 'fullCT_morph', 'CV' + str(cvI))
        if os.path.isdir(str(SaveDir_cv)) == False:
            os.mkdir(str(SaveDir_cv))

        dataDir = Path(crossvalid_dir, 'CV' + str(cvI))
        n_imagesTrain = len(glob.glob(os.path.join(str(Path(dataDir, 'train', 'image')), "*.png")))
        print('n_imagesTrain', n_imagesTrain)
        n_imagesValidate = len(glob.glob(os.path.join(str(Path(dataDir, 'validate', 'image')), "*.png")))
        print('n_imagesValidate', n_imagesValidate)
        n_imagesTest = len(glob.glob(os.path.join(str(Path(dataDir, 'test', 'fullCT', 'image')), "*.png")))
        print('n_imagesTest: ', n_imagesTest)
        n_imagesTest = len(glob.glob(os.path.join(str(Path(dataDir, 'test', 'crops', 'image')), "*.png")))
        trainGener = trainGenerator(batch_size, str(Path(dataDir, 'train')), 'image', 'label', data_gen_args,
                                    save_to_dir=None, target_size=(128, 128))
        valGener = validateGenerator(batch_size, str(Path(dataDir, 'validate')), 'image', 'label', save_to_dir=None,
                                     target_size=(128, 128))
        modelUnet = dr_unet(pretrained_weights=pretrained_weights, input_size=(windowLen, windowLen, 1))
        model_checkpoint = ModelCheckpoint(save_model_path,
                                           mode='min',
                                           verbose=1, save_freq=NumEpochEval)

        history1 = modelUnet.fit(trainGener, epochs=NumEpochs,
                                 steps_per_epoch=int(n_imagesTrain / batch_size),
                                 # validation_data=valGener, validation_steps=n_imagesValidate,
                                 callbacks=[model_checkpoint])

        modelUnet.save(save_model_path)

        # Loading and testing the model with lowest validation loss
        print('Testing the best U-Net model on testing data and saving the results to: ' + str(SaveDir_crops_cv))
        testModel(save_model_path, str(Path(dataDir, 'test', 'crops', 'image')),
                  str(SaveDir_crops_cv))

        # Creating full image mask from the crops predictions
        if num_CV != 1:
            if cvI < num_CV - 1:
                subjectNums_cvI_testing = subject_nums_shaffled[
                                          cvI * int(numSubj / num_CV):cvI * int(numSubj / num_CV) + int(
                                              numSubj / num_CV)]
            else:
                subjectNums_cvI_testing = subject_nums_shaffled[cvI * int(numSubj / num_CV):numSubj]
        else:
            subjectNums_cvI_testing = [31, 39, 21, 33, 34, 5, 54, 2, 67, 68, 53, 29, 44, 76, 59, 73, 77, 71, 61, 69, 50,
                                     32, 6, 37,
                                     57, 75, 80, 41, 27, 40, 46, 79, 45, 55, 62, 7, 66, 58, 78, 4, 47, 52, 28, 20, 24,
                                     51, 36, 63, 30,
                                     48, 26, 60, 49, 25, 42, 18, 43, 72, 0, 35, 81, 70, 22, 64, 1, 3, 17, 74, 23, 38, 8,
                                     65, 19,
                                     56, 9]

        # Finding the predictions or ICH segmentation for the whole slice
        print(
            'Combining the crops masks to find the full CT mask after performing morphological operations and saving the results to: ' + str(
                SaveDir_full_cv))
        for subItest in range(0, len(subjectNums_cvI_testing)):
            slicenum_s = hemorrhageDiagnosisArray[
                hemorrhageDiagnosisArray[:, 0] == subjectNums_cvI_testing[subItest], 1]
            sliceInds = np.where(hemorrhageDiagnosisArray[:, 0] == subjectNums_cvI_testing[
                subItest])  # using the slice index to keep the predictions have the same sequence as the ground truth.
            counterSlice = 0
            for sliceI in range(slicenum_s.size):
                # reading the predicted segmentation for each window
                CTslicePredict = np.zeros((imageLen, imageLen))
                windowOcc = np.zeros((imageLen, imageLen))  # number of predictions for each pixel in the CT scan
                counterCrop = 0
                for i in range(num_Moves):
                    for j in range(num_Moves):
                        windowI = imread(Path(SaveDir_crops_cv, str(subjectNums_cvI_testing[subItest])
                                              + '_' + str(sliceI) + '_' + str(counterCrop) + '.png'))
                        windowI = windowI / 255
                        CTslicePredict[
                        int(i * imageLen / (num_Moves + 1)):int(i * imageLen / (num_Moves + 1) + windowLen),
                        int(j * imageLen / (num_Moves + 1)):int(
                            j * imageLen / (num_Moves + 1) + windowLen)] \
                            = CTslicePredict[
                              int(i * imageLen / (num_Moves + 1)):int(
                                  i * imageLen / (
                                          num_Moves + 1) + windowLen),
                              int(j * imageLen / (num_Moves + 1)):int(
                                  j * imageLen / (
                                          num_Moves + 1) + windowLen)] + windowI

                        windowOcc[
                        int(i * imageLen / (num_Moves + 1)):int(i * imageLen / (num_Moves + 1) + windowLen),
                        int(j * imageLen / (num_Moves + 1)):int(
                            j * imageLen / (num_Moves + 1) + windowLen)] \
                            = windowOcc[
                              int(i * imageLen / (num_Moves + 1)):int(
                                  i * imageLen / (
                                          num_Moves + 1) + windowLen),
                              int(j * imageLen / (num_Moves + 1)):int(
                                  j * imageLen / (
                                          num_Moves + 1) + windowLen)] + 1

                        counterCrop = counterCrop + 1

                CTslicePredict = CTslicePredict / windowOcc * 255
                # CTslicePredict = CTslicePredict * 255
                img = np.uint8(CTslicePredict)
                imsave(Path(SaveDir_full_cv, str(subjectNums_cvI_testing[subItest])
                            + '_' + str(sliceI) + '.png'), img)

                img = np.int16(np.where(img > detectionThreshold, 255, 0))

                # imsave(Path(SaveDir_full_cv, str(subjectNums_cvI_testing[subItest])
                #             + '_' + str(sliceI) + '.png'), img)

                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_closing)  # Filling the gaps
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_opening)
                imsave(Path(SaveDir_cv, str(subjectNums_cvI_testing[subItest])
                            + '_' + str(sliceI) + '.png'), np.uint8(img))

                testPredictions[sliceInds[0][counterSlice]] = np.uint8(np.where(img > (0.5 * 256), 1, 0))

                counterSlice += 1
