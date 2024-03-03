import numpy as np
from PIL import Image
from imageio import imread
from glob import glob
from sklearn.metrics import roc_auc_score
import os
import cv2

DATASET_PATH = '/content/drive/MyDrive/MRI_Anomaly'

def resize(image, shape=(182, 182)):
    return np.array(Image.fromarray(image).resize(shape[::-1]))


def bilinears(images, shape) -> np.ndarray:
    N = images.shape[0]
    new_shape = (N,) + shape
    ret = np.zeros(new_shape, dtype=images.dtype)
    for i in range(N):
        ret[i] = cv2.resize(images[i], dsize=shape[::-1], interpolation=cv2.INTER_LINEAR)
    return ret

# def gray2rgb(images):
#     # tile_shape = tuple(np.ones(len(images.shape), dtype=int))
#     # tile_shape += (3,)

#     # images = np.tile(np.expand_dims(images, axis=-1), tile_shape)
#     images = np.expand_dims(images, axis=-1)
#     # print(images.shape)
#     return images

def set_root_path(new_path):
    global DATASET_PATH
    DATASET_PATH = new_path


def get_x(mode='train'):
    fpattern = os.path.join(DATASET_PATH, f'{mode}/*/*/*.png')
    print(f'fpattern : {fpattern}')
    fpaths = sorted(glob(fpattern))
    print(f'fpattern 내 개수 : {len(fpaths)}')

    if mode == 'test':
        # test 폴더 내 Normal(정상)이 아닌 이미지 경로
        fpaths1 = list(filter(lambda fpath: 'Normal' not in fpath.split(os.path.sep), fpaths))
        print(f'Abnormal 경로 : {len(fpaths1)}')
        # test 폴더 내 Normal(정상)인 이미지 경로
        fpaths2 = list(filter(lambda fpath: 'Normal' in fpath.split(os.path.sep), fpaths))
        print(f'Normal 경로 : {len(fpaths2)}')

        images1 = np.asarray(list(map(imread, fpaths1)))
        print(images1.shape)
        # images1 = np.expand_dims(images1, axis=-1) 
        images2 = np.asarray(list(map(imread, fpaths2)))
        print(images2.shape)
        # images2 = np.expand_dims(images2, axis=-1) 
        images = np.concatenate([images1, images2])

    else:
        images = np.asarray(list(map(imread, fpaths)))
        # images = np.expand_dims(images, axis=-1) 
        
        print(f'불러들어온 image 개수 : {len(images)}')

    print(f'이미지 텐서 shape : {images.shape}')
    # MRI 이미지는 흑백이므로 아래 코드는 주석 처리
    print('텐서 shape 변환!!! \n')
    # images = np.expand_dims(images, axis=-1)
    
    # if images.shape[-1] != 3:
    #     print('텐서 shape 변환!!! \n')
    #     images = np.expand_dims(images, axis=-1)
    #     # images = gray2rgb(images)

    images = list(map(resize, images))
    images = np.asarray(images)
    images = np.expand_dims(images, axis=-1) 
    print(f'이미지 텐서 차원 : {images.shape}\n')
    return images


def get_x_standardized(mode='train'):
    x = get_x(mode=mode)
    mean = get_mean(x)
    return (x.astype(np.float32) - mean) / 255


def get_label():
    fpattern = os.path.join(DATASET_PATH, f'test/*/*/*.png')
    fpaths = sorted(glob(fpattern))
    # test 폴더 내 Normal(정상)이 아닌 이미지 경로
    fpaths1 = list(filter(lambda fpath: 'Normal' not in fpath.split(os.path.sep), fpaths))
    # test 폴더 내 Normal(정상)인 이미지 경로
    fpaths2 = list(filter(lambda fpath: 'Normal' in fpath.split(os.path.sep), fpaths))

    Nanomaly = len(fpaths1)
    Nnormal = len(fpaths2)
    labels = np.zeros(Nanomaly + Nnormal, dtype=np.int32)
    labels[:Nanomaly] = 1
    return labels


def get_mask():
    # maks는 test 폴더에만 존재
    fpattern = os.path.join(DATASET_PATH, f'ground_truth/*/*.png')
    fpaths = sorted(glob(fpattern))
    masks = np.asarray(list(map(lambda fpath: resize(imread(fpath), (182, 182)), fpaths)))
    # 테스트 폴더 내 비정상 이미지 개수
    Nanomaly = masks.shape[0]
    # 테스트 폴더 내 정상 이미지 개수
    Nnormal = len(glob(os.path.join(DATASET_PATH, f'test/Normal/*/*.png')))

    # 픽셀 값이 128 이하면 0
    masks[masks <= 128] = 0
    # 픽셀 값이 128 초과하면 255
    masks[masks > 128] = 255
    results = np.zeros((Nanomaly + Nnormal,) + masks.shape[1:], dtype=masks.dtype)
    results[:Nanomaly] = masks

    return results


def get_mean(images):
    # images = get_x(mode='train')
    mean = images.astype(np.float32).mean(axis=0)
    return mean


def detection_auroc(anomaly_scores):
    label = get_label()  # 1: anomaly 0: normal
    auroc = roc_auc_score(label, anomaly_scores)
    return auroc


def segmentation_auroc(anomaly_maps):
    gt = get_mask()
    gt = gt.astype(np.int32)
    gt[gt == 255] = 1  # 1: anomaly

    # bilinears(anomaly_maps, (256, 256)) --> bilinears(anomaly_maps, (182, 182))
    anomaly_maps = bilinears(anomaly_maps, (182, 182))
    auroc = roc_auc_score(gt.flatten(), anomaly_maps.flatten())
    return auroc