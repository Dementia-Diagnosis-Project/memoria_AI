# inference.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model

# GradCAM 클래스
class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.gradient_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output])

    def compute_heatmap(self, image, class_index):
        with tf.GradientTape() as tape:
            conv_output, predictions = self.gradient_model(image)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]

        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

# 보조 함수들
def load_and_process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (299, 299))
    image = np.expand_dims(image, axis=0)
    return image

def load_image(image_path, gender, age_group, image_number):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    return image, (gender, age_group, image_number)

def calculate_percentage(probabilities):
    neg_percentage = np.mean(probabilities[:, 0]) * 100.0
    pos_percentage = np.mean(probabilities[:, 1]) * 100.0
    return round(neg_percentage, 1), round(pos_percentage, 1)

def generate_heatmap_for_patient(patient_folder, model, class_index=1, image_index=18):
    image_path = os.path.join(patient_folder, f"plane{image_index}.png")
    image = load_and_process_image(image_path)

    gradcam = GradCAM(model, 'conv_7b')  # Inception-ResNetV2의 마지막 conv layer
    heatmap = gradcam.compute_heatmap(image, class_index)

    heatmap_resized = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    heatmap_rescaled = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)

    return heatmap_color

def visualize_heatmap(heatmap_color):
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Main Inference 함수
def inference(folder_path, gender, age_group, model_path, batch_size=30):
    """
    Args:
        folder_path: 검사할 환자 이미지 폴더
        gender: 0 (여성) 또는 1 (남성)
        age_group: 0 (50대), 1 (60대), 2 (70대), 3 (80대), 4 (90대)
        model_path: 저장된 모델 경로
    Returns:
        heatmap, (neg_percent, pos_percent)
    """

    global full_model

    if 'full_model' not in globals():
        full_model = load_model(model_path)

    model = full_model
    model_CNN = model.pretrained_model.inception_res_partial  # backbone만 추출

    # 데이터셋 준비
    image_paths = []
    genders = []
    age_groups = []
    image_numbers = []

    for i in range(30):
        img_path = os.path.join(folder_path, f"plane{i}.png")
        image_paths.append(img_path)
        genders.append(gender)
        age_groups.append(age_group)
        image_numbers.append(i)

    df = pd.DataFrame({
        'image_path': image_paths,
        'gender': genders,
        'age_group': age_groups,
        'image_number': image_numbers
    })

    image_paths = df['image_path'].values
    genders = tf.cast(df['gender'].values, tf.int64)
    age_groups = tf.cast(df['age_group'].values, tf.int64)
    image_numbers = tf.cast(df['image_number'].values, tf.int64)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, genders, age_groups, image_numbers))
    dataset = dataset.map(load_image)
    dataset = dataset.batch(batch_size)

    for images, diagnoses in dataset.take(1):
        logits = model([images, diagnoses])

    neg_percentage, pos_percentage = calculate_percentage(logits.numpy())

    print("음성(정상) 확률 평균: {:.1f}%".format(neg_percentage))
    print("양성(AD) 확률 평균: {:.1f}%".format(pos_percentage))

    # GradCAM 히트맵 생성
    heatmap_color = generate_heatmap_for_patient(folder_path, model_CNN, class_index=1)

    # 히트맵 시각화
    visualize_heatmap(heatmap_color)

    return heatmap_color, (neg_percentage, pos_percentage)

# Example usage
if __name__ == "__main__":
    folder_path = "/content/drive/MyDrive/Dataset/Dementia_sample/Real_data/Mild AD/002_S_0729_110816"
    gender = 0
    age_group = 0
    model_path = "/content/drive/MyDrive/Dataset/inception_resnet"

    heatmap_color, percentages = inference(folder_path, gender, age_group, model_path)
    print("히트맵 생성 완료")
    print("음성 및 양성 퍼센트:", percentages)
