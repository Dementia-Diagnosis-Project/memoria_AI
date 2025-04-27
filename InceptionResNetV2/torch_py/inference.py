# inference.py (PyTorch version)

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from model import FinalModel
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# GradCAM 클래스
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)

    def compute_heatmap(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)
        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=[1, 2], keepdim=True)
        heatmap = torch.sum(weights * activations, dim=0)

        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.cpu().detach().numpy()

# 보조 함수들
def load_and_process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (299, 299))
    image = np.expand_dims(image, axis=0)
    return image

def load_image(image_path, gender, age_group, image_number):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image, (torch.tensor(gender).long(), torch.tensor(age_group).long(), torch.tensor(image_number).long())

def calculate_percentage(probabilities):
    neg_percentage = np.mean(probabilities[:, 0]) * 100.0
    pos_percentage = np.mean(probabilities[:, 1]) * 100.0
    return round(neg_percentage, 1), round(pos_percentage, 1)

def generate_heatmap_for_patient(patient_folder, model, device, class_index=1, image_index=18):
    image_path = os.path.join(patient_folder, f"plane{image_index}.png")
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    gradcam = GradCAM(model.cnn.backbone, target_layer='Mixed_7c')
    heatmap = gradcam.compute_heatmap(image, class_idx=class_index)

    heatmap_resized = cv2.resize(heatmap, (299, 299))
    heatmap_rescaled = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)

    return heatmap_color

def visualize_heatmap(heatmap_color):
    plt.imshow(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Main Inference 함수
def inference(folder_path, gender, age_group, model_path, batch_size=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FinalModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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

    images = []
    genders = []
    age_groups = []
    image_numbers = []

    for idx in range(len(df)):
        img, (g, a, n) = load_image(df.iloc[idx]['image_path'], df.iloc[idx]['gender'], df.iloc[idx]['age_group'], df.iloc[idx]['image_number'])
        images.append(img)
        genders.append(g)
        age_groups.append(a)
        image_numbers.append(n)

    images = torch.stack(images).to(device)
    genders = torch.stack(genders).to(device)
    age_groups = torch.stack(age_groups).to(device)
    image_numbers = torch.stack(image_numbers).to(device)

    with torch.no_grad():
        outputs = model((images, (genders, age_groups, image_numbers)))
        probs = F.softmax(outputs, dim=1).cpu().numpy()

    neg_percentage, pos_percentage = calculate_percentage(probs)

    print("음성(정상) 확률 평균: {:.1f}%".format(neg_percentage))
    print("양성(AD) 확률 평균: {:.1f}%".format(pos_percentage))

    # GradCAM 히트맵 생성
    heatmap_color = generate_heatmap_for_patient(folder_path, model, device, class_index=1)

    # 히트맵 시각화
    visualize_heatmap(heatmap_color)

    return heatmap_color, (neg_percentage, pos_percentage)

# Example usage
if __name__ == "__main__":
    folder_path = "/content/drive/MyDrive/Dataset/Dementia_sample/Real_data/Mild AD/002_S_0729_110816"
    gender = 0
    age_group = 0
    model_path = "/content/drive/MyDrive/Dataset/before_augmentation_False/best_model.pth"

    heatmap_color, percentages = inference(folder_path, gender, age_group, model_path)
    print("히트맵 생성 완료")
    print("음성 및 양성 퍼센트:", percentages)