# train.py (PyTorch version)

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR
from utils import seed_everything, fold_path, create_dataset, make_dataloader
from model import FinalModel
import re

# 1. 설정
seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_path = '/content/drive/MyDrive/Dataset/MRI_dataset/Sharpening_data_2'
csv_path = '/content/drive/MyDrive/Dataset/ADNI_tabular/ADNI_subjects.xlsx'

# 2. 데이터 로드 및 정리
df = pd.read_excel(csv_path)
df = df.drop(['PHASE', 'AD severity', 'Certain or Not', 'CDR'], axis=1)

fold_data = fold_path(base_path)

data_list = []
for class_label, subject_folders in fold_data.items():
    for subject_folder in subject_folders:
        parts = subject_folder.split('_')
        subject_id, examdate = '_'.join(parts[0:3]), '20' + parts[3]
        result_row = df[(df['PTID'] == subject_id) & (df['EXAMDATE'] == examdate)]

        if not result_row.empty:
            image_dir = os.path.join(base_path, class_label, subject_folder)
            image_numbers = [int(re.search(r'plane(\d+)', f).group(1)) for f in os.listdir(image_dir) if f.endswith('.png')]
            data_list.append({
                'image_path': image_dir,
                'Gender': result_row['Gender'].values[0],
                'Age': result_row['Age'].values[0],
                'Image_Number': sorted(image_numbers),
                'label': class_label
            })

final_df = pd.DataFrame(data_list)

# 라벨, 나이 그룹 변환
final_df['label'] = final_df['label'].map({'Normal': 0, 'Mild AD': 1})
bins = [50, 60, 70, 80, 90, 100]
labels = ['50대', '60대', '70대', '80대', '90대']
final_df['Age_Group'] = pd.cut(final_df['Age'], bins=bins, labels=labels, right=False)
age_group_mapping = {'50대': 0, '60대': 1, '70대': 2, '80대': 3, '90대': 4}
final_df['Age_Group'] = final_df['Age_Group'].map(age_group_mapping)

final_df = final_df.drop('Age', axis=1)
final_df['Gender'] = final_df['Gender'].astype('category')
final_df['Age_Group'] = final_df['Age_Group'].astype('category')
final_df['label'] = final_df['label'].astype('int')

# 3. Train/Valid/Test 분할
train_data, valid_data, train_labels, valid_labels = train_test_split(
    final_df[['image_path', 'Gender', 'Age_Group', 'Image_Number']], final_df['label'],
    test_size=0.2, random_state=42)

train_data, test_data, train_labels, test_labels = train_test_split(
    train_data, train_labels,
    test_size=0.2, random_state=42)

# 4. Dataset 생성
train_dataset = create_dataset(train_data, train_labels)
valid_dataset = create_dataset(valid_data, valid_labels)
test_dataset = create_dataset(test_data, test_labels)

train_loader = make_dataloader(train_dataset, augment=True)
valid_loader = make_dataloader(valid_dataset, shuffle=False)
test_loader = make_dataloader(test_dataset, shuffle=False)

# 5. 모델 학습
model = FinalModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = ExponentialLR(optimizer, gamma=0.1)

best_val_acc = 0.0
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for (images, (gender, age_group, image_number)), labels in train_loader:
        images = images.to(device)
        gender = gender.to(device)
        age_group = age_group.to(device)
        image_number = image_number.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model((images, (gender, age_group, image_number)))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for (images, (gender, age_group, image_number)), labels in valid_loader:
            images = images.to(device)
            gender = gender.to(device)
            age_group = age_group.to(device)
            image_number = image_number.to(device)
            labels = labels.to(device)

            outputs = model((images, (gender, age_group, image_number)))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100. * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '/content/drive/MyDrive/Dataset/before_augmentation_False/best_model.pth')
        print("Best model saved!")

    scheduler.step()

# 테스트 평가
model.load_state_dict(torch.load('/content/drive/MyDrive/Dataset/before_augmentation_False/best_model.pth'))
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for (images, (gender, age_group, image_number)), labels in test_loader:
        images = images.to(device)
        gender = gender.to(device)
        age_group = age_group.to(device)
        image_number = image_number.to(device)
        labels = labels.to(device)

        outputs = model((images, (gender, age_group, image_number)))
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"Test Accuracy: {100. * correct / total:.2f}%")