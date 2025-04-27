# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from utils import seed_everything, fold_path, create_dataset, make_tf_dataset
from model import FinalModel

# 1. 설정
seed_everything(42)
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

train_ds = make_tf_dataset(train_dataset, augment=True)
valid_ds = make_tf_dataset(valid_dataset)
test_ds = make_tf_dataset(test_dataset)

# 5. 모델 학습
model = FinalModel()

# 학습 설정
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-4,
    decay_steps=len(train_ds)*5,
    decay_rate=0.1
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# 콜백
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('/content/drive/MyDrive/Dataset/before_augmentation_False/best_model.h5', save_best_only=True)
]

# 학습 시작
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    callbacks=callbacks
)

# 테스트 평가
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")
