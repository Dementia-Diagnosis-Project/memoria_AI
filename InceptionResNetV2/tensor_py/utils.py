# utils.py
import random
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import collections
import re
import glob

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def fold_path(data_path):
    fold = collections.defaultdict(list)
    for folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder)):
            fold[folder] = os.listdir(os.path.join(data_path, folder))
    return fold

def create_dataset(data, labels):
    data_list = []
    for idx, row in data.iterrows():
        for num in row['Image_Number']:
            data_list.append({
                'image_path': os.path.join(row['image_path'], f"sharp2_plane{num}.png"),
                'Gender': row['Gender'],
                'Age_Group': row['Age_Group'],
                'image_number': num,
                'label': labels[idx]
            })
    return pd.DataFrame(data_list)

def preprocess(image_path, gender, age_group, image_number, label, augment=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if augment:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.resize(image, [299, 299])

    gender = tf.cast(gender, tf.int64)
    age_group = tf.cast(age_group, tf.int64)
    image_number = tf.cast(image_number, tf.int64)
    label = tf.cast(label, tf.int64)
    return image, (gender, age_group, image_number), label

def make_tf_dataset(df, augment=False, batch_size=30):
    ds = tf.data.Dataset.from_tensor_slices((
        df['image_path'].astype(str).values,
        df['Gender'].values,
        df['Age_Group'].values,
        df['image_number'].values,
        df['label'].values
    ))
    ds = ds.map(lambda x, y, z, w, l: preprocess(x, y, z, w, l, augment), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds
