# -*- coding: utf-8 -*-
"""
Raspberry Pi 上で動かす7クラス分類CNNサンプル
- クラス: ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses', 'human'] (7クラス)
- データ構造: dataset_tv/images/train/ と /val/ を想定
- 画像サイズ: 128x128
"""


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

from PIL import Image, UnidentifiedImageError
import os
import pandas as pd
import json  

# --- 定数の定義 ---
NUM_CLASSES = 7  # 判別するクラスの数
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
# データが配置されているベースディレクトリ
BASE_DIR = 'dataset_tvr/images'


def remove_invalid_images(base_dir):
    """
    base_dir 以下のすべての画像ファイルをチェックし、
    壊れた画像は削除。
    削除数をディレクトリ単位で表示。
    """
    for root, dirs, files in os.walk(base_dir):
        deleted_count = 0
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                path = os.path.join(root, f)
                try:
                    img = Image.open(path)
                    img.verify()  # 破損チェック
                except (UnidentifiedImageError, IOError):
                    os.remove(path)
                    deleted_count += 1
        if deleted_count > 0:
            print(f"{root}: {deleted_count} 件削除しました")


# -----------------------------------------
# ディレクトリ指定（構造に合わせて変更）
# -----------------------------------------
# dataset_tv/images/train と dataset_tv/images/val を参照
train_dir = os.path.join(BASE_DIR, 'train')
val_dir   = os.path.join(BASE_DIR, 'val')

# train/val 両方チェック
remove_invalid_images(train_dir)
remove_invalid_images(val_dir)


# -----------------------------------------
# 1. データジェネレータの作成
# -----------------------------------------
# 画像をリスケール（0-1に正規化）して、学習時に少し拡張
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,     
    width_shift_range=0.1, 
    height_shift_range=0.1,
    horizontal_flip=True   
)

val_datagen = ImageDataGenerator(rescale=1./255)  # 検証データは正規化のみ


# データジェネレータを作成
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE, 
    batch_size=BATCH_SIZE,
    class_mode='categorical' # ★★★ 変更点1: 7クラス分類のため 'categorical' に変更 ★★★
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical' # ★★★ 変更点1: 7クラス分類のため 'categorical' に変更 ★★★
)

# -----------------------------------------
# 2. CNNモデルの定義
# -----------------------------------------
model = models.Sequential()

# 畳み込み層1
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(layers.MaxPooling2D((2,2)))

# 畳み込み層2
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# 畳み込み層3
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# Flatten
model.add(layers.Flatten())

# 全結合層
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # 過学習防止

# ★★★ 変更点2: 出力層を7ユニット、活性化関数を softmax に変更 ★★★
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))  # 出力層（7クラス分類）

# -----------------------------------------
# 3. モデルのコンパイル
# -----------------------------------------
model.compile(
    optimizer='adam',
    # ★★★ 変更点3: 損失関数を 'categorical_crossentropy' に変更 ★★★
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# モデル構造の確認（オプション）
model.summary()


# -----------------------------------------
# 4. 学習
# -----------------------------------------

epochs = 10 

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# -----------------------------------------
# 5. 学習履歴の保存
# -----------------------------------------
# 学習履歴（history.history）をDataFrameに変換
history_df = pd.DataFrame(history.history)
# CSVファイルとして保存
history_df.to_csv("training_history_7class.csv", index=False, encoding='utf-8-sig')
# history.history は dict形式
with open('training_history_7class.json', 'w') as f:
    json.dump(history.history, f)

print("学習履歴を training_history_7class.csv に保存しました。")

# -----------------------------------------
# 6. 学習済みモデルの保存
# -----------------------------------------
model.save('7class_cnn.h5')
print("モデル保存完了: 7class_cnn.h5")