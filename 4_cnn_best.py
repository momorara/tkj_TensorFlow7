# -*- coding: utf-8 -*-
"""
7クラス分類CNNサンプル
- EarlyStoppingを削除し、指定エポック数まで学習を継続。
- ModelCheckpointで「ベストモデル」と「指定間隔ごと」のモデルを保存。
- 最終エポックのモデルを保存。
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint # モデル保存機能（コールバック）をインポート
import os
import pandas as pd
import json  
from PIL import Image, UnidentifiedImageError
import numpy as np 

# --- 定数の定義 ---
NUM_CLASSES = 7                             # 判別するクラスの数 (バイク、車、猫など7種類)
IMAGE_SIZE = (128, 128)                     # モデルに入力する画像のサイズ
BATCH_SIZE = 8                              # 一度に学習させる画像の枚数
BASE_DIR = 'dataset_tvr/images'             # 学習データ（train/valフォルダ）の親ディレクトリ
BEST_MODEL_FILE = 'best_7class_cnn.keras'      # 検証損失が最低のベストモデルの保存ファイル名
INTERVAL_MODEL_DIR = 'checkpoints'          # 指定エポックごと保存するモデルのディレクトリ名
INTERVAL_FREQ = 10                          # モデルを保存するエポック間隔 (10エポックごと)
EPOCHS = 800                                # 最大の学習回数
# ---

# -----------------------------------------
# 画像ファイル破損チェック関数
# -----------------------------------------
def remove_invalid_images(base_dir):
    """
    指定ディレクトリ以下のすべての画像ファイルをチェックし、
    破損しているものは削除する。（学習中のエラーを防ぐ）
    """
    for root, dirs, files in os.walk(base_dir):
        deleted_count = 0
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                path = os.path.join(root, f)
                try:
                    img = Image.open(path)
                    img.verify()  # 画像の破損チェック
                except (UnidentifiedImageError, IOError):
                    os.remove(path) # 破損していたら削除
                    deleted_count += 1
        if deleted_count > 0:
            print(f"{root}: {deleted_count} 件削除しました")


# -----------------------------------------
# ディレクトリ指定と破損チェックの実行
# -----------------------------------------
train_dir = os.path.join(BASE_DIR, 'train') # 学習用画像フォルダのパスを設定
val_dir   = os.path.join(BASE_DIR, 'val')   # 検証用画像フォルダのパスを設定

remove_invalid_images(train_dir) # 学習用フォルダの画像をチェック
remove_invalid_images(val_dir)   # 検証用フォルダの画像をチェック


# -----------------------------------------
# 1. データジェネレータの作成
# -----------------------------------------
# 学習用データジェネレータ（データ拡張を行う）
train_datagen = ImageDataGenerator(
    rescale=1./255,                 # 画像のピクセル値を0〜1に正規化（必須）
    rotation_range=20,              # 20度までのランダム回転
    width_shift_range=0.1,          # 左右へのランダムなシフト
    horizontal_flip=True            # 左右反転
)
# 検証用データジェネレータ（正規化のみ、拡張はしない）
val_datagen = ImageDataGenerator(rescale=1./255)

# データをフォルダ構造から読み込み、ジェネレータを作成
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=IMAGE_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='categorical' # 7クラス分類のため 'categorical' を指定
)
val_generator = val_datagen.flow_from_directory(
    val_dir, 
    target_size=IMAGE_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='categorical'
)

# -----------------------------------------
# 2. CNNモデルの定義とコンパイル
# -----------------------------------------
# モデルの構造を定義
model = models.Sequential([
    # 第1畳み込み層とプーリング層
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)), 
    # 第2畳み込み層とプーリング層
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    # 第3畳み込み層とプーリング層
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    # 2次元データを1次元に変換
    layers.Flatten(),
    # 全結合層とDropout
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), 
    # 出力層: 7クラスの確率を出力（多クラス分類のためsoftmax）
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# モデルの学習方法を設定（コンパイル）
model.compile(
    optimizer='adam',                   # 最適化手法: Adam
    loss='categorical_crossentropy',    # 損失関数: 多クラス分類用
    metrics=['accuracy']                # 評価指標: 正解率
)

# -----------------------------------------
# 3. モデル保存のためのコールバックの定義
# -----------------------------------------
# 1. ベストモデルの保存 (検証損失 val_loss が最低のモデルを保存)
best_model_checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_FILE,
    monitor='val_loss',         # 監視する値は検証損失
    mode='min',                 # 損失が最小になった時を「ベスト」とする
    save_best_only=True,        # ベストなものだけを保存
    verbose=1                   # 進捗メッセージを表示
)

# 2. 指定間隔ごとのモデル保存 (10エポックごと)
os.makedirs(INTERVAL_MODEL_DIR, exist_ok=True) # 保存ディレクトリを確実に作成

# 10エポックごとに保存するためのステップ数（バッチ数）を計算
steps_per_epoch = train_generator.samples // train_generator.batch_size
steps_to_save = steps_per_epoch * INTERVAL_FREQ 

interval_checkpoint = ModelCheckpoint(
    # ファイル名にエポック番号を埋め込む ('model_epoch_010.keras' のように保存)
    filepath=os.path.join(INTERVAL_MODEL_DIR, 'model_epoch_{epoch:03d}.keras'),
    monitor='val_loss',         
    mode='min',         
    save_freq=steps_to_save,    # 10エポック分のバッチ数（ステップ）ごとに保存を実行
    save_best_only=False,       # ベストかどうかにかかわらず、指定間隔で保存
    verbose=0 
)

callbacks_list = [best_model_checkpoint, interval_checkpoint] # 使用するコールバックのリスト

# -----------------------------------------
# 4. 学習の実行
# -----------------------------------------
print(f"学習を開始します。最大エポック数: {EPOCHS} (10エポックごとにチェックポイントを保存)")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_list # 定義したモデル保存のルールを適用
)


# -----------------------------------------
# 5. 学習履歴の保存
# -----------------------------------------
# 学習中の損失や精度を記録したhistoryオブジェクトをデータフレームに変換
history_df = pd.DataFrame(history.history)
# CSVファイルとして保存
history_df.to_csv("training_history_7class.csv", index=False, encoding='utf-8-sig')
# JSON形式でも保存
with open('training_history_7class.json', 'w') as f:
    json.dump(history.history, f)

print("学習履歴を training_history_7class.csv に保存しました。")


# -----------------------------------------
# 6. 最終モデルの保存
# -----------------------------------------
# 学習が最後まで完了した時点のモデルを保存（ラストモデル）
model.save('final_7class_cnn.keras')
print("最終モデル保存完了: final_7class_cnn.keras")
print(f"ベストモデルは {BEST_MODEL_FILE} に保存されました。")
print(f"10エポックごとのモデルは {INTERVAL_MODEL_DIR} フォルダ内に保存されました。")