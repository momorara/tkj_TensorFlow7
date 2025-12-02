# -*- coding: utf-8 -*-
"""
7カテゴリー分類モデルの検証データに対する性能評価スクリプト。
全画像に対して推論を行い、カテゴリーごとの正解数、不正解数、正解率を計算する。
"""
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import pandas as pd # 結果を整形するためにPandasを使用

# --- 定数 ---
# 判別するクラスのリスト
CATEGORIES = ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses', 'human']
# モデルの入力サイズ（学習時と同じにする）
TARGET_SIZE = (128, 128)
# 学習済みモデルのパス
MODEL_PATH = '7class_cnn.keras'
# 推論対象フォルダ（valデータセットのルート）
BASE_DIR = 'dataset_tvr/images/val'
# ---

# --- 1. モデルの読み込みとデータ構造の確認 ---
try:
    model = load_model(MODEL_PATH)
except:
    print(f"エラー: モデルファイルが見つかりません: {MODEL_PATH}")
    exit()

# 結果を保存するための辞書を初期化
# キーはカテゴリ名、値は [正解数, 不正解数]
results = {cat: [0, 0] for cat in CATEGORIES}
all_data = [] # 推論対象の全画像パスと正解ラベルを格納

print(f"--- 評価を開始します ---")
print(f"評価対象ディレクトリ: {BASE_DIR}")
print(f"評価クラス: {CATEGORIES}")

# 2. 全画像パスと正解ラベルのリストを作成
for true_index, true_category in enumerate(CATEGORIES):
    folder_path = os.path.join(BASE_DIR, true_category)
    
    if os.path.isdir(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in files:
            img_path = os.path.join(folder_path, filename)
            # (画像パス, 正解クラス名, 正解クラスID) をリストに追加
            all_data.append((img_path, true_category, true_index))

if not all_data:
    print(f"エラー: {BASE_DIR} 以下で画像ファイルが見つかりません。パスを確認してください。")
    exit()

print(f"合計 {len(all_data)} 枚の画像を処理します...")

# 3. 推論の実行と結果の集計
for img_path, true_category, true_index in all_data:
    try:
        # 画像の前処理
        img = image.load_img(img_path, target_size=TARGET_SIZE)
        x = image.img_to_array(img)
        x = x / 255.0  # 正規化
        x = np.expand_dims(x, axis=0) # バッチ次元を追加

        # 推論の実行
        predictions = model.predict(x, verbose=0)[0]
        
        # 最も高い確率のインデックスを取得
        predicted_index = np.argmax(predictions)
        predicted_class = CATEGORIES[predicted_index]
        
        # 結果の集計
        if predicted_index == true_index:
            results[true_category][0] += 1 # 正解
        else:
            results[true_category][1] += 1 # 不正解
            # どのクラスに間違えたかを表示したい場合はここで記録 (例: print(f"Miss: {true_category} -> {predicted_class}"))

    except Exception as e:
        print(f"警告: {img_path} の処理中にエラーが発生しました: {e}")
        
# 4. 結果の整形と表示
data = []
total_correct = 0
total_incorrect = 0

print("\n--- カテゴリー別 評価結果 ---")

for category, counts in results.items():
    correct = counts[0]
    incorrect = counts[1]
    total = correct + incorrect
    
    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0.0

    total_correct += correct
    total_incorrect += incorrect
    
    data.append([
        category, 
        correct, 
        incorrect, 
        total, 
        f"{accuracy:.4f}" # 正解率を小数点4桁で表示
    ])

# Pandas DataFrameを作成して整形
df = pd.DataFrame(data, columns=['Category', 'Correct', 'Incorrect', 'Total', 'Accuracy'])

# 全体の合計行を追加
overall_total = total_correct + total_incorrect
overall_accuracy = total_correct / overall_total if overall_total > 0 else 0.0
overall_row = [
    "TOTAL (Overall)", 
    total_correct, 
    total_incorrect, 
    overall_total, 
    f"{overall_accuracy:.4f}"
]
df.loc[len(df)] = overall_row

print(df.to_string(index=False))
print("\n評価が完了しました。")