# -*- coding: utf-8 -*-
"""
TensorFlowのライセンスは、Apache License 2.0です。
"""
# -----------------------------------------
# ランダム画像を選んで推論し、
# スペースで次の画像、qで終了
# -----------------------------------------
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# --- 定数 ---
# 判別するクラスのリスト
CATEGORIES = ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses', 'human']
# モデルの入力サイズ（学習時と同じにする）
TARGET_SIZE = (128, 128)
# 学習済みモデルの読み込み
model = load_model('7class_cnn.keras')

# 推論対象フォルダ（valデータセットのルート）
base_dir = 'dataset_tvr/images/val'

# --- 画像ファイルのリストを作成 ---
img_list = []

for category in CATEGORIES:
    folder_path = os.path.join(base_dir, category)
    # フォルダが存在することを確認
    if os.path.isdir(folder_path):
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        img_list.extend(files)

# 隠しメタデータファイルなどを削除
img_list = [f for f in img_list if "/._" not in f and "/.DS_Store" not in f]

if not img_list:
    print(f"エラー: 画像ファイルが見つかりません。パス: {base_dir} を確認してください。")
    exit()

print(f"全{len(img_list)}枚の画像ファイルが見つかりました。")

# -----------------------------------------
# matplotlib ウィンドウを作成
# -----------------------------------------
fig, ax = plt.subplots(figsize=(6, 6)) # ウィンドウサイズを調整
plt.axis('off')  # 軸を非表示

# グローバル変数で表示用画像
img_show = None

def show_random_image(event=None):
    """ランダムに画像を選んで推論＆表示"""
    global img_show
    img_path = random.choice(img_list)
    
    # 1. 画像の前処理
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    x = image.img_to_array(img)
    x = x / 255.0  # 正規化（学習時と同じ処理）
    x = np.expand_dims(x, axis=0) # バッチ次元を追加 (1, 128, 128, 3)

    # 2. 推論の実行
    # モデルの出力は7つのクラスそれぞれの確率 ([p1, p2, p3, p4, p5, p6, p7])
    predictions = model.predict(x, verbose=0)[0]
    
    # 最も高い確率のインデックスを取得
    predicted_index = np.argmax(predictions)
    
    # 予測されたクラス名と確率を取得
    predicted_class = CATEGORIES[predicted_index]
    confidence = predictions[predicted_index]
    
    # 3. 検証情報（正解ラベル）の取得
    # ファイルパスからフォルダ名（＝正解ラベル）を抽出
    true_category = os.path.basename(os.path.dirname(img_path))
    
    # 4. 表示結果の文字列を作成
    is_correct = "OK" if predicted_class == true_category else "NG"
    result_title = (
        f"Predict: {predicted_class} ({confidence:.3f})\n"
        f"Truth: {true_category} [{is_correct}]"
    )

    print("---------------------------------------")
    print("選ばれた画像:", img_path)
    print("予測結果:", result_title.replace('\n', ' | '))
    # 全ての確率を表示（デバッグ用）
    # print("全確率:", dict(zip(CATEGORIES, predictions.round(3))))
    print("---------------------------------------")

    # 5. 画像の表示
    img_show = image.load_img(img_path)
    ax.clear()
    ax.imshow(img_show)
    ax.axis('off')
    
    # タイトルを表示
    # 不正解の場合は赤、正解の場合は緑で色付け
    title_color = 'green' if is_correct == 'OK' else 'red'
    ax.set_title(result_title, color=title_color, fontsize=12, fontweight='bold')
    
    fig.canvas.draw()

def on_key(event):
    """キー入力イベント"""
    if event.key == ' ':
        try:
            show_random_image()
        except Exception as e:
            print(f"次の画像への切り替え中にエラーが発生しました: {e}")
            pass
    elif event.key.lower() == 'q': # 'q'キーで終了
        plt.close(fig)

# キーイベントを接続
fig.canvas.mpl_connect('key_press_event', on_key)

# 最初の画像を表示
show_random_image()

# ウィンドウを閉じるまで表示を維持
plt.show()