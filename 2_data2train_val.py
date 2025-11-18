# -*- coding: utf-8 -*-
"""
dataset_jをyolowで処理しやすいように

dataset_tvフォルダに分ける：

現在のデータセットは
dataset_j/
├── bike
├── cars
├── cats
├── dogs
├── flowers
├── horses
└── human

これを

dataset_tv/
├─ images/           # すべての画像ファイルをここに配置
   ├── train/        # 学習用画像
   └── val/          # 検証用画像

に分ける

"""
import os
import shutil
import random

# --- 設定 ---
SOURCE_ROOT = "dataset_j"   # 元のクラス別データセットのルート
TARGET_ROOT = "dataset_tv"  # 新しいYOLO形式のデータセットのルート
VAL_RATIO = 0.20            # 検証データに割り当てる割合 (20%に設定)

# 処理対象とするクラスフォルダ名
CLASSES = ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses', 'human']

# --- メイン処理 ---

def prepare_yolo_dataset():
    """
    クラス別フォルダの画像を読み込み、クラス名プレフィックスを付けて
    ランダムに images/train と images/val に分割・コピーします。
    """
    
    # ターゲットディレクトリの作成
    for split in ['train', 'val']:
        os.makedirs(os.path.join(TARGET_ROOT, 'images', split), exist_ok=True)
    
    print(f"ターゲットディレクトリ '{TARGET_ROOT}/images/' に train/val フォルダを作成しました。")
    print(f"分割比率: Train={1 - VAL_RATIO:.0%} / Val={VAL_RATIO:.0%}")
    print("-" * 50)
    
    total_images_processed = 0
    
    for class_name in CLASSES:
        source_dir = os.path.join(SOURCE_ROOT, class_name)
        
        if not os.path.exists(source_dir):
            print(f"⚠️ 警告: クラスフォルダ '{source_dir}' が見つかりません。スキップします。")
            continue

        # 1. クラスフォルダ内のすべての画像ファイルを取得
        all_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not all_files:
            print(f"  クラス '{class_name}': ファイルが見つかりません。スキップ。")
            continue

        # 2. ファイルリストをランダムにシャッフル
        random.shuffle(all_files)
        
        # 3. 分割点を計算
        num_val = int(len(all_files) * VAL_RATIO)
        
        # 4. ファイルを train と val に分割
        val_files = all_files[:num_val]
        train_files = all_files[num_val:]
        
        # 5. ファイルを新しい構造にコピー (クラス名プレフィックスを付与)
        
        def copy_and_rename_files(file_list, split_name):
            target_image_path = os.path.join(TARGET_ROOT, 'images', split_name)
            prefix = class_name.lower() + "_"
            
            for filename in file_list:
                src_file = os.path.join(source_dir, filename)
                
                # 新しいファイル名: クラス名 + 元のファイル名
                # 例: bike/123.jpg -> bike_123.jpg
                dst_filename = prefix + filename
                dst_file = os.path.join(target_image_path, dst_filename) 
                
                try:
                    shutil.copy2(src_file, dst_file)
                except Exception as e:
                    print(f"❌ コピー失敗: {filename} -> {split_name}。原因: {e}")


        copy_and_rename_files(train_files, 'train')
        copy_and_rename_files(val_files, 'val')
        
        print(f"  クラス '{class_name}' 処理完了: Train={len(train_files)}枚, Val={len(val_files)}枚")
        total_images_processed += len(all_files)

    print("-" * 50)
    print(f"🎉 データセットの分割とリネームが完了しました。総画像数: {total_images_processed}枚")
    print(f"新しい画像は '{TARGET_ROOT}/images/' フォルダに保存されました。")
    print("\n次のステップ: この新しい画像名に対応する **YOLO形式のラベル (.txt) ファイル**を作成する必要があります。")

if __name__ == "__main__":
    # 既存のターゲットディレクトリが存在する場合は削除または警告を出すことも可能ですが、
    # このスクリプトでは単に上書きします。
    if os.path.exists(TARGET_ROOT):
         print(f"⚠️ 警告: ターゲットフォルダ '{TARGET_ROOT}' が既に存在します。内容を上書きします。")
    
    prepare_yolo_dataset()