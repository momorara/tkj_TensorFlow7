# -*- coding: utf-8 -*-
"""
ファイル名の先頭からクラス名を識別し、リサイズして
新しい7クラス構造に振り分けるスクリプト。
"""
import os
from PIL import Image

# --- 定数設定 ---
src_base = 'dataset_tv'                                      # 元のデータセットディレクトリ
dst_base = 'dataset_tvr'                                     # 出力先データセットディレクトリ
CLASSES = ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses', 'human'] # クラスリスト
IMG_SIZE = (128, 128)                                        # リサイズ後の画像サイズ (幅, 高さ)
# 元のデータセット内の画像がある共通パス (例: dataset_tv/images)
SRC_IMAGE_SUBDIR = 'images' 
# 出力先で作成する構造 (例: dataset_tvr/images)
DST_IMAGE_SUBDIR = 'images'
# ---

def resize_and_structure_by_filename(src_base_dir, dst_base_dir, classes, size):
    """
    元のデータセット構造から画像を読み込み、ファイル名でクラスを判別し、
    リサイズして新しいディレクトリ構造に保存する。
    """
    print(f"--- リサイズと振り分け処理を開始します ---")
    
    # 元の画像ディレクトリ
    src_images_dir = os.path.join(src_base_dir, SRC_IMAGE_SUBDIR)
    
    if not os.path.exists(src_images_dir):
        print(f"エラー: 元の画像ディレクトリが見つかりません: {src_images_dir}")
        return

    total_files = 0
    classified_files = 0

    # train と val のサブディレクトリを走査
    for subset in ['train', 'val']:
        src_subset_dir = os.path.join(src_images_dir, subset)
        
        if not os.path.exists(src_subset_dir):
            print(f"警告: 元の {subset} ディレクトリが見つかりません: {src_subset_dir}")
            continue

        print(f"\n--- {subset} フォルダを処理中 ---")
        
        # フォルダ内を再帰的に走査（元の構造がフラットでなくても対応可能）
        for root, dirs, files in os.walk(src_subset_dir):
            for filename in files:
                total_files += 1
                
                # 画像ファイルかチェック
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                # ファイル名からクラス名を特定
                detected_class = None
                for class_name in classes:
                    # ファイル名が 'class_name_' や 'class_name-' で始まっているかを確認
                    if filename.lower().startswith(f"{class_name.lower()}_") or \
                       filename.lower().startswith(f"{class_name.lower()}-"):
                        detected_class = class_name
                        break
                
                if detected_class:
                    classified_files += 1
                    
                    src_path = os.path.join(root, filename)
                    
                    # 出力先のパスを決定 (dst_base/images/train/クラス名/ファイル名)
                    dst_class_dir = os.path.join(dst_base_dir, DST_IMAGE_SUBDIR, subset, detected_class)
                    dst_path = os.path.join(dst_class_dir, filename) 
                    
                    # 出力先ディレクトリを作成
                    os.makedirs(dst_class_dir, exist_ok=True)

                    try:
                        # 画像を開く
                        img = Image.open(src_path)
                        
                        # リサイズを実行 (高画質リサンプリング)
                        resized_img = img.resize(size, Image.Resampling.LANCZOS)
                        
                        # 出力先に保存（ファイル名はそのまま維持）
                        resized_img.save(dst_path)
                        
                    except Exception as e:
                        print(f"スキップ: {src_path} の処理中にエラーが発生しました: {e}")
                
                # else:
                #     print(f"警告: {filename} は定義されたクラス名で始まりませんでした。")


    print("--- 処理完了 ---")
    print(f"合計画像ファイル数: {total_files}")
    print(f"クラスに分類され処理されたファイル数: {classified_files}")


if __name__ == '__main__':
    # スクリプトを実行
    resize_and_structure_by_filename(src_base, dst_base, CLASSES, IMG_SIZE)