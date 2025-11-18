# -*- coding: utf-8 -*-
"""
解凍した元データdataのファル形式をjpgに統一する

現在のデータセットは
data/
├── bike
├── cars
├── cats
├── dogs
├── flowers
├── horses
└── human
です。

これを
dataset_j/
├── bike
├── cars
├── cats
├── dogs
├── flowers
├── horses
└── human
とする
"""
import os
from PIL import Image
import shutil

# --- 設定 ---
SOURCE_ROOT = "data"     # 変換したい画像ファイルがあるルートフォルダ
TARGET_ROOT = "dataset_j" # 変換後のJPEGファイルを保存するルートフォルダ

# 処理対象とするサブディレクトリ名（クラス名）のリスト
# フォルダ名が正確にこれと一致していることを確認してください
CLASSES = ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses', 'human']

# JPEG変換時に画質を調整（1〜100、高いほど高画質/ファイルサイズ大）
JPEG_QUALITY = 90

# --- メイン処理 ---

def convert_categorized_images_to_jpg():
    """
    クラスディレクトリ構造を維持したまま、画像をJPEG形式に変換します。
    """
    print(f"ソースディレクトリ: {SOURCE_ROOT}")
    print(f"ターゲットディレクトリ: {TARGET_ROOT}")
    print("-" * 40)
    
    total_converted_count = 0
    total_skipped_count = 0
    
    if not os.path.exists(SOURCE_ROOT):
        print(f"❌ エラー: ソースディレクトリ '{SOURCE_ROOT}' が見つかりません。プログラムを終了します。")
        return

    for class_name in CLASSES:
        source_dir = os.path.join(SOURCE_ROOT, class_name)
        target_dir = os.path.join(TARGET_ROOT, class_name)
        
        # ターゲットディレクトリの作成
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\n--- クラス '{class_name}' の処理を開始 ---")
        
        converted_count = 0
        skipped_count = 0

        if not os.path.exists(source_dir):
            print(f"⚠️ 警告: クラスフォルダ '{source_dir}' が見つかりません。スキップします。")
            continue

        for filename in os.listdir(source_dir):
            input_path = os.path.join(source_dir, filename)
            
            # ディレクトリはスキップ
            if os.path.isdir(input_path):
                continue

            name, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            # 処理対象の画像形式をチェック
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # 出力ファイル名：拡張子を強制的に .jpg に設定
                output_path = os.path.join(target_dir, name + ".jpg") 
                
                try:
                    # 1. 画像のロード
                    img = Image.open(input_path)
                    
                    # 2. RGB形式に変換 (JPEGはアルファチャンネル非対応のため)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 3. JPEG形式で保存
                    img.save(output_path, 'JPEG', quality=JPEG_QUALITY)
                    
                    # print(f"  ✅ 変換成功: {filename} -> {name}.jpg") # ファイル数が多い場合はコメントアウト推奨
                    converted_count += 1
                    
                except Exception as e:
                    print(f"  ❌ 変換失敗: {filename}。原因: {e}")
                    skipped_count += 1
            
            else:
                # サポートされていない形式のファイルはスキップ
                skipped_count += 1

        print(f"  結果: {converted_count} 個のファイルをJPEGに変換しました。スキップ: {skipped_count} 個。")
        total_converted_count += converted_count
        total_skipped_count += skipped_count

    print("-" * 40)
    print(f"🎉 全てのクラスの処理が完了しました。")
    print(f"総変換ファイル数: {total_converted_count} 個。")
    print(f"新しいデータセットは '{TARGET_ROOT}' に保存されました。")

if __name__ == "__main__":
    convert_categorized_images_to_jpg()