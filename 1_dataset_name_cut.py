# -*- coding: utf-8 -*-
"""
dataset_j/
├── bike
├── cars
├── cats
├── dogs
├── flowers
├── horses
└── human
フォルダの画像データの名前を数字のみにする


"""
import os
import re # 正規表現モジュールを使用

# --- 設定 ---
ROOT_DIR = "dataset_j" 

# 処理対象とするクラスフォルダ名
CLASSES = ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses', 'human']

# 画像として処理する拡張子
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

# --- メイン処理 ---

def rename_files_to_numbers_only():
    """
    クラスフォルダ内のファイルを走査し、ファイル名の先頭にある英字プレフィックスを
    取り除いて数字のみの名前にリネームします。
    """
    print(f"ターゲットディレクトリ: {ROOT_DIR}")
    print("-" * 40)
    
    total_renamed_count = 0
    
    for class_name in CLASSES:
        target_dir = os.path.join(ROOT_DIR, class_name)
        
        if not os.path.exists(target_dir):
            print(f"⚠️ 警告: クラスフォルダ '{target_dir}' が見つかりません。スキップします。")
            continue
            
        print(f"\n--- クラス '{class_name}' の処理を開始 ---")
        renamed_count = 0

        for filename in os.listdir(target_dir):
            src_path = os.path.join(target_dir, filename)
            
            # ディレクトリや隠しファイルはスキップ
            if os.path.isdir(src_path) or filename.startswith('.'):
                continue
            
            name, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            # 画像ファイルであるかチェック
            if ext not in IMAGE_EXTENSIONS:
                continue
                
            # 1. プレフィックスの検出と除去
            
            # 正規表現: ファイル名の先頭にある英字とアンダースコア（_）を無視し、
            # その後に続く数字の連続を抽出する
            match = re.search(r'([a-zA-Z_]+)?(\d+)', name) 
            
            new_name = None
            if match:
                # グループ2が数字部分
                new_name = match.group(2) 
            
            # 2. リネームの実行
            if new_name and new_name != name:
                new_filename = new_name + ext
                dst_path = os.path.join(target_dir, new_filename)

                try:
                    # ファイル名を変更
                    os.rename(src_path, dst_path)
                    renamed_count += 1
                    # print(f"  リネーム: {filename} -> {new_filename}") 
                except Exception as e:
                    print(f"❌ リネーム失敗: {filename}。原因: {e}")
            
        print(f"  結果: {renamed_count} 個のファイルをリネームしました。")
        total_renamed_count += renamed_count
        
    print("-" * 40)
    print(f"🎉 全ての処理が完了しました。総リネーム数: {total_renamed_count} 個。")

if __name__ == "__main__":
    rename_files_to_numbers_only()