# -*- coding: utf-8 -*-
"""
TensorFlowのライセンスは、Apache License 2.0です。
"""

import json
import matplotlib.pyplot as plt

# JSONファイルを読み込む
with open('training_history_7class.json', 'r') as f:
    history = json.load(f)

# 学習履歴のキーを確認
print("保存されているデータ:", list(history.keys()))

# -----------------------------
# グラフで表示
# -----------------------------
plt.figure(figsize=(10, 4))

# 損失 (Loss)
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
if 'val_loss' in history:
    plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 精度 (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
if 'val_accuracy' in history:
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('graph.png')
try:
    plt.show()
except:
    print('表示されない場合はGUI環境等で問題があるようです')