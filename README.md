# CNN_-PyTorch-
# 画像分類 PyTorch プロジェクト

## 概要
このプロジェクトは、グレースケール画像を使用して6クラスの分類を行うPyTorchベースの深層学習モデルを実装しています。K分割交差検証を使用して、モデルの性能を評価します。

## プロジェクト構造
```
.
├── dataset/
│   ├── 1/
│   │   └── */
│   │       └── Pytorch_Architecture.png
│   ├── 2/
│   └── .../
├── model.py
└── README.md
```

## 主な機能

### データセット
- 入力画像サイズ: 128x64 ピクセル（グレースケール）
- 6クラスの分類
- データ拡張: 回転、反転、アフィン変換などを適用

### モデルアーキテクチャ
- 3つの畳み込みブロック
  - 各ブロック: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> MaxPool
- 全結合層
  - 256ユニット -> 128ユニット -> 6クラス出力
- ドロップアウト（0.5）を使用して過学習を防止

### トレーニング設定
- 最適化アルゴリズム: Adam（学習率 0.001）
- 損失関数: CrossEntropyLoss
- バッチサイズ: 32
- エポック数: 200
- 学習率スケジューラ: ReduceLROnPlateau
- K分割交差検証（K=5）

## 性能モニタリング
- TensorBoardによる訓練過程の可視化
  - 訓練損失
  - 検証損失
  - 検証精度
- クラスごとの予測精度をヒートマップで表示

## 使用方法

### 環境設定
```bash
pip install torch torchvision numpy pillow sklearn matplotlib seaborn tensorboard
```

### データセットの準備
1. `dataset`フォルダを作成
2. 各クラスのフォルダ（1-6）を作成
3. 画像ファイル（Pytorch_Architecture.png）を配置

### トレーニングの実行
```bash
python model.py
```

### 結果の確認
- モデルの重みは`YYYY-MM-DD_HH-MM-SS`フォルダに保存
- TensorBoardログは`logs`フォルダに保存
- 精度ヒートマップは`Average_Prediction_Accuracy_per_Class.png`として保存

## 注意事項
- GPUが利用可能な場合は自動的に使用します
- 十分なメモリを確保してください
- データセットのパスは必要に応じて調整してください

## 機能の拡張
- データ拡張のパラメータ調整
- モデルアーキテクチャの変更
- ハイパーパラメータのチューニング
- 新しい評価指標の追加
