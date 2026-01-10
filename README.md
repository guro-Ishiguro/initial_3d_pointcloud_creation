# 3D Point Cloud Generation from Stereo Images

ステレオ画像から高精度な3D点群を生成するためのツールである。PatchMatch MVS（Multi-View Stereo）アルゴリズムを使用して深度マップを最適化し、複数ビューからの情報を統合して高品質な3D点群を生成する。

## 特徴

- **高精度な深度推定**: PatchMatch MVSアルゴリズムによる深度マップの最適化
- **GPU高速化**: CUDAを使用した並列処理による高速計算
- **複数ビュー統合**: 複数のカメラ視点からの情報を統合して高品質な点群を生成
- **フィルタリング機能**: 光度一貫性と幾何学的一貫性による品質向上
- **GUI/CLI対応**: PyQt5によるGUIとコマンドラインインターフェースの両方を提供
- **柔軟な設定**: YAMLファイルによる詳細なパラメータ設定

## 動作環境

### 必須要件

- Python 3.10以上
- NVIDIA GPU（CUDA対応）
- CUDA 11.x以上

### 推奨環境

- メモリ: 16GB以上
- GPU: NVIDIA GPU with 8GB以上のVRAM

## インストール

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd initial_3d_pointcloud_creation
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

主要な依存パッケージ:
- PyQt5 (GUI)
- NumPy, OpenCV
- Numba (CUDA対応)
- Open3D (点群処理)
- OpenEXR (深度マップ読み書き)

### 3. データセットの準備

データセットは以下の構造で配置する:

```
data/
  └── <dataset_name>/
      ├── images/
      │   ├── image_0/          # 左カメラ画像
      │   ├── image_1/          # 右カメラ画像
      │   └── depth/            # 正解深度マップ
      └── txt/
          ├── camera_params.csv      # カメラ内部パラメータ
          ├── left_camera_poses.csv  # 左カメラのポーズ情報
          └── right_camera_poses.csv # 右カメラのポーズ情報
```

データセットは[Unityで開発された建設現場のシミュレータ](https://github.com/guro-Ishiguro/ConstructionSiteSimulator)を動かすと、このような構造のデータを取得することができる。

## 使用方法

### GUIを使用する場合

```bash
python app/gui.py
```

GUIでは以下が可能である:
- データセットの選択
- パラメータの設定（タブ形式で整理）
- リアルタイムログ表示
- 処理の実行と進捗確認

### コマンドラインを使用する場合

```bash
# データセットを対話的に選択
python app/cli.py

# データセットを指定
python app/cli.py --dataset <dataset_name>

# 設定ファイルを指定
python app/cli.py --config app/mvs.yaml --dataset <dataset_name>
```

### 直接実行する場合

```bash
# 環境変数でデータセットを指定
export DATA_TYPE=<dataset_name>
python -m mvs.main
```

## 設定ファイル

`app/mvs.yaml`でパラメータを設定できる。主要なパラメータ:

### PatchMatch基本パラメータ

- `PATCHMATCH_ITERATIONS`: PatchMatchの反復回数（デフォルト: 10）
- `PATCHMATCH_PATCH_SIZE`: パッチサイズ（デフォルト: 7）
- `PATCHMATCH_DECAY_RATE`: ランダムサーチの減衰率（デフォルト: 0.9）

### フィルタリングパラメータ

- `FILTERING_COLOR_DIFFERENCE_THRESHOLD`: 光度一貫性フィルタリングの色差閾値
- `GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD`: 幾何学的一貫性のエラー閾値
- `GEOMETRIC_MIN_CONSISTENT_VIEWS`: 最小一貫ビュー数

### フレーム選択パラメータ

- `FRAME_STRIDE`: 処理するフレームの間隔（デフォルト: 15）

詳細は`app/mvs.yaml`を参照する。

## ツール

### データセット統合ツール

複数のデータセットを1つのデータセットに統合する。

```bash
# 対話的にデータセットを選択して統合
python tools/merge_datasets.py

# コマンドライン引数でデータセットを指定
python tools/merge_datasets.py dataset1 dataset2 dataset3
```

統合後のデータセットは`data/<merged_name>/`に作成され、通常のデータセットと同様に処理できる。

### バッチ処理ツール

複数回のMVS処理を連続実行する。各実行で設定を変更しながら処理を実行する。

```bash
python tools/run_batch_mvs.py --dataset <dataset_name>
```

### 評価結果の集約

複数の評価結果CSVファイルを集約して平均値を計算する。

```bash
# CSVディレクトリを指定して集約
python tools/aggregate.py --csv_dir output/<dataset_name>/csv

# 出力ファイル名を指定
python tools/aggregate.py --csv_dir output/<dataset_name>/csv --output result.csv
```

## 評価

### 深度評価

推定深度と真値深度を比較し、評価指標を計算してCSVファイルに保存する。

```bash
# データセットを指定して評価
python evaluation/depth_evaluate.py --dataset <dataset_name>

# 出力ディレクトリを指定
python evaluation/depth_evaluate.py --output_dir output/<dataset_name>
```

評価指標:
- `abs_rel`: 絶対相対誤差
- `sq_rel`: 二乗相対誤差
- `rmse`: 平均二乗平方根誤差
- `rmse_log`: 対数空間でのRMSE
- `mae`: 平均絶対誤差
- `delta1`, `delta2`, `delta3`: 閾値内の精度

### 点群評価

生成された点群の品質を評価する。

```bash
python evaluation/pointcloud_evaluation.py --predicted <predicted.ply> --ground_truth <ground_truth.ply>
```

## 出力

処理が完了すると、`output/<dataset_name>/`以下に以下のファイルが生成される:

- `point_cloud/output.ply`: 統合された3D点群（PLY形式）
- `depth/<frame_name>/`: 各フレームの深度マップ（PNG/EXR形式）
- `normal/<frame_name>/`: 各フレームの法線マップ（PNG形式）
- `csv/<frame_name>/time.csv`: 各処理ステージの処理時間
- `disparity/`: 視差マップ（PNG形式）

## 処理フロー

1. **初期深度推定**: ステレオ画像ペアから視差を計算し、深度マップを生成
2. **PatchMatch最適化**: GPUを使用したPatchMatch MVSアルゴリズムで深度マップを最適化
3. **フィルタリング**: 
   - 光度一貫性フィルタリング
   - 幾何学的一貫性フィルタリング
4. **点群生成**: 深度マップから3D点群を生成
5. **点群統合**: 複数ビューからの点群を統合

## プロジェクト構造

```
initial_3d_pointcloud_creation/
├── app/
│   ├── gui.py              # GUIアプリケーション
│   ├── cli.py              # コマンドラインインターフェース
│   ├── mvs.yaml            # デフォルト設定ファイル
│   ├── data_loader.py      # データ読み込み
│   └── settings.py         # 設定管理
├── mvs/
│   ├── main.py             # メイン処理
│   ├── depth_optimization.py  # PatchMatch MVS実装
│   ├── depth_estimation.py    # 深度推定
│   ├── disparity_estimation.py # 視差推定
│   ├── point_cloud_integrator.py # 点群統合
│   └── config.py           # 設定管理
├── evaluation/
│   ├── depth_evaluate.py   # 深度評価
│   ├── pointcloud_evaluation.py # 点群評価
│   └── aggregate.py        # 評価結果の集約
├── tools/
│   ├── merge_datasets.py   # データセット統合ツール
│   └── run_batch_mvs.py    # バッチ処理ツール
├── data/                   # データセットディレクトリ
├── output/                 # 出力ディレクトリ
└── requirements.txt        # 依存パッケージ
```

## トラブルシューティング

### GPUが認識されない場合

- CUDAが正しくインストールされているか確認する
- `nvidia-smi`でGPUが認識されているか確認する
- NumbaのCUDAサポートが有効か確認する

### メモリ不足エラー

- `FRAME_STRIDE`を大きくして処理フレーム数を減らす
- `MAX_NEIGHBORS`を減らす
- 画像解像度を下げる

### CSVファイルが保存されない場合

- 出力ディレクトリの書き込み権限を確認する
- ログでエラーメッセージを確認する

