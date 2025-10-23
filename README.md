# 3D点群生成パイプライン

本プロジェクトは、ステレオカメラから取得した連続画像（動画）とカメラの姿勢データを用いて、高密度な3D点群を生成するためのパイプラインである。

主な処理として、ステレオマッチングによる初期深度推定、複数視点の情報を用いたPatchMatch MVSによる深度マップの最適化、そして複数フレームから得られた点群の統合を行う。処理の一部はCUDAを利用してGPUで高速化することが可能である。

## 主な機能

* **初期深度推定**: ステレオ画像ペアからSGBMアルゴリズムを用いて視差を計算し、初期深度マップを生成する。
* **深度マップ最適化 (PatchMatch MVS)**: 参照画像と複数の近傍画像の幾何学的な整合性を利用して、深度マップを高精度化・高密度化する。この処理はCPU版とGPU(CUDA)版が利用可能である。
* **信頼性フィルタリング**: 光度一貫性チェックと幾何学的一貫性チェックにより、信頼性の低い深度値を除去する。
* **点群統合**: 各フレームで生成された点群を一つのワールド座標系に統合し、ボクセルグリッド内で中央値を計算することでノイズを低減する。
* **可視化と評価**: 処理中に点群をリアルタイムで表示したり、真値の深度データと比較して評価指標を算出したりする機能が含まれる。

## 動作環境

* Python 3.8以上
* 必要なライブラリは `requirements.txt` を参照。
* **（オプション）GPU高速化を利用する場合:**
    * NVIDIA製GPU
    * CUDA Toolkit
    * `cupy-cuda11x` (または環境に合わせたバージョン)

## セットアップ手順

1.  **リポジトリのクローン**
    ```bash
    git clone https://github.com/guro-Ishiguro/initial_3d_pointcloud_creation
    cd initial_3d_pointcloud_creation
    ```

2.  **Python仮想環境の構築（推奨）**
    ```bash
    python3 -m venv .venv
    ```

    ```bash
    source .venv/bin/activate（Mac用）
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force（Win用）
    .venv\Scripts\Activate.ps1（Win用）
    ```

3.  **依存ライブラリのインストール**
    ```bash
    pip install -r requirements.txt
    ```
## データ準備

[ConstructionSiteSimulator](https://github.com/guro-Ishiguro/ConstructionSiteSimulator)でクレーン周辺の動画像及び姿勢情報を取得する。

1.  **データセットフォルダの作成**

    ルートディレクトリに `data/<dataset_name>` フォルダを作成する。<dataset_name>の命名は、`config.py`がパラメータを読み込むために、以下の形式に従う必要がある。

    **フォーマット:**
    `<width>_<height>_<camera_height>_<fov_v>_<fov_h>_<B>`

    **各パラメータの説明:**
    * `width`: 画像の幅 (ピクセル)
    * `height`: 画像の高さ (ピクセル)
    * `camera_height`: カメラの高さ (メートル)
    * `fov_v`: カメラの垂直視野角 (度)
    * `fov_h`: カメラの水平視野角 (度)
    * `B`: ステレオカメラのベースライン長 (メートル)

    **命名例:**
    `3840_2160_16_74.73365_92_0.3`

2.  **RGB画像ファイルの配置**
    ステレオカメラで撮影した左カメラと右カメラのRGB画像ファイルを、作成した `data/<dataset_name>/images/stereo/` ディレクトリに配置する。

2.  **真値深度画像ファイルの配置**
    ステレオカメラで撮影した左カメラの真値深度画像を、作成した `data/<dataset_name>/images/depth/` ディレクトリに配置する。

3.  **カメラ姿勢ログの配置**
    各フレームに対応するカメラの位置と姿勢が記録されたログファイル（例: `drone_image_log.txt`）を `data/<dataset_name>/txt/` ディレクトリに配置する。

## 実行方法

1.  **設定ファイルの編集**
    `mvs/config.py` を開き、処理対象のデータセットを選択する。スクリプトを実行すると、`data` ディレクトリ内のデータセットが一覧表示されるので、対応する番号を入力する。
    また、PatchMatchの反復回数や各種パラメータもこのファイルで調整できる。

2.  **エントリポイントの実行**
    以下のコマンドで3D点群生成パイプラインを実行する。
    ```bash
    python app/cli.py
    ```
    既存のエントリ（互換）も引き続き利用可能です。
    ```bash
    python mvs/main.py
    ```

    処理が完了すると、最終的な点群データが `output/<dataset_name>/point_cloud/output.ply` として保存される。また、各フレームの深度マップや法線マップなどの中間生成物も `output` ディレクトリ以下に保存される。

### GPU/最適化オプション

- 伝播方向の切替（4/8方向）
  - `mvs/config.py` の `PROPAGATION_NEIGHBOR_DIRECTIONS` に 4 または 8 を設定（既定: 4）。
  - 8方向は精度安定、4方向は高速化に有利。

- priority 伝播の固定順・スイープ回数
  - GPUのpriority伝播は、bin内画素を「行→列」固定順に処理。
  - bin内での内部スイープ回数は `PRIORITY_MAX_SWEEPS`（既定: 8、環境変数 `PM_PRIORITY_SWEEPS` で上書き可）。
  - 例: `PM_PRIORITY_SWEEPS=10 python3 mvs/main.py`

- 伝播方式の選択
  - `CHOICED_PROPAGATION_METHOD` に `"checkerboard"` か `"priority"` を指定。
  - 速度重視: `checkerboard`、収束深さ重視: `priority`（スイープ回数を増やす）。

- Top-K集約のロバスト化
  - `USE_MEDIAN_TOP_K=1` で Top-K を中央値集約。
  - 速度重視なら `USE_MEDIAN_TOP_K=0`。

- イテレーションログの累積時間
  - GPUログに `cum=...s` を追加。これは「最初のGPUカーネル実行完了以降」の累積時間で、初回JITコンパイル時間を含まない。

- GPUウォームアップ
  - 起動直後にバックグラウンドで主要カーネルを小規模入力で一度起動。
  - 環境変数で無効化: `PM_GPU_WARMUP=0`。

## 処理フローの概要

本パイプラインは `mvs/main.py` によって制御され、以下のステップで処理が進められる。

1.  **データ読み込み (`data_loader.py`)**: `config.py` で指定されたデータセットのステレオ画像とカメラ姿勢データを読み込む。
2.  **初期深度推定 (`disparity_estimation.py`, `depth_estimation.py`)**: 各フレームのステレオ画像ペアから視差マップを計算し、これを深度マップに変換する。
3.  **深度マップ最適化 (`depth_optimization.py`)**: PatchMatchアルゴリズムを用いて、近傍フレームの情報を活用しながら深度と法線を繰り返し最適化する。このステップはGPUで高速化できる (`depth_optimization_gpu.py`)。
4.  **フィルタリング**: 最適化された深度マップに対し、光度一貫性と幾何学的一貫性のチェックを行い、信頼性の低い推定結果を除去する。
5.  **点群への変換と統合 (`depth_estimation.py`, `point_cloud_integrator.py`)**: フィルタリング後の深度マップを3D点群に変換し、ワールド座標系で統合する。複数のフレームからの点群を重ね合わせることで、より密でノイズの少ない点群を生成する。
6.  **出力**: 最終的な点群データをPLYファイルとして保存する。

## 連絡

プロジェクトに関する質問や連絡は、以下の連絡先までお願いします。

- **GitHub:** [guro-Ishiguro](https://github.com/guro-Ishiguro)
- **Email1:** guro120411@gmail.com
- **Email2:** ishiguro.ryunosuke.62w@st.kyoto-u.ac.jp
