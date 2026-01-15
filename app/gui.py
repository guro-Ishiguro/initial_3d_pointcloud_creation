import logging
import os
import sys
import tempfile

import yaml

try:
    from PyQt5.QtCore import QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    try:
        from PySide2.QtCore import QThread  # noqa: F401
        from PySide2.QtCore import Signal as pyqtSignal
        from PySide2.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QProgressBar,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print("エラー: PyQt5またはPySide2が必要です。")
        print("インストール方法: pip install PyQt5")
        sys.exit(1)


def _list_datasets(project_root: str):
    """データセットのリストを取得"""
    data_dir = os.path.join(project_root, "data")
    if not os.path.isdir(data_dir):
        return []
    dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    dirs.sort()
    return dirs


def _load_default_config(config_path: str):
    """デフォルト設定を読み込む"""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
            # 数値型に変換
            for key, value in config.items():
                if isinstance(value, str):
                    # 数値文字列を数値に変換
                    try:
                        if "." in value or "e" in value.lower():
                            config[key] = float(value)
                        else:
                            config[key] = int(value)
                    except ValueError:
                        # 変換できない場合はそのまま（boolやNoneなど）
                        if value.lower() in ("true", "false"):
                            config[key] = value.lower() == "true"
                        elif value.lower() == "null":
                            config[key] = None
            return config
    return {}


class PipelineThread(QThread):
    """パイプラインを実行するスレッド"""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, project_root, dataset, config_dict):
        super().__init__()
        self.project_root = project_root
        self.dataset = dataset
        self.config_dict = config_dict

    def run(self):
        """パイプラインを実行"""
        try:
            # 環境変数を設定
            os.environ["DATA_TYPE"] = self.dataset

            # 一時的なYAMLファイルを作成
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(
                    self.config_dict, f, default_flow_style=False, allow_unicode=True
                )
                temp_config_path = f.name

            try:
                # 設定ファイルの適用
                try:
                    from app.settings import apply_env_overrides

                    # 設定を環境変数に設定（ブール値は文字列に変換）
                    for key, value in self.config_dict.items():
                        if value is not None:
                            # ブール値の場合は小文字の文字列に変換
                            if isinstance(value, bool):
                                os.environ[key] = "true" if value else "false"
                            else:
                                os.environ[key] = str(value)

                    # 一時YAMLファイルのパスを環境変数に設定（mvs/config.pyが読み込む）
                    os.environ["APP_MVS_CONFIG"] = temp_config_path

                    apply_env_overrides(temp_config_path)
                    self.log_signal.emit(f"設定を適用しました。")
                except Exception as e:
                    self.log_signal.emit(f"警告: 設定の適用に失敗しました: {e}")

                # ログハンドラを設定
                root_logger = logging.getLogger()
                handler = LogHandler(self.log_signal)
                handler.setLevel(logging.INFO)
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                root_logger.addHandler(handler)

                try:
                    # mvs.mainをインポートして実行
                    try:
                        from mvs import main as mvs_main
                    except Exception:
                        import importlib.util

                        main_path = os.path.join(self.project_root, "mvs", "main.py")
                        spec = importlib.util.spec_from_file_location(
                            "mvs.main", main_path
                        )
                        m = importlib.util.module_from_spec(spec)
                        assert spec and spec.loader
                        spec.loader.exec_module(m)
                        mvs_main = m

                    # 実行
                    result = mvs_main.run()
                    self.finished_signal.emit(result)

                finally:
                    root_logger.removeHandler(handler)
            finally:
                # 一時ファイルを削除
                try:
                    os.unlink(temp_config_path)
                except Exception:
                    pass

        except Exception as e:
            self.log_signal.emit(f"エラーが発生しました: {e}")
            import traceback

            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(1)


class LogHandler(logging.Handler):
    """GUIのログエリアにログを出力するハンドラ"""

    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        """ログメッセージをGUIに表示"""
        try:
            msg = self.format(record)
            self.signal.emit(msg)
        except Exception:
            pass


class ConfigWidget(QWidget):
    """設定入力ウィジェット"""

    def __init__(self, default_config):
        super().__init__()
        self.config_widgets = {}
        self.default_config = default_config
        self._create_widgets()

    def _get_int_value(self, key, default):
        """整数値を取得（型変換付き）"""
        value = self.default_config.get(key, default)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return int(value) if value is not None else default

    def _get_float_value(self, key, default):
        """浮動小数点値を取得（型変換付き）"""
        value = self.default_config.get(key, default)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return float(value) if value is not None else default

    def _get_bool_value(self, key, default):
        """ブール値を取得（型変換付き）"""
        value = self.default_config.get(key, default)
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value) if value is not None else default

    def _create_widgets(self):
        """ウィジェットを作成"""
        layout = QVBoxLayout(self)

        # タブウィジェット
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # 各カテゴリのタブを作成
        tabs.addTab(self._create_patchmatch_tab(), "PatchMatch基本")
        tabs.addTab(self._create_filtering_tab(), "フィルタリング")
        tabs.addTab(self._create_frame_tab(), "フレーム選択")
        tabs.addTab(self._create_neighbor_tab(), "近傍ビュー")
        tabs.addTab(self._create_debug_tab(), "デバッグ・出力")
        tabs.addTab(self._create_visualization_tab(), "可視化")
        tabs.addTab(self._create_other_tab(), "その他")

    def _create_patchmatch_tab(self):
        """PatchMatch基本パラメータのタブ"""
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)

        # PATCHMATCH_ITERATIONS
        self.config_widgets["PATCHMATCH_ITERATIONS"] = QSpinBox()
        self.config_widgets["PATCHMATCH_ITERATIONS"].setRange(1, 100)
        self.config_widgets["PATCHMATCH_ITERATIONS"].setValue(
            self._get_int_value("PATCHMATCH_ITERATIONS", 10)
        )
        layout.addRow(
            QLabel("PatchMatch反復回数:"), self.config_widgets["PATCHMATCH_ITERATIONS"]
        )
        layout.addRow(
            QLabel(""),
            QLabel(
                "PatchMatchアルゴリズムの反復回数。多いほど精度が上がりますが時間がかかります。"
            ),
        )

        # PATCHMATCH_PATCH_SIZE
        self.config_widgets["PATCHMATCH_PATCH_SIZE"] = QSpinBox()
        self.config_widgets["PATCHMATCH_PATCH_SIZE"].setRange(3, 15)
        self.config_widgets["PATCHMATCH_PATCH_SIZE"].setValue(
            self._get_int_value("PATCHMATCH_PATCH_SIZE", 7)
        )
        layout.addRow(
            QLabel("パッチサイズ:"), self.config_widgets["PATCHMATCH_PATCH_SIZE"]
        )
        layout.addRow(
            QLabel(""),
            QLabel(
                "マッチングに使用するパッチのサイズ（奇数）。大きいほど安定しますが計算コストが増えます。"
            ),
        )

        # ZNCC_EPSILON
        self.config_widgets["ZNCC_EPSILON"] = QDoubleSpinBox()
        self.config_widgets["ZNCC_EPSILON"].setDecimals(10)
        self.config_widgets["ZNCC_EPSILON"].setRange(1e-10, 1e-3)
        self.config_widgets["ZNCC_EPSILON"].setSingleStep(1e-6)
        self.config_widgets["ZNCC_EPSILON"].setValue(
            self._get_float_value("ZNCC_EPSILON", 1e-6)
        )
        layout.addRow(QLabel("ZNCCイプシロン:"), self.config_widgets["ZNCC_EPSILON"])
        layout.addRow(QLabel(""), QLabel("ZNCC計算時のゼロ除算を防ぐための小さな値。"))

        # TOP_K_COSTS
        self.config_widgets["TOP_K_COSTS"] = QSpinBox()
        self.config_widgets["TOP_K_COSTS"].setRange(1, 20)
        self.config_widgets["TOP_K_COSTS"].setValue(
            self._get_int_value("TOP_K_COSTS", 5)
        )
        layout.addRow(QLabel("Top-Kコスト数:"), self.config_widgets["TOP_K_COSTS"])
        layout.addRow(QLabel(""), QLabel("コスト集約時に使用する上位K個のコスト値。"))

        # PATCHMATCH_DECAY_RATE
        self.config_widgets["PATCHMATCH_DECAY_RATE"] = QDoubleSpinBox()
        self.config_widgets["PATCHMATCH_DECAY_RATE"].setDecimals(2)
        self.config_widgets["PATCHMATCH_DECAY_RATE"].setRange(0.1, 1.0)
        self.config_widgets["PATCHMATCH_DECAY_RATE"].setSingleStep(0.1)
        self.config_widgets["PATCHMATCH_DECAY_RATE"].setValue(
            self._get_float_value("PATCHMATCH_DECAY_RATE", 0.9)
        )
        layout.addRow(QLabel("減衰率:"), self.config_widgets["PATCHMATCH_DECAY_RATE"])
        layout.addRow(
            QLabel(""),
            QLabel(
                "ランダムサーチの探索範囲を減衰させる率。イテレーションごとに探索範囲が狭くなります。"
            ),
        )

        # PATCHMATCH_NORMAL_SEARCH_ANGLE
        self.config_widgets["PATCHMATCH_NORMAL_SEARCH_ANGLE"] = QDoubleSpinBox()
        self.config_widgets["PATCHMATCH_NORMAL_SEARCH_ANGLE"].setDecimals(1)
        self.config_widgets["PATCHMATCH_NORMAL_SEARCH_ANGLE"].setRange(0.0, 90.0)
        self.config_widgets["PATCHMATCH_NORMAL_SEARCH_ANGLE"].setValue(
            self._get_float_value("PATCHMATCH_NORMAL_SEARCH_ANGLE", 20.0)
        )
        layout.addRow(
            QLabel("法線探索角度:"),
            self.config_widgets["PATCHMATCH_NORMAL_SEARCH_ANGLE"],
        )
        layout.addRow(
            QLabel(""), QLabel("ランダムサーチで法線を探索する角度範囲（度）。")
        )

        # ADAPTIVE_WEIGHT_SIGMA_COLOR
        self.config_widgets["ADAPTIVE_WEIGHT_SIGMA_COLOR"] = QDoubleSpinBox()
        self.config_widgets["ADAPTIVE_WEIGHT_SIGMA_COLOR"].setDecimals(1)
        self.config_widgets["ADAPTIVE_WEIGHT_SIGMA_COLOR"].setRange(1.0, 50.0)
        self.config_widgets["ADAPTIVE_WEIGHT_SIGMA_COLOR"].setValue(
            self._get_float_value("ADAPTIVE_WEIGHT_SIGMA_COLOR", 10.0)
        )
        layout.addRow(
            QLabel("適応重みシグマ（色）:"),
            self.config_widgets["ADAPTIVE_WEIGHT_SIGMA_COLOR"],
        )
        layout.addRow(
            QLabel(""),
            QLabel(
                "色差に基づく適応的重み付けのシグマ値。大きいほど色差の影響が小さくなります。"
            ),
        )

        # 視差推定パラメータ
        layout.addRow(QLabel(""), QLabel(""))  # セパレータ
        layout.addRow(QLabel(""), QLabel("--- 視差推定パラメータ ---"))

        # WINDOW_SIZE
        self.config_widgets["WINDOW_SIZE"] = QSpinBox()
        self.config_widgets["WINDOW_SIZE"].setRange(3, 21)
        self.config_widgets["WINDOW_SIZE"].setValue(
            self._get_int_value("WINDOW_SIZE", 7)
        )
        layout.addRow(
            QLabel("視差推定ウィンドウサイズ:"), self.config_widgets["WINDOW_SIZE"]
        )
        layout.addRow(
            QLabel(""),
            QLabel(
                "ステレオマッチングで使用するブロックサイズ（奇数）。大きいほど安定しますが計算コストが増えます。"
            ),
        )

        # MIN_DISP
        self.config_widgets["MIN_DISP"] = QSpinBox()
        self.config_widgets["MIN_DISP"].setRange(0, 1000)
        self.config_widgets["MIN_DISP"].setValue(self._get_int_value("MIN_DISP", 0))
        layout.addRow(QLabel("最小視差:"), self.config_widgets["MIN_DISP"])
        layout.addRow(
            QLabel(""),
            QLabel("視差探索の最小値。通常は0です。"),
        )

        # NUM_DISP
        self.config_widgets["NUM_DISP"] = QSpinBox()
        self.config_widgets["NUM_DISP"].setRange(16, 512)
        self.config_widgets["NUM_DISP"].setValue(self._get_int_value("NUM_DISP", 216))
        layout.addRow(QLabel("視差範囲:"), self.config_widgets["NUM_DISP"])
        layout.addRow(
            QLabel(""),
            QLabel(
                "視差探索の範囲（16の倍数）。大きいほど遠距離まで検出できますが計算コストが増えます。"
            ),
        )

        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        return scroll

    def _create_filtering_tab(self):
        """フィルタリングパラメータのタブ"""
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)

        # FILTERING_COLOR_DIFFERENCE_THRESHOLD
        self.config_widgets["FILTERING_COLOR_DIFFERENCE_THRESHOLD"] = QDoubleSpinBox()
        self.config_widgets["FILTERING_COLOR_DIFFERENCE_THRESHOLD"].setDecimals(1)
        self.config_widgets["FILTERING_COLOR_DIFFERENCE_THRESHOLD"].setRange(0.0, 255.0)
        self.config_widgets["FILTERING_COLOR_DIFFERENCE_THRESHOLD"].setValue(
            self._get_float_value("FILTERING_COLOR_DIFFERENCE_THRESHOLD", 20.0)
        )
        layout.addRow(
            QLabel("色差閾値:"),
            self.config_widgets["FILTERING_COLOR_DIFFERENCE_THRESHOLD"],
        )
        layout.addRow(QLabel(""), QLabel("光度一貫性フィルタリングでの色差の閾値。"))

        # FILTERING_MIN_CONSISTENT_VIEWS
        self.config_widgets["FILTERING_MIN_CONSISTENT_VIEWS"] = QSpinBox()
        self.config_widgets["FILTERING_MIN_CONSISTENT_VIEWS"].setRange(1, 20)
        self.config_widgets["FILTERING_MIN_CONSISTENT_VIEWS"].setValue(
            self._get_int_value("FILTERING_MIN_CONSISTENT_VIEWS", 3)
        )
        layout.addRow(
            QLabel("最小一貫ビュー数（光度）:"),
            self.config_widgets["FILTERING_MIN_CONSISTENT_VIEWS"],
        )
        layout.addRow(
            QLabel(""), QLabel("光度一貫性チェックで必要な最小の一貫ビュー数。")
        )

        # GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD
        self.config_widgets["GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD"] = QDoubleSpinBox()
        self.config_widgets["GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD"].setDecimals(4)
        self.config_widgets["GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD"].setRange(0.0, 1.0)
        self.config_widgets["GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD"].setValue(
            self._get_float_value("GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD", 0.05)
        )
        layout.addRow(
            QLabel("幾何一貫性エラー閾値:"),
            self.config_widgets["GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD"],
        )
        layout.addRow(
            QLabel(""), QLabel("幾何学的な一貫性チェックでの相対深度差の閾値。")
        )

        # GEOMETRIC_MIN_CONSISTENT_VIEWS
        self.config_widgets["GEOMETRIC_MIN_CONSISTENT_VIEWS"] = QSpinBox()
        self.config_widgets["GEOMETRIC_MIN_CONSISTENT_VIEWS"].setRange(1, 20)
        self.config_widgets["GEOMETRIC_MIN_CONSISTENT_VIEWS"].setValue(
            self._get_int_value("GEOMETRIC_MIN_CONSISTENT_VIEWS", 2)
        )
        layout.addRow(
            QLabel("最小一貫ビュー数（幾何）:"),
            self.config_widgets["GEOMETRIC_MIN_CONSISTENT_VIEWS"],
        )
        layout.addRow(
            QLabel(""), QLabel("幾何学的な一貫性チェックで必要な最小の一貫ビュー数。")
        )

        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        return scroll

    def _create_frame_tab(self):
        """フレーム選択パラメータのタブ"""
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)

        # FRAME_STRIDE
        self.config_widgets["FRAME_STRIDE"] = QSpinBox()
        self.config_widgets["FRAME_STRIDE"].setRange(1, 100)
        self.config_widgets["FRAME_STRIDE"].setValue(
            self._get_int_value("FRAME_STRIDE", 15)
        )
        layout.addRow(
            QLabel("フレームストライド:"), self.config_widgets["FRAME_STRIDE"]
        )
        layout.addRow(
            QLabel(""),
            QLabel(
                "処理するフレームの間隔。1なら全フレーム、15なら15フレームごとに処理します。"
            ),
        )

        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        return scroll

    def _create_neighbor_tab(self):
        """近傍ビュー選択パラメータのタブ"""
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)

        # MAX_NEIGHBORS
        self.config_widgets["MAX_NEIGHBORS"] = QSpinBox()
        self.config_widgets["MAX_NEIGHBORS"].setRange(1, 50)
        self.config_widgets["MAX_NEIGHBORS"].setValue(
            self._get_int_value("MAX_NEIGHBORS", 8)
        )
        layout.addRow(QLabel("最大近傍数:"), self.config_widgets["MAX_NEIGHBORS"])
        layout.addRow(QLabel(""), QLabel("使用する近傍ビューの最大数。"))

        # NEIGHBOR_SELECTION_MODE
        self.config_widgets["NEIGHBOR_SELECTION_MODE"] = QComboBox()
        self.config_widgets["NEIGHBOR_SELECTION_MODE"].addItems(["nearest", "keyframe"])
        mode = self.default_config.get("NEIGHBOR_SELECTION_MODE", "nearest")
        index = 0 if mode == "nearest" else 1
        self.config_widgets["NEIGHBOR_SELECTION_MODE"].setCurrentIndex(index)
        layout.addRow(
            QLabel("近傍選択モード:"), self.config_widgets["NEIGHBOR_SELECTION_MODE"]
        )
        layout.addRow(
            QLabel(""), QLabel("nearest: 距離が近い順、keyframe: キーフレームベース。")
        )

        # NEIGHBOR_NEAREST_COUNT
        self.config_widgets["NEIGHBOR_NEAREST_COUNT"] = QSpinBox()
        self.config_widgets["NEIGHBOR_NEAREST_COUNT"].setRange(1, 50)
        self.config_widgets["NEIGHBOR_NEAREST_COUNT"].setValue(
            self._get_int_value("NEIGHBOR_NEAREST_COUNT", 10)
        )
        layout.addRow(
            QLabel("近傍数（nearest）:"), self.config_widgets["NEIGHBOR_NEAREST_COUNT"]
        )
        layout.addRow(QLabel(""), QLabel("nearestモードで選択する近傍の数。"))

        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        return scroll

    def _create_debug_tab(self):
        """デバッグ・出力設定のタブ"""
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)

        # DEBUG_SAVE_DEPTH_MAPS
        self.config_widgets["DEBUG_SAVE_DEPTH_MAPS"] = QCheckBox()
        self.config_widgets["DEBUG_SAVE_DEPTH_MAPS"].setChecked(
            self._get_bool_value("DEBUG_SAVE_DEPTH_MAPS", True)
        )
        layout.addRow(
            QLabel("深度マップを保存:"), self.config_widgets["DEBUG_SAVE_DEPTH_MAPS"]
        )
        layout.addRow(
            QLabel(""), QLabel("各イテレーションの深度マップをPNG形式で保存します。")
        )

        # DEBUG_SAVE_NORMAL_MAPS
        self.config_widgets["DEBUG_SAVE_NORMAL_MAPS"] = QCheckBox()
        self.config_widgets["DEBUG_SAVE_NORMAL_MAPS"].setChecked(
            self._get_bool_value("DEBUG_SAVE_NORMAL_MAPS", True)
        )
        layout.addRow(
            QLabel("法線マップを保存:"), self.config_widgets["DEBUG_SAVE_NORMAL_MAPS"]
        )
        layout.addRow(
            QLabel(""), QLabel("各イテレーションの法線マップをPNG形式で保存します。")
        )

        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        return scroll

    def _create_visualization_tab(self):
        """可視化パラメータのタブ"""
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)

        # SHOW_POINT_CLOUD
        self.config_widgets["SHOW_POINT_CLOUD"] = QCheckBox()
        self.config_widgets["SHOW_POINT_CLOUD"].setChecked(
            self._get_bool_value("SHOW_POINT_CLOUD", False)
        )
        layout.addRow(QLabel("点群を表示:"), self.config_widgets["SHOW_POINT_CLOUD"])
        layout.addRow(
            QLabel(""),
            QLabel("処理完了後に点群を3Dビューアで表示します。"),
        )

        # VIZ_DEPTH_MIN
        self.config_widgets["VIZ_DEPTH_MIN"] = QDoubleSpinBox()
        self.config_widgets["VIZ_DEPTH_MIN"].setDecimals(1)
        self.config_widgets["VIZ_DEPTH_MIN"].setRange(0.0, 1000.0)
        self.config_widgets["VIZ_DEPTH_MIN"].setValue(
            self._get_float_value("VIZ_DEPTH_MIN", 17.0)
        )
        layout.addRow(QLabel("可視化深度最小値:"), self.config_widgets["VIZ_DEPTH_MIN"])
        layout.addRow(QLabel(""), QLabel("深度マップ可視化時の最小深度値。"))

        # VIZ_DEPTH_MAX
        self.config_widgets["VIZ_DEPTH_MAX"] = QDoubleSpinBox()
        self.config_widgets["VIZ_DEPTH_MAX"].setDecimals(1)
        self.config_widgets["VIZ_DEPTH_MAX"].setRange(0.0, 1000.0)
        self.config_widgets["VIZ_DEPTH_MAX"].setValue(
            self._get_float_value("VIZ_DEPTH_MAX", 36.0)
        )
        layout.addRow(QLabel("可視化深度最大値:"), self.config_widgets["VIZ_DEPTH_MAX"])
        layout.addRow(QLabel(""), QLabel("深度マップ可視化時の最大深度値。"))

        # VIZ_CMAP
        self.config_widgets["VIZ_CMAP"] = QComboBox()
        cmap_options = [
            "jet",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "turbo",
            "hot",
            "cool",
            "spring",
            "summer",
            "autumn",
            "winter",
            "gray",
            "bone",
            "copper",
            "pink",
        ]
        self.config_widgets["VIZ_CMAP"].addItems(cmap_options)
        current_cmap = self.default_config.get("VIZ_CMAP", "jet")
        if current_cmap in cmap_options:
            index = cmap_options.index(current_cmap)
            self.config_widgets["VIZ_CMAP"].setCurrentIndex(index)
        else:
            self.config_widgets["VIZ_CMAP"].setCurrentText(str(current_cmap))
        layout.addRow(QLabel("カラーマップ:"), self.config_widgets["VIZ_CMAP"])
        layout.addRow(
            QLabel(""),
            QLabel("深度マップ可視化時に使用するカラーマップ。"),
        )

        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        return scroll

    def _create_other_tab(self):
        """その他のタブ"""
        scroll = QScrollArea()
        widget = QWidget()
        layout = QFormLayout(widget)

        # POSITION_ERROR_SCALE
        self.config_widgets["POSITION_ERROR_SCALE"] = QDoubleSpinBox()
        self.config_widgets["POSITION_ERROR_SCALE"].setDecimals(4)
        self.config_widgets["POSITION_ERROR_SCALE"].setRange(0.0, 10.0)
        self.config_widgets["POSITION_ERROR_SCALE"].setSingleStep(0.01)
        self.config_widgets["POSITION_ERROR_SCALE"].setValue(
            self._get_float_value("POSITION_ERROR_SCALE", 0.0)
        )
        layout.addRow(
            QLabel("位置エラースケール (m):"),
            self.config_widgets["POSITION_ERROR_SCALE"],
        )
        layout.addRow(
            QLabel(""),
            QLabel(
                "カメラ位置の誤差の標準偏差（メートル単位）。0.0でエラーなし。例: 0.1m = 10cm"
            ),
        )

        # ROTATION_ERROR_SCALE
        self.config_widgets["ROTATION_ERROR_SCALE"] = QDoubleSpinBox()
        self.config_widgets["ROTATION_ERROR_SCALE"].setDecimals(4)
        self.config_widgets["ROTATION_ERROR_SCALE"].setRange(0.0, 1.0)
        self.config_widgets["ROTATION_ERROR_SCALE"].setSingleStep(0.001)
        self.config_widgets["ROTATION_ERROR_SCALE"].setValue(
            self._get_float_value("ROTATION_ERROR_SCALE", 0.0)
        )
        layout.addRow(
            QLabel("回転エラースケール (rad):"),
            self.config_widgets["ROTATION_ERROR_SCALE"],
        )
        layout.addRow(
            QLabel(""),
            QLabel(
                "カメラ回転の誤差の標準偏差（ラジアン単位）。0.0でエラーなし。例: 0.01rad ≈ 0.6度, 0.1rad ≈ 6度"
            ),
        )

        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        return scroll

    def get_config(self):
        """現在の設定値を取得"""
        config = {}
        for key, widget in self.config_widgets.items():
            if isinstance(widget, QCheckBox):
                config[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                config[key] = widget.currentText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                config[key] = widget.value()
        return config


class MVSGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Point Cloud Pipeline")
        self.setGeometry(100, 100, 900, 700)

        # プロジェクトルートを取得
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.mvs_dir = os.path.join(self.project_root, "mvs")

        # sys.pathに追加
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)
        if self.mvs_dir not in sys.path:
            sys.path.insert(0, self.mvs_dir)

        # デフォルト設定を読み込み
        default_config_path = os.path.join(self.project_root, "app", "mvs.yaml")
        self.default_config = _load_default_config(default_config_path)

        # 変数
        self.pipeline_thread = None

        self._create_widgets()
        self._load_datasets()

    def _create_widgets(self):
        """ウィジェットを作成"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)

        # データセット選択
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("データセット:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumWidth(300)
        dataset_layout.addWidget(self.dataset_combo)
        dataset_layout.addStretch()
        layout.addLayout(dataset_layout)

        # 設定タブ
        self.config_widget = ConfigWidget(self.default_config)
        layout.addWidget(self.config_widget)

        # 実行ボタン
        self.run_button = QPushButton("実行")
        self.run_button.clicked.connect(self._run_pipeline)
        self.run_button.setMinimumHeight(40)
        layout.addWidget(self.run_button)

        # ログ表示エリア
        layout.addWidget(QLabel("ログ:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFontFamily("Courier")
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)

        # クリアボタン
        clear_log_btn = QPushButton("ログをクリア")
        clear_log_btn.clicked.connect(self._clear_log)
        layout.addWidget(clear_log_btn)

        # 進捗バー
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # 不定進捗
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

    def _load_datasets(self):
        """データセットのリストを読み込み"""
        datasets = _list_datasets(self.project_root)
        if datasets:
            self.dataset_combo.addItems(datasets)
            if datasets:
                self.dataset_combo.setCurrentIndex(0)
        else:
            self._log("警告: データセットが見つかりませんでした。")
            self.run_button.setEnabled(False)

    def _log(self, message):
        """ログを表示"""
        self.log_text.append(message)
        # 自動スクロール
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _clear_log(self):
        """ログをクリア"""
        self.log_text.clear()

    def _run_pipeline(self):
        """パイプラインを実行"""
        if self.pipeline_thread and self.pipeline_thread.isRunning():
            self._log("既に実行中です。")
            return

        # データセットの確認
        dataset = self.dataset_combo.currentText()
        if not dataset:
            self._log("エラー: データセットを選択してください。")
            return

        datasets = _list_datasets(self.project_root)
        if dataset not in datasets:
            self._log(f"エラー: データセット '{dataset}' が見つかりません。")
            return

        # 設定を取得
        config_dict = self.config_widget.get_config()

        # UIを無効化
        self.run_button.setEnabled(False)
        self.dataset_combo.setEnabled(False)
        self.config_widget.setEnabled(False)
        self.progress.setVisible(True)

        self._log(f"データセット '{dataset}' でパイプラインを開始します...")

        # スレッドで実行
        self.pipeline_thread = PipelineThread(self.project_root, dataset, config_dict)
        self.pipeline_thread.log_signal.connect(self._log)
        self.pipeline_thread.finished_signal.connect(self._on_pipeline_finished)
        self.pipeline_thread.start()

    def _on_pipeline_finished(self, result):
        """パイプライン完了時の処理"""
        if result == 0:
            self._log("パイプラインが正常に完了しました。")
        else:
            self._log(f"パイプラインがエラーで終了しました（コード: {result}）。")

        # UIを有効化
        self.run_button.setEnabled(True)
        self.dataset_combo.setEnabled(True)
        self.config_widget.setEnabled(True)
        self.progress.setVisible(False)
        self.pipeline_thread = None


def main():
    """GUIアプリケーションを起動"""
    app = QApplication(sys.argv)
    window = MVSGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
