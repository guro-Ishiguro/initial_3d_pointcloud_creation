import argparse
import os
import sys
import threading
import logging
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path


def _list_datasets(project_root: str):
    """データセットのリストを取得"""
    data_dir = os.path.join(project_root, "data")
    if not os.path.isdir(data_dir):
        return []
    dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    dirs.sort()
    return dirs


class GUILogHandler(logging.Handler):
    """GUIのログエリアにログを出力するハンドラ"""
    def __init__(self, text_widget, root):
        super().__init__()
        self.text_widget = text_widget
        self.root = root
        self.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    
    def emit(self, record):
        """ログメッセージをGUIに表示"""
        try:
            msg = self.format(record)
            # メインスレッドで実行
            self.root.after(0, lambda: self._append_log(msg))
        except Exception:
            pass
    
    def _append_log(self, msg):
        """ログを追加（メインスレッドで実行）"""
        self.text_widget.insert(tk.END, msg + "\n")
        self.text_widget.see(tk.END)


class MVSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Point Cloud Pipeline")
        self.root.geometry("800x600")
        
        # プロジェクトルートを取得
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.mvs_dir = os.path.join(self.project_root, "mvs")
        
        # sys.pathに追加
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)
        if self.mvs_dir not in sys.path:
            sys.path.insert(0, self.mvs_dir)
        
        # 変数
        self.selected_dataset = tk.StringVar()
        self.config_path = tk.StringVar()
        self.is_running = False
        self.log_handler = None
        
        self._create_widgets()
        self._load_datasets()
    
    def _create_widgets(self):
        """ウィジェットを作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # グリッドの重み設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # データセット選択
        ttk.Label(main_frame, text="データセット:").grid(row=0, column=0, sticky=tk.W, pady=5)
        dataset_combo = ttk.Combobox(main_frame, textvariable=self.selected_dataset, state="readonly", width=50)
        dataset_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        self.dataset_combo = dataset_combo
        
        # 設定ファイル選択
        ttk.Label(main_frame, text="設定ファイル:").grid(row=1, column=0, sticky=tk.W, pady=5)
        config_frame = ttk.Frame(main_frame)
        config_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        config_frame.columnconfigure(0, weight=1)
        
        config_entry = ttk.Entry(config_frame, textvariable=self.config_path, width=40)
        config_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(config_frame, text="参照...", command=self._browse_config).grid(row=0, column=1)
        ttk.Button(config_frame, text="クリア", command=lambda: self.config_path.set("")).grid(row=0, column=2, padx=(5, 0))
        
        # 実行ボタン
        self.run_button = ttk.Button(main_frame, text="実行", command=self._run_pipeline, width=20)
        self.run_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        # ログ表示エリア
        ttk.Label(main_frame, text="ログ:").grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, width=80, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # クリアボタン
        ttk.Button(main_frame, text="ログをクリア", command=self._clear_log).grid(row=5, column=0, columnspan=2, pady=5)
        
        # 進捗バー
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
    
    def _load_datasets(self):
        """データセットのリストを読み込み"""
        datasets = _list_datasets(self.project_root)
        if datasets:
            self.dataset_combo['values'] = datasets
            if len(datasets) == 1:
                self.selected_dataset.set(datasets[0])
            elif datasets:
                self.selected_dataset.set(datasets[0])
        else:
            self._log("警告: データセットが見つかりませんでした。")
            self.run_button.config(state='disabled')
    
    def _browse_config(self):
        """設定ファイルを選択"""
        initial_dir = os.path.join(self.project_root, "app")
        if not os.path.exists(initial_dir):
            initial_dir = self.project_root
        
        filename = filedialog.askopenfilename(
            title="設定ファイルを選択",
            initialdir=initial_dir,
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if filename:
            self.config_path.set(filename)
    
    def _log(self, message):
        """ログを表示"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def _clear_log(self):
        """ログをクリア"""
        self.log_text.delete(1.0, tk.END)
    
    def _run_pipeline(self):
        """パイプラインを実行"""
        if self.is_running:
            self._log("既に実行中です。")
            return
        
        # データセットの確認
        dataset = self.selected_dataset.get()
        if not dataset:
            self._log("エラー: データセットを選択してください。")
            return
        
        datasets = _list_datasets(self.project_root)
        if dataset not in datasets:
            self._log(f"エラー: データセット '{dataset}' が見つかりません。")
            return
        
        # 設定ファイルの確認
        config_path = self.config_path.get().strip()
        if config_path and not os.path.exists(config_path):
            self._log(f"警告: 設定ファイル '{config_path}' が見つかりません。デフォルト設定を使用します。")
            config_path = None
        
        # UIを無効化
        self.is_running = True
        self.run_button.config(state='disabled')
        self.dataset_combo.config(state='disabled')
        self.progress.start()
        
        # 別スレッドで実行
        thread = threading.Thread(target=self._execute_pipeline, args=(dataset, config_path))
        thread.daemon = True
        thread.start()
    
    def _execute_pipeline(self, dataset, config_path):
        """パイプラインを実行（別スレッド）"""
        try:
            self._log(f"データセット '{dataset}' でパイプラインを開始します...")
            
            # 環境変数を設定
            os.environ["DATA_TYPE"] = dataset
            
            # 設定ファイルの適用
            if config_path:
                try:
                    from app.settings import apply_env_overrides
                    apply_env_overrides(config_path)
                    self._log(f"設定ファイル '{config_path}' を読み込みました。")
                except Exception as e:
                    self._log(f"警告: 設定ファイルの読み込みに失敗しました: {e}")
            
            # ログハンドラを設定
            root_logger = logging.getLogger()
            
            # GUI用のログハンドラを追加
            self.log_handler = GUILogHandler(self.log_text, self.root)
            self.log_handler.setLevel(logging.INFO)
            root_logger.addHandler(self.log_handler)
            
            try:
                # mvs.mainをインポートして実行
                try:
                    from mvs import main as mvs_main
                except Exception:
                    import importlib.util
                    main_path = os.path.join(self.project_root, "mvs", "main.py")
                    spec = importlib.util.spec_from_file_location("mvs.main", main_path)
                    m = importlib.util.module_from_spec(spec)
                    assert spec and spec.loader
                    spec.loader.exec_module(m)
                    mvs_main = m
                
                # 実行
                result = mvs_main.run()
                
                if result == 0:
                    self._log("パイプラインが正常に完了しました。")
                else:
                    self._log(f"パイプラインがエラーで終了しました（コード: {result}）。")
                    
            finally:
                # ログハンドラを削除
                if self.log_handler:
                    root_logger.removeHandler(self.log_handler)
                    self.log_handler = None
                
        except Exception as e:
            self._log(f"エラーが発生しました: {e}")
            import traceback
            self._log(traceback.format_exc())
        finally:
            # UIを有効化
            self.root.after(0, self._reset_ui)
    
    def _reset_ui(self):
        """UIをリセット"""
        self.is_running = False
        self.run_button.config(state='normal')
        self.dataset_combo.config(state='readonly')
        self.progress.stop()


def main():
    """GUIアプリケーションを起動"""
    root = tk.Tk()
    app = MVSGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

