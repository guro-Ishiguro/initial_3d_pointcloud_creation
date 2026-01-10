import argparse
import pathlib
from typing import Dict, Optional, Union, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm


ArrayNx3 = np.ndarray


def save_point_cloud(points: ArrayNx3, file_path: Union[str, pathlib.Path]) -> None:
    """
    点群をPLY形式で保存する。
    
    Open3Dを使用して高速に保存（利用可能な場合）。

    Parameters
    ----------
    points : (N, 3) ndarray
        点群の座標 [m]。
    file_path : str | pathlib.Path
        保存先のファイルパス。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("`points` は形状 (N, 3) の配列である必要があります。")

    file_path = pathlib.Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Open3Dを使用して保存（高速、バイナリ形式）
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # バイナリ形式で保存（高速）
        success = o3d.io.write_point_cloud(str(file_path), pcd, write_ascii=False)
        if success:
            print(f"点群を保存しました (Open3D): {file_path} ({len(points)}点)")
            return
    except (ImportError, Exception):
        pass  # Open3Dが利用できない、またはエラーが発生した場合はフォールバック

    # フォールバック: ASCII形式で保存
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    with open(file_path, "w") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.6f")

    print(f"点群を保存しました: {file_path} ({len(points)}点)")


def load_point_cloud(file_path: Union[str, pathlib.Path]) -> ArrayNx3:
    """
    点群ファイル（PLYなど）を読み込む。

    Parameters
    ----------
    file_path : str | pathlib.Path
        点群ファイルのパス。

    Returns
    -------
    points : (N, 3) ndarray
        点群の座標 [m]。
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"点群ファイルが見つかりません: {file_path}")

    # まずopen3dを試す（PLYファイルの場合、より確実）
    if file_path.suffix.lower() == ".ply":
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(file_path))
            if pcd is not None and len(pcd.points) > 0:
                points = np.asarray(pcd.points, dtype=np.float64)
                return points
        except ImportError:
            pass  # open3dが利用できない場合はtrimeshにフォールバック
        except Exception as e:
            # open3dで読み込めなかった場合、trimeshにフォールバック
            pass

    # trimeshで点群を読み込む
    # process=Falseにすることで、点群専用ファイルでも読み込める
    loaded = trimesh.load(str(file_path), process=False)

    # リストとして読み込まれた場合（複数のメッシュ/点群）
    if isinstance(loaded, list):
        if len(loaded) == 0:
            raise ValueError(f"ファイルにデータが含まれていません: {file_path}")
        # 最初の要素を使用（通常は1つのみ）
        loaded = loaded[0]

    # Trimeshオブジェクトの場合
    if isinstance(loaded, trimesh.Trimesh):
        vertices = loaded.vertices
        if vertices is not None and vertices.size > 0:
            return np.asarray(vertices, dtype=np.float64)
        else:
            raise ValueError(
                f"メッシュに頂点が含まれていません: {file_path}\n"
                f"verticesの形状: {vertices.shape if vertices is not None else 'None'}"
            )

    # PointCloudオブジェクトの場合
    if isinstance(loaded, trimesh.PointCloud):
        vertices = loaded.vertices
        if vertices is not None and vertices.size > 0:
            return np.asarray(vertices, dtype=np.float64)
        else:
            raise ValueError(
                f"点群に頂点が含まれていません: {file_path}\n"
                f"verticesの形状: {vertices.shape if vertices is not None else 'None'}"
            )

    # その他の場合、vertices属性があるか確認
    if hasattr(loaded, "vertices"):
        vertices = loaded.vertices
        if vertices is not None and vertices.size > 0:
            return np.asarray(vertices, dtype=np.float64)

    raise ValueError(
        f"点群ファイルを読み込めませんでした: {file_path}\n"
        f"読み込まれたオブジェクトの型: {type(loaded)}\n"
        f"オブジェクトの属性: {dir(loaded) if hasattr(loaded, '__dict__') else 'N/A'}\n"
        f"ヒント: PLYファイルの場合、open3dがインストールされていると読み込みが改善される可能性があります。"
    )


def load_mesh(
    mesh: Union[str, pathlib.Path, trimesh.Trimesh, None] = None,
    vertices: Optional[ArrayNx3] = None,
    faces: Optional[np.ndarray] = None,
) -> trimesh.Trimesh:
    """
    真値メッシュ M を読み込む / 構築するユーティリティ関数。

    Parameters
    ----------
    mesh : str | pathlib.Path | trimesh.Trimesh | None
        - メッシュファイルパス（PLY/OBJなど）、
        - 既に構築済みの `trimesh.Trimesh` インスタンス、
        - もしくは None（この場合は `vertices` / `faces` から構築）。
    vertices : (V, 3) ndarray, optional
        メモリ上の頂点座標 [m]。
    faces : (F, 3) ndarray[int], optional
        メモリ上の三角形ポリゴンの頂点インデックス。

    Returns
    -------
    mesh : trimesh.Trimesh
        構築された真値メッシュ。
    """
    if isinstance(mesh, trimesh.Trimesh):
        return mesh

    if mesh is not None:
        # ファイルから読み込み
        return trimesh.load_mesh(str(mesh), process=True)

    if vertices is None or faces is None:
        raise ValueError("`vertices` と `faces` の両方、または `mesh` を指定してください。")

    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces, dtype=np.int64), process=True)


def crop_mesh_with_bbox(
    mesh: trimesh.Trimesh,
    points: ArrayNx3,
    margin: float = 0.0,
) -> trimesh.Trimesh:
    """
    再構成点群 P のバウンディングボックスで真値メッシュ M をクロップする。

    Parameters
    ----------
    mesh : trimesh.Trimesh
        真値メッシュ M。
    points : (N, 3) ndarray
        再構成点群 P [m]。
    margin : float, optional
        バウンディングボックスに付与するマージン [m]。

    Returns
    -------
    cropped : trimesh.Trimesh
        クロップ後の真値メッシュ。

    Raises
    ------
    ValueError
        クロップ後にメッシュが空になった場合。
    """
    if points.size == 0:
        raise ValueError("`points` が空です。")

    pts = np.asarray(points, dtype=np.float64)
    p_min = pts.min(axis=0) - margin
    p_max = pts.max(axis=0) + margin

    v = mesh.vertices
    inside_mask = np.all((v >= p_min) & (v <= p_max), axis=1)

    # 全頂点がバウンディングボックス内にある三角形のみ残す
    faces_keep_mask = inside_mask[mesh.faces].all(axis=1)
    faces_keep = mesh.faces[faces_keep_mask]

    if faces_keep.size == 0:
        raise ValueError("バウンディングボックスでクロップした結果、メッシュが空になりました。")

    cropped = trimesh.Trimesh(vertices=v, faces=faces_keep, process=True)
    return cropped


def sample_points_from_mesh(
    mesh: trimesh.Trimesh,
    num_points: int,
    random_state: Optional[int] = None,
    show_progress: bool = True,
) -> ArrayNx3:
    """
    メッシュ表面から一様サンプリングで点群 Q を生成する。

    Parameters
    ----------
    mesh : trimesh.Trimesh
        クロップ後の真値メッシュ M。
    num_points : int
        サンプリングする点数 |Q| = |P|。
    random_state : int, optional
        乱数シード（再現性のため）。
    show_progress : bool, optional
        プログレスバーを表示するかどうか。

    Returns
    -------
    points : (num_points, 3) ndarray
        真値点群 Q [m]。
    """
    if num_points <= 0:
        raise ValueError("`num_points` は正の整数である必要があります。")

    rng = np.random.default_rng(random_state)
    # trimesh の sample メソッドは内部で乱数を使うが、グローバル乱数を使うため、
    # 一時的に NumPy のシードを設定して再現性を担保する。
    state = np.random.get_state()
    np.random.seed(rng.integers(0, 2**31 - 1))
    try:
        if show_progress:
            # trimeshのsampleは一括処理なので、プログレスバーは処理全体を表示
            with tqdm(total=1, desc="真値点群をサンプリング中", unit="処理") as pbar:
                sampled = mesh.sample(num_points)
                pbar.update(1)
        else:
            sampled = mesh.sample(num_points)
    finally:
        np.random.set_state(state)

    return np.asarray(sampled, dtype=np.float64)


def bidirectional_consistency_error(P: ArrayNx3, Q: ArrayNx3, show_progress: bool = True) -> float:
    """
    Bidirectional Consistency Error (BCE) を計算する。

    Eq. (BCE):
        NN_Q(p) : 点 p に対する Q 上の最近傍点
        NN_P(x) : 点 x に対する P 上の最近傍点
        BCE = (1/|P|) sum_{p in P} || p - NN_P( NN_Q(p) ) ||

    Parameters
    ----------
    P : (N, 3) ndarray
        再構成点群 P [m]。
    Q : (N, 3) ndarray
        真値点群 Q [m]。
    show_progress : bool, optional
        プログレスバーを表示するかどうか。

    Returns
    -------
    bce : float
        Bidirectional Consistency Error [m]。
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    if P.shape[0] == 0 or Q.shape[0] == 0:
        raise ValueError("P と Q は共に1点以上を含む必要があります。")

    # P 上と Q 上の KD-tree を構築（効率のため再利用）
    tree_Q = cKDTree(Q)
    tree_P = cKDTree(P)

    # 各 p ∈ P に対して NN_Q(p) を求める
    if show_progress and P.shape[0] > 50000:
        batch_size = 50000
        idx_Q_list = []
        with tqdm(total=P.shape[0], desc="  BCE: P→Q最近傍を計算中", unit="点") as pbar:
            for i in range(0, P.shape[0], batch_size):
                batch = P[i:i + batch_size]
                _, batch_idx = tree_Q.query(batch, k=1)
                idx_Q_list.append(batch_idx)
                pbar.update(len(batch))
        idx_Q = np.concatenate(idx_Q_list)
    else:
        _, idx_Q = tree_Q.query(P, k=1)
    
    nn_Q = Q[idx_Q]  # shape: (N, 3)

    # さらに NN_Q(p) に対して NN_P(NN_Q(p)) を求める
    if show_progress and nn_Q.shape[0] > 50000:
        batch_size = 50000
        idx_P_back_list = []
        with tqdm(total=nn_Q.shape[0], desc="  BCE: Q→P最近傍を計算中", unit="点") as pbar:
            for i in range(0, nn_Q.shape[0], batch_size):
                batch = nn_Q[i:i + batch_size]
                _, batch_idx = tree_P.query(batch, k=1)
                idx_P_back_list.append(batch_idx)
                pbar.update(len(batch))
        idx_P_back = np.concatenate(idx_P_back_list)
    else:
        _, idx_P_back = tree_P.query(nn_Q, k=1)
    
    nn_P_back = P[idx_P_back]  # shape: (N, 3)

    # p と NN_P(NN_Q(p)) の距離
    diff = P - nn_P_back
    dists = np.linalg.norm(diff, axis=1)

    return float(dists.mean())


def point_to_mesh_distance(P: ArrayNx3, mesh: trimesh.Trimesh, show_progress: bool = True) -> float:
    """
    点群 P から真値メッシュ M への最短距離の平均 d_{p->m} を計算する。

    Eq. (point_to_mesh_distance):
        d_{p->m} = (1/|P|) sum_{p in P} min_{x in M} || p - x ||

    Parameters
    ----------
    P : (N, 3) ndarray
        再構成点群 P [m]。
    mesh : trimesh.Trimesh
        クロップ後の真値メッシュ M。
    show_progress : bool, optional
        プログレスバーを表示するかどうか。

    Returns
    -------
    d : float
        平均点群→メッシュ距離 [m]。
    """
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] == 0:
        raise ValueError("P は1点以上を含む必要があります。")

    # trimesh の最近傍探索で点からメッシュ表面への最短距離を計算
    # 大量の点の場合はバッチ処理でプログレスバーを表示
    # バッチサイズを大きくして、メッシュのKD-tree構築コストを減らす
    if show_progress and P.shape[0] > 50000:
        # 大量の点の場合は大きなバッチサイズを使用
        batch_size = min(100000, P.shape[0] // 10)  # 10バッチ程度に分割、最大100,000点
        distances_list = []
        with tqdm(total=P.shape[0], desc="点群→メッシュ距離を計算中", unit="点", miniters=1000) as pbar:
            for i in range(0, P.shape[0], batch_size):
                batch = P[i:i + batch_size]
                _, batch_distances, _ = mesh.nearest.on_surface(batch)
                distances_list.append(batch_distances)
                pbar.update(len(batch))
                pbar.refresh()  # 強制的に更新
        distances = np.concatenate(distances_list)
    else:
        _, distances, _ = mesh.nearest.on_surface(P)
    
    # `distances` は各点に対応する距離
    return float(np.asarray(distances, dtype=np.float64).mean())


def _compute_point_to_mesh_and_outlier_ratio(
    P: ArrayNx3,
    mesh: trimesh.Trimesh,
    threshold: float = 0.1,
    show_progress: bool = True,
    mesh_file_path: Optional[Union[str, pathlib.Path]] = None,
    return_distances: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, ArrayNx3]]:
    """
    点群→メッシュ距離と外れ値割合を同時に計算する（効率化のため）。
    
    Open3DのRaycastingSceneを使用して高速化（利用可能な場合）。
    メッシュファイルパスが指定されている場合、Open3Dで直接読み込んで変換をスキップ。

    Parameters
    ----------
    P : (N, 3) ndarray
        再構成点群 P [m]。
    mesh : trimesh.Trimesh
        クロップ後の真値メッシュ M。
    threshold : float, optional
        外れ値判定の閾値 T [m]。
    show_progress : bool, optional
        プログレスバーを表示するかどうか。
    mesh_file_path : str | pathlib.Path | None, optional
        メッシュファイルのパス（指定されている場合、Open3Dで直接読み込む）。

    return_distances : bool, optional
        距離情報も返すかどうか（可視化用）。

    Returns
    -------
    d_pm : float
        平均点群→メッシュ距離 [m]。
    r_out : float
        外れ値割合 [0〜1]。
    distances : (N,) ndarray, optional
        各点のメッシュへの距離 [m]（return_distances=Trueの場合）。
    """
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] == 0:
        raise ValueError("P は1点以上を含む必要があります。")

    # Open3DのRaycastingSceneを使用（高速化のため）
    try:
        import open3d as o3d
        
        # メッシュファイルパスが指定されている場合、Open3Dで直接読み込む（変換をスキップ）
        if mesh_file_path is not None:
            try:
                o3d_mesh = o3d.io.read_triangle_mesh(str(mesh_file_path))
                if len(o3d_mesh.vertices) > 0:
                    o3d_mesh.compute_triangle_normals()
                    # RaycastingSceneにメッシュを登録
                    scene = o3d.t.geometry.RaycastingScene()
                    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
                    _ = scene.add_triangles(tmesh)
                else:
                    raise ValueError("Open3Dで読み込んだメッシュが空です")
            except Exception:
                # Open3Dで読み込めなかった場合、trimeshから変換
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                faces = np.asarray(mesh.faces, dtype=np.int32)
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
                o3d_mesh.compute_triangle_normals()
                scene = o3d.t.geometry.RaycastingScene()
                tmesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
                _ = scene.add_triangles(tmesh)
        else:
            # trimeshのメッシュをOpen3DのTriangleMeshに変換
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            
            # Open3Dのレガシー形式のメッシュを作成
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d_mesh.compute_triangle_normals()
            
            # RaycastingSceneにメッシュを登録
            scene = o3d.t.geometry.RaycastingScene()
            tmesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
            _ = scene.add_triangles(tmesh)
        
        # 符号付き距離を計算（バッチ処理でプログレスバーを表示）
        if show_progress and P.shape[0] > 50000:
            batch_size = min(200000, P.shape[0] // 5)  # Open3Dは高速なので大きなバッチサイズを使用
            distances_list = []
            with tqdm(total=P.shape[0], desc="点群→メッシュ距離を計算中 (Open3D)", unit="点", miniters=1000) as pbar:
                for i in range(0, P.shape[0], batch_size):
                    batch = P[i:i + batch_size].astype(np.float32)
                    query = o3d.core.Tensor(batch, dtype=o3d.core.Dtype.Float32)
                    signed_dist = scene.compute_signed_distance(query).numpy().flatten()
                    # 符号付き距離の絶対値を取る（距離は常に正）
                    batch_distances = np.abs(signed_dist).astype(np.float64)
                    distances_list.append(batch_distances)
                    pbar.update(len(batch))
                    pbar.refresh()
            distances = np.concatenate(distances_list)
        else:
            points = P.astype(np.float32)
            query = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
            signed_dist = scene.compute_signed_distance(query).numpy().flatten()
            distances = np.abs(signed_dist).astype(np.float64)
        
    except (ImportError, AttributeError, RuntimeError) as e:
        # Open3Dが利用できない、またはエラーが発生した場合はtrimeshにフォールバック
        if show_progress:
            print(f"  Open3Dが利用できないため、trimeshを使用します: {e}")
        
        # 一度だけメッシュへの最近傍距離を計算
        if show_progress and P.shape[0] > 50000:
            batch_size = min(100000, P.shape[0] // 10)  # 10バッチ程度に分割、最大100,000点
            distances_list = []
            with tqdm(total=P.shape[0], desc="点群→メッシュ距離を計算中 (trimesh)", unit="点", miniters=1000) as pbar:
                for i in range(0, P.shape[0], batch_size):
                    batch = P[i:i + batch_size]
                    _, batch_distances, _ = mesh.nearest.on_surface(batch)
                    distances_list.append(batch_distances)
                    pbar.update(len(batch))
                    pbar.refresh()  # 強制的に更新
            distances = np.concatenate(distances_list)
        else:
            _, distances, _ = mesh.nearest.on_surface(P)
        
        distances = np.asarray(distances, dtype=np.float64)
    
    # 平均距離
    d_pm = float(distances.mean())
    
    # 外れ値割合
    num_outliers = np.count_nonzero(distances > threshold)
    r_out = num_outliers / float(P.shape[0])
    
    if return_distances:
        return d_pm, float(r_out), distances
    else:
            return d_pm, float(r_out)


def visualize_point_cloud_with_distances(
    points: ArrayNx3,
    distances: np.ndarray,
    gt_points: Optional[ArrayNx3] = None,
    title: str = "点群の距離可視化",
) -> None:
    """
    距離に応じて色分けした点群を可視化する。
    推定点群と真値点群の両方を表示する。

    Parameters
    ----------
    points : (N, 3) ndarray
        推定点群の座標 [m]。
    distances : (N,) ndarray
        各点のメッシュへの距離 [m]。
    gt_points : (M, 3) ndarray, optional
        真値点群の座標 [m]（表示する場合）。
    title : str, optional
        ウィンドウタイトル。
    """
    try:
        import open3d as o3d
        
        points = np.asarray(points, dtype=np.float64)
        distances = np.asarray(distances, dtype=np.float64)
        
        if points.shape[0] != distances.shape[0]:
            raise ValueError("点群と距離の数が一致しません。")
        
        # 距離に応じて色を設定（ヒートマップ: 青→緑→黄→赤）
        # 距離が小さい（良い）→ 青、距離が大きい（悪い）→ 赤
        dist_min = distances.min()
        dist_max = distances.max()
        dist_range = dist_max - dist_min
        
        if dist_range < 1e-10:
            # 距離がほぼ一定の場合、すべて青にする
            colors = np.tile([0.0, 0.0, 1.0], (len(points), 1))
        else:
            # 正規化（0〜1）
            normalized = (distances - dist_min) / dist_range
            
            # カラーマップ: 青(0.0) → シアン(0.33) → 緑(0.5) → 黄(0.67) → 赤(1.0)
            colors = np.zeros((len(points), 3))
            # 青からシアンへ
            mask1 = normalized < 0.33
            t1 = normalized[mask1] / 0.33
            colors[mask1, 0] = 0.0
            colors[mask1, 1] = t1
            colors[mask1, 2] = 1.0
            
            # シアンから緑へ
            mask2 = (normalized >= 0.33) & (normalized < 0.5)
            t2 = (normalized[mask2] - 0.33) / 0.17
            colors[mask2, 0] = 0.0
            colors[mask2, 1] = 1.0
            colors[mask2, 2] = 1.0 - t2
            
            # 緑から黄へ
            mask3 = (normalized >= 0.5) & (normalized < 0.67)
            t3 = (normalized[mask3] - 0.5) / 0.17
            colors[mask3, 0] = t3
            colors[mask3, 1] = 1.0
            colors[mask3, 2] = 0.0
            
            # 黄から赤へ
            mask4 = normalized >= 0.67
            t4 = (normalized[mask4] - 0.67) / 0.33
            colors[mask4, 0] = 1.0
            colors[mask4, 1] = 1.0 - t4
            colors[mask4, 2] = 0.0
        
        # Open3Dの推定点群を作成（距離に応じて色分け）
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(points)
        pcd_pred.colors = o3d.utility.Vector3dVector(colors)
        
        # 可視化するオブジェクトのリスト
        geometries = [pcd_pred]
        
        # 真値点群も表示する場合
        if gt_points is not None:
            try:
                gt_points_array = np.asarray(gt_points, dtype=np.float64)
                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(gt_points_array)
                # 真値点群は白で表示
                pcd_gt.paint_uniform_color([1.0, 1.0, 1.0])
                geometries.append(pcd_gt)
            except Exception:
                pass  # 真値点群の変換に失敗した場合はスキップ
        
        # 可視化
        print(f"\n可視化ウィンドウを表示しています...")
        print(f"  推定点群: 距離に応じて色分け（青（距離小）→ 緑 → 黄 → 赤（距離大））")
        if gt_points is not None:
            print(f"  真値点群: 白色で表示")
        print(f"  距離範囲: {dist_min:.4f} 〜 {dist_max:.4f} [m]")
        print(f"  ウィンドウを閉じると処理が続行されます。")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=title,
            width=1280,
            height=720,
            mesh_show_back_face=True,
        )
        
    except ImportError:
        print("  Open3Dが利用できないため、可視化をスキップします。")
    except Exception as e:
        print(f"  可視化中にエラーが発生しました: {e}")


def outlier_ratio(P: ArrayNx3, mesh: trimesh.Trimesh, threshold: float = 0.1, show_progress: bool = True) -> float:
    """
    点群 P のうち、真値メッシュ M からの距離が閾値 T を超える外れ値の割合 r を計算する。

    Eq. (outlier_ratio):
        r = | { p in P | min_{x in M} ||p - x|| > T } | / |P|

    Parameters
    ----------
    P : (N, 3) ndarray
        再構成点群 P [m]。
    mesh : trimesh.Trimesh
        クロップ後の真値メッシュ M。
    threshold : float, optional
        外れ値判定の閾値 T [m]。
    show_progress : bool, optional
        プログレスバーを表示するかどうか。

    Returns
    -------
    r : float
        外れ値割合（0〜1）。
    """
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] == 0:
        raise ValueError("P は1点以上を含む必要があります。")

    # point_to_mesh_distanceと同じバッチ処理を使用
    # 大量の点の場合は大きなバッチサイズを使用
    if show_progress and P.shape[0] > 50000:
        batch_size = min(100000, P.shape[0] // 10)  # 10バッチ程度に分割、最大100,000点
        distances_list = []
        with tqdm(total=P.shape[0], desc="外れ値割合を計算中", unit="点", miniters=1000) as pbar:
            for i in range(0, P.shape[0], batch_size):
                batch = P[i:i + batch_size]
                _, batch_distances, _ = mesh.nearest.on_surface(batch)
                distances_list.append(batch_distances)
                pbar.update(len(batch))
                pbar.refresh()  # 強制的に更新
        distances = np.concatenate(distances_list)
    else:
        _, distances, _ = mesh.nearest.on_surface(P)
    
    distances = np.asarray(distances, dtype=np.float64)
    num_outliers = np.count_nonzero(distances > threshold)
    r = num_outliers / float(P.shape[0])
    return float(r)


def evaluate_point_cloud(
    pred_points: ArrayNx3,
    gt_mesh: Union[str, pathlib.Path, trimesh.Trimesh, None] = None,
    gt_points: Optional[ArrayNx3] = None,
    *,
    gt_vertices: Optional[ArrayNx3] = None,
    gt_faces: Optional[np.ndarray] = None,
    threshold: float = 0.1,
    random_state: Optional[int] = 0,
) -> Dict[str, float]:
    """
    再構成点群と真値メッシュ（および真値点群）に対する総合評価関数。

    内部で以下を行う：
      1. 真値メッシュの読み込み
      2. 真値点群 Q の生成（gt_points がない場合はメッシュからサンプリング）
      3. BCE, 点群→メッシュ距離, 外れ値割合の計算

    Parameters
    ----------
    pred_points : (N, 3) ndarray
        再構成点群 P [m]。
    gt_mesh : str | pathlib.Path | trimesh.Trimesh | None
        真値メッシュ M。ファイルパスまたは Trimesh インスタンス。
        None の場合は gt_vertices / gt_faces から構築。
    gt_points : (N, 3) ndarray, optional
        既に用意されている真値点群 Q [m]。
        None の場合は真値メッシュから一様サンプリングで生成。
    gt_vertices : (V, 3) ndarray, optional
        メモリ上の真値メッシュ頂点座標 [m]。
    gt_faces : (F, 3) ndarray[int], optional
        メモリ上の真値メッシュ面インデックス。
    threshold : float, optional
        外れ値割合 r の閾値 T [m]。
    random_state : int, optional
        真値点群 Q サンプリングの乱数シード。

    Returns
    -------
    metrics : dict
        {
            "bce": float,            # Bidirectional Consistency Error [m]
            "point_to_mesh": float,  # 平均点群→メッシュ距離 [m]
            "outlier_ratio": float,  # 外れ値割合 [0〜1]
            "gt_points": ndarray,   # サンプリングされた真値点群 Q (オプション)
            "distances": ndarray,   # 各点のメッシュへの距離 [m] (可視化用)
        }
    """
    P = np.asarray(pred_points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("`pred_points` は形状 (N, 3) の配列である必要があります。")

    # 1. 真値メッシュを構築
    print("真値メッシュを読み込んでいます...")
    mesh = load_mesh(mesh=gt_mesh, vertices=gt_vertices, faces=gt_faces)
    print(f"  メッシュ頂点数: {len(mesh.vertices)}, 面数: {len(mesh.faces)}")

    # 2. 真値点群 Q を準備
    if gt_points is not None:
        Q = np.asarray(gt_points, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[1] != 3:
            raise ValueError("`gt_points` は形状 (N, 3) の配列である必要があります。")
        print(f"真値点群を読み込みました: {Q.shape[0]}点")
    else:
            Q = sample_points_from_mesh(mesh, num_points=P.shape[0]*10, random_state=random_state, show_progress=True)

    # 4. 各種指標を計算
    print("\n評価指標を計算しています...")
    bce = bidirectional_consistency_error(P, Q, show_progress=True)
    
    # point_to_mesh_distanceとoutlier_ratioは同じ計算を使用するため、一度だけ計算して再利用
    print("点群→メッシュ距離を計算中（外れ値割合も同時に計算）...")
    # メッシュファイルパスを渡すことで、Open3Dで直接読み込んで変換をスキップ
    mesh_file_path = gt_mesh if isinstance(gt_mesh, (str, pathlib.Path)) else None
    d_pm, r_out, distances = _compute_point_to_mesh_and_outlier_ratio(
        P, mesh, threshold, show_progress=True, mesh_file_path=mesh_file_path, return_distances=True
    )

    result = {
        "bce": bce,
        "point_to_mesh": d_pm,
        "outlier_ratio": r_out,
    }
    
    # サンプリングされた真値点群も返す（保存用）
    result["gt_points"] = Q
    # 距離情報も返す（可視化用）
    result["distances"] = distances
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="点群と真値メッシュの評価を実行します。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python evaluation/pointcloud_evaluation.py \\
    --pred_pointcloud output/height_30_radius_15/point_cloud/output.ply \\
    --gt_mesh data/ground_truth_mesh.ply \\
    --threshold 0.1
        """,
    )
    parser.add_argument(
        "--pred_pointcloud",
        type=str,
        required=True,
        help="推定点群ファイルのパス",
    )
    parser.add_argument(
        "--gt_mesh",
        type=str,
        required=True,
        help="真値メッシュファイルのパス",
    )
    parser.add_argument(
        "--gt_points",
        type=str,
        default=None,
        help="真値点群ファイルのパス",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="外れ値判定の閾値 [m] (デフォルト: 0.1)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="真値点群サンプリングの乱数シード (デフォルト: 0)",
    )
    parser.add_argument(
        "--save_gt_points",
        type=str,
        default=None,
        help="サンプリングされた真値点群を保存するパス（PLY形式）",
    )
    parser.add_argument(
        "--flip_y_axis",
        action="store_true",
        help="推定点群のY軸を反転する（座標系の変換用）",
    )
    parser.add_argument(
        "--save_pred_pointcloud",
        type=str,
        default=None,
        help="推定点群を保存するパス（PLY形式）。--flip_y_axisが指定されている場合は反転後の点群を保存",
    )

    args = parser.parse_args()

    # 推定点群を読み込む
    print(f"推定点群を読み込んでいます: {args.pred_pointcloud}")
    pred_points = load_point_cloud(args.pred_pointcloud)
    if args.flip_y_axis:
        pred_points[:, 1] *= -1.0
        print("  Y軸を反転しました")
    print(f"  読み込んだ点数: {pred_points.shape[0]}")
    
    # 推定点群を保存（指定されている場合）
    if args.save_pred_pointcloud is not None:
        save_point_cloud(pred_points, args.save_pred_pointcloud)

    # 真値点群を読み込む（指定されている場合）
    gt_points = None
    if args.gt_points is not None:
        print(f"真値点群を読み込んでいます: {args.gt_points}")
        gt_points = load_point_cloud(args.gt_points)
        print(f"  読み込んだ点数: {gt_points.shape[0]}")

    # 評価を実行
    print(f"真値メッシュを読み込んでいます: {args.gt_mesh}")
    metrics = evaluate_point_cloud(
        pred_points=pred_points,
        gt_mesh=args.gt_mesh,
        gt_points=gt_points,
        threshold=args.threshold,
        random_state=args.random_state,
    )

    # サンプリングされた真値点群を保存（指定されている場合）
    if args.save_gt_points is not None and "gt_points" in metrics:
        save_point_cloud(metrics["gt_points"], args.save_gt_points)

    # 結果を表示
    print("\n評価結果:")
    print("=" * 50)
    for k, v in metrics.items():
        if k not in ("gt_points", "distances"):  # 点群データと距離データは表示しない
            print(f"  {k:20s}: {v:.6f}")
    print("=" * 50)
    
    # 距離に応じて色分けした点群を可視化（デフォルトで実行）
    if "distances" in metrics and "gt_points" in metrics:
        # 推定点群と真値点群の両方を表示
        visualize_point_cloud_with_distances(
            pred_points,
            metrics["distances"],
            gt_points=metrics["gt_points"],
            title="推定点群と真値点群の比較",
        )


