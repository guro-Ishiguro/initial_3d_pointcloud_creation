import pathlib
from typing import Dict, Optional, Union, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree


ArrayNx3 = np.ndarray


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
        sampled = mesh.sample(num_points)
    finally:
        np.random.set_state(state)

    return np.asarray(sampled, dtype=np.float64)


def chamfer_distance(P: ArrayNx3, Q: ArrayNx3) -> float:
    """
    Chamfer Distance (CD) を計算する。

    Eq. (chamfer_distance):
        d_Chamfer = (1/|P|) sum_{p in P} min_{q in Q} ||p - q||
                  + (1/|Q|) sum_{q in Q} min_{p in P} ||q - p||

    Parameters
    ----------
    P : (N, 3) ndarray
        再構成点群 P [m]。
    Q : (N, 3) ndarray
        真値点群 Q [m]。

    Returns
    -------
    d : float
        Chamfer Distance [m]。
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    if P.shape[0] == 0 or Q.shape[0] == 0:
        raise ValueError("P と Q は共に1点以上を含む必要があります。")

    # P -> Q 最近傍距離
    tree_Q = cKDTree(Q)
    dists_P_to_Q, _ = tree_Q.query(P, k=1)

    # Q -> P 最近傍距離
    tree_P = cKDTree(P)
    dists_Q_to_P, _ = tree_P.query(Q, k=1)

    term1 = dists_P_to_Q.mean()
    term2 = dists_Q_to_P.mean()

    return float(term1 + term2)


def bidirectional_consistency_error(P: ArrayNx3, Q: ArrayNx3) -> float:
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
    _, idx_Q = tree_Q.query(P, k=1)
    nn_Q = Q[idx_Q]  # shape: (N, 3)

    # さらに NN_Q(p) に対して NN_P(NN_Q(p)) を求める
    _, idx_P_back = tree_P.query(nn_Q, k=1)
    nn_P_back = P[idx_P_back]  # shape: (N, 3)

    # p と NN_P(NN_Q(p)) の距離
    diff = P - nn_P_back
    dists = np.linalg.norm(diff, axis=1)

    return float(dists.mean())


def point_to_mesh_distance(P: ArrayNx3, mesh: trimesh.Trimesh) -> float:
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

    Returns
    -------
    d : float
        平均点群→メッシュ距離 [m]。
    """
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] == 0:
        raise ValueError("P は1点以上を含む必要があります。")

    # trimesh の最近傍探索で点からメッシュ表面への最短距離を計算
    closest_points, distances, _ = mesh.nearest.on_surface(P)
    # `distances` は各点に対応する距離
    return float(np.asarray(distances, dtype=np.float64).mean())


def outlier_ratio(P: ArrayNx3, mesh: trimesh.Trimesh, threshold: float = 0.1) -> float:
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

    Returns
    -------
    r : float
        外れ値割合（0〜1）。
    """
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] == 0:
        raise ValueError("P は1点以上を含む必要があります。")

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
    mesh_margin: float = 0.0,
    random_state: Optional[int] = 0,
) -> Dict[str, float]:
    """
    再構成点群と真値メッシュ（および真値点群）に対する総合評価関数。

    内部で以下を行う：
      1. メッシュのクロップ（P のバウンディングボックス + マージン）
      2. 真値点群 Q の生成（gt_points がない場合はメッシュからサンプリング）
      3. Chamfer Distance, BCE, 点群→メッシュ距離, 外れ値割合の計算

    Parameters
    ----------
    pred_points : (N, 3) ndarray
        再構成点群 P [m]。
    gt_mesh : str | pathlib.Path | trimesh.Trimesh | None
        真値メッシュ M。ファイルパスまたは Trimesh インスタンス。
        None の場合は gt_vertices / gt_faces から構築。
    gt_points : (N, 3) ndarray, optional
        既に用意されている真値点群 Q [m]。
        None の場合はクロップ済みメッシュから一様サンプリングで生成。
    gt_vertices : (V, 3) ndarray, optional
        メモリ上の真値メッシュ頂点座標 [m]。
    gt_faces : (F, 3) ndarray[int], optional
        メモリ上の真値メッシュ面インデックス。
    threshold : float, optional
        外れ値割合 r の閾値 T [m]。
    mesh_margin : float, optional
        P のバウンディングボックスに与えるマージン [m]。
    random_state : int, optional
        真値点群 Q サンプリングの乱数シード。

    Returns
    -------
    metrics : dict
        {
            "chamfer": float,        # Chamfer Distance [m]
            "bce": float,            # Bidirectional Consistency Error [m]
            "point_to_mesh": float,  # 平均点群→メッシュ距離 [m]
            "outlier_ratio": float,  # 外れ値割合 [0〜1]
        }
    """
    P = np.asarray(pred_points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("`pred_points` は形状 (N, 3) の配列である必要があります。")

    # 1. 真値メッシュを構築
    mesh = load_mesh(mesh=gt_mesh, vertices=gt_vertices, faces=gt_faces)

    # 2. 再構成点群 P のバウンディングボックスでメッシュをクロップ
    cropped_mesh = crop_mesh_with_bbox(mesh, P, margin=mesh_margin)

    # 3. 真値点群 Q を準備
    if gt_points is not None:
        Q = np.asarray(gt_points, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[1] != 3:
            raise ValueError("`gt_points` は形状 (N, 3) の配列である必要があります。")
    else:
        Q = sample_points_from_mesh(cropped_mesh, num_points=P.shape[0], random_state=random_state)

    # 4. 各種指標を計算
    cd = chamfer_distance(P, Q)
    bce = bidirectional_consistency_error(P, Q)
    d_pm = point_to_mesh_distance(P, cropped_mesh)
    r_out = outlier_ratio(P, cropped_mesh, threshold=threshold)

    return {
        "chamfer": cd,
        "bce": bce,
        "point_to_mesh": d_pm,
        "outlier_ratio": r_out,
    }


if __name__ == "__main__":
    # 簡単な使用例（ダミーデータ）
    # ここでは単純な平面メッシュと、それに近い点群を評価する。
    # 実際には PLY/OBJ からの読み込みや実データを使用する想定。

    # 正方形平面メッシュ (z=0)
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int64,
    )

    # 真値メッシュから真値点群 Q をサンプリング
    mesh_gt = load_mesh(vertices=vertices, faces=faces)
    Q_true = sample_points_from_mesh(mesh_gt, num_points=1000, random_state=42)

    # Q_true を少しだけ z 方向にずらして再構成点群 P を生成
    P_pred = Q_true + np.array([0.0, 0.0, 0.01], dtype=np.float64)

    metrics = evaluate_point_cloud(
        pred_points=P_pred,
        gt_mesh=mesh_gt,
        gt_points=None,  # 内部でメッシュからサンプリング
        threshold=0.1,
        mesh_margin=0.0,
        random_state=42,
    )

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


