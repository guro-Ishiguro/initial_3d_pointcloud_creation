import numpy as np
import matplotlib.pyplot as plt


# 距離分解能を計算する関数
def calc_distance_resolution(z, c, B, S, W):
    dz = z**2 * S / (B * c * W)
    return dz


# パラメータ
B = 0.6  # ベースライン (m)
c = 3.222654 * 10 ** (-3)  # 焦点距離 (m)
S = 6.3 * 10 ** (-3)  # センサーサイズ (m)
W = 3840  # 画素数 (ピクセル)
z_values = np.linspace(0.1, 20, 100)  # z=0から12までの距離

# 距離分解能を計算
dz_values = calc_distance_resolution(z_values, c, B, S, W)

# プロット
plt.figure(figsize=(8, 6))
plt.plot(z_values, dz_values, label="Distance Resolution (dz)", color="blue")
plt.title("Distance Resolution vs. Distance (z)", fontsize=14)
plt.xlabel("Distance (z) [m]", fontsize=12)
plt.ylabel("Distance Resolution (dz) [m]", fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
output_path = "distance_resolution_plot.png"
plt.savefig(output_path)
print(f"Plot saved as {output_path}")
