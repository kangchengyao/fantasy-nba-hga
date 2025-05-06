import numpy as np
import matplotlib.pyplot as plt

# 設定論文格式字體與大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# === 讀取資料 ===
last_year_normal = np.load('normal_last_year.npy')
last_year_chaos = np.load('chaos_last_year.npy')

this_mid_normal = np.load('normal.npy')
this_mid_chaos = np.load('chaos.npy')

this_end_normal = np.load('normal_this_end.npy')
this_end_chaos = np.load('chaos_this_end.npy')


# === 圖片 1：去年賽季 ===
plt.figure(figsize=(8, 5))
plt.plot(range(len(last_year_normal)), last_year_normal, label='Normal GA', color='blue', marker='o', linestyle='-')
plt.plot(range(len(last_year_chaos)), last_year_chaos, label='Chaos GA', color='red', marker='x', linestyle='--')
plt.title("Convergence Curve – Last Season")
plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("convergence_last_season.jpg", dpi=300, bbox_inches='tight')
plt.close()


# === 圖片 2：今年季中 ===
plt.figure(figsize=(8, 5))
plt.plot(range(len(this_mid_normal)), this_mid_normal, label='Normal GA', color='blue', marker='o', linestyle='-')
plt.plot(range(len(this_mid_chaos)), this_mid_chaos, label='Chaos GA', color='red', marker='x', linestyle='--')
plt.title("Convergence Curve – This Season Mid")
plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("convergence_this_season_mid.jpg", dpi=300, bbox_inches='tight')
plt.close()


# === 圖片 3：今年季末 ===
plt.figure(figsize=(8, 5))
plt.plot(range(len(this_end_normal)), this_end_normal, label='Normal GA', color='blue', marker='o', linestyle='-')
plt.plot(range(len(this_end_chaos)), this_end_chaos, label='Chaos GA', color='red', marker='x', linestyle='--')
plt.title("Convergence Curve – This Season End")
plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("convergence_this_season_end.jpg", dpi=300, bbox_inches='tight')
plt.close()
