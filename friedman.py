import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import scipy.stats as stats


def rank_matrix(matrix):
    cnum = matrix.shape[1]
    rnum = matrix.shape[0]
    ## 升序排序索引
    sorts = np.argsort(matrix)
    # sorts = matrix.argsort()[:, ::-1]##降序排序索引用于r2
    for i in range(rnum):
        k = 1
        n = 0
        flag = False
        nsum = 0
        for j in range(cnum):
            n = n+1
            ## 相同排名评分序值
            if j < cnum-1 and matrix[i, sorts[i,j]] == matrix[i, sorts[i,j + 1]]:
                flag = True;
                k = k + 1;
                nsum += j + 1;
            elif (j == cnum-1 or (j < cnum-1 and matrix[i, sorts[i,j]] != matrix[i, sorts[i,j + 1]])) and flag:
                nsum += j + 1
                flag = False;
                for q in range(k):
                    matrix[i,sorts[i,j - k + q + 1]] = nsum / k
                k = 1
                flag = False
                nsum = 0
            else:
                matrix[i, sorts[i,j]] = j + 1
                continue
    matrixtt = matrix
    return matrixtt

def friedman(n, k, rank_matrix):
    # 计算每一列的排序和
    sumr = sum(list(map(lambda x: np.mean(x) ** 2, rank_matrix.T)))
    result = 12 * n / (k * ( k + 1)) * (sumr - k * (k + 1) ** 2 / 4)
    result = (n - 1) * result /(n * (k - 1) - result)
    return result

def nemenyi(n, k, q):
    return q * (np.sqrt(k * (k + 1) / (6 * n)))

# 目标检测算法在3个数据集上的AP值
# 每行代表一个数据集，每列代表一个算法
visdrone_ap = [0.45, 0.52, 0.41, 0.48, 0.39]  # visdrone数据集上各算法的AP
dota_ap =     [0.38, 0.43, 0.35, 0.40, 0.33]  # dota数据集上各算法的AP  
aitod_ap =    [0.51, 0.55, 0.47, 0.53, 0.44]  # aitod数据集上各算法的AP

# 构建数据矩阵：3个数据集 x 5个算法
load = np.array([visdrone_ap, dota_ap, aitod_ap])

print("原始AP值矩阵:")
print("行: 数据集 (visdrone, dota, aitod)")
print("列: 算法")
print(load)

# 复制一份用于排序（因为rank_matrix会修改原矩阵）
matrix1 = load.copy()
tt = matrix1.copy()

# 执行排序
matrix_r = rank_matrix(tt)
print("\n排序矩阵:")
print(matrix_r)

# 执行Friedman检验
# 参数：n=3(数据集数), k=5(算法数), 排序矩阵
n_datasets = 3
n_algorithms = 5
Friedman_stat = friedman(n_datasets, n_algorithms, matrix_r)
print(f"\nFriedman统计量: {Friedman_stat:.4f}")

# 计算p值
df1 = n_algorithms - 1  
df2 = (n_algorithms - 1) * (n_datasets - 1)  
p_value = 1 - stats.f.cdf(Friedman_stat, df1, df2)

print(f"p值: {p_value:.2e}")
print(f"自由度: F({df1}, {df2})")

# 显著性判断
alpha = 0.05
is_significant = p_value < alpha
print(f"显著性检验结果: {'显著' if is_significant else '不显著'} (α = {alpha})")

# 修改可视化部分，添加p值信息
plt.text(0.02, 0.90, f'p-value: {p_value:.2e}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.86, f'Significant: {"Yes" if is_significant else "No"}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

# 计算临界差值 (CD)
# 对于α=0.05, k=5, 查临界值表得q≈2.728（您可能需要根据实际情况调整）
q_value = 2.728  # 需要根据α=0.05, k=5查表确定
CD = nemenyi(n_datasets, n_algorithms, q_value)
print(f"临界差值 (CD): {CD:.4f}")

# 计算平均排名
rank_means = list(map(lambda x: np.mean(x), matrix_r.T))
print(f"\n各算法平均排名: {rank_means}")

# 算法名称
algorithm_names = ['YOLO', 'R-CNN', 'SSD', 'RetinaNet', 'FCOS']

# 绘制CD图
min_1 = [x - CD/2 for x in rank_means]
max_1 = [x + CD/2 for x in rank_means]

plt.figure(figsize=(12, 8))
# plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 600

plt.title("Friedman Test with AP Values - Object Detection Algorithms", fontsize=14)
plt.scatter(rank_means, algorithm_names, s=100, color='red')
plt.hlines(algorithm_names, min_1, max_1, colors='blue', linewidth=2)

# 添加CD线和标注
plt.xlabel('Average Rank', fontsize=12)
plt.ylabel('Detection Algorithms', fontsize=12)

# 添加文本说明
plt.text(0.02, 0.98, f'Friedman statistic: {Friedman_stat:.4f}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.94, f'Critical Difference (CD): {CD:.4f}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('AP_friedman_detection.jpg', bbox_inches='tight')
plt.show()

# 进行两两比较
print("\n两两算法比较结果:")
for i in range(n_algorithms):
    for j in range(i+1, n_algorithms):
        rank_diff = abs(rank_means[i] - rank_means[j])
        is_significant = rank_diff > CD
        print(f"{algorithm_names[i]} vs {algorithm_names[j]}: "
              f"排名差={rank_diff:.4f}, {'显著差异' if is_significant else '无显著差异'}")