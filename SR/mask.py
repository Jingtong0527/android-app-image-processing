import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# 示例数据
np.random.seed(42)
x = np.random.rand(100)
y = np.random.rand(100)

# 特定几个点的索引
selected_indices = [5, 12, 35, 42, 68]

# 将数据组合成一个特征矩阵
points = np.column_stack((x, y))

# 通过凸包找到特定几个点围成的区域
selected_points = points[selected_indices]
hull = ConvexHull(selected_points)

# 绘制散点图
plt.scatter(x, y, c='black', alpha=0.8)

# 绘制特定几个点
plt.scatter(selected_points[:, 0], selected_points[:, 1], c='red', marker='x')

# 绘制凸包区域
for simplex in hull.simplices:
    plt.plot(selected_points[simplex, 0], selected_points[simplex, 1], 'k-')

# 将凸包区域填充为白色
plt.fill(selected_points[hull.vertices, 0], selected_points[hull.vertices, 1], color='white', alpha=0.5)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Selected Points Enclosed Area in White')
plt.show()
