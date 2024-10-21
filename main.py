import numpy as np
from scipy.optimize import minimize

# 定义网络中的节点数
n_nodes = 3

# 定义每个节点的局部目标函数（这里假设是简单的二次函数）
def local_objective(x, a, b, c):
    return a * x**2 + b * x + c

# 定义每个节点的梯度（对应的二次函数的梯度）
def local_gradient(x, a, b):
    return 2 * a * x + b

# 初始化每个节点的状态估计
x = np.random.rand(n_nodes)
print(f"Initial state estimates: {x}")

# 定义每个节点的局部目标函数参数
params = [
    (1, -2, 1),  # 节点1的参数 (a, b, c)
    (2, -3, 1),  # 节点2的参数 (a, b, c)
    (3, -4, 1)   # 节点3的参数 (a, b, c)
]

# 定义学习率和最大迭代次数
alpha = 0.1
max_iter = 100

# 定义网络的邻接矩阵（假设是一个完全连接的网络）
adj_matrix = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)

# 初始化噪声向量
s = np.zeros((n_nodes, n_nodes))

# RSS-NB算法
for k in range(max_iter):
    # 计算每个节点的梯度
    gradients = np.array([local_gradient(x[i], params[i][0], params[i][1]) for i in range(n_nodes)])
    
    # 生成随机噪声
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] == 1:
                s[i, j] = np.random.randn()
    
    # 计算每个节点的扰动
    d = np.sum(s, axis=1) - np.sum(s, axis=0)
    
    # 更新每个节点的状态估计
    for i in range(n_nodes):
        x[i] = x[i] - alpha * (gradients[i] + d[i])
    
    # 打印每次迭代后的状态估计
    if (k + 1) % 10 == 0:
        print(f"Iteration {k + 1}: {x}")

# 最终状态估计
print(f"Final state estimates: {x}")


# 使用scipy求解最优参数

def objective(x):
    return local_objective(x,params[0])+local_objective(x,params[1])+local_objective(x,params[2])

x0 = np.array([0.0])
result = minimize(objective, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
print(result.x)