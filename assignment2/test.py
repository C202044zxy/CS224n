import numpy as np

# 示例数据
gradOutsideVecs = np.array([1, 2, 3, 4, 5])
indices = np.array([0, 1, 3, 1])

# 要添加的梯度
grad = 2

for i in gradOutsideVecs:
    print(i, id(i)) 
print("--------")
for i in gradOutsideVecs[indices]:
    print(id(i))

gradOutsideVecs[indices] += grad
