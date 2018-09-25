import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 加载糖尿病数据
diabetes = datasets.load_diabetes()

# 列举一下特征名称
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# 年龄，性别，身体质量指数，血压，s1-s6是6中血清的化验数据
print(diabetes.feature_names)

# 选择bmi作为预测的指标（特征）
diabetes_X = diabetes.data[:, np.newaxis, 2]

# 除了最后20个都是训练集
diabetes_X_train = diabetes_X[:-20]
# 最后20个是测试集
diabetes_X_test = diabetes_X[-20:]

# 训练集的标注结果
diabetes_y_train = diabetes.target[:-20]
# 测试集的标注结果
diabetes_y_test = diabetes.target[-20:]

# 创建线性回归分类器
regr = linear_model.LinearRegression()
# 使用训练数据训练
regr.fit(diabetes_X_train, diabetes_y_train)
# 使用测试数据，让模型生成对应的结果
diabetes_y_pred = regr.predict(diabetes_X_test)

# 比较测试集的真实结果与模型的预测结果，使用均方误差作为评估标准
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# 画点（真实数据）
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# 画线（模型的直线）
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.show()