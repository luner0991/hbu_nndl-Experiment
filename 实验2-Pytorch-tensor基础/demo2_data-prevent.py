import pandas as pd
import numpy as np
import torch
'''
# 读取CSV文件
data = pd.read_csv('house_tiny.csv')
print("原始数据\n", data)

# ====== 处理缺失值
# 只对数值列填充缺失值为均值
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(value=data[numeric_cols].mean())

# 对于非数值列（如字符串），填充为 'Unknown'
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
data[non_numeric_cols] = data[non_numeric_cols].fillna(value='Unknown')

print("处理缺失值后\n", data)

# ======== 处理布尔值，将 True/False 转换为 1/0
data = data.applymap(lambda x: 1 if x is True else (0 if x is False else x))

# ======== 处理离散值
# 使用 get_dummies 对离散值进行独热编码
data_encoded = pd.get_dummies(data)
# 替换掉 True/False，确保所有布尔类型都转为 0/1
data_encoded = data_encoded.replace({True: 1, False: 0})  # 重新赋值给 data_encoded
print("处理离散值后\n", data_encoded)

# ======== 转换为张量形式
data_array = np.array(data_encoded)  # 转换为 NumPy 数组
data_tensor = torch.tensor(data_array)  # 转换为张量，指定类型为 float32
print("转换为张量后\n", data_tensor)
#==========================================================
data = pd.read_csv('boston_house_prices.csv')
print("数据集展示\n",data)

# 检查DataFrame中的缺失值
# missing_values = data.isna()  # 或者 df.isnull()
missing_count = data.isna().sum()  # 或者 df.isnull().sum()
print("缺失值数量统计\n",missing_count)
# 转换为张量形式
data = np.array(data)
data = torch.tensor(data)
print("转换为张量形式\n",data)'''
#==================================================
data = pd.read_csv('Iris.csv')
print("数据集展示\n",data)

# 检查DataFrame中的缺失值
# missing_values = data.isna()  # 或者 df.isnull()
missing_count = data.isna().sum()  # 或者 df.isnull().sum()
print("缺失值数量统计\n",missing_count)

data['Species'], _ = pd.factorize(data['Species'])
# 转换为张量形式
data = np.array(data)
data = torch.tensor(data)
print("转换为张量形式\n",data)