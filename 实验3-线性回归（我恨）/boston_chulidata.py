import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('boston_house_prices.csv')

# 缺失值分析
missing_values = data.isna().sum()
print("缺失值统计:\n", missing_values)

# 绘制箱线图以检测异常值
def boxplot(data, fig_name):
    """绘制每个属性的箱线图，帮助识别异常值"""
    num_plots = len(data.columns)
    cols = 3  # 每行显示3个子图
    rows = (num_plots + cols - 1) // cols  # 计算需要的行数

    plt.figure(figsize=(20, 5 * rows), dpi=300)  # 调整图形大小
    plt.subplots_adjust(wspace=0.4, hspace=0.6)  # 调整子图间距

    for i, col_name in enumerate(data.columns):
        plt.subplot(rows, cols, i + 1)
        plt.boxplot(data[col_name],
                    showmeans=True,
                    meanprops={"markersize": 1, "marker": "D", "markeredgecolor": "#C54680"},
                    medianprops={"color": "#946279"},
                    whiskerprops={"color": "#8E004D", "linewidth": 0.4, 'linestyle': "--"},
                    flierprops={"markersize": 0.4})
        plt.title(col_name, fontdict={"size": 10}, pad=2)
        plt.xticks([])

    plt.savefig(fig_name)
    plt.show()

# 绘制初始箱线图
boxplot(data, 'initial_boxplot.pdf')

# 四分位数异常值处理
num_features = data.select_dtypes(exclude=['object', 'bool']).columns.tolist()
for feature in num_features:
    if feature == 'CHAS':  # CHAS 特征不处理
        continue
    Q1 = data[feature].quantile(0.25)  # 下四分位
    Q3 = data[feature].quantile(0.75)  # 上四分位
    IQR = Q3 - Q1  # 四分位距
    # 计算上下限
    upper_limit = Q3 + 1.5 * IQR  # 上限
    lower_limit = Q1 - 1.5 * IQR  # 下限
    # 替换异常值
    data[feature] = data[feature].clip(lower=lower_limit, upper=upper_limit)

# 再次绘制箱线图，查看异常值处理效果
boxplot(data, 'final_boxplot.pdf')
