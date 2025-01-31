import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata

# 定义CSV文件列表
files = ['data/ncf.csv', 'data/resnet.csv', 'data/yolo.csv', 'data/bert.csv', 'data/opt.csv']
z_columns = {
    'health': "AUPC",
}  # z轴的列名
model_names = ['(a) NeuMF', '(b) ResNet18', '(c) YoloV3', '(d) Bert', '(e) OPT-1.3B']  # 模型名称

# 创建3行5列的子图布局
fig, axes = plt.subplots(len(z_columns), 5, figsize=(25, 4.5), subplot_kw={'projection': '3d'})

fontsize = 15


def sub_plot(x, y, z, ax, x_label='Learning rate', y_label='Batch size', log_scale=True):
    # # 创建网格数据用于插值
    # grid_x, grid_y = np.meshgrid(
    #     np.linspace(x.min(), x.max(), 50),
    #     np.linspace(y.min(), y.max(), 50)
    # )
    # grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')  # 插值
    #
    # # 创建3D曲面图
    # ax = axes[j, i]
    # surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='Spectral_r', edgecolor='none', alpha=1)  # 使用反转的颜色映射
    #
    # # 绘制xy平面的热力图投影（使用反转的颜色映射）
    # heatmap = ax.contourf(grid_x, grid_y, grid_z, zdir='z', offset=z.min() - 1e-3, cmap='Spectral_r', alpha=1)

    # 使用三角剖分绘制3D曲面
    triang = tri.Triangulation(x, y)
    surf = ax.plot_trisurf(triang, z, cmap='Spectral_r', alpha=0.7)

    # 在 xy 平面投影颜色
    ax.tricontourf(triang, z, cmap='Spectral_r', alpha=1, zdir='z', offset=z.min())

    # 设置坐标轴
    ax.set_xlabel(x_label, fontsize=fontsize + 2, labelpad=10)
    ax.set_ylabel(y_label, fontsize=fontsize + 2, labelpad=10)
    # ax.set_zlabel(f'{z_col}', fontsize=fontsize)

    # 设置坐标轴刻度为原始数值
    if log_scale:
        # 自定义格式化函数
        def format_10log(val):
            # 使用 .1e 格式化科学计数法
            formatted = f'{10 ** val:.1e}'
            # 去掉指数部分的补零
            base, exponent = formatted.split('e')
            exponent = exponent.replace('-0', '-')  # 去掉多余的零
            exponent = exponent.replace('+0', '-')  # 去掉多余的零
            return f'{base}e{exponent}'

        ax.set_xticks(np.linspace(x.min(), x.max(), 4))
        ax.set_xticklabels([format_10log(val) for val in np.linspace(x.min(), x.max(), 4)], fontsize=fontsize)
        ax.set_yticks(np.linspace(y.min(), y.max(), 4))
        ax.set_yticklabels([f'{int(2 ** val)}' for val in np.linspace(y.min(), y.max(), 4)], fontsize=fontsize)
        ax.set_zticks(np.linspace(z.min(), z.max(), 4))
        ax.set_zticklabels([f'{val:.1f}' for val in np.linspace(z.min(), z.max(), 4)], fontsize=fontsize)
    else:
        ax.set_xticks(np.linspace(x.min(), x.max(), 4))
        ax.set_xticklabels([f'{val:.1f}' for val in np.linspace(x.min(), x.max(), 4)], fontsize=fontsize)
        ax.set_yticks(np.linspace(y.min(), y.max(), 4))
        ax.set_yticklabels([f'{val:.1f}' for val in np.linspace(y.min(), y.max(), 4)], fontsize=fontsize)
        ax.set_zticks(np.linspace(z.min(), z.max(), 4))
        ax.set_zticklabels([f'{val:.1f}' for val in np.linspace(z.min(), z.max(), 4)], fontsize=fontsize)

    ax.invert_xaxis()  # 翻转x轴
    ax.zaxis.set_ticks_position('default')  # 将z轴刻度移动到左侧

    # 颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12)
    cbar.set_label(f'{z_columns[z_col]}', fontsize=fontsize, labelpad=10, rotation=0)
    cbar.ax.yaxis.set_label_coords(0.5, 1.1)  # (x, y) 坐标，y=1.1 表示在colorbar上方
    cbar.ax.tick_params(labelsize=fontsize)  # 调整刻度字体大小


for j, z_col in enumerate(z_columns):  # 遍历每个z轴列
    for i, file in enumerate(files):  # 遍历每个文件
        data = pd.read_csv(file)  # 读取CSV文件
        ax = axes[i]

        # 提取x, y, z数据
        x = data['GRT']
        y = data['GRS']
        z = data['health']  # z轴原始数据
        sub_plot(x, y, z, ax, x_label='GR$_T$', y_label='GR$_S$', log_scale=False)

# 在每一列下方标注模型名称
for i, model_name in enumerate(model_names):
    fig.text(
        (i + 0.5) / len(files),  # x坐标（居中）
        0.01,  # y坐标（底部）
        model_name,  # 模型名称
        ha='center',  # 水平居中
        va='bottom',  # 垂直对齐底部
        fontsize=25,  # 字体大小
        family='Times New Roman',  # 字体
        weight='bold'  # 设置字体为粗体
    )

# 手动调整子图间距
plt.subplots_adjust(left=0.005, right=0.995, bottom=0.12, top=1.08, wspace=0., hspace=-0.15)
plt.savefig('figure/heath.pdf', dpi=300)
plt.show()
