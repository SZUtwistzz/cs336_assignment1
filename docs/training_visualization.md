# 训练曲线：运行方法与代码阅读指南

脚本是 `cs336_basics/train_LM/plot_training.py`。只读取日志，不加载模型，不重新训练。
依赖 Matplotlib 已写入项目配置和 uv.lock。

## 1. 运行

在 PowerShell 中执行（反引号表示同一条命令还没结束，其后不能有空格）：

```powershell
Set-Location E:/cs336/assignment1-basics
uv run --cache-dir E:/cs336/.uv-cache python -m cs336_basics.train_LM.plot_training `
  --run-dir E:/cs336/saves/runs/tinystories_20260918_080507_208128 `
  --title TinyStories `
  --target-loss 1.45 `
  --smooth-window 20
```

只需指定最后一段实验目录。脚本根据其中 config.json 的 resume 找到前一段实验。
以后换实验，修改 `--run-dir`；OWT 不适用 TinyStories 的目标线，可省略 `--target-loss`。
默认输出到指定实验目录下的 `plots/`，重复运行会更新这些可视化文件。
可用 `--out-dir E:/cs336/saves/figures/my_experiment` 自定义输出位置。

| 文件 | 内容 |
| --- | --- |
| training_overview.png | loss 对步数、loss 对累计时间、学习率，共三个子图 |
| training_overview.svg | 相同总览图的矢量版本，放大不会模糊 |
| validation_detail.png | 后半程验证 loss，便于观察小幅变化 |
| merged_metrics.csv | 拼接好时间的原始数据，不进行 loss 平滑 |
| summary.json | 最终与最佳验证 loss、最佳步数、累计记录时间 |

## 2. 从最小画图代码开始理解

```python
import matplotlib.pyplot as plt

steps = [100, 200, 300]
losses = [4.0, 3.0, 2.5]
fig, ax = plt.subplots()                 # fig 是整张图，ax 是其中的坐标轴
ax.plot(steps, losses, label="Train")    # (100,4.0)、(200,3.0)、(300,2.5)
ax.set(xlabel="Step", ylabel="Loss")    # 设置坐标轴文字
ax.legend()                             # 显示 label 对应的图例
fig.savefig("example.png", dpi=180)     # 保存图片
plt.close(fig)                          # 释放画图占用的资源
```

主脚本里的 `plt.subplots(1, 3)` 创建一行三个子图，因此可以通过
`axes[0]`、`axes[1]`、`axes[2]` 分别控制它们。

## 3. 按这个顺序阅读主脚本

### read_metrics：读取 CSV

`csv.DictReader(f)` 把一行变成字典，例如：

```python
{"step": "200", "loss": "3.2", "split": "train", ...}
```

CSV 读取到的值是字符串。`int(raw["step"])` 和 `float(raw["loss"])`
把它们转换成数值，否则无法可靠计算、排序或画数值坐标。
`with ... as f` 会在代码块结束时自动关闭文件。
缺列、空日志、非有限值和倒退的时间会明确报错，不会悄悄丢掉数据。

### load_history：接续恢复训练

假设旧日志第 15000 步累计耗时为 44890.97 秒，新日志第 15010 步耗时为
236.29 秒，则新图上的第 15010 步时间为 `44890.97 + 236.29` 秒。

同一步有 train、valid 两条记录时，采用较晚时间作为拼接点，以包含验证耗时。
若从更早的检查点恢复，只保留旧分支在检查点及之前的数据。
多次恢复使用相同规则递归处理；`visited` 集合防止配置互相引用造成无限递归。
前一段目录必须还在，resume 必须是有效的绝对路径；搬动实验目录后应更新路径。

横轴是**累计已记录的进程运行时间**：包括日志计时区间内的验证等开销，
不包含关机、暂停的间隔，也不包含没有被日志覆盖的最后一次保存和启动耗时。
所以它不是纯 GPU 计算时间，也不是日历上经过的总时间。
如果检查点对应步没有记录，脚本会报错，不用猜测的时间补齐。

### moving_average：平滑训练曲线

```python
train = [row for row in rows if row["split"] == "train"]
x = [row["step"] for row in train]
y = [row["loss"] for row in train]
```

这是列表推导式：第一行筛选训练记录，后两行分别提取横轴、纵轴数据。
主脚本同时绘制浅色原始训练 loss 和深色尾随平均，验证 loss 保留原值。
默认平均最近 20 条日志；若每 10 步记录一次，就是最多 20 个采样点，
并不是每一步 loss 的精确 200 步平均。`--smooth-window 1` 可关闭平滑。
前面的记录不足 20 条时，使用已有记录平均。

### plot_history：画图

- `ax.plot(x, y, ...)` 画折线；`alpha` 越小越透明。
- `ax.axhline(1.45, ...)` 画水平目标线。
- `ax.scatter(...)` 标出最低验证 loss 的采样点。
- `elapsed_seconds / 3600` 把秒转换为小时。
- `fig.savefig(...)` 保存图；英文轴标题避免没有中文字体时显示方框。

学习率图也来自实际日志，不重新计算理论曲线，所以未记录的最初几步不会补画。
训练 loss 是采样 batch 的损失，验证 loss 是多批次平均，两条曲线的噪声大小不同。
后半程详情图缩小了显示范围，阅读时注意纵轴刻度，不能只凭折线陡峭程度判断变化大小。

### main：连接各个步骤

`argparse` 接收命令行参数，`type=Path` 把路径字符串转换为路径对象，
`type=int` 把平滑窗口转换为整数。
随后依次读取/合并数据、绘图、导出数据和摘要。
`if __name__ == "__main__"` 使直接运行模块时执行 main，导入函数测试时不自动画图。

## 4. 核对结果

本次 TinyStories 数据应得到：20000 步，最终验证 loss 约 1.4422，
最佳验证 loss 约 1.4343（19200 步），两段日志合计 2100 条。
这只是本次运行记录，不代表做过多个学习率或多次随机种子实验。

恢复日志的针对性测试：

```powershell
uv run --cache-dir E:/cs336/.uv-cache pytest tests/test_plot_training.py -q
```

你可以先尝试改 `--smooth-window` 为 1、10、50，观察可读性和细节的变化；
再在 `plot_history` 里修改颜色、标题或 `figsize`，理解这些参数的作用。
