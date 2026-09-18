"""读取已有训练日志画图，不加载模型、不占用 GPU。配套讲解见 docs/training_visualization.md。"""

import argparse
import csv
import json
import math
from pathlib import Path


FIELDS = ["step", "elapsed_seconds", "split", "loss", "lr"]


def read_metrics(run_dir):
    """CSV 中的值都是字符串，画图前先转换成 int/float。"""
    rows = []
    with (run_dir / "metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {run_dir}")
        for raw in reader:
            row = {"step": int(raw["step"]), "split": raw["split"]}
            for name in ("elapsed_seconds", "loss", "lr"):
                row[name] = float(raw[name])
                if not math.isfinite(row[name]):
                    raise ValueError(f"Non-finite {name} at step {row['step']}: {run_dir}")
            if row["split"] not in ("train", "valid"):
                raise ValueError(f"Unknown split: {row['split']}")
            if rows and (row["step"] < rows[-1]["step"]
                         or row["elapsed_seconds"] < rows[-1]["elapsed_seconds"]):
                raise ValueError(f"Log step/time goes backwards: {run_dir}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty log: {run_dir}")
    return rows


def load_history(run_dir, visited=None):
    """递归追溯 resume，把各次进程的时间变为累计时间。无需读取大型 .pt 文件。"""
    run_dir = Path(run_dir).resolve()
    visited = set() if visited is None else visited
    if run_dir in visited:
        raise ValueError(f"Resume cycle: {run_dir}")
    visited.add(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rows = read_metrics(run_dir)
    if not config.get("resume"):
        return rows

    # 本项目保存的 resume 是绝对路径：实验目录/checkpoints/step_N.pt。
    checkpoint = Path(config["resume"])
    if not checkpoint.is_absolute():
        raise ValueError("Please use an absolute resume path in config.json")
    if config.get("elapsed_seconds_scope", "current_session") != "current_session":
        raise ValueError("Unsupported elapsed_seconds_scope")
    start_step = int(config["start_step"])
    previous = load_history(checkpoint.parent.parent, visited)
    # 从第 N 步恢复时，旧分支 N 之后的结果不属于此次训练，必须丢掉。
    previous = [row for row in previous if row["step"] <= start_step]
    boundary = [row for row in previous if row["step"] == start_step]
    if not boundary:
        raise ValueError(f"No timestamp for checkpoint step {start_step}; cannot join time reliably")
    if rows[0]["step"] <= start_step:
        raise ValueError("Resumed log must start after start_step")
    # 同一步可能有 train 和 valid 两行，取较晚时间，包含当次验证耗时。
    offset = max(row["elapsed_seconds"] for row in boundary)
    for row in rows:
        row["elapsed_seconds"] += offset
    return previous + rows


def moving_average(values, window):
    """尾随平均：第 i 个点仅使用当前及之前的记录，不使用未来值。"""
    return [sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window)
            for i in range(len(values))]


def plot_history(rows, out_dir, title, smooth_window, target_loss):
    # 延迟导入使日志读取函数可以单独使用。Agg 后端直接保存图片，无需弹出窗口。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    train = [row for row in rows if row["split"] == "train"]
    valid = [row for row in rows if row["split"] == "valid"]
    if not train:
        raise ValueError("No training records to plot")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    blue, orange = "#2563a6", "#c45d20"
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")

    # plot(x, y)：两个列表中同一下标的元素，组成图上的一个点。
    for ax, x_key, scale, x_label in (
        (axes[0], "step", 1, "Gradient steps"),
        (axes[1], "elapsed_seconds", 3600, "Cumulative logged runtime (hours)"),
    ):
        x = [row[x_key] / scale for row in train]
        y = [row["loss"] for row in train]
        ax.plot(x, y, color=blue, alpha=.22, linewidth=.7, label="Train (raw)")
        ax.plot(x, moving_average(y, smooth_window), color=blue, linewidth=1.6,
                label=f"Train (mean of {smooth_window} records)")
        if valid:
            ax.plot([row[x_key] / scale for row in valid], [row["loss"] for row in valid],
                    color=orange, linewidth=1.5, label="Validation (raw)")
            best = min(valid, key=lambda row: row["loss"])
            ax.scatter([best[x_key] / scale], [best["loss"]], color=orange, s=28, zorder=5)
        if target_loss is not None:
            ax.axhline(target_loss, color="#537457", linestyle="--", linewidth=1,
                       label=f"Target = {target_loss:g}")
        ax.set(xlabel=x_label, ylabel="Cross-entropy loss", title=f"Loss vs. {x_label.lower()}")
        ax.legend(fontsize=8)

    axes[2].plot([row["step"] for row in train], [row["lr"] for row in train], color=blue)
    axes[2].set(xlabel="Gradient steps", ylabel="Learning rate", title="Logged learning rate")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes:
        ax.grid(alpha=.18)
    subtitle = f"Final step: {max(row['step'] for row in rows):,}"
    if valid:
        subtitle += f"   |   Final valid: {valid[-1]['loss']:.4f}   |   Best valid: {best['loss']:.4f}"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=14)
    fig.savefig(out_dir / "training_overview.png")
    fig.savefig(out_dir / "training_overview.svg")
    plt.close(fig)

    # 另画后半程的验证曲线：总览图的早期高 loss 会压缩后期差异。
    if valid:
        tail = [row for row in valid if row["step"] >= max(r["step"] for r in rows) / 2]
        if tail:
            fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
            ax.plot([r["step"] for r in tail], [r["loss"] for r in tail],
                    color=orange, marker=".", label="Validation (raw)")
            if target_loss is not None:
                ax.axhline(target_loss, color="#537457", linestyle="--", label=f"Target = {target_loss:g}")
            ax.set(title=f"{title}: validation in second half", xlabel="Gradient steps", ylabel="Loss")
            ax.grid(alpha=.18)
            ax.legend()
            fig.savefig(out_dir / "validation_detail.png")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize CSV logs, including resumed training")
    parser.add_argument("--run-dir", type=Path, required=True, help="最后一段训练的实验目录")
    parser.add_argument("--out-dir", type=Path, help="默认：实验目录/plots")
    parser.add_argument("--title", default="Language model training")
    parser.add_argument("--smooth-window", type=int, default=20, help="平均最近多少条日志，不是训练步数")
    parser.add_argument("--target-loss", type=float, default=None, help="可选目标线；TinyStories 用 1.45")
    args = parser.parse_args()
    if args.smooth_window < 1:
        parser.error("--smooth-window must be positive")
    out_dir = args.out_dir or args.run_dir / "plots"
    rows = load_history(args.run_dir)
    plot_history(rows, out_dir, args.title, args.smooth_window, args.target_loss)
    # 导出合并后的数据，方便检查时间拼接，或以后用其他工具画图。
    with (out_dir / "merged_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row["split"] == "valid"]
    summary = {"source_run": str(args.run_dir.resolve()), "records": len(rows),
               "final_step": max(row["step"] for row in rows),
               "logged_runtime_hours": max(row["elapsed_seconds"] for row in rows) / 3600,
               "final_validation": valid[-1] if valid else None,
               "best_validation": min(valid, key=lambda r: r["loss"]) if valid else None,
               "time_note": "Sum of logged session durations; excludes offline gaps and unlogged save/startup time."}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved figures and merged data to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
