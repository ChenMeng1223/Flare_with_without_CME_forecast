from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def plot_gong_halpha(file_path, save_dir="Halpha_addition_data/"):
    try:
        with fits.open(file_path) as hdu_list:
            print("=== 文件结构 ===")
            hdu_list.info()

            # 读取真实数据：HDU 1
            data = hdu_list[1].data
            header = hdu_list[1].header

            print(f"图像数据形状: {data.shape}")

            # ===================== 1. 自动生成文件名（核心需求） =====================
            # 从表头读取观测时间
            date_obs = header.get("DATE-OBS", "")
            # 解析时间，格式化为：RSM20241223T000042_HA_GONG
            if date_obs:
                # 处理时间字符串：2024-12-23T00:00:42 → 20241223T000042
                dt = datetime.fromisoformat(date_obs.split('.')[0])
                time_str = dt.strftime("%Y%m%dT%H%M%S")
                filename = f"RSM{time_str}_HA_GONG.png"
            else:
                filename = "RSM_UNKNOWN_HA_GONG.png"
            save_path = save_dir + filename

            # ===================== 2. 绘制标准Hα图像（修复色差） =====================
            plt.figure(figsize=(10, 10), dpi=150)

            # GONG Hα专用拉伸参数
            vmin, vmax = np.percentile(data, (0,100))
            # vmin, vmax = np.percentile(data, (5, 95))

            # 🔥 关键修复：afmhot_r 反转色标（标准Hα样式）
            im = plt.imshow(
                data,
                cmap='afmhot',  # 反转色标！黑底+亮特征，标准太阳Hα
                origin='lower',
                vmin=vmin, vmax=vmax
            )

            # 美化
            # plt.title(f"GONG Hα 全日面\n观测时间: {date_obs}", fontsize=12, pad=20)
            plt.axis('off')

            # 保存高清图
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='black')
            plt.show()
            print(f"\n标准Hα图像已保存: {save_path}")

    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
    except Exception as e:
        print(f"绘图失败: {str(e)}")


# ===================== 运行 =====================
if __name__ == "__main__":
    GONG_FILE = "Halpha_addition_data/20241223000042Lh.fits.fz"
    plot_gong_halpha(GONG_FILE)