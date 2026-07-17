#!/usr/bin/env python3  # 指定脚本使用Python3解释器执行
"""
批量图像绘制脚本

将下载的FITS文件批量转换为图像文件，供后续数据集创建使用。
支持多进程并行处理，提高效率。
"""
# 导入系统相关模块
import sys
import os
# 导入类型注解相关模块，用于类型提示
from typing import Dict, List, Optional, Tuple
# 导入路径处理模块，简化文件路径操作
from pathlib import Path
# 导入日期时间处理模块
from datetime import datetime
# 导入日志模块
import logging
# 导入命令行参数解析模块
import argparse

# 导入数据分析相关库
import pandas as pd
# 导入yaml配置文件解析库
import yaml
# 导入数值计算库
import numpy as np
# 导入PIL图像处理库
from PIL import Image
# 导入matplotlib绘图库
import matplotlib.pyplot as plt
# 导入多进程并行处理相关模块
from concurrent.futures import ProcessPoolExecutor, as_completed
# 导入matplotlib颜色处理模块
from matplotlib import colors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 添加项目根目录到Python路径，确保本地自定义包能被导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入本地日志模块和日志配置工具
from data import logger
from utils.logging_utils import setup_logging


class FITSImageProcessor:
    """FITS图像处理器：负责批量将FITS天文数据文件转换为可视化图像"""

    # 类变量：不同EUV波段的绘图配置（最大值、最小值、色标、标签）
    BAND_CONFIG = {
        94: {'dmax': 10000.0, 'dmin': 1.0, 'cmap': 'sdoaia94', 'label': '94Å'},
        171: {'dmax': 5000.0, 'dmin': 5.0, 'cmap': 'sdoaia171', 'label': '171Å'},
        193: {'dmax': 8000.0, 'dmin': 10.0, 'cmap': 'sdoaia193', 'label': '193Å'},
    }

    def __init__(self, config: Dict):
        """
        初始化处理器

        Args:
            config: 配置字典（包含数据路径、模态等配置）
        """
        # 保存配置字典
        self.config = config
        # 从配置中获取数据模态（如magnetogram、euv_193等）
        self.modalities = config['data']['modalities']

        # 创建图像输出基础目录（确保目录层级与后续处理对齐）
        self.image_base_dir = Path(config['data']['processed_data_dir'])
        # 递归创建目录（父目录不存在则创建，已存在则不报错）
        self.image_base_dir.mkdir(parents=True, exist_ok=True)

        # 检查astropy库是否可用（用于读取FITS文件）
        try:
            from astropy.io import fits
            import astropy.visualization as vis
            # 保存fits和vis模块引用
            self.fits = fits
            self.vis = vis
            # 标记astropy可用
            self.astropy_available = True
            # 记录日志
            logger.info("Astropy库可用")
        except ImportError:
            # 记录警告日志
            logger.warning("Astropy库不可用，请安装: pip install astropy")
            # 标记astropy不可用
            self.astropy_available = False

        # 图像对齐参数：设置输出图像尺寸（与原始数据一致，避免信息损失）
        self.target_size = (2048, 2048)
        # 计算目标太阳半径像素值（取输出尺寸一半的85%，留少量边距）
        self.target_solar_radius_pix = min(self.target_size[0], self.target_size[1]) / 2.0 * 0.85
        # 初始化sunpy可用性标记
        self.sunpy_available = False
        try:
            # 导入sunpy色标库（用于天文图像专用色标）
            import sunpy.visualization.colormaps
            # 标记sunpy可用
            self.sunpy_available = True
        except ImportError:
            pass
        # 初始化波段配置
        self.band_config = self.BAND_CONFIG

    def find_fits_files(self, download_dir: Path) -> Dict[str, List[Path]]:
        """查找指定目录下所有FITS文件，按模态分类"""
        # 初始化FITS文件字典（key:模态名，value:该模态下的FITS文件路径列表）
        fits_files = {}

        # 遍历下载目录下的所有子目录
        for modality_dir in download_dir.iterdir():
            # 筛选：是目录 且 目录名在配置的模态列表中
            if modality_dir.is_dir() and modality_dir.name in self.modalities:
                # 获取模态名
                modality = modality_dir.name
                # 查找该目录下所有.fits文件
                fits_files[modality] = list(modality_dir.glob('*.fits'))
                # 按文件名排序（保证按时间顺序处理）
                fits_files[modality].sort()

                # 记录该模态下找到的FITS文件数量
                logger.info(f"模态 {modality}: 找到 {len(fits_files[modality])} 个FITS文件")

        # 返回分类后的FITS文件路径字典
        return fits_files

    def _get_exposure_time(self, fits_path: Path) -> float:
        """从FITS文件头中获取曝光时间（私有方法）"""
        try:
            # 以只读模式打开FITS文件
            with self.fits.open(str(fits_path), mode='readonly') as h:
                # 遍历可能的曝光时间关键字（兼容不同FITS文件格式）
                for key in ('EXPTIME', 'EXPOSURE', 'EXPOSUR'):
                    # 检查关键字是否存在于文件头中
                    if key in h[0].header:
                        # 转换为浮点型
                        v = float(h[0].header[key])
                        # 返回曝光时间（确保值大于0，否则返回1.0）
                        return v if v > 0 else 1.0
        except:
            # 异常时不处理
            pass
        # 未找到则返回默认值1.0
        return 1.0

    def _get_fits_time(self, fits_path: Path) -> Optional[datetime]:
        """从FITS文件头中获取观测时间（私有方法）"""
        try:
            # 导入astropy时间处理模块
            from astropy.time import Time
            # 以只读模式打开FITS文件
            with self.fits.open(str(fits_path), mode='readonly') as h:
                # 遍历可能的时间关键字（兼容不同FITS文件格式）
                for key in ('DATE-OBS', 'T_OBS', 'DATE_OBS', 'TSTART'):
                    # 检查关键字是否存在
                    if key in h[0].header:
                        # 解析时间
                        t = Time(h[0].header[key])
                        # 转换为Python datetime对象并返回
                        return t.to_datetime()
        except:
            pass
        # 解析失败返回None
        return None

    def get_pixel_scale_and_zoom(self, fits_path: Path, modality: str = "unknown") -> tuple:
        """计算像素缩放比例和图像缩放因子（用于太阳圆盘对齐）"""
        try:
            # 打开FITS文件
            with self.fits.open(fits_path) as hdul:
                # 诊断：遍历所有HDU，找出包含WCS信息的HDU
                print(f"\n[DEBUG] {modality} - FITS结构诊断（{fits_path.name}）:")
                print(f"  总HDU数: {len(hdul)}")
                
                # 遍历所有HDU找出有数据和WCS的HDU
                target_hdu = None
                for hdu_idx, hdu in enumerate(hdul):
                    data_shape = hdu.data.shape if hdu.data is not None else 'None'
                    print(f"  HDU[{hdu_idx}]: name={hdu.name}, data_shape={data_shape}")
                    if hdu.data is not None and hdu.data.size > 0:
                        header = hdu.header
                        # 检查WCS关键字
                        wcs_keys_found = []
                        for key in ['CDELT1', 'CDELT2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CRPIX1', 'CRVAL1', 'CTYPE1']:
                            if key in header:
                                wcs_keys_found.append(f"{key}={header[key]}")
                        if wcs_keys_found:
                            print(f"    WCS信息: {', '.join(wcs_keys_found[:5])}")  # 只打前5个
                            if target_hdu is None:
                                target_hdu = (hdu_idx, header)
                
                # 如果没找到参考HDU，默认使用HDU[0]
                if target_hdu is None:
                    target_hdu = (0, hdul[0].header)
                
                hdu_idx, header = target_hdu
                print(f"  使用HDU[{hdu_idx}]的头文件")
                
                # 优先尝试CDELT1/2
                cdelt1_raw = header.get('CDELT1', None)
                cdelt2_raw = header.get('CDELT2', None)
                
                # 备用：如果没有CDELT，尝试CD矩阵
                if cdelt1_raw is None and 'CD1_1' in header:
                    cdelt1_raw = abs(header.get('CD1_1', 0.5))
                    print(f"  [备用] 使用CD1_1={cdelt1_raw}")
                
                if cdelt2_raw is None and 'CD2_2' in header:
                    cdelt2_raw = abs(header.get('CD2_2', 0.5))
                    print(f"  [备用] 使用CD2_2={cdelt2_raw}")
                
                # 如果还是没有，使用默认值0.5
                if cdelt1_raw is None:
                    cdelt1_raw = 0.5
                    print(f"  [默认] CDELT1=0.5（未在header中找到）")
                if cdelt2_raw is None:
                    cdelt2_raw = 0.5
                    print(f"  [默认] CDELT2=0.5（未在header中找到）")
                
                # 取绝对值
                cdelt1 = abs(cdelt1_raw)
                cdelt2 = abs(cdelt2_raw)
                # 计算平均像素尺度
                pixel_scale = (cdelt1 + cdelt2) / 2
                # 太阳半径（固定值：479.63角秒）
                solar_radius_arcsec = 479.63
                # 计算当前太阳半径对应的像素数
                current_solar_radius_pix = solar_radius_arcsec / pixel_scale
                # 计算缩放因子
                zoom_factor = self.target_solar_radius_pix / current_solar_radius_pix if current_solar_radius_pix > 0 else 1.0
                # 输出最终结果
                print(f"[ALIGN] modality={modality}, CDELT1_raw={cdelt1_raw}, CDELT2_raw={cdelt2_raw}, CDELT1={cdelt1:.6f}, CDELT2={cdelt2:.6f}, pixel_scale={pixel_scale:.6f}, zoom_factor={zoom_factor:.4f}\n")
                # 返回缩放因子和像素尺度
                return zoom_factor, pixel_scale
        except Exception as e:
            # 记录警告日志
            logger.warning(f"Failed to get pixel scale from {fits_path}: {e}, using default zoom 1.0")
            # 异常时返回默认值
            return 1.0, 0.5

    def get_solar_disk_params(self, fits_path: Path, data_shape: tuple, modality: str = "unknown") -> Dict[str, float]:
        """从FITS头读取日心与日面半径，并计算缩放因子"""
        # 默认值：图像中心和经验半径
        height, width = data_shape[:2]
        center_x = width / 2.0
        center_y = height / 2.0
        radius_pix = min(height, width) / 2.0 * 0.9
        zoom_factor = 1.0
        pixel_scale = 0.5
        try:
            with self.fits.open(str(fits_path)) as hdul:
                header = None
                for hdu in hdul:
                    if hdu.data is not None and hdu.data.size > 0:
                        header = hdu.header
                        break
                if header is None:
                    header = hdul[0].header

                # CRPIX是1-based像素坐标，转换为0-based
                crpix1 = header.get('CRPIX1', None)
                crpix2 = header.get('CRPIX2', None)
                if crpix1 is not None and crpix2 is not None:
                    center_x = float(crpix1) - 1.0
                    center_y = float(crpix2) - 1.0

                # 优先读取像素半径
                rsun_pix = header.get('RSUN_PIX', None)
                if rsun_pix is not None and float(rsun_pix) > 0:
                    radius_pix = float(rsun_pix)
                else:
                    # 没有像素半径时，使用角秒半径 / 像素尺度
                    cdelt1_raw = header.get('CDELT1', None)
                    cdelt2_raw = header.get('CDELT2', None)
                    if cdelt1_raw is None and 'CD1_1' in header:
                        cdelt1_raw = abs(header.get('CD1_1', 0.5))
                    if cdelt2_raw is None and 'CD2_2' in header:
                        cdelt2_raw = abs(header.get('CD2_2', 0.5))
                    if cdelt1_raw is None:
                        cdelt1_raw = 0.5
                    if cdelt2_raw is None:
                        cdelt2_raw = 0.5
                    pixel_scale = (abs(float(cdelt1_raw)) + abs(float(cdelt2_raw))) / 2.0
                    rsun_obs = header.get('RSUN_OBS', header.get('SOLAR_R', 479.63))
                    rsun_obs = float(rsun_obs) if rsun_obs is not None else 479.63
                    if pixel_scale > 0 and rsun_obs > 0:
                        radius_pix = rsun_obs / pixel_scale

                if radius_pix > 0:
                    zoom_factor = self.target_solar_radius_pix / radius_pix

        except Exception as e:
            logger.warning(f"Failed to read solar disk params from {fits_path}: {e}")

        print(
            f"[DISK] modality={modality}, center=({center_x:.2f},{center_y:.2f}), "
            f"radius_pix={radius_pix:.2f}, zoom_factor={zoom_factor:.4f}, pixel_scale={pixel_scale:.6f}"
        )
        return {
            'center_x': center_x,
            'center_y': center_y,
            'radius_pix': radius_pix,
            'zoom_factor': zoom_factor,
            'pixel_scale': pixel_scale
        }

    def align_image(self, data: np.ndarray, fits_path: Path, modality: str = "unknown", alignment_mode: str = "full") -> np.ndarray:
        """
        将图像按日面参数对齐并映射到目标画布。
        
        alignment_mode:
            - 'full'：整幅图缩放+按日心平移（尽量保留日面外信息）
            - 'disk_only'：仅缩放日面区域，并将日面映射到目标画布（画布外置零）
        """
        # 提取日面参数（中心、半径、缩放因子）
        disk_params = self.get_solar_disk_params(fits_path, data.shape, modality=modality)
        center_x = disk_params['center_x']
        center_y = disk_params['center_y']
        radius = disk_params['radius_pix']
        zoom_factor = disk_params['zoom_factor']

        th, tw = self.target_size
        out = np.zeros((th, tw), dtype=np.float32)

        # 按“缩放后的日心 -> 目标日心”进行平移贴回目标画布
        scaled_center_x = center_x * zoom_factor
        scaled_center_y = center_y * zoom_factor
        target_center_x = tw / 2.0
        target_center_y = th / 2.0
        offset_x = int(round(target_center_x - scaled_center_x))
        offset_y = int(round(target_center_y - scaled_center_y))

        if alignment_mode == "disk_only":
            # 先构建原图日面掩码，仅保留日面数据
            height, width = data.shape[:2]
            y, x = np.ogrid[:height, :width]
            src_disk_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            src_data = data.astype(np.float32, copy=False)
            disk_only = np.zeros_like(src_data, dtype=np.float32)
            disk_only[src_disk_mask] = src_data[src_disk_mask]

            # 缩放日面数据
            if abs(zoom_factor - 1.0) < 1e-6:
                scaled = disk_only
            else:
                try:
                    from scipy.ndimage import zoom
                    scaled = zoom(disk_only, zoom_factor, order=1)
                except Exception:
                    new_h = max(1, int(disk_only.shape[0] * zoom_factor))
                    new_w = max(1, int(disk_only.shape[1] * zoom_factor))
                    scaled_img = Image.fromarray(disk_only)
                    scaled_img = scaled_img.resize((new_w, new_h), resample=Image.BILINEAR)
                    scaled = np.array(scaled_img)

        else:
            # 'full'：对整幅图缩放
            src_data = data.astype(np.float32, copy=False)
            if abs(zoom_factor - 1.0) < 1e-6:
                scaled = src_data
            else:
                try:
                    from scipy.ndimage import zoom
                    scaled = zoom(src_data, zoom_factor, order=1)
                except Exception:
                    new_h = max(1, int(src_data.shape[0] * zoom_factor))
                    new_w = max(1, int(src_data.shape[1] * zoom_factor))
                    scaled_img = Image.fromarray(src_data)
                    scaled_img = scaled_img.resize((new_w, new_h), resample=Image.BILINEAR)
                    scaled = np.array(scaled_img)

        src_h, src_w = scaled.shape[:2]
        dst_x0 = max(0, offset_x)
        dst_y0 = max(0, offset_y)
        dst_x1 = min(tw, offset_x + src_w)
        dst_y1 = min(th, offset_y + src_h)
        src_x0 = max(0, -offset_x)
        src_y0 = max(0, -offset_y)
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = scaled[src_y0:src_y1, src_x0:src_x1]

        return out.astype(data.dtype, copy=False)

    def plot_magnetogram_aligned(self, fits_path: Path, output_path: Path) -> bool:
        """绘制磁图并对齐，保存为PNG文件"""
        try:
            # 打开FITS文件
            with self.fits.open(fits_path) as hdul:
                # 优先读取第2个HDU的数据（磁图常见存储位置）
                if len(hdul) > 1 and hdul[1].data is not None:
                    data = hdul[1].data
                # 备用：读取第1个HDU的数据
                elif hdul[0].data is not None:
                    data = hdul[0].data
                else:
                    # 无数据返回失败
                    return False
            # 处理NaN/无穷值（替换为0）
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            # 设置磁图数值范围（裁剪异常值）
            vmin, vmax = -2000, 2000
            data = np.clip(data, vmin, vmax)
            # 图像对齐并缩放到目标尺寸
            data = self.align_image(data, fits_path, modality="magnetogram", alignment_mode="disk_only")
            # 获取图像尺寸
            height, width = data.shape
            # 计算图像中心坐标（目标画布中心）
            center_y, center_x = height // 2, width // 2
            # 使用目标太阳半径生成最终圆盘掩码
            radius = int(self.target_solar_radius_pix)
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            # 初始化输出图像（圆盘外保持全黑）
            output_image = np.zeros((height, width), dtype=np.uint8)
            # 归一化数据到0-255并仅写入圆盘内
            data_normalized = ((data - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            output_image[mask] = data_normalized[mask]
            # 转换为PIL图像
            img = Image.fromarray(output_image)
            # 保存为PNG文件
            img.save(str(output_path))
            # 返回成功
            return True
        except Exception as e:
            # 记录错误日志
            logger.error(f"Plot magnetogram failed {fits_path}: {e}")
            # 返回失败
            return False

    def plot_euv_aligned(self, fits_path: Path, output_path: Path, wavelength: int) -> bool:
        """绘制EUV图像并对齐，保存为PNG文件"""
        # 获取该波段的配置（默认使用193Å配置）
        cfg = self.band_config.get(wavelength, self.band_config[193])
        dmax = cfg['dmax']
        dmin = cfg['dmin']
        cmap_name = cfg['cmap']
        try:
            # 打开FITS文件
            with self.fits.open(str(fits_path)) as hdul:
                # 优先读取第2个HDU的数据
                if len(hdul) > 1 and hdul[1].data is not None:
                    data = np.array(hdul[1].data, dtype=float)
                # 备用：读取第1个HDU的数据
                elif hdul[0].data is not None:
                    data = np.array(hdul[0].data, dtype=float)
                else:
                    return False
            # 获取曝光时间并归一化数据
            original_exposure = self._get_exposure_time(fits_path)
            data = data / original_exposure
            # 处理NaN/无穷值
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            # 确保数据最小值不低于配置的dmin
            data = np.maximum(data, dmin)
            # 图像对齐并缩放到目标尺寸
            modality_name = f"euv_{wavelength}"
            data = self.align_image(data, fits_path, modality=modality_name, alignment_mode="full")
            # 设置对数归一化（适配EUV数据分布）
            norm = colors.LogNorm(vmin=dmin, vmax=dmax, clip=True)
            # 选择色标（sunpy可用则用专用色标，否则用hot色标）
            cmap = cmap_name if self.sunpy_available else 'hot'
            # 创建标量映射器（将数据值映射到颜色）
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            # 将数据转换为RGBA颜色
            rgba = sm.to_rgba(data)
            # 提取RGB通道并转换为0-255整数
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            # 转换为PIL图像
            img = Image.fromarray(rgb)
            # 保存为PNG文件
            img.save(str(output_path))
            return True
        except Exception as e:
            # 记录错误日志
            logger.error(f"Plot euv failed {fits_path}: {e}")
            return False

    def plot_halpha_aligned(self, fits_path: Path, output_path: Path, slice_index: Optional[int] = 69, use_color: bool = True) -> bool:
        """绘制H-alpha图像并对齐，保存为PNG文件"""
        try:
            # 以只读模式打开FITS文件
            with self.fits.open(str(fits_path), mode='readonly') as hdu_list:
                # 检查数据是否存在
                if len(hdu_list) < 2 or hdu_list[1].data is None:
                    return False
                # 获取数据
                hdu_data = hdu_list[1].data
                # 处理3D数据（H-alpha常为多切片数据）
                if hdu_data.ndim == 3:
                    # 获取切片数量
                    n_slices = hdu_data.shape[0]
                    # 确定切片索引（默认69，或中间切片）
                    idx = slice_index if slice_index is not None else n_slices // 2
                    # 确保索引在有效范围内
                    idx = max(0, min(idx, n_slices - 1))
                    # 提取指定切片数据
                    image_data = hdu_data[idx, :, :].astype(np.float32)
                # 处理2D数据
                elif hdu_data.ndim == 2:
                    image_data = hdu_data.astype(np.float32)
                else:
                    # 不支持的维度
                    return False
            # 处理NaN/无穷值
            image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
            # 确保数据非负
            image_data = np.maximum(image_data, 0.0)
            # 图像对齐并缩放到目标尺寸
            image_data = self.align_image(image_data, fits_path, modality="halpha", alignment_mode="disk_only")
            # 与磁图保持一致：圆盘外置黑，避免视觉不一致并保证归一化只在圆盘内统计
            height, width = image_data.shape
            center_y, center_x = height // 2, width // 2
            radius = int(self.target_solar_radius_pix)
            y, x = np.ogrid[:height, :width]
            disk_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            image_data = image_data.copy()
            image_data[~disk_mask] = 0.0
            # 彩色绘制
            if use_color:
                # 设置数值范围（均值的4倍作为最大值）
                vmin = 0.0
                vmax = float(image_data.mean() * 4) if image_data.mean() > 0 else float(image_data.max())
                # 确保最大值大于最小值
                if vmax <= vmin:
                    vmax = vmin + 1.0
                # 线性归一化
                norm = Normalize(vmin=vmin, vmax=vmax)
                # 选择色标
                cmap = 'afmhot'
                # 创建标量映射器
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                # 转换为RGBA颜色
                rgba = sm.to_rgba(image_data)
                # 提取RGB通道并转换为0-255整数
                rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                # 圆盘外强制置黑（避免 colormap 在 0 值处仍显示颜色）
                rgb[~disk_mask] = 0
                # 转换为RGB模式PIL图像
                img = Image.fromarray(rgb, 'RGB')
            # 灰度绘制
            else:
                # 数据有差异则归一化
                if image_data.max() > image_data.min():
                    normalized = (image_data - image_data.min()) * 255 / (image_data.max() - image_data.min())
                else:
                    # 数据无差异则全黑
                    normalized = np.zeros_like(image_data)
                # 裁剪到0-255并转换为uint8
                normalized = np.clip(normalized, 0, 255).astype(np.uint8)
                # 圆盘外强制置黑
                normalized[~disk_mask] = 0
                # 转换为灰度模式PIL图像
                img = Image.fromarray(normalized, 'L')
            # 保存为PNG文件
            img.save(str(output_path))
            return True
        except Exception as e:
            # 记录错误日志
            logger.error(f"Plot halpha failed {fits_path}: {e}")
            return False

    def process_fits_file(self, fits_path: Path, modality: str,
                         output_dir: Path) -> Optional[str]:
        """处理单个FITS文件，转换为图像并保存"""
        # 检查astropy是否可用
        if not self.astropy_available:
            logger.error("Astropy不可用，无法处理FITS文件")
            return None

        try:
            # 构造输出文件名（保留原文件名，后缀改为png）
            output_filename = f"{fits_path.stem}.png"
            # 构造输出文件路径
            output_path = output_dir / output_filename

            # 处理磁图模态
            if modality == 'magnetogram':
                success = self.plot_magnetogram_aligned(fits_path, output_path)
            # 处理EUV模态（格式：euv_波长）
            elif modality.startswith('euv_'):
                # 提取波长数值
                wavelength = int(modality.split('_')[1])
                success = self.plot_euv_aligned(fits_path, output_path, wavelength)
            # 处理H-alpha模态
            elif modality == 'halpha':
                success = self.plot_halpha_aligned(fits_path, output_path)
            else:
                # 未知模态记录警告
                logger.warning(f"未知模态 {modality}")
                return None

            # 处理成功则返回输出路径
            if success:
                return str(output_path)
            else:
                return None

        except Exception as e:
            # 记录处理失败日志
            logger.error(f"处理FITS文件 {fits_path} 失败: {e}")
            return None

    def preprocess_data(self, data: np.ndarray, modality: str) -> np.ndarray:
        """数据预处理：处理异常值、归一化到0-255"""
        # 处理NaN和无穷值（替换为0）
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # 获取该模态的配置
        modality_config = self.modalities[modality]
        # 获取归一化配置
        normalization = modality_config.get('normalization')

        # 使用配置的归一化范围
        if normalization:
            vmin, vmax = normalization
            # 裁剪数据到指定范围
            data = np.clip(data, vmin, vmax)
            # 归一化到0-255并转换为uint8
            data = ((data - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            # 默认归一化：使用1%和99%分位数去除异常值
            data_min, data_max = np.percentile(data, [1, 99])
            # 数据有差异则归一化
            if data_max > data_min:
                data = np.clip(data, data_min, data_max)
                data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                # 数据无差异则返回全黑图像
                data = np.zeros_like(data, dtype=np.uint8)

        # 返回预处理后的数据
        return data

    def data_to_image(self, data: np.ndarray, modality: str) -> np.ndarray:
        """将原始数据转换为指定尺寸的图像数组"""
        # 确保数据是2D的
        if data.ndim == 3:
            # 3D数据取第一个通道
            data = data[0]
        elif data.ndim != 2:
            # 抛出不支持维度的异常
            raise ValueError(f"不支持的数据维度: {data.ndim}")

        # 获取该模态的目标分辨率
        target_size = self.modalities[modality]['resolution']
        # 尺寸不匹配则缩放
        if data.shape != tuple(target_size):
            from scipy.ndimage import zoom
            # 计算缩放因子
            zoom_factors = (target_size[0] / data.shape[0], target_size[1] / data.shape[1])
            # 双线性插值缩放
            data = zoom(data, zoom_factors, order=1)

        # 返回调整后的数据
        return data

    def process_event_images(self, event_id: str) -> tuple:
        """处理单个事件的所有FITS文件，转换为图像"""
        # 构造该事件的FITS文件下载目录
        download_dir = Path(self.config['data']['raw_data_dir']) / 'downloaded' / event_id
        # 构造该事件的图像输出目录
        image_dir = self.image_base_dir / event_id

        # 检查下载目录是否存在
        if not download_dir.exists():
            logger.warning(f"下载目录不存在: {download_dir}")
            # 返回0成功、0总数
            return 0, 0

        # 创建图像输出目录
        image_dir.mkdir(parents=True, exist_ok=True)

        # 查找该事件下的所有FITS文件
        fits_files = self.find_fits_files(download_dir)

        # 无FITS文件则返回
        if not fits_files:
            logger.warning(f"事件 {event_id} 没有找到FITS文件")
            return 0, 0

        # 初始化成功计数和总数
        success_count = 0
        total_count = 0

        # 遍历每个模态的FITS文件
        for modality, files in fits_files.items():
            # 构造该模态的图像输出目录
            modality_image_dir = image_dir / modality
            # 创建目录
            modality_image_dir.mkdir(exist_ok=True)

            # 记录处理该模态的文件数量
            logger.info(f"处理模态 {modality}: {len(files)} 个文件")

            # 遍历该模态下的所有FITS文件（带索引）
            for idx, fits_path in enumerate(files, start=1):
                # 总数加1
                total_count += 1
                # 处理单个FITS文件
                result = self.process_fits_file(fits_path, modality, modality_image_dir)
                # 处理成功
                if result:
                    success_count += 1
                    logger.info(f"已完成 {event_id} / {modality} [{idx}/{len(files)}]: {fits_path.name} -> 成功")
                else:
                    logger.warning(f"已完成 {event_id} / {modality} [{idx}/{len(files)}]: {fits_path.name} -> 失败")

        # 记录该事件处理完成日志
        logger.info(f"事件 {event_id} 图像处理完成: {success_count}/{total_count}")
        # 返回成功数和总数
        return success_count, total_count

    def process_all_events(self, events_df: pd.DataFrame, max_workers: int = 4) -> None:
        """批量处理所有事件的图像（多进程并行）"""
        # 记录开始批量处理的日志
        logger.info(f"开始批量处理 {len(events_df)} 个事件的图像")

        # 初始化成功事件数和已处理事件数
        successful_events = 0
        processed_events = 0

        # 创建进程池（指定最大进程数）
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有事件的处理任务到进程池
            future_to_event = {
                executor.submit(_process_event_images_worker, row['event_id'], self.config): row['event_id']
                for _, row in events_df.iterrows()
            }

            # 遍历已完成的任务
            for future in as_completed(future_to_event):
                # 获取对应的事件ID
                event_id = future_to_event[future]
                try:
                    # 获取任务结果
                    _event_id, success_count, total_count = future.result()
                    # 已处理事件数加1
                    processed_events += 1
                    # 有文件且处理成功则计数
                    if total_count > 0 and success_count > 0:
                        successful_events += 1
                        logger.info(f"事件 {_event_id} 图像处理成功: {success_count}/{total_count}")
                    elif total_count == 0:
                        logger.warning(f"事件 {_event_id} 无可处理 FITS（total=0）")
                    else:
                        logger.warning(f"事件 {_event_id} 图像处理失败: {success_count}/{total_count}")
                except Exception as e:
                    # 记录任务异常日志
                    logger.error(f"事件 {event_id} 图像处理异常: {e}")

        # 记录批量处理完成日志
        logger.info(f"批量图像处理完成: {successful_events}/{len(events_df)} 个事件成功（processed_events={processed_events}）")


def _process_event_images_worker(event_id: str, config: Dict) -> tuple:
    """多进程工作函数：重新实例化处理器并处理单个事件"""
    # 创建处理器实例
    processor = FITSImageProcessor(config)
    # 处理该事件的图像
    success_count, total_count = processor.process_event_images(event_id)
    # 返回事件ID、成功数、总数
    return event_id, success_count, total_count


def load_events_metadata(events_file: str) -> pd.DataFrame:
    """加载事件元数据文件（支持Excel和CSV格式）"""
    # 读取Excel文件
    if events_file.endswith('.xlsx'):
        df = pd.read_excel(events_file)
    # 读取CSV文件
    elif events_file.endswith('.csv'):
        df = pd.read_csv(events_file)
    else:
        # 抛出不支持格式的异常
        raise ValueError(f"不支持的文件格式: {events_file}")

    # 检查必要列是否存在
    required_columns = ['event_id']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"文件缺少必要列: {col}")

    # 记录加载的事件数量
    logger.info(f"加载了 {len(df)} 个事件")
    # 返回事件数据框
    return df


def parse_args():
    """解析命令行参数"""
    # 创建参数解析器（描述信息）
    parser = argparse.ArgumentParser(description='批量绘制FITS图像')
    # 添加配置文件路径参数（默认值）
    parser.add_argument('--config', type=str, default='configs/data_config.yaml',
                        help='配置文件路径')
    # 添加最大进程数参数（默认4）
    parser.add_argument('--max_workers', type=int, default=4,
                        help='最大并发进程数')
    # 添加日志目录参数（默认logs）
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    # 解析参数并返回
    return parser.parse_args()


def main():
    """主函数：程序入口"""
    # 解析命令行参数
    args = parse_args()

    # 设置日志配置（日志目录、日志级别）
    setup_logging(log_dir=args.log_dir, level=logging.INFO)
    # 获取当前模块的日志器
    logger = logging.getLogger(__name__)

    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建FITS图像处理器实例
    processor = FITSImageProcessor(config)

    # 从配置中获取事件元数据文件路径
    events_file = config['data']['events_file']
    logger.info(f"使用事件文件: {events_file}")

    # 加载事件元数据
    events_df = load_events_metadata(events_file)

    # 批量处理所有事件的图像
    processor.process_all_events(events_df, max_workers=args.max_workers)


# 程序入口
if __name__ == "__main__":
    # 执行主函数
    main()