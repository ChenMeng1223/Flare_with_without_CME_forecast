import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ====================== 【在这里修改你的时间范围和保存路径】 ======================
START_TIME = "2024-12-23 00:00:00"  # 起始时间
END_TIME = "2024-12-23 00:02:00"  # 结束时间
SAVE_DIR = "./haf_data"  # 文件保存目录


# ================================================================================

# 配置请求重试策略（避免网络波动导致下载失败）
def create_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


session = create_session()


def get_links(url):
    """获取指定URL下的所有文件夹/文件链接（排除 Parent Directory）"""
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        for a in soup.find_all('a'):
            href = a.get('href')
            if href and not href.startswith('?') and not href.startswith('/') and href != '../':
                links.append(href.rstrip('/'))  # 统一去掉末尾的/
        return links
    except Exception as e:
        print(f"访问 {url} 失败: {e}")
        return []


def parse_filename_to_datetime(filename):
    """解析 .fits.fz 文件名中的时间，例如 '20241223000042Lh.fits.fz' → datetime 对象"""
    try:
        time_str = filename[:14]  # 提取前14位：YYYYMMDDHHMMSS
        return datetime.strptime(time_str, '%Y%m%d%H%M%S')
    except ValueError as e:
        print(f"文件名 {filename} 时间解析失败: {e}")
        return None


def download_file(url, save_dir):
    """下载文件到指定目录，若文件已存在则自动跳过"""
    filename = os.path.basename(url)
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        print(f"✅ 文件已存在，跳过: {filename}")
        return

    try:
        print(f"⬇️ 正在下载: {filename}")
        response = session.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ 下载完成: {filename}")
    except Exception as e:
        print(f"❌ 下载 {filename} 失败: {e}")


def main(start_time_str, end_time_str, save_dir='./haf_data'):
    # 解析输入的时间范围（格式：YYYY-MM-DD HH:MM:SS）
    try:
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
    except ValueError as e:
        print(f"时间格式错误，请使用 'YYYY-MM-DD HH:MM:SS' 格式: {e}")
        return

    # 创建本地保存目录
    os.makedirs(save_dir, exist_ok=True)
    base_url = 'https://gong2.nso.edu/ftp/HA/haf/'

    # 第一步：遍历所有「年月」文件夹（格式：YYYYMM）
    year_month_folders = get_links(base_url)
    year_month_folders = [ym for ym in year_month_folders if ym.isdigit() and len(ym) == 6]

    for ym in year_month_folders:
        # 解析年月
        year = int(ym[:4])
        month = int(ym[4:])
        # 计算当前月的时间范围，判断是否与目标时间重叠
        first_day = datetime(year, month, 1)
        if month == 12:
            last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = datetime(year, month + 1, 1) - timedelta(days=1)

        # 若当前月完全不在目标时间范围内，跳过
        if last_day < start_time or first_day > end_time:
            continue

        # 第二步：遍历当前「年月」下的「日」文件夹（格式：YYYYMMDD）
        ym_url = f"{base_url}{ym}/"
        day_folders = get_links(ym_url)
        day_folders = [d for d in day_folders if d.isdigit() and len(d) == 8]

        for day in day_folders:
            day_dt = datetime.strptime(day, '%Y%m%d')
            # 若当前日完全不在目标时间范围内，跳过
            if day_dt.date() < start_time.date() or day_dt.date() > end_time.date():
                continue

            # 第三步：遍历当前「日」下的 .fits.fz 文件
            day_url = f"{ym_url}{day}/"
            files = get_links(day_url)
            fits_files = [f for f in files if f.endswith('.fits.fz')]

            for fits_file in fits_files:
                file_time = parse_filename_to_datetime(fits_file)
                if not file_time:
                    continue
                # 筛选出在目标时间范围内的文件
                if start_time <= file_time <= end_time:
                    file_url = f"{day_url}{fits_file}"
                    download_file(file_url, save_dir)


# 直接运行主程序
if __name__ == '__main__':
    main(START_TIME, END_TIME, SAVE_DIR)