"""
Kaggle 竞赛数据集下载脚本
竞赛: comp-5434-20261-project-task-2
数据存放: /Volumes/T9/code/5434_project/data
"""

import os
import shutil
import kagglehub
from dotenv import load_dotenv

def main():
    # 加载项目根目录的 .env 文件
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    load_dotenv(env_path)

    # 脚本所在目录即项目的 data 目录
    data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(data_dir, exist_ok=True)

    print(f"正在下载竞赛数据集到: {data_dir} ...")

    # 下载数据集（kagglehub 默认下载到缓存目录）
    cache_path = kagglehub.competition_download("comp-5434-20261-project-task-2")
    print(f"数据集已缓存到: {cache_path}")

    # 将缓存目录中的所有文件拷贝到目标数据目录
    for filename in os.listdir(cache_path):
        src = os.path.join(cache_path, filename)
        dst = os.path.join(data_dir, filename)
        shutil.copy2(src, dst)
        print(f"  已拷贝: {filename}")

    print(f"全部文件已拷贝到: {data_dir}")
    print("下载完成！")

if __name__ == "__main__":
    main()
