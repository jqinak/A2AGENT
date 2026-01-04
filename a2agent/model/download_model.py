import os
import argparse
from huggingface_hub import snapshot_download

def download_dataset(repo_id, token, local_dir, hf_home=None, repo_type="dataset", allow_patterns=None):
    """
    从 Hugging Face Hub 下载数据集
    
    Args:
        repo_id: 仓库ID
        token: Hugging Face token
        local_dir: 本地保存目录
        hf_home: HF缓存目录（可选）
        repo_type: 仓库类型（默认：dataset）
        allow_patterns: 允许下载的文件模式（可选）
    """
    # 设置 HF_HOME 环境变量（如果提供）
    if hf_home:
        os.environ['HF_HOME'] = hf_home
        print(f"设置 HF_HOME 为: {hf_home}")
    
    # 打印下载信息
    print(f"开始下载数据集: {repo_id}")
    print(f"保存到: {local_dir}")
    if allow_patterns:
        print(f"下载模式: {allow_patterns}")
    
    try:
        # 下载数据集
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            token=token,
            local_dir=local_dir
        )
        print(f"下载完成！数据集已保存到: {local_dir}")
    except Exception as e:
        print(f"下载失败: {str(e)}")
        raise

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='从 Hugging Face Hub 下载数据集')
    
    # 添加参数
    parser.add_argument('--repo-id', required=True, help='Hugging Face 仓库ID')
    parser.add_argument('--token', required=True, help='Hugging Face 访问令牌')
    parser.add_argument('--local-dir', required=True, help='本地保存目录')
    parser.add_argument('--hf-home', help='HF缓存目录（可选）')
    parser.add_argument('--repo-type', default='dataset', help='仓库类型（默认: dataset）')
    parser.add_argument('--allow-patterns', help='允许下载的文件模式（可选，例如: "folder/*"）')
    
    # 解析参数
    args = parser.parse_args()
    
    # 调用下载函数
    download_dataset(
        repo_id=args.repo_id,
        token=args.token,
        local_dir=args.local_dir,
        hf_home=args.hf_home,
        repo_type=args.repo_type,
        allow_patterns=args.allow_patterns
    )

if __name__ == "__main__":
    main()




import os
from huggingface_hub import snapshot_download

# 设置 HF_HOME 环境变量
os.environ['HF_HOME'] = '/project/peilab/qjl/HFCache'

snapshot_download(
    repo_id="hiyouga/geometry3k",
    repo_type="dataset",
    # allow_patterns="2_3_m_academic_v0_1/*",  # 指定要下载的文件夹
    token="",
    local_dir="/project/peilab/qjl/CODE/DATA/geometry3k"
)