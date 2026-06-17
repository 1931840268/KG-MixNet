import os
import sys

# =================【关键修复】=================
# 强制使用国内镜像，解决 DSW 无法连接 HuggingFace 的问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ============================================

import torch
# 1. 确保能导入 utils (根据 build_kg.py 的路径逻辑)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from utils.taxonomy import CLASS_TO_IDX

# ================= 配置区域 =================
# 必须与 build_kg.py 保持一致，控制变量！
MODEL_NAME = 'all-mpnet-base-v2' 
SAVE_PATH = "class_embeddings_simple.pt"  # 保存为新文件名

# 【关键】简单类别名映射表
SIMPLE_NAMES = {
    "N": "Nitrogen deficiency",
    "P": "Phosphorus deficiency",
    "K": "Potassium deficiency",
    "Ca": "Calcium deficiency",
    "Mg": "Magnesium deficiency",
    "Fe": "Iron deficiency",
    "Mn": "Manganese deficiency",
    "B": "Boron deficiency",
    "Healthy": "Healthy coffee leaf"
}
# ===========================================

def generate_simple():
    print(f"🚀 开始生成简单语义嵌入 (Simple Class Names)...")
    print(f"   > 模型: {MODEL_NAME}")
    print(f"   > 正在尝试从镜像站下载模型，请稍候...")
    
    # 加载模型
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"\n❌ 模型加载失败。请检查网络或尝试手动下载。错误信息:\n{e}")
        return
    
    ordered_texts = []
    print("\n正在准备简单文本:")

    # 严格按照 0-8 的索引顺序提取
    for idx in range(len(CLASS_TO_IDX)):
        # 1. 找到对应的缩写键 (如 'B', 'N')
        abbr_key = [k for k, v in CLASS_TO_IDX.items() if v == idx][0]
        
        # 2. 转换为全称 (如 'Boron deficiency')
        simple_text = SIMPLE_NAMES.get(abbr_key, abbr_key)
        ordered_texts.append(simple_text)
        
        print(f"   [{idx}] {abbr_key} -> '{simple_text}'")

    # 编码
    print("\n开始计算语义向量...")
    with torch.no_grad():
        embeddings = model.encode(ordered_texts, convert_to_tensor=True)

    # 维度检查
    if embeddings.shape[1] != 768:
        print(f"❌ 错误: 维度为 {embeddings.shape[1]}，但 KG-MixNet 需要 768 (MPNet)！")
        return

    print(f"\n✅ 生成完成! Shape: {embeddings.shape}")
    
    # 保存
    torch.save(embeddings, SAVE_PATH)
    print(f"💾 已保存至: {os.path.abspath(SAVE_PATH)}")

if __name__ == "__main__":
    generate_simple()