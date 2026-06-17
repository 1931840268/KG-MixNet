import torch
import os
import sys

# 将父目录加入 path 以便导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from utils.taxonomy import CLASS_DESCRIPTIONS, CLASS_TO_IDX


def build_knowledge_graph_embeddings():
    # ================= 配置区域 =================
    # 使用更强的 MPNet Base 模型 (768维)
    # 这能更好地捕捉我们在 taxonomy.py 中定义的复杂病理学语义
    MODEL_NAME = 'all-mpnet-base-v2'
    SAVE_PATH = "class_embeddings.pt"
    # ===========================================

    print(f"正在加载 SBERT 模型: {MODEL_NAME} ...")
    # 如果第一次运行，会自动下载模型权重 (约 400MB)
    model = SentenceTransformer(MODEL_NAME)

    # 准备文本列表 (严格按照索引 0-8 排序)
    ordered_texts = []
    print("\n正在编码以下类别描述:")

    # 按照 0 到 8 的顺序提取文本，确保与 label 索引对齐
    for idx in range(len(CLASS_TO_IDX)):
        # 反查类别名 (例如 0 -> 'B')
        class_name = [k for k, v in CLASS_TO_IDX.items() if v == idx][0]
        text = CLASS_DESCRIPTIONS[class_name]
        ordered_texts.append(text)

        # 打印预览，确认文本内容
        print(f"[{idx}] {class_name}: {text[:60]}...")

        # 编码 (Encoding)
    print("\n开始计算语义向量...")
    with torch.no_grad():
        # convert_to_tensor=True 会直接返回 PyTorch Tensor
        embeddings = model.encode(ordered_texts, convert_to_tensor=True)

    # 检查形状
    expected_dim = 768
    if embeddings.shape[1] != expected_dim:
        print(f"[警告] 模型输出维度为 {embeddings.shape[1]}，预期为 {expected_dim}。请确认模型选择。")

    print(f"\n编码完成! 最终语义矩阵形状: {embeddings.shape}")
    print(f"  - 类别数 (Rows): {embeddings.shape[0]}")
    print(f"  - 语义维度 (Cols): {embeddings.shape[1]} (匹配 MPNet-Base)")

    # 保存
    torch.save(embeddings, SAVE_PATH)
    print(f"语义向量已保存至: {os.path.abspath(SAVE_PATH)}")


if __name__ == "__main__":
    build_knowledge_graph_embeddings()