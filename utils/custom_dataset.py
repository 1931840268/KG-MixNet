import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from .taxonomy import CLASS_TO_IDX


class CoffeeDataset(Dataset):
    """
    通用咖啡叶片数据集封装 (GZSL 专用)
    """

    def __init__(self, img_paths, labels, transform=None, mode='source'):
        """
        Args:
            img_paths: 图片路径列表
            labels: 标签列表
            transform: 预处理 (Resize, Normalize)
            mode:
                - 'source': 训练模式。标签必须是单类别字符串 (如 'N')。返回 int 索引。
                - 'target': 测试模式。标签是多类别列表 (如 ['N', 'P'])。返回原始字符串列表用于评估。
        """
        assert mode in ('source', 'target'), "Mode must be 'source' or 'target'"
        self.img_paths = list(img_paths)
        self.labels = list(labels)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        raw_label = self.labels[idx]

        # 1. 健壮的图像读取
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Corrupt image skipped: {path}, error: {e}")
            # 返回全黑图像防止训练中断
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        # 2. 标签处理逻辑
        if self.mode == 'source':
            # --- 训练模式 (必须是单标签) ---
            if not isinstance(raw_label, str):
                # 如果万一混入了非字符串标签，强制取第一个或报错
                if isinstance(raw_label, (list, tuple)):
                    raw_label = raw_label[0]
                else:
                    raise ValueError(f"Source label must be str, got: {type(raw_label)} at {path}")

            # 转为数字索引 (0-8)
            label_out = CLASS_TO_IDX.get(raw_label, -1)
            if label_out == -1:
                print(f"[Warning] Unknown label '{raw_label}' in source data.")

            # 返回: (图片, 索引, 占位符)
            # 这里的 raw_label 只是为了方便调试打印，训练用不到
            return image, label_out, raw_label

        else:
            # --- 测试/目标域模式 (支持多标签) ---
            # 统一转为 list 格式: ['N', 'P']
            if isinstance(raw_label, str):
                raw_label_list = [raw_label]
            else:
                raw_label_list = list(raw_label)

            # 返回: (图片, 占位索引, 真实标签列表)
            # label_idx 返回 -1，因为测试集不需要计算 CrossEntropyLoss
            # raw_label_list 将被 DataLoader 收集起来，用于计算 Top-k Accuracy
            return image, -1, raw_label_list