#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test.py - 独立测试脚本
只计算zidong（argmax）指标，与原代码逻辑完全一致
修复权重加载问题
"""

import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryF1Score, BinaryRecall, BinaryPrecision,
    BinarySpecificity, BinaryAccuracy, BinaryAUROC
)
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    BertTokenizer, BertForTokenClassification
)
import tqdm

# 添加自定义模块路径
sys.path.append('/kaggle/input/zidingyi-xiuendata1')
from dataloader import EntDataset, get_data


# ============================================================================
# 配置参数（根据实际情况修改）
# ============================================================================

# 模型配置
MODEL_PATH = "facebook/esm2_t30_150M_UR50D"
WEIGHTS_PATH = '/kaggle/input/9pcy-new-esm-t30-jiudaima/val_model_besteo_2_dp_0.1_lr_5e-05_bz_1.pkl'  # 修改为实际路径

# 数据路径
TEST_FILE = '/kaggle/input/zhou-2wan-part/zhou_2wan_part.csv'

# 其他配置
BATCH_SIZE = 1
#MAX_LEN = 1024
MAX_LEN = 2048
DROPO = 0.1
SEED = 681
OUTPUT_DIR = '/kaggle/working'


# ============================================================================
# 设置随机种子
# ============================================================================

def set_seed(seed: int = 681) -> None:
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# 模型定义（兼容原代码权重）
# ============================================================================

class EpiAntModel(nn.Module):
    def __init__(self, model):
        super(EpiAntModel, self).__init__()
        self.model = model

    def forward(self, batch_token, batch_attention_mask):
        out = self.model(input_ids=batch_token, attention_mask=batch_attention_mask)
        return out['logits']


# ============================================================================
# 调试函数：查看权重结构
# ============================================================================

def debug_weights_structure(weights_path: str):
    """查看权重文件的结构"""
    print("=" * 60)
    print("调试：查看权重文件结构")
    print("=" * 60)

    try:
        state_dict = torch.load(weights_path, map_location='cpu')

        print(f"权重文件路径: {weights_path}")
        print(f"总键数量: {len(state_dict)}")

        print("\n前20个键:")
        for i, k in enumerate(list(state_dict.keys())[:20]):
            print(f"  {i+1:2d}. {k} -> shape: {state_dict[k].shape}")

        # 统计不同前缀
        prefixes = {}
        for k in state_dict.keys():
            prefix = k.split('.')[0]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1

        print("\n键前缀统计:")
        for prefix, count in prefixes.items():
            print(f"  {prefix}: {count}")

        return state_dict
    except Exception as e:
        print(f"加载权重文件失败: {e}")
        return None


# ============================================================================
# 加载模型（修复权重加载问题）
# ============================================================================

def load_model(model_path: str, weights_path: str, dropo: float, device: torch.device):
    """加载模型和权重，修复权重不匹配问题"""
    print(f"Loading pretrained model from: {model_path}")

    # 调试：先查看权重结构
    debug_weights_structure(weights_path)

    if 'esm' in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
        base_model = AutoModelForTokenClassification.from_pretrained(
            model_path, num_labels=2, classifier_dropout=dropo
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        base_model = BertForTokenClassification.from_pretrained(
            model_path, num_labels=2, classifier_dropout=dropo
        )

    net = EpiAntModel(base_model)

    print(f"\nLoading weights from: {weights_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    # 加载权重
    state_dict = torch.load(weights_path, map_location=device)

    # 方法1: 直接尝试加载，允许不匹配
    print("尝试加载权重 (strict=False)...")
    missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)

    print(f"缺失的键: {len(missing_keys)}")
    for k in missing_keys[:10]:  # 只显示前10个
        print(f"  - {k}")
    if len(missing_keys) > 10:
        print(f"  ... 还有 {len(missing_keys) - 10} 个")

    print(f"\n意外的键: {len(unexpected_keys)}")
    for k in unexpected_keys[:10]:  # 只显示前10个
        print(f"  - {k}")
    if len(unexpected_keys) > 10:
        print(f"  ... 还有 {len(unexpected_keys) - 10} 个")

    # 方法2: 如果strict=False不行，尝试调整键名
    if len(unexpected_keys) > 0 or len(missing_keys) > 0:
        print("\n尝试修复键名...")
        new_state_dict = {}
        for k, v in state_dict.items():
            # 尝试去掉可能的冗余前缀
            if k.startswith('module.'):  # 多GPU训练保存的
                new_key = k[7:]
            elif k.startswith('model.'):  # EpiAntModel包装
                new_key = k[6:]
            else:
                new_key = k
            new_state_dict[new_key] = v

        # 再次尝试加载
        print("重新加载权重...")
        missing_keys2, unexpected_keys2 = net.load_state_dict(new_state_dict, strict=False)

        if len(missing_keys2) < len(missing_keys) or len(unexpected_keys2) < len(unexpected_keys):
            print("键名修复成功！")
            print(f"修复后 - 缺失的键: {len(missing_keys2)}")
            print(f"修复后 - 意外的键: {len(unexpected_keys2)}")

    net.to(device)
    net.eval()

    print("\n模型加载完成！")
    return net, tokenizer


# ============================================================================
# 测试函数（与原代码逻辑完全一致）
# ============================================================================

def test_model(
    net: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[Dict, List]:
    """
    执行测试，只计算zidong指标
    与原代码逻辑完全一致
    """

    # 创建独立的指标实例（避免与其他代码混淆）
    f1fun = BinaryF1Score(ignore_index=-100).to(device)
    recallfun = BinaryRecall(ignore_index=-100).to(device)
    precisionfun = BinaryPrecision(ignore_index=-100).to(device)
    specificityfun = BinarySpecificity(ignore_index=-100).to(device)
    accfun = BinaryAccuracy(ignore_index=-100).to(device)
    aucfun = BinaryAUROC(thresholds=None, ignore_index=-100).to(device)

    # 确保指标是干净的
    f1fun.reset()
    recallfun.reset()
    precisionfun.reset()
    specificityfun.reset()
    accfun.reset()
    aucfun.reset()

    net.eval()
    all_predictions = []

    print("\n" + "="*60)
    print("Starting Test...")
    print("="*60)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm.tqdm(test_loader, desc="Testing")):
            raw_text_list, labels, input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 前向传播
            out = net(input_ids, attention_mask)
            out = out[:, 1:-1, :]  # 去掉[CLS]和[SEP]

            b, l, n = out.size()
            attention_mask = attention_mask[:, 1:-1]

            # 维度对齐（与原代码一致）
            if attention_mask.size(1) > labels.size(1):
                attention_mask = attention_mask[:, :labels.size(1)]
            if out.size(1) > labels.size(1):
                out = out[:, :labels.size(1), :]

            # 与原代码完全一致的label masking
            labels = labels + (attention_mask - 1) * 100

            # 更新指标（与原代码完全一致）
            f1fun(out.argmax(dim=-1), labels.long())
            recallfun(out.argmax(dim=-1), labels.long())
            precisionfun(out.argmax(dim=-1), labels.long())
            specificityfun(out.argmax(dim=-1), labels.long())
            accfun(out.argmax(dim=-1), labels.long())
            aucfun(out[:, :, 1], labels.long())  # AUC使用logits

            # 收集预测结果（与原代码一致，使用sigmoid转换为概率）
            probabilities = torch.sigmoid(out[:, :, 1]).tolist()
            probabilities = [[round(p, 4) for p in prob] for prob in probabilities]

            for i in range(len(raw_text_list)):
                all_predictions.append({
                    'true_labels': labels[i].int().tolist(),
                    'pred_probs': probabilities[i],
                })

    # 计算最终结果
    results = {
        'auc': round(aucfun.compute().item(), 4),
        'acc': round(accfun.compute().item(), 4),
        'f1': round(f1fun.compute().item(), 4),
        'recall': round(recallfun.compute().item(), 4),
        'precision': round(precisionfun.compute().item(), 4),
        'specificity': round(specificityfun.compute().item(), 4),
    }

    return results, all_predictions


# ============================================================================
# 主函数
# ============================================================================

def main():
    # 初始化
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    net, tokenizer = load_model(MODEL_PATH, WEIGHTS_PATH, DROPO, device)

    # 加载测试数据
    print(f"\nLoading test data from: {TEST_FILE}")
    test_data = get_data(TEST_FILE)
    print(f"Test samples: {len(test_data)}")

    test_dataset = EntDataset(test_data, tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=test_dataset.collate, 
        shuffle=False
    )

    # 执行测试
    results, predictions = test_model(net, test_loader, device)

    # 打印结果
    print("\n" + "="*60)
    print("                    TEST RESULTS (zidong)")
    print("="*60)
    print(f"  AUC:         {results['auc']:.4f}")
    print(f"  Accuracy:    {results['acc']:.4f}")
    print(f"  F1:          {results['f1']:.4f}")
    print(f"  Recall:      {results['recall']:.4f}")
    print(f"  Precision:   {results['precision']:.4f}")
    print(f"  Specificity: {results['specificity']:.4f}")
    print("="*60)

    # 保存指标结果
    results_df = pd.DataFrame([results])
    results_df.insert(0, 'type', 'zidong')
    metrics_path = os.path.join(OUTPUT_DIR, 'test_result.csv')
    results_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")

    # 保存预测分数
    pred_df = pd.DataFrame(predictions)
    pred_df['auc'] = results['auc']  # 添加整体AUC
    pred_path = os.path.join(OUTPUT_DIR, 'forecast_fenshu.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to: {pred_path}")

    print("\nTesting Complete!")


if __name__ == "__main__":
    main()

