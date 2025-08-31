#!/usr/bin/env python3
"""
超参数搜索结果分析脚本
分析不同超参数组合的性能表现
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def format_maybe_float(value, fmt):
    try:
        if value is None:
            return "N/A"
        v = float(value)
        return format(v, fmt)
    except Exception:
        return str(value)


def parse_hp_from_dirname(dirname):
    """从目录名解析超参数"""
    # 格式: dalia_lr0.001_alpha0.3_beta0.4_gamma0.5 或 stanfordAF_lr0.001_alpha0.3_beta0.4_gamma0.5
    try:
        parts = dirname.split('_')
        dataset = parts[0]  # 第一个部分是数据集名
        lr = float(parts[1].replace('lr', ''))
        alpha = float(parts[2].replace('alpha', ''))
        beta = float(parts[3].replace('beta', ''))
        gamma = float(parts[4].replace('gamma', '')) if len(parts) > 4 else None
        return dataset, lr, alpha, beta, gamma
    except:
        return None, None, None, None, None

def load_training_log(log_file):
    """加载训练日志"""
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
        return data
    except:
        return None

def extract_metrics(data):
    """提取关键指标"""
    if not data:
        return {}
    
    metrics = {}
    
    # 实验信息
    exp_info = data.get('experiment_info', {})
    metrics['compression_ratio'] = exp_info.get('compression_ratio', 'N/A')
    metrics['teacher_params'] = exp_info.get('teacher_params', {}).get('total', 'N/A')
    metrics['student_params'] = exp_info.get('student_params', {}).get('total', 'N/A')
    
    # 测试结果
    test_results = data.get('test_results', {})
    metrics['test_loss'] = test_results.get('test_loss', 'N/A')
    
    test_metrics = test_results.get('test_metrics', {})
    if 'mse' in test_metrics:
        metrics['test_mse'] = test_metrics['mse']
    if 'mae' in test_metrics:
        metrics['test_mae'] = test_metrics['mae']
    if 'r2' in test_metrics:
        metrics['test_r2'] = test_metrics['r2']
    if 'accuracy' in test_metrics:
        metrics['test_accuracy'] = test_metrics['accuracy']
    if 'f1' in test_metrics:
        metrics['test_f1'] = test_metrics['f1']
    
    # 性能统计
    performance_stats = test_results.get('test_performance', {})
    if 'avg_test_batches_per_second' in performance_stats:
        metrics['avg_test_batches_per_second'] = performance_stats['avg_test_batches_per_second']
    
    # 从experiment_info中获取训练性能统计
    exp_info = data.get('experiment_info', {})
    performance_stats = exp_info.get('performance_stats', {})
    if 'avg_train_batches_per_second' in performance_stats:
        metrics['avg_train_batches_per_second'] = performance_stats['avg_train_batches_per_second']
    
    return metrics

def analyze_hp_search(output_dir="./output_s/hp_search"):
    """分析超参数搜索结果"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        return
    
    results = []
    
    # 遍历所有超参数组合目录
    for hp_dir in output_path.iterdir():
        if not hp_dir.is_dir():
            continue
            
        dirname = hp_dir.name
        
        # 解析超参数
        dataset, lr, alpha, beta, gamma = parse_hp_from_dirname(dirname)
        if lr is None:
            continue
            
        print(f"📁 分析目录: {dirname}")
        print(f"   数据集: {dataset}")
        if gamma is not None:
            print(f"   超参数: lr={lr}, alpha={alpha}, beta={beta}, gamma={gamma}")
        else:
            print(f"   超参数: lr={lr}, alpha={alpha}, beta={beta}")
        
        # 查找训练日志
        log_files = list(hp_dir.glob("*distill_training_log.json"))
        if not log_files:
            print(f"   ❌ 未找到训练日志")
            continue
            
        log_file = log_files[0]
        print(f"   📄 训练日志: {log_file.name}")
        
        # 加载并分析日志
        data = load_training_log(log_file)
        metrics = extract_metrics(data)
        
        # 记录结果
        result = {
            'dataset': dataset,
            'lr': lr,
            'alpha': alpha,
            'beta': beta,
            'dirname': dirname,
            **metrics
        }
        if gamma is not None:
            result['gamma'] = gamma
        results.append(result)
        
        # 打印关键指标
        if 'test_loss' in metrics and metrics['test_loss'] != 'N/A':
            print(f"   📊 测试损失: {metrics['test_loss']:.6f}")
        
        # 根据数据集类型显示不同指标
        if dataset == 'dalia':
            if 'test_mse' in metrics:
                print(f"   📊 测试MSE: {metrics['test_mse']:.6f}")
            if 'test_mae' in metrics:
                print(f"   📊 测试MAE: {metrics['test_mae']:.6f}")
        elif dataset == 'stanfordAF':
            if 'test_accuracy' in metrics:
                print(f"   📊 测试准确率: {metrics['test_accuracy']:.4f}")
            if 'test_f1' in metrics:
                print(f"   📊 测试F1: {metrics['test_f1']:.4f}")
        
        # 显示性能指标
        if 'avg_train_batches_per_second' in metrics:
            print(f"   🚀 训练批次/秒: {format_maybe_float(metrics['avg_train_batches_per_second'], '.2f')}")
        if 'avg_test_batches_per_second' in metrics:
            print(f"   🚀 测试批次/秒: {format_maybe_float(metrics['avg_test_batches_per_second'], '.2f')}")
        
        if 'compression_ratio' in metrics and metrics['compression_ratio'] != 'N/A':
            print(f"   📊 压缩比: {metrics['compression_ratio']}x")
        print()
    
    if not results:
        print("❌ 未找到有效的超参数搜索结果")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 排序（按测试损失，如果有的话）
    if 'test_loss' in df.columns and df['test_loss'].dtype != 'object':
        df = df.sort_values('test_loss')
    
    # 保存分析结果
    output_file = "hp_search_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ 分析结果已保存到: {output_file}")

    # 将最佳test结果汇总到一个文件中（按数据集标准：dalia 使用 test_mae 最小；stanfordAF 使用 test_f1 最大）
    best_summary_rows = []
    best_overall_row = None

    # 规范化数值列
    df_numeric = df.copy()
    if 'test_mae' in df_numeric.columns:
        df_numeric['test_mae'] = pd.to_numeric(df_numeric['test_mae'], errors='coerce')
    if 'test_f1' in df_numeric.columns:
        df_numeric['test_f1'] = pd.to_numeric(df_numeric['test_f1'], errors='coerce')

    # dalia 最佳（test_mae 最小）
    if 'dataset' in df_numeric.columns and 'test_mae' in df_numeric.columns:
        dalia_df = df_numeric[df_numeric['dataset'] == 'dalia'].dropna(subset=['test_mae'])
        if not dalia_df.empty:
            best_dalia = dalia_df.loc[dalia_df['test_mae'].idxmin()]
            best_summary_rows.append({
                'scope': 'dataset:dalia',
                **{k: best_dalia.get(k, 'N/A') for k in [
                    'dataset','dirname','lr','alpha','beta','gamma',
                    'test_loss','test_mse','test_mae','test_accuracy','test_f1',
                    'avg_train_batches_per_second','avg_test_batches_per_second',
                    'compression_ratio','teacher_params','student_params'
                ]}
            })

    # stanfordAF 最佳（test_f1 最大）
    if 'dataset' in df_numeric.columns and 'test_f1' in df_numeric.columns:
        saf_df = df_numeric[df_numeric['dataset'] == 'stanfordAF'].dropna(subset=['test_f1'])
        if not saf_df.empty:
            best_saf = saf_df.loc[saf_df['test_f1'].idxmax()]
            best_summary_rows.append({
                'scope': 'dataset:stanfordAF',
                **{k: best_saf.get(k, 'N/A') for k in [
                    'dataset','dirname','lr','alpha','beta','gamma',
                    'test_loss','test_mse','test_mae','test_accuracy','test_f1',
                    'avg_train_batches_per_second','avg_test_batches_per_second',
                    'compression_ratio','teacher_params','student_params'
                ]}
            })

    # 整体最佳（使用可比较分数：dalia 用 test_mae，stanfordAF 用 -test_f1，分数越小越好）
    candidates = []
    for row in best_summary_rows:
        if row.get('scope') == 'dataset:dalia' and pd.notna(row.get('test_mae')):
            candidates.append((row['test_mae'], row))
        elif row.get('scope') == 'dataset:stanfordAF' and pd.notna(row.get('test_f1')):
            candidates.append((-row['test_f1'], row))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_overall_row = candidates[0][1].copy()
        best_overall_row['scope'] = 'overall'
        best_summary_rows.insert(0, best_overall_row)

    if best_summary_rows:
        best_summary_df = pd.DataFrame(best_summary_rows)
        best_output_file = "best_test_results.csv"
        best_summary_df.to_csv(best_output_file, index=False)
        print(f"✅ 最佳测试结果已汇总到: {best_output_file}")
    else:
        print("⚠️ 未能根据数据集标准生成最佳结果汇总（可能缺少 test_mae 或 test_f1）")

    # 显示最佳结果（依据数据集标准）
    print("\n🏆 最佳超参数组合:")
    if 'best_overall_row' in locals() and best_overall_row is not None:
        best_row = best_overall_row
        if pd.notna(best_row.get('test_loss')):
            print(f"   测试损失: {best_row['test_loss']:.6f}")
        print(f"   数据集: {best_row['dataset']}")

        if best_row['dataset'] == 'dalia':
            if pd.notna(best_row.get('test_mse')):
                print(f"   测试MSE: {best_row['test_mse']:.6f}")
            if pd.notna(best_row.get('test_mae')):
                print(f"   测试MAE: {best_row['test_mae']:.6f}")
        elif best_row['dataset'] == 'stanfordAF':
            if pd.notna(best_row.get('test_accuracy')):
                print(f"   测试准确率: {best_row['test_accuracy']:.4f}")
            if pd.notna(best_row.get('test_f1')):
                print(f"   测试F1: {best_row['test_f1']:.4f}")

        if pd.notna(best_row.get('avg_train_batches_per_second')):
            print(f"   训练批次/秒: {format_maybe_float(best_row['avg_train_batches_per_second'], '.2f')}")
        if pd.notna(best_row.get('avg_test_batches_per_second')):
            print(f"   测试批次/秒: {format_maybe_float(best_row['avg_test_batches_per_second'], '.2f')}")

        print(f"   学习率: {best_row['lr']}")
        print(f"   Alpha: {best_row['alpha']}")
        print(f"   Beta: {best_row['beta']}")
        if 'gamma' in best_row and pd.notna(best_row['gamma']):
            print(f"   Gamma: {best_row['gamma']}")
        print(f"   目录: {best_row['dirname']}")
    
    # 显示所有结果
    print(f"\n📋 所有超参数组合结果 (共{len(df)}个):")
    print(df.to_string(index=False))
    
    # 统计分析
    print(f"\n📊 统计分析:")
    if 'test_loss' in df.columns and df['test_loss'].dtype != 'object':
        print(f"   测试损失范围: {df['test_loss'].min():.6f} - {df['test_loss'].max():.6f}")
        print(f"   测试损失均值: {df['test_loss'].mean():.6f}")
        print(f"   测试损失标准差: {df['test_loss'].std():.6f}")
    
    # 根据数据集类型显示不同指标的统计
    dalia_df = df[df['dataset'] == 'dalia']
    stanfordAF_df = df[df['dataset'] == 'stanfordAF']
    
    if not dalia_df.empty:
        print(f"\n   📊 DALIA数据集统计 ({len(dalia_df)}个结果):")
        if 'test_mse' in dalia_df.columns and dalia_df['test_mse'].dtype != 'object':
            print(f"     测试MSE范围: {dalia_df['test_mse'].min():.6f} - {dalia_df['test_mse'].max():.6f}")
            print(f"     测试MSE均值: {dalia_df['test_mse'].mean():.6f}")
        if 'test_mae' in dalia_df.columns and dalia_df['test_mae'].dtype != 'object':
            print(f"     测试MAE范围: {dalia_df['test_mae'].min():.6f} - {dalia_df['test_mae'].max():.6f}")
            print(f"     测试MAE均值: {dalia_df['test_mae'].mean():.6f}")
    
    if not stanfordAF_df.empty:
        print(f"\n   📊 StanfordAF数据集统计 ({len(stanfordAF_df)}个结果):")
        if 'test_accuracy' in stanfordAF_df.columns and stanfordAF_df['test_accuracy'].dtype != 'object':
            print(f"     测试准确率范围: {stanfordAF_df['test_accuracy'].min():.4f} - {stanfordAF_df['test_accuracy'].max():.4f}")
            print(f"     测试准确率均值: {stanfordAF_df['test_accuracy'].mean():.4f}")
        if 'test_f1' in stanfordAF_df.columns and stanfordAF_df['test_f1'].dtype != 'object':
            print(f"     测试F1范围: {stanfordAF_df['test_f1'].min():.4f} - {stanfordAF_df['test_f1'].max():.4f}")
            print(f"     测试F1均值: {stanfordAF_df['test_f1'].mean():.4f}")
    
    # 性能统计
    print(f"\n   🚀 性能统计:")
    if 'avg_train_batches_per_second' in df.columns and df['avg_train_batches_per_second'].dtype != 'object':
        valid_train_speeds = df['avg_train_batches_per_second'].dropna()
        if not valid_train_speeds.empty:
            print(f"     训练批次/秒范围: {valid_train_speeds.min():.2f} - {valid_train_speeds.max():.2f}")
            print(f"     训练批次/秒均值: {valid_train_speeds.mean():.2f}")
    
    if 'avg_test_batches_per_second' in df.columns and df['avg_test_batches_per_second'].dtype != 'object':
        valid_test_speeds = df['avg_test_batches_per_second'].dropna()
        if not valid_test_speeds.empty:
            print(f"     测试批次/秒范围: {valid_test_speeds.min():.2f} - {valid_test_speeds.max():.2f}")
            print(f"     测试批次/秒均值: {valid_test_speeds.mean():.2f}")
    
    # 超参数影响分析
    print(f"\n🔍 超参数影响分析:")
    
    # 学习率影响
    lr_groups = df.groupby('lr')
    if 'test_loss' in df.columns and df['test_loss'].dtype != 'object':
        print("   学习率影响:")
        for lr, group in lr_groups:
            avg_loss = group['test_loss'].mean()
            print(f"     lr={lr}: 平均损失={avg_loss:.6f}")
    
    # Alpha影响
    alpha_groups = df.groupby('alpha')
    if 'test_loss' in df.columns and df['test_loss'].dtype != 'object':
        print("   Alpha影响:")
        for alpha, group in alpha_groups:
            avg_loss = group['test_loss'].mean()
            print(f"     alpha={alpha}: 平均损失={avg_loss:.6f}")
    
    # Beta影响
    beta_groups = df.groupby('beta')
    if 'test_loss' in df.columns and df['test_loss'].dtype != 'object':
        print("   Beta影响:")
        for beta, group in beta_groups:
            avg_loss = group['test_loss'].mean()
            print(f"     beta={beta}: 平均损失={avg_loss:.6f}")
    
    # Gamma影响（如果存在）
    if 'gamma' in df.columns:
        gamma_groups = df.groupby('gamma')
        if 'test_loss' in df.columns and df['test_loss'].dtype != 'object':
            print("   Gamma影响:")
            for gamma, group in gamma_groups:
                avg_loss = group['test_loss'].mean()
                print(f"     gamma={gamma}: 平均损失={avg_loss:.6f}")
    
    # 数据集影响（显示数据集特定的指标）
    dataset_groups = df.groupby('dataset')
    print("   数据集影响:")
    for dataset, group in dataset_groups:
        if dataset == 'dalia':
            if 'test_mse' in group.columns and group['test_mse'].dtype != 'object':
                avg_mse = group['test_mse'].mean()
                print(f"     {dataset}: 平均MSE={avg_mse:.6f}")
            if 'test_mae' in group.columns and group['test_mae'].dtype != 'object':
                avg_mae = group['test_mae'].mean()
                print(f"     {dataset}: 平均MAE={avg_mae:.6f}")
        elif dataset == 'stanfordAF':
            if 'test_accuracy' in group.columns and group['test_accuracy'].dtype != 'object':
                avg_acc = group['test_accuracy'].mean()
                print(f"     {dataset}: 平均准确率={avg_acc:.4f}")
            if 'test_f1' in group.columns and group['test_f1'].dtype != 'object':
                avg_f1 = group['test_f1'].mean()
                print(f"     {dataset}: 平均F1={avg_f1:.4f}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='分析超参数搜索结果')
    parser.add_argument('--output_dir', type=str, default='./output_s/hp_search',
                       help='超参数搜索输出目录')
    
    args = parser.parse_args()
    
    print("🔍 开始分析超参数搜索结果...")
    print(f"📁 输出目录: {args.output_dir}")
    print()
    
    results_df = analyze_hp_search(args.output_dir)
    
    if results_df is not None:
        print(f"\n✅ 分析完成！共分析了 {len(results_df)} 个超参数组合")
    else:
        print("\n❌ 分析失败")

if __name__ == "__main__":
    main() 