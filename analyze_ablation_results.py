#!/usr/bin/env python
"""
PPG-GPT 知识蒸馏消融实验结果分析脚本
"""

import os
import json
import pandas as pd
from pathlib import Path
import argparse

def analyze_ablation_results(output_dir="./output_ablation", dataset="dalia", 
                           teacher_type="gpt_19m", student_type="gpt_1m"):
    """分析消融实验结果"""
    
    results = []
    
    # 消融实验名称映射
    experiment_names = {
        'full_distill': '完整蒸馏 (baseline)',
        'no_feature_distill': '去除 Feature-level',
        'no_label_distill': '去除 Label-level', 
        'no_patch_contrastive': '去除 Patch Contrastive (禁用patch特征蒸馏)',
        'no_patch_relational': '去除 Patch Relational',
        'no_ground_truth': '去除 Ground Truth'
    }
    
    print("=== PPG-GPT 知识蒸馏消融实验结果分析 ===")
    print(f"数据集: {dataset}")
    print(f"Teacher模型: {teacher_type}")
    print(f"Student模型: {student_type}")
    print(f"输出目录: {output_dir}")
    print("")
    
    for config_name, exp_name in experiment_names.items():
        # 构建日志文件路径
        exp_dir = f"{config_name}_{teacher_type}_to_{student_type}_{dataset}"
        log_pattern = f"ablation_{config_name}_{teacher_type}_to_{student_type}_{dataset}_distill_training_log.json"
        log_file = os.path.join(output_dir, exp_dir, log_pattern)
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                
                # 提取关键信息
                best_info = log_data.get('best_model_info', {})
                test_results = log_data.get('test_results', {})
                exp_info = log_data.get('experiment_info', {})
                
                result = {
                    'experiment': exp_name,
                    'config_name': config_name,
                    'best_epoch': best_info.get('epoch', 'N/A'),
                    'best_val_loss': best_info.get('val_loss', 'N/A'),
                    'test_loss': test_results.get('test_loss', 'N/A'),
                    'compression_ratio': exp_info.get('compression_ratio', 'N/A'),
                    'early_stopped': exp_info.get('early_stopped', False),
                    'total_training_time': exp_info.get('total_training_time', 'N/A'),
                    'teacher_params': exp_info.get('teacher_params', {}).get('total', 'N/A'),
                    'student_params': exp_info.get('student_params', {}).get('total', 'N/A')
                }
                
                # 添加测试指标
                test_metrics = test_results.get('test_metrics', {})
                for metric_name, metric_value in test_metrics.items():
                    result[f'test_{metric_name}'] = metric_value
                
                # 添加训练损失组件信息（来自最佳模型）
                train_losses = best_info.get('train_losses', {})
                for loss_name, loss_value in train_losses.items():
                    result[f'train_{loss_name}'] = loss_value
                
                results.append(result)
                print(f"✓ 已分析: {exp_name}")
                
            except Exception as e:
                print(f"✗ 分析 {exp_name} 时出错: {e}")
        else:
            print(f"✗ 未找到日志文件: {log_file}")
    
    if results:
        # 创建DataFrame并保存结果
        df = pd.DataFrame(results)
        
        # 保存详细结果
        csv_file = os.path.join(output_dir, f"ablation_results_summary_{teacher_type}_to_{student_type}_{dataset}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"\n详细结果已保存到: {csv_file}")
        
        # 显示汇总表格
        print("\n消融实验结果汇总:")
        print("=" * 120)
        
        # 选择关键列显示
        display_cols = ['experiment', 'best_val_loss', 'test_loss']
        if 'test_mse' in df.columns:
            display_cols.append('test_mse')
        if 'test_mae' in df.columns:
            display_cols.append('test_mae')
        display_cols.extend(['best_epoch', 'early_stopped'])
        
        display_df = df[display_cols].copy()
        
        # 格式化数值列
        for col in ['best_val_loss', 'test_loss', 'test_mse', 'test_mae']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else str(x))
        
        print(display_df.to_string(index=False))
        print("=" * 120)
        
        # 计算相对于baseline的性能变化
        baseline_idx = df[df['config_name'] == 'full_distill'].index
        if len(baseline_idx) > 0:
            baseline = df.iloc[baseline_idx[0]]
            print(f"\n相对于baseline的性能变化:")
            print("-" * 80)
            
            for _, row in df.iterrows():
                if row['config_name'] != 'full_distill':
                    if isinstance(row['test_loss'], (int, float)) and isinstance(baseline['test_loss'], (int, float)):
                        loss_change = ((row['test_loss'] - baseline['test_loss']) / baseline['test_loss']) * 100
                        print(f"{row['experiment']}: 测试损失变化 {loss_change:+.2f}%")
                    
                    # 显示MSE变化（如果有）
                    if 'test_mse' in df.columns and isinstance(row['test_mse'], (int, float)) and isinstance(baseline['test_mse'], (int, float)):
                        mse_change = ((row['test_mse'] - baseline['test_mse']) / baseline['test_mse']) * 100
                        print(f"    MSE变化: {mse_change:+.2f}%")
        
        # 显示训练损失组件分析
        print(f"\n训练损失组件分析:")
        print("-" * 80)
        
        loss_components = ['train_total', 'train_gt', 'train_pred_distill', 'train_feature_distill', 
                          'train_patch_feature_distill', 'train_patch_distance_distill']
        
        for component in loss_components:
            if component in df.columns:
                print(f"\n{component}:")
                for _, row in df.iterrows():
                    if isinstance(row[component], (int, float)):
                        print(f"  {row['experiment']}: {row[component]:.6f}")
        
        # 保存训练损失组件分析
        loss_analysis_file = os.path.join(output_dir, f"ablation_loss_components_{teacher_type}_to_{student_type}_{dataset}.csv")
        loss_df = df[['experiment', 'config_name'] + [col for col in loss_components if col in df.columns]]
        loss_df.to_csv(loss_analysis_file, index=False, encoding='utf-8')
        print(f"\n训练损失组件分析已保存到: {loss_analysis_file}")
        
    else:
        print("未找到任何有效的实验结果")
        print(f"请确保在 {output_dir} 目录下存在以下子目录:")
        for config_name in experiment_names.keys():
            exp_dir = f"{config_name}_{teacher_type}_to_{student_type}_{dataset}"
            print(f"  - {exp_dir}/")

def main():
    parser = argparse.ArgumentParser(description='分析PPG知识蒸馏消融实验结果')
    parser.add_argument('--output_dir', type=str, default='./output_ablation', help='实验输出目录')
    parser.add_argument('--dataset', type=str, default='dalia', help='数据集名称')
    parser.add_argument('--teacher_type', type=str, default='gpt_19m', help='Teacher模型类型')
    parser.add_argument('--student_type', type=str, default='gpt_1m', help='Student模型类型')
    
    args = parser.parse_args()
    
    analyze_ablation_results(args.output_dir, args.dataset, args.teacher_type, args.student_type)

if __name__ == "__main__":
    main() 