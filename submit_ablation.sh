#!/usr/bin/env bash
#SBATCH --job-name=ppg_ablation
#SBATCH --partition=i.q
#SBATCH --mem=65000
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=log/ablation_%A_%a.out
#SBATCH --error=log/ablation_%A_%a.err
#SBATCH --array=0-5

source ~/anaconda3/etc/profile.d/conda.sh
conda activate py312

# 创建日志目录
mkdir -p log
mkdir -p output_ablation

# ============================================================================
# 消融实验配置区域
# ============================================================================

# 基础参数设置
DATASET="dalia"
TEACHER_TYPE="gpt_19m"
STUDENT_TYPE="gpt_1m"
SAVE_DIR="./output"
SAVE_DIR_STUDENT="./output_ablation"
SEED=42

# 消融实验配置列表
ABLATION_CONFIGS=(
    "full_distill:完整蒸馏(baseline)"
    "no_feature_distill:去除Feature-level蒸馏"
    "no_label_distill:去除Label-level蒸馏"
    "no_patch_contrastive:去除Patch_Contrastive_Level"
    "no_patch_relational:去除Patch_Relational_Level"
    "no_ground_truth:去除Ground_Truth_Loss"
)

TOTAL_EXPERIMENTS=${#ABLATION_CONFIGS[@]}
MAX_ARRAY_INDEX=$((TOTAL_EXPERIMENTS - 1))

echo "=== PPG-GPT 知识蒸馏消融实验 ==="
echo "数据集: $DATASET"
echo "Teacher模型: $TEACHER_TYPE"
echo "Student模型: $STUDENT_TYPE"
echo "总实验数: $TOTAL_EXPERIMENTS"
echo "数组索引范围: 0-$MAX_ARRAY_INDEX"
echo ""

# 检查SLURM数组任务ID是否在有效范围内
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    if [ $SLURM_ARRAY_TASK_ID -gt $MAX_ARRAY_INDEX ]; then
        echo "❌ 错误: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) 超出有效范围 (0-$MAX_ARRAY_INDEX)"
        echo "请更新SBATCH --array参数为: --array=0-$MAX_ARRAY_INDEX"
        exit 1
    fi
    exp_id=$SLURM_ARRAY_TASK_ID
else
    # 如果不在SLURM环境中，使用默认值进行测试
    exp_id=0
    echo "⚠️  警告: 不在SLURM环境中，使用默认实验ID: $exp_id"
fi

# 获取当前实验配置
current_config=${ABLATION_CONFIGS[$exp_id]}
config_name=${current_config%%:*}
exp_description=${current_config##*:}

echo "=== 当前实验配置 ==="
echo "实验 ID: $exp_id"
echo "配置名称: $config_name"
echo "实验描述: $exp_description"
echo "开始时间: $(date)"
echo ""

# ============================================================================
# 检查Teacher模型
# ============================================================================

teacher_model_path="$SAVE_DIR/${TEACHER_TYPE}_${DATASET}_best.pth"
if [ ! -f "$teacher_model_path" ]; then
    echo "❌ Teacher模型不存在: $teacher_model_path"
    echo "请先训练teacher模型或检查路径是否正确"
    exit 1
fi

echo "✅ 找到Teacher模型: $teacher_model_path"
teacher_size=$(du -h "$teacher_model_path" | cut -f1)
echo "Teacher模型大小: $teacher_size"

# ============================================================================
# 创建临时配置文件
# ============================================================================

# 创建临时配置目录
TEMP_CONFIG_DIR="./config/ablation_temp_${SLURM_JOB_ID}_${exp_id}"
mkdir -p $TEMP_CONFIG_DIR

echo "创建消融实验配置文件: $config_name"

# 根据实验ID创建对应的配置文件
case $exp_id in
    0)  # 完整蒸馏 (baseline)
        cat > $TEMP_CONFIG_DIR/${config_name}.yaml << 'EOF'
distillation:
  alpha: 0.1      # 预测蒸馏损失权重
  beta: 0.1       # 全局特征蒸馏损失权重
  gamma: 0.5      # patch级别蒸馏损失权重
  temperature: 2.0
  
  use_feature_distill: true
  feature_loss_weight: 1.0
  
  use_patch_feature_distill: true
  patch_feature_loss_weight: 1.0
  patch_distill_mode: 'contrastive'
  contrastive_temperature: auto
  
  use_patch_distance_distill: true
  patch_distance_loss_weight: 1.0
  
  tie_feature_and_patch: true

train_config:
  epochs: 120
  batch_size: 64
  lr_init: 1e-5
  lr_max: 0.001
  lr_final: 1e-6
  lr_schedule_ratio: 1
  lr_warm_up: 0.25
  optimizer: "AdamW"
  weight_decay: 0.1
  
  save_freq: 10
  early_stop_patience: 20
  early_stop_min_delta: 1e-6
EOF
        ;;
    1)  # 去除 Feature-level 蒸馏
        cat > $TEMP_CONFIG_DIR/${config_name}.yaml << 'EOF'
distillation:
  alpha: 0.1      # 预测蒸馏损失权重
  beta: 0.0       # 全局特征蒸馏损失权重 (设为0)
  gamma: 0.5      # patch级别蒸馏损失权重
  temperature: 2.0
  
  use_feature_distill: false
  feature_loss_weight: 0.0
  
  use_patch_feature_distill: true
  patch_feature_loss_weight: 1.0
  patch_distill_mode: 'contrastive'
  contrastive_temperature: auto
  
  use_patch_distance_distill: true
  patch_distance_loss_weight: 1.0
  
  tie_feature_and_patch: true

train_config:
  epochs: 120
  batch_size: 64
  lr_init: 1e-5
  lr_max: 0.001
  lr_final: 1e-6
  lr_schedule_ratio: 1
  lr_warm_up: 0.25
  optimizer: "AdamW"
  weight_decay: 0.1
  
  save_freq: 10
  early_stop_patience: 20
  early_stop_min_delta: 1e-6
EOF
        ;;
    2)  # 去除 Label-level (预测蒸馏)
        cat > $TEMP_CONFIG_DIR/${config_name}.yaml << 'EOF'
distillation:
  alpha: 0.0      # 预测蒸馏损失权重 (设为0)
  beta: 0.1       # 全局特征蒸馏损失权重
  gamma: 0.5      # patch级别蒸馏损失权重
  temperature: 2.0
  
  use_feature_distill: true
  feature_loss_weight: 1.0
  
  use_patch_feature_distill: true
  patch_feature_loss_weight: 1.0
  patch_distill_mode: 'contrastive'
  contrastive_temperature: auto
  
  use_patch_distance_distill: true
  patch_distance_loss_weight: 1.0
  
  tie_feature_and_patch: true

train_config:
  epochs: 120
  batch_size: 64
  lr_init: 1e-5
  lr_max: 0.001
  lr_final: 1e-6
  lr_schedule_ratio: 1
  lr_warm_up: 0.25
  optimizer: "AdamW"
  weight_decay: 0.1
  
  save_freq: 10
  early_stop_patience: 20
  early_stop_min_delta: 1e-6
EOF
        ;;
    3)  # 去除 Patch Contrastive Level
        cat > $TEMP_CONFIG_DIR/${config_name}.yaml << 'EOF'
distillation:
  alpha: 0.1      # 预测蒸馏损失权重
  beta: 0.1       # 全局特征蒸馏损失权重
  gamma: 0.5      # patch级别蒸馏损失权重
  temperature: 2.0
  
  use_feature_distill: true
  feature_loss_weight: 1.0
  
  # 完全禁用patch特征蒸馏（去除contrastive）
  use_patch_feature_distill: false
  patch_feature_loss_weight: 0.0
  patch_distill_mode: 'direct'
  contrastive_temperature: auto
  
  # 保持patch距离蒸馏
  use_patch_distance_distill: true
  patch_distance_loss_weight: 1.0
  
  tie_feature_and_patch: true

train_config:
  epochs: 120
  batch_size: 64
  lr_init: 1e-5
  lr_max: 0.001
  lr_final: 1e-6
  lr_schedule_ratio: 1
  lr_warm_up: 0.25
  optimizer: "AdamW"
  weight_decay: 0.1
  
  save_freq: 10
  early_stop_patience: 20
  early_stop_min_delta: 1e-6
EOF
        ;;
    4)  # 去除 Patch Relational Level (距离蒸馏)
        cat > $TEMP_CONFIG_DIR/${config_name}.yaml << 'EOF'
distillation:
  alpha: 0.1      # 预测蒸馏损失权重
  beta: 0.1       # 全局特征蒸馏损失权重
  gamma: 0.5      # patch级别蒸馏损失权重
  temperature: 2.0
  
  use_feature_distill: true
  feature_loss_weight: 1.0
  
  # 保持patch特征蒸馏
  use_patch_feature_distill: true
  patch_feature_loss_weight: 1.0
  patch_distill_mode: 'contrastive'
  contrastive_temperature: auto
  
  # 禁用patch距离蒸馏
  use_patch_distance_distill: false
  patch_distance_loss_weight: 0.0
  
  tie_feature_and_patch: true

train_config:
  epochs: 120
  batch_size: 64
  lr_init: 1e-5
  lr_max: 0.001
  lr_final: 1e-6
  lr_schedule_ratio: 1
  lr_warm_up: 0.25
  optimizer: "AdamW"
  weight_decay: 0.1
  
  save_freq: 10
  early_stop_patience: 20
  early_stop_min_delta: 1e-6
EOF
        ;;
    5)  # 去除 Ground Truth Loss
        cat > $TEMP_CONFIG_DIR/${config_name}.yaml << 'EOF'
distillation:
  alpha: 0.8      # 增加预测蒸馏损失权重来补偿
  beta: 0.1       # 全局特征蒸馏损失权重
  gamma: 0.5      # patch级别蒸馏损失权重
  temperature: 2.0
  
  use_feature_distill: true
  feature_loss_weight: 1.0
  
  use_patch_feature_distill: true
  patch_feature_loss_weight: 1.0
  patch_distill_mode: 'contrastive'
  contrastive_temperature: auto
  
  use_patch_distance_distill: true
  patch_distance_loss_weight: 1.0
  
  tie_feature_and_patch: true

train_config:
  epochs: 120
  batch_size: 64
  lr_init: 1e-5
  lr_max: 0.001
  lr_final: 1e-6
  lr_schedule_ratio: 1
  lr_warm_up: 0.25
  optimizer: "AdamW"
  weight_decay: 0.1
  
  save_freq: 10
  early_stop_patience: 20
  early_stop_min_delta: 1e-6
EOF
        ;;
    *)
        echo "❌ 错误: 无效的实验ID: $exp_id"
        exit 1
        ;;
esac

echo "✅ 创建配置文件完成: $TEMP_CONFIG_DIR/${config_name}.yaml"

# ============================================================================
# 创建消融实验训练脚本
# ============================================================================

# 创建修改版的训练脚本来支持自定义配置文件
cat > train_distill_ablation.py << 'EOF'
#!/usr/bin/env python
"""
消融实验版本的知识蒸馏训练脚本
支持自定义配置文件路径和去除ground truth损失
"""

import os
import sys
sys.path.append('.')
import argparse
import torch
import logging
from logging import info as lprint
import yaml
import json

# 导入原始训练脚本的所有组件
from train_distill import *

def load_custom_config(distill_config_path, student_config_path, data_config_path):
    """加载自定义配置文件"""
    distill_config = load_config(distill_config_path)
    student_model_config = load_config(student_config_path)
    data_config = load_config(data_config_path)
    
    # 合并配置
    config = {
        'distillation': distill_config['distillation'],
        'train_config': distill_config['train_config'],
        'data_config': data_config['data_config']
    }
    
    # 获取任务类型
    task_type = data_config['data_config'].get('task_type', 'regression')
    
    return config, student_model_config['model_config'], task_type

class AblationDistillationLoss(DistillationLoss):
    """消融实验版本的蒸馏损失，支持完全去除ground truth损失"""
    
    def __init__(self, *args, use_ground_truth=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ground_truth = use_ground_truth
    
    def forward(self, student_output, teacher_output, student_features=None, 
                teacher_features=None, targets=None, task_type='regression',
                student_patch_features=None, teacher_patch_features=None):
        """
        计算蒸馏损失（支持去除ground truth损失的消融实验）
        """
        
        # 调用父类方法获取所有损失
        loss_dict = super().forward(
            student_output, teacher_output, student_features, 
            teacher_features, targets, task_type,
            student_patch_features, teacher_patch_features
        )
        
        # 如果禁用ground truth损失，重新计算总损失
        if not self.use_ground_truth:
            total = 0.0
            # 只包含蒸馏损失，不包含ground truth损失
            total += self.alpha * loss_dict['pred_distill_loss']  # 预测蒸馏损失
            
            if self.tie_feature_and_patch:
                # 合并模式
                if loss_dict['feature_distill_loss'].item() > 0:
                    total += self.beta * loss_dict['feature_distill_loss']
                
                # 合并patch级别损失
                patch_combined = 0.0
                if loss_dict['patch_feature_distill_loss'].item() > 0:
                    patch_combined += loss_dict['patch_feature_distill_loss']
                if loss_dict['patch_distance_distill_loss'].item() > 0:
                    patch_combined += loss_dict['patch_distance_distill_loss']
                
                if patch_combined > 0:
                    total += self.gamma * patch_combined
            else:
                # 分离模式
                if loss_dict['feature_distill_loss'].item() > 0:
                    total += self.beta * loss_dict['feature_distill_loss']
                if loss_dict['patch_feature_distill_loss'].item() > 0:
                    total += self.gamma * loss_dict['patch_feature_distill_loss']
                if loss_dict['patch_distance_distill_loss'].item() > 0:
                    total += self.gamma * loss_dict['patch_distance_distill_loss']
            
            loss_dict['total_loss'] = total
            # 将ground truth损失设为0（用于记录）
            loss_dict['gt_loss'] = torch.tensor(0.0, device=total.device)
        
        return loss_dict

class AblationDistillationTrainer(DistillationTrainer):
    """消融实验版本的蒸馏训练器"""
    
    def __init__(self, teacher_model, student_model, train_loader, val_loader, 
                 config, device, save_path, task_type='regression', test_loader=None,
                 use_ground_truth=True):
        
        # 先调用父类初始化（但不使用父类的distill_loss）
        super().__init__(teacher_model, student_model, train_loader, val_loader, 
                        config, device, save_path, task_type, test_loader)
        
        # 创建消融版本的蒸馏损失
        distill_config = config['distillation']
        self.distill_loss = AblationDistillationLoss(
            alpha=distill_config.get('alpha', 0.5),
            beta=distill_config.get('beta', 0.5),
            gamma=distill_config.get('gamma', 0.5),
            temperature=distill_config.get('temperature', 4.0),
            use_feature_distill=distill_config.get('use_feature_distill', True),
            feature_loss_weight=distill_config.get('feature_loss_weight', 1.0),
            use_patch_feature_distill=distill_config.get('use_patch_feature_distill', False),
            patch_feature_loss_weight=distill_config.get('patch_feature_loss_weight', 1.0),
            patch_distill_mode=distill_config.get('patch_distill_mode', 'direct'),
            contrastive_temperature=distill_config.get('contrastive_temperature', 0.1),
            use_patch_distance_distill=distill_config.get('use_patch_distance_distill', False),
            patch_distance_loss_weight=distill_config.get('patch_distance_loss_weight', 1.0),
            tie_feature_and_patch=distill_config.get('tie_feature_and_patch', True),
            use_ground_truth=use_ground_truth  # 新增参数
        )
        
        # 重新设置特征适配器
        teacher_dim = self.get_feature_dim(teacher_model)
        student_dim = self.get_feature_dim(student_model)
        
        teacher_patch_dim = None
        student_patch_dim = None
        if self.distill_loss.use_patch_feature_distill:
            teacher_patch_dim = self.get_patch_feature_dim(teacher_model)
            student_patch_dim = self.get_patch_feature_dim(student_model)
        
        self.distill_loss.setup_feature_adapter(teacher_dim, student_dim, teacher_patch_dim, student_patch_dim)
        
        # 将适配器移动到设备
        if self.distill_loss.feature_adapter is not None:
            self.distill_loss.feature_adapter = self.distill_loss.feature_adapter.to(device)
        
        if self.distill_loss.patch_distance_distiller is not None:
            self.distill_loss.patch_distance_distiller = self.distill_loss.patch_distance_distiller.to(device)
        
        # 重新创建优化器（包含新的适配器参数）
        optimizer_params = list(self.student_model.parameters())
        if self.distill_loss.feature_adapter is not None:
            optimizer_params.extend(self.distill_loss.feature_adapter.parameters())
        
        train_config = config['train_config']
        optimizer_type = train_config.get('optimizer', 'Adam').lower()
        lr = float(train_config.get('lr_init', 1e-5))
        weight_decay = float(train_config.get('weight_decay', 0.0))
        
        if optimizer_type == 'adam':
            self.optimizer = Adam(optimizer_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            self.optimizer = AdamW(optimizer_params, lr=lr, weight_decay=weight_decay)

def main():
    parser = argparse.ArgumentParser(description='PPG模型知识蒸馏消融实验')
    parser.add_argument('--teacher_type', type=str, required=True, help='Teacher模型类型')
    parser.add_argument('--student_type', type=str, required=True, help='Student模型类型')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--teacher_path', type=str, help='Teacher模型路径')
    parser.add_argument('--save_dir', type=str, default='./output', help='Teacher模型保存目录')
    parser.add_argument('--save_dir_student', type=str, default='./output_ablation', help='Student模型保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--config_path', type=str, required=True, help='蒸馏配置文件路径')
    parser.add_argument('--no_ground_truth', action='store_true', help='去除ground truth损失')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载自定义配置
    student_config_path = f"config/models/gpt_config_{args.dataset}.yaml"
    data_config_path = f"config/data/{args.dataset}_data.yaml"
    
    config, student_config, task_type = load_custom_config(
        args.config_path, student_config_path, data_config_path
    )
    
    # 自动推断teacher模型路径
    if args.teacher_path is None:
        teacher_path = auto_infer_teacher_path(args.teacher_type, args.dataset, args.save_dir)
    else:
        teacher_path = args.teacher_path
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lprint(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir_student, exist_ok=True)
    
    # 构建保存路径
    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    save_path = os.path.join(args.save_dir_student, 
                           f"ablation_{config_name}_{args.teacher_type}_to_{args.student_type}_{args.dataset}")
    
    # 数据加载
    data_cfg = config['data_config']
    patch_size = student_config.get('patch_size', 40)
    n_patches = calc_n_patches(data_cfg['train_data_path'], patch_size)
    student_config['n_patches'] = n_patches
    
    # 加载模型
    teacher_model = load_teacher_model(teacher_path, args.teacher_type, device, n_patches)
    student_model = create_model(student_config, args.student_type)
    student_model = student_model.to(device)
    
    # 创建数据加载器
    train_dataset = PretrainDataset(
        data_cfg['train_data_path'],
        patch_size=patch_size,
        train_labels_dataset_path=data_cfg.get('train_label_path', ''),
        data_red_factor=data_cfg.get('data_red_factor', 1)
    )
    
    val_dataset = None
    if data_cfg.get('val_data_path'):
        val_dataset = PretrainDataset(
            data_cfg['val_data_path'],
            patch_size=patch_size,
            train_labels_dataset_path=data_cfg.get('val_label_path', ''),
            data_red_factor=data_cfg.get('data_red_factor', 1)
        )
    
    test_dataset = None
    if data_cfg.get('test_data_path'):
        test_dataset = PretrainDataset(
            data_cfg['test_data_path'],
            patch_size=patch_size,
            train_labels_dataset_path=data_cfg.get('test_label_path', ''),
            data_red_factor=data_cfg.get('data_red_factor', 1)
        )
    
    train_config = config['train_config']
    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=0) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=0) if test_dataset else None
    
    # 创建消融版本的训练器
    trainer = AblationDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_path=save_path,
        task_type=task_type,
        test_loader=test_loader,
        use_ground_truth=not args.no_ground_truth  # 控制是否使用ground truth损失
    )
    
    # 开始训练
    test_loss, test_metrics = trainer.train()
    
    if test_loader and test_loss is not None:
        lprint(f"\n最终测试损失: {test_loss:.4f}")
        lprint(f"最终测试指标: {test_metrics}")

if __name__ == "__main__":
    main()
EOF

# ============================================================================
# 创建实验特定的输出目录
# ============================================================================

exp_output_dir="$SAVE_DIR_STUDENT/${config_name}_${TEACHER_TYPE}_to_${STUDENT_TYPE}_${DATASET}"
mkdir -p "$exp_output_dir"

echo "✅ 创建实验输出目录: $exp_output_dir"

# ============================================================================
# 运行知识蒸馏消融实验
# ============================================================================

echo ""
echo "=== 开始消融实验 ==="
echo "实验: $exp_description"
echo "配置: $config_name"
echo "模型: $TEACHER_TYPE -> $STUDENT_TYPE"
echo "数据集: $DATASET"
echo "输出目录: $exp_output_dir"

# 设置额外的参数（如果是去除ground truth实验）
extra_args=""
if [ "$config_name" = "no_ground_truth" ]; then
    extra_args="--no_ground_truth"
fi

python train_distill_ablation.py \
    --teacher_type $TEACHER_TYPE \
    --student_type $STUDENT_TYPE \
    --dataset $DATASET \
    --teacher_path $teacher_model_path \
    --save_dir $SAVE_DIR \
    --save_dir_student "$exp_output_dir" \
    --seed $SEED \
    --config_path $TEMP_CONFIG_DIR/${config_name}.yaml \
    $extra_args

training_exit_code=$?
echo ""
echo "完成时间: $(date)"

# ============================================================================
# 检查训练结果
# ============================================================================

if [ $training_exit_code -eq 0 ]; then
    echo "✅ 消融实验成功完成!"
    echo ""
    
    # 检查生成的文件
    student_model_file="$exp_output_dir/ablation_${config_name}_${TEACHER_TYPE}_to_${STUDENT_TYPE}_${DATASET}_best.pth"
    log_file="$exp_output_dir/ablation_${config_name}_${TEACHER_TYPE}_to_${STUDENT_TYPE}_${DATASET}_distill_training_log.json"
    metrics_file="$exp_output_dir/ablation_${config_name}_${TEACHER_TYPE}_to_${STUDENT_TYPE}_${DATASET}_distill_metrics.csv"
    
    echo "=== 生成文件检查 ==="
    if [ -f "$student_model_file" ]; then
        student_size=$(du -h "$student_model_file" | cut -f1)
        echo "✅ Student模型: $student_model_file (大小: $student_size)"
        
        # 计算模型大小比较
        teacher_size_bytes=$(stat -f%z "$teacher_model_path" 2>/dev/null || stat -c%s "$teacher_model_path" 2>/dev/null)
        student_size_bytes=$(stat -f%z "$student_model_file" 2>/dev/null || stat -c%s "$student_model_file" 2>/dev/null)
        if [ -n "$teacher_size_bytes" ] && [ -n "$student_size_bytes" ] && [ $teacher_size_bytes -gt 0 ]; then
            compression_ratio=$(echo "scale=2; $teacher_size_bytes / $student_size_bytes" | bc 2>/dev/null || echo "计算失败")
            echo "📊 文件压缩比: ${compression_ratio}x (${teacher_size} -> ${student_size})"
        fi
    else
        echo "❌ Student模型文件未找到: $student_model_file"
    fi
    
    if [ -f "$log_file" ]; then
        echo "✅ 训练日志: $log_file"
        
        # 提取关键结果信息
        echo ""
        echo "=== 实验结果摘要 ==="
        ablation_result=$(python3 -c "
import json
try:
    with open('$log_file', 'r') as f:
        data = json.load(f)
    
    exp_info = data.get('experiment_info', {})
    test_results = data.get('test_results', {})
    
    # 压缩比信息
    compression_ratio = exp_info.get('compression_ratio', 'N/A')
    teacher_params = exp_info.get('teacher_params', {}).get('total', 'N/A')
    student_params = exp_info.get('student_params', {}).get('total', 'N/A')
    
    print(f'参数压缩比: {compression_ratio}x')
    if isinstance(teacher_params, int) and isinstance(student_params, int):
        print(f'参数数量: {teacher_params:,} -> {student_params:,}')
    
    # 训练信息
    best_epoch = exp_info.get('best_epoch', 'N/A')
    best_val_loss = exp_info.get('best_val_loss', 'N/A')
    early_stopped = exp_info.get('early_stopped', False)
    
    print(f'最佳epoch: {best_epoch}')
    print(f'最佳验证损失: {best_val_loss:.6f}' if isinstance(best_val_loss, (int, float)) else f'最佳验证损失: {best_val_loss}')
    print(f'早停: {\"是\" if early_stopped else \"否\"}')
    
    # 测试结果
    if 'test_loss' in test_results:
        print(f'测试损失: {test_results[\"test_loss\"]:.6f}')
        test_metrics = test_results.get('test_metrics', {})
        if 'mse' in test_metrics:
            print(f'测试MSE: {test_metrics[\"mse\"]:.6f}')
        if 'mae' in test_metrics:
            print(f'测试MAE: {test_metrics[\"mae\"]:.6f}')
    else:
        print('测试结果: 未找到')
        
except Exception as e:
    print(f'读取结果失败: {e}')
" 2>/dev/null)
        echo "$ablation_result" | sed 's/^/  /'
    else
        echo "❌ 训练日志未找到: $log_file"
    fi
    
    if [ -f "$metrics_file" ]; then
        echo "✅ 指标CSV: $metrics_file"
    else
        echo "❌ 指标CSV未找到: $metrics_file"
    fi
    
else
    echo "❌ 消融实验失败 (退出代码: $training_exit_code)"
    echo "实验: $exp_description"
    echo "配置: $config_name"
fi

# ============================================================================
# 清理和完成
# ============================================================================

# 清理临时文件
rm -rf $TEMP_CONFIG_DIR
rm -f train_distill_ablation.py

echo ""
echo "=== 实验完成 ==="
echo "实验: $exp_description"
echo "配置: $config_name"
echo "模型: $TEACHER_TYPE -> $STUDENT_TYPE"
echo "数据集: $DATASET"
echo "输出目录: $exp_output_dir" 