#!/usr/bin/env bash
#SBATCH --job-name=ppg_distill_hp
#SBATCH --partition=i.q
#SBATCH --mem=65000
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=log/distill_hp_%A_%a.out
#SBATCH --error=log/distill_hp_%A_%a.err
#SBATCH --array=0-127

source ~/anaconda3/etc/profile.d/conda.sh
conda activate py312

# 创建日志目录
mkdir -p log
mkdir -p output_s

# ============================================================================
# 超参数配置区域 - 只需在此处修改扫参范围
# ============================================================================

# 学习率范围 (lr_max) - 使用科学记数法
# LR_VALUES=(1e-3 5e-3)
LR_VALUES=(0.001)

# Alpha值范围 (预测蒸馏损失权重)
ALPHA_VALUES=(0 0.1 0.5 1)

# Beta值范围 (全局特征蒸馏损失权重)  
BETA_VALUES=(0 0.1 0.5 1)

# Gamma值范围 (patch级别蒸馏损失权重)
GAMMA_VALUES=(00.1 0.5 1)

# Weight Decay值范围
WEIGHT_DECAY_VALUES=(0.1 0.0001)

# 数据集范围
DATASETS=(dalia stanfordAF)
# DATASETS=(stanfordAF)

# 模型组合配置
# ============================================================================
# 可选的模型组合 - 取消注释想要使用的组合
# ============================================================================

# 选项1: Papagei -> GPT-1M (支持patch级别蒸馏)
TEACHER_TYPE="gpt_19m"
STUDENT_TYPE="gpt_1m"

# 选项2: Papagei -> MLP (仅全局特征蒸馏)
# TEACHER_TYPE="papagei"
# STUDENT_TYPE="mlp"

# 选项3: GPT-19M -> GPT-1M (支持patch级别蒸馏)
# TEACHER_TYPE="gpt_19m"
# STUDENT_TYPE="gpt_1m"

# 选项4: GPT-19M -> MLP (仅全局特征蒸馏)
# TEACHER_TYPE="gpt_19m"
# STUDENT_TYPE="mlp"

# 其他固定的学习率参数
LR_INIT="1e-5"
LR_FINAL="1e-6" 
LR_SCHEDULE_RATIO="1"
LR_WARM_UP="0.25"

# ============================================================================
# 自动计算总任务数和数组范围 - 无需修改
# ============================================================================

NUM_LR=${#LR_VALUES[@]}
NUM_ALPHA=${#ALPHA_VALUES[@]}
NUM_BETA=${#BETA_VALUES[@]}
NUM_GAMMA=${#GAMMA_VALUES[@]}
NUM_WEIGHT_DECAY=${#WEIGHT_DECAY_VALUES[@]}
NUM_DATASETS=${#DATASETS[@]}

TOTAL_TASKS=$((NUM_LR * NUM_ALPHA * NUM_BETA * NUM_GAMMA * NUM_WEIGHT_DECAY * NUM_DATASETS))
MAX_ARRAY_INDEX=$((TOTAL_TASKS - 1))

echo "=== 超参数扫描配置 ==="
echo "学习率范围 (${NUM_LR}个): ${LR_VALUES[*]}"
echo "Alpha范围 (${NUM_ALPHA}个): ${ALPHA_VALUES[*]}"
echo "Beta范围 (${NUM_BETA}个): ${BETA_VALUES[*]}"
echo "Gamma范围 (${NUM_GAMMA}个): ${GAMMA_VALUES[*]}"
echo "Weight Decay范围 (${NUM_WEIGHT_DECAY}个): ${WEIGHT_DECAY_VALUES[*]}"
echo "数据集范围 (${NUM_DATASETS}个): ${DATASETS[*]}"
echo "总任务数: $TOTAL_TASKS"
echo "数组索引范围: 0-$MAX_ARRAY_INDEX"
echo ""

# 检查SLURM数组任务ID是否在有效范围内
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    if [ $SLURM_ARRAY_TASK_ID -gt $MAX_ARRAY_INDEX ]; then
        echo "❌ 错误: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) 超出有效范围 (0-$MAX_ARRAY_INDEX)"
        echo "请更新SBATCH --array参数为: --array=0-$MAX_ARRAY_INDEX"
        echo ""
        echo "当前参数范围产生的总任务数: $TOTAL_TASKS"
        echo "建议运行以下命令更新脚本:"
        echo "sed -i 's/^#SBATCH --array=.*/#SBATCH --array=0-$MAX_ARRAY_INDEX/' $0"
        exit 1
    fi
    hp_id=$SLURM_ARRAY_TASK_ID
else
    # 如果不在SLURM环境中，使用默认值进行测试
    hp_id=0
    echo "⚠️  警告: 不在SLURM环境中，使用默认任务ID: $hp_id"
fi

# ============================================================================
# 自动计算当前任务的超参数组合 - 无需修改
# ============================================================================

lr_idx=$((hp_id % NUM_LR))
alpha_idx=$(((hp_id / NUM_LR) % NUM_ALPHA))
beta_idx=$(((hp_id / (NUM_LR * NUM_ALPHA)) % NUM_BETA))
gamma_idx=$(((hp_id / (NUM_LR * NUM_ALPHA * NUM_BETA)) % NUM_GAMMA))
weight_decay_idx=$(((hp_id / (NUM_LR * NUM_ALPHA * NUM_BETA * NUM_GAMMA)) % NUM_WEIGHT_DECAY))
dataset_idx=$(((hp_id / (NUM_LR * NUM_ALPHA * NUM_BETA * NUM_GAMMA * NUM_WEIGHT_DECAY)) % NUM_DATASETS))

# 获取对应的超参数值
lr_max=${LR_VALUES[$lr_idx]}
alpha=${ALPHA_VALUES[$alpha_idx]}
beta=${BETA_VALUES[$beta_idx]}
gamma=${GAMMA_VALUES[$gamma_idx]}
weight_decay=${WEIGHT_DECAY_VALUES[$weight_decay_idx]}
dataset=${DATASETS[$dataset_idx]}

# ============================================================================
# 显示当前任务信息
# ============================================================================

echo "=== 当前任务配置 ==="
echo "任务 ID: $hp_id"
echo "Teacher模型: $TEACHER_TYPE"
echo "Student模型: $STUDENT_TYPE"
echo "数据集: $dataset"

# 显示蒸馏类型信息
if [[ "$TEACHER_TYPE" == "papagei" && ("$STUDENT_TYPE" == "gpt_1m" || "$STUDENT_TYPE" == "gpt_19m") ]]; then
    echo "蒸馏类型: Papagei -> GPT (支持patch级别蒸馏)"
elif [[ "$TEACHER_TYPE" == "papagei" ]]; then
    echo "蒸馏类型: Papagei -> $STUDENT_TYPE (仅全局特征蒸馏)"
elif [[ ("$TEACHER_TYPE" == "gpt_19m" || "$TEACHER_TYPE" == "gpt_1m") && ("$STUDENT_TYPE" == "gpt_1m" || "$STUDENT_TYPE" == "gpt_19m") ]]; then
    echo "蒸馏类型: GPT -> GPT (支持patch级别蒸馏)"
else
    echo "蒸馏类型: $TEACHER_TYPE -> $STUDENT_TYPE (仅全局特征蒸馏)"
fi

echo "学习率参数:"
echo "  lr_init: $LR_INIT"
echo "  lr_max: $lr_max"
echo "  lr_final: $LR_FINAL"
echo "  lr_schedule_ratio: $LR_SCHEDULE_RATIO"
echo "  lr_warm_up: $LR_WARM_UP"
echo "蒸馏损失权重:"
echo "  Alpha (预测蒸馏): $alpha"
echo "  Beta (全局特征蒸馏): $beta"
echo "  Gamma (patch级别蒸馏): $gamma"
echo "优化器参数:"
echo "  Weight Decay: $weight_decay"
echo "开始时间: $(date)"
echo ""

# ============================================================================
# 检查Teacher模型
# ============================================================================

teacher_model_path="./output/${TEACHER_TYPE}_${dataset}_best.pth"
if [ ! -f "$teacher_model_path" ]; then
    echo "❌ Teacher模型不存在: $teacher_model_path"
    echo "请先训练teacher模型或检查路径是否正确"
    exit 1
fi

echo "✅ 找到Teacher模型: $teacher_model_path"
teacher_size=$(du -h "$teacher_model_path" | cut -f1)
echo "Teacher模型大小: $teacher_size"

# ============================================================================
# 创建输出目录和配置文件
# ============================================================================

# 创建超参数特定的输出目录
hp_output_dir="./output_s/hp_search/${dataset}_lr${lr_max}_alpha${alpha}_beta${beta}_gamma${gamma}_wd${weight_decay}"
mkdir -p "$hp_output_dir"

# 使用对应的配置文件
if [[ "$TEACHER_TYPE" == "papagei" ]]; then
    if [[ "$STUDENT_TYPE" == "gpt_1m" ]] || [[ "$STUDENT_TYPE" == "gpt_19m" ]]; then
        config_file="config/distillation/papagei_to_gpt_distill.yaml"
    elif [[ "$STUDENT_TYPE" == "mlp" ]]; then
        config_file="config/distillation/papagei_to_mlp_distill.yaml"
    elif [[ "$STUDENT_TYPE" == "linear" ]]; then
        config_file="config/distillation/papagei_to_linear_distill.yaml"
    else
        echo "❌ 不支持的学生模型: $STUDENT_TYPE"
        exit 1
    fi
elif [[ "$TEACHER_TYPE" == "gpt_19m" ]] || [[ "$TEACHER_TYPE" == "gpt_1m" ]]; then
    if [[ "$STUDENT_TYPE" == "gpt_1m" ]] || [[ "$STUDENT_TYPE" == "gpt_19m" ]]; then
        config_file="config/distillation/gpt_to_gpt_patch_distill.yaml"
    elif [[ "$STUDENT_TYPE" == "mlp" ]]; then
        config_file="config/distillation/gpt_to_mlp_distill.yaml"
    elif [[ "$STUDENT_TYPE" == "linear" ]]; then
        config_file="config/distillation/gpt_to_linear_distill.yaml"
    else
        echo "❌ 不支持的学生模型: $STUDENT_TYPE"
        exit 1
    fi
else
    echo "❌ 不支持的教师模型: $TEACHER_TYPE"
    exit 1
fi
echo "✅ 使用配置文件: $config_file"

# 备份原配置文件
backup_config="${config_file}.backup.${hp_id}"
cp "$config_file" "$backup_config"
echo "✅ 备份原配置文件: $backup_config"

# ============================================================================
# 临时修改配置文件
# ============================================================================

# 修改蒸馏损失权重
sed -i "s/alpha: [0-9.]*/alpha: $alpha/" "$config_file"
sed -i "s/beta: [0-9.]*/beta: $beta/" "$config_file"
sed -i "s/gamma: [0-9.]*/gamma: $gamma/" "$config_file"

# 修改学习率参数
sed -i "s/lr_init: [0-9.e-]*/lr_init: $LR_INIT/" "$config_file"
sed -i "s/lr_max: [0-9.e-]*/lr_max: $lr_max/" "$config_file"
sed -i "s/lr_final: [0-9.e-]*/lr_final: $LR_FINAL/" "$config_file"
sed -i "s/lr_schedule_ratio: [0-9.]*/lr_schedule_ratio: $LR_SCHEDULE_RATIO/" "$config_file"
sed -i "s/lr_warm_up: [0-9.]*/lr_warm_up: $LR_WARM_UP/" "$config_file"

# 修改weight decay参数
sed -i "s/weight_decay: [0-9.e-]*/weight_decay: $weight_decay/" "$config_file"

# 确保patch级别蒸馏选项存在（仅对支持的组合）
if [[ "$TEACHER_TYPE" == "papagei" && ("$STUDENT_TYPE" == "gpt_1m" || "$STUDENT_TYPE" == "gpt_19m") ]]; then
    # 确保启用patch级别特征蒸馏
    if ! grep -q "use_patch_feature_distill:" "$config_file"; then
        echo "  use_patch_feature_distill: true" >> "$config_file"
    else
        sed -i "s/use_patch_feature_distill: .*/use_patch_feature_distill: true/" "$config_file"
    fi
    
    # 确保启用patch距离蒸馏
    if ! grep -q "use_patch_distance_distill:" "$config_file"; then
        echo "  use_patch_distance_distill: true" >> "$config_file"
    else
        sed -i "s/use_patch_distance_distill: .*/use_patch_distance_distill: true/" "$config_file"
    fi
    
    echo "✅ 启用Papagei到GPT的patch级别蒸馏功能"
fi

# 确保降参优化选项存在
if ! grep -q "tie_feature_and_patch:" "$config_file"; then
    echo "  tie_feature_and_patch: true" >> "$config_file"
fi

echo "✅ 临时修改配置文件完成"
echo "  蒸馏权重: alpha=$alpha, beta=$beta, gamma=$gamma"
echo "  学习率: lr_max=$lr_max, lr_init=$LR_INIT, lr_final=$LR_FINAL"
echo "  优化器: weight_decay=$weight_decay"

# ============================================================================
# 运行知识蒸馏训练
# ============================================================================

echo ""
echo "=== 开始知识蒸馏训练 ==="
echo "模型: $TEACHER_TYPE -> $STUDENT_TYPE"
echo "数据集: $dataset"
echo "输出目录: $hp_output_dir"

python train_distill.py \
    --teacher_type $TEACHER_TYPE \
    --student_type $STUDENT_TYPE \
    --dataset $dataset \
    --teacher_path $teacher_model_path \
    --save_dir ./output \
    --save_dir_student "$hp_output_dir" \
    --seed 42

training_exit_code=$?
echo ""
echo "完成时间: $(date)"

# ============================================================================
# 检查训练结果
# ============================================================================

if [ $training_exit_code -eq 0 ]; then
    echo "✅ 知识蒸馏训练成功完成!"
    echo ""
    
    # 检查生成的文件
    student_model_file="$hp_output_dir/distill_${TEACHER_TYPE}_to_${STUDENT_TYPE}_${dataset}_best.pth"
    log_file="$hp_output_dir/distill_${TEACHER_TYPE}_to_${STUDENT_TYPE}_${dataset}_distill_training_log.json"
    metrics_file="$hp_output_dir/distill_${TEACHER_TYPE}_to_${STUDENT_TYPE}_${dataset}_distill_metrics.csv"
    
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
        echo "=== 训练结果摘要 ==="
        distill_result=$(python3 -c "
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
        echo "$distill_result" | sed 's/^/  /'
    else
        echo "❌ 训练日志未找到: $log_file"
    fi
    
    if [ -f "$metrics_file" ]; then
        echo "✅ 指标CSV: $metrics_file"
    else
        echo "❌ 指标CSV未找到: $metrics_file"
    fi
    
else
    echo "❌ 知识蒸馏训练失败 (退出代码: $training_exit_code)"
    echo "超参数: lr_max=$lr_max, alpha=$alpha, beta=$beta, gamma=$gamma, weight_decay=$weight_decay"
fi

# ============================================================================
# 清理和完成
# ============================================================================

# 恢复原配置文件
cp "$backup_config" "$config_file"
rm "$backup_config"
echo ""
echo "✅ 已恢复原配置文件"

echo ""
echo "=== 任务完成 ==="
echo "模型: $TEACHER_TYPE -> $STUDENT_TYPE"
echo "数据集: $dataset"
echo "超参数: lr_max=$lr_max, alpha=$alpha, beta=$beta, gamma=$gamma, weight_decay=$weight_decay"
echo "输出目录: $hp_output_dir" 