#!/usr/bin/env bash
#SBATCH --job-name=ppg_train
#SBATCH --partition=i.q
#SBATCH --mem=65000
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=log/train_%A_%a.out
#SBATCH --error=log/train_%A_%a.err
#SBATCH --array=0-1


source ~/anaconda3/etc/profile.d/conda.sh
conda activate py312


# 创建日志目录
mkdir -p log
mkdir -p output

# 定义模型类型和数据集的组合
models=("papagei")
datasets=("dalia" "stanfordAF")

# 计算当前任务的模型和数据集
model_idx=$((SLURM_ARRAY_TASK_ID / 2))
dataset_idx=$((SLURM_ARRAY_TASK_ID % 2))

model_type=${models[$model_idx]}
dataset=${datasets[$dataset_idx]}

echo "任务 ID: $SLURM_ARRAY_TASK_ID"
echo "模型类型: $model_type"
echo "数据集: $dataset"
echo "开始时间: $(date)"

# 运行训练
echo "开始训练: $model_type on $dataset"
python train.py \
    --model_type $model_type \
    --dataset $dataset \
    --save_dir ./output \
    --seed 42

training_exit_code=$?
echo "完成时间: $(date)"

# 检查训练是否成功
if [ $training_exit_code -eq 0 ]; then
    echo "✅ 训练成功完成: $model_type on $dataset"
    
    # 检查生成的文件
    model_file="./output/${model_type}_${dataset}_best.pth"
    log_file="./output/${model_type}_${dataset}_training_log.json"
    metrics_file="./output/${model_type}_${dataset}_metrics.csv"
    
    echo "检查生成的文件:"
    if [ -f "$model_file" ]; then
        model_size=$(du -h "$model_file" | cut -f1)
        echo "  ✅ 模型文件: $model_file (大小: $model_size)"
    else
        echo "  ❌ 模型文件未找到: $model_file"
    fi
    
    if [ -f "$log_file" ]; then
        echo "  ✅ 训练日志: $log_file"
        # 尝试提取测试结果
        test_result=$(python3 -c "
import json
try:
    with open('$log_file', 'r') as f:
        data = json.load(f)
    test_results = data.get('test_results', {})
    if 'test_loss' in test_results:
        print(f'测试损失: {test_results[\"test_loss\"]:.6f}')
        test_metrics = test_results.get('test_metrics', {})
        if 'mse' in test_metrics:
            print(f'测试MSE: {test_metrics[\"mse\"]:.6f}')
    else:
        print('测试结果未找到')
except Exception as e:
    print(f'读取测试结果失败: {e}')
" 2>/dev/null)
        echo "  📊 $test_result"
    else
        echo "  ❌ 训练日志未找到: $log_file"
    fi
    
    if [ -f "$metrics_file" ]; then
        echo "  ✅ 指标CSV: $metrics_file"
    else
        echo "  ❌ 指标CSV未找到: $metrics_file"
    fi
    
else
    echo "❌ 训练失败: $model_type on $dataset (退出代码: $training_exit_code)"
fi

echo "任务完成: $model_type on $dataset"

