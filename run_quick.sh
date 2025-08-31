python train.py --model_type gpt_1m --dataset dalia
python train.py --model_type mlp --dataset dalia
python train.py --model_type linear --dataset dalia

python train.py --model_type mlp --dataset stanfordAF

python train_distill.py --teacher_type gpt_19m --student_type gpt_1m --dataset dalia
python train_distill.py --teacher_type gpt_19m --student_type mlp --dataset dalia
python train_distill.py --teacher_type gpt_19m --student_type linear --dataset dalia

python train.py --model_type gpt_19m --dataset dalia --eval_only
python train.py --model_type gpt_1m --dataset dalia --eval_only
python train.py --model_type mlp --dataset stanfordAF --eval_only

python train_distill.py \
    --teacher_type gpt_19m \
    --student_type gpt_1m \
    --dataset dalia

python train_distill.py \
    --teacher_type gpt_19m \
    --student_type gpt_1m \
    --dataset stanfordAF