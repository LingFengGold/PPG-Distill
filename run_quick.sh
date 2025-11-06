python train.py --model_type gpt_1m --dataset dalia
python train.py --model_type papagei --dataset dalia
python train.py --model_type mlp --dataset dalia

python train.py --model_type gpt_1m --dataset stanfordAF
python train.py --model_type papagei --dataset stanfordAF
python train.py --model_type mlp --dataset stanfordAF

python train_distill.py --teacher_type papagei --student_type gpt_1m --dataset dalia
python train_distill.py --teacher_type papagei --student_type mlp --dataset dalia

python train_distill.py --teacher_type papagei --student_type gpt_1m --dataset stanfordAF
python train_distill.py --teacher_type papagei --student_type mlp --dataset stanfordAF

python train.py --model_type papagei --dataset dalia --eval_only