CUDA_VISIBLE_DEVICES=2 python main.py \
    --do-train \
    --do-eval \
    --checkpoint ./outputs/checkpoint_soft_best.pt \
    --emotion-model ../model/emotion/best_emotion.pt \
    --bert-score-baseline /home/liuyuhan/datasets/roberta-large-en/roberta-large.tsv \
    --bert-score-model /home/liuyuhan/datasets/roberta-large-en \
    --output-dir ../outputs
