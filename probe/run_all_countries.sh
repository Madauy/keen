#!/bin/bash

PROMPTS=("paper" "en" "es")
COUNTRIES=("argentina" "chile" "colombia" "costa_rica" "cuba" "ecuador" "el_salvador" "guatemala" "honduras" "mexico" "nicaragua" "panama" "paraguay" "peru" "republica_dominicana" "usa" "venezuela")

LR=0.01
MAX_ITER=100
BATCH_SIZE=32

total=$((${#PROMPTS[@]} * ${#COUNTRIES[@]}))
current=0

for prompt in "${PROMPTS[@]}"; do
    for country in "${COUNTRIES[@]}"; do
        current=$((current + 1))
        echo "[$current/$total] Running: prompt=$prompt, country=$country"
        python main.py --prompt "$prompt" --country "$country" \
            --learning_rate $LR --max_iter $MAX_ITER --batch_size $BATCH_SIZE
    done
done

echo "Completado: $total experimentos ejecutados"
