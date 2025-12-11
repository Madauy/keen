#!/bin/bash

# Ejecuta evaluate_probe.py para todas las combinaciones de país/prompt,
# para el dataset completo con cada prompt y para regiones (latam, usa).

PROMPTS=("paper" "en" "es")
COUNTRIES=("argentina" "chile" "colombia" "costa_rica" "cuba" "ecuador" "el_salvador" "guatemala" "honduras" "mexico" "nicaragua" "panama" "paraguay" "peru" "republica_dominicana" "usa" "venezuela")
REGIONS=("latam" "usa")

LR=0.01
MAX_ITER=100
BATCH_SIZE=32

total_country_runs=$((${#PROMPTS[@]} * ${#COUNTRIES[@]}))
total_full_dataset_runs=${#PROMPTS[@]}
total_region_runs=$((${#PROMPTS[@]} * ${#REGIONS[@]}))
total_runs=$((total_country_runs + total_full_dataset_runs + total_region_runs))
current=0

# País + prompt
for prompt in "${PROMPTS[@]}"; do
    for country in "${COUNTRIES[@]}"; do
        current=$((current + 1))
        echo "[$current/$total_runs] Evaluando: prompt=$prompt, country=$country"
        uv run evaluate_probe.py --prompt "$prompt" --country "$country" \
            --learning_rate $LR --max_iter $MAX_ITER --batch_size $BATCH_SIZE
    done
done

# Dataset completo (sin country/region)
for prompt in "${PROMPTS[@]}"; do
    current=$((current + 1))
    echo "[$current/$total_runs] Evaluando dataset completo: prompt=$prompt"
    uv run evaluate_probe.py --prompt "$prompt" \
        --learning_rate $LR --max_iter $MAX_ITER --batch_size $BATCH_SIZE
done

# Regiones (latam, usa) por prompt
for prompt in "${PROMPTS[@]}"; do
    for region in "${REGIONS[@]}"; do
        current=$((current + 1))
        echo "[$current/$total_runs] Evaluando: prompt=$prompt, region=$region"
        uv run evaluate_probe.py --prompt "$prompt" --region "$region" \
            --learning_rate $LR --max_iter $MAX_ITER --batch_size $BATCH_SIZE
    done
done

echo "Completado: $total_runs evaluaciones ejecutadas"
