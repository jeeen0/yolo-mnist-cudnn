#!/bin/bash

PGM_DIR="pgm_output" #pgm 저장된 폴더
EXE="./mnistCUDNN"     #mnist 실행파일

# 정답 라벨: 최대 ?개
answers=(6 6 6 6 1 5 6 8 6 3 8 6) #예시

total=0
correct=0
total_infer_ms=0

for img in $(find "$PGM_DIR" -name "*.pgm" | sort -V | head -n 10)
do
    echo "================================"
    echo "INPUT: $img"

    output=$(${EXE} image="$img")

    result=$(echo "$output" | grep "Result of classification")
    infer_line=$(echo "$output" | grep "Inference time:")

    digit=$(echo "$result" | awk '{print $4}')
    infer_ms=$(echo "$infer_line" | awk '{print $3}')

    gt=${answers[$total]}

    echo "정답: $gt , 추론: $digit"
    echo "Inference time: $infer_ms ms"

    if [[ "$digit" == "$gt" ]]; then
        echo "결과: O"
        correct=$((correct+1))
    else
        echo "결과: X"
    fi

    if [[ "$infer_ms" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        total_infer_ms=$(echo "$total_infer_ms + $infer_ms" | bc -l)
    fi

    total=$((total+1))
done

if [ "$total" -gt 0 ]; then
    total_infer_sec=$(echo "$total_infer_ms / 1000" | bc -l)
    avg_infer_ms=$(echo "$total_infer_ms / $total" | bc -l)
else
    total_infer_sec=0
    avg_infer_ms=0
fi

echo
echo "============= SUMMARY ============="
echo "Total Images              : $total"
echo "Correct Predictions       : $correct"
echo "Total Inference Time      : $total_infer_ms ms"
echo "Total Inference Time      : $total_infer_sec sec"
echo "Average Inference / Image : $avg_infer_ms ms"
echo "==================================="