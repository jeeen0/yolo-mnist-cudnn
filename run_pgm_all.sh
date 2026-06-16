#!/usr/bin/env bash
# 로컬 테스트용 — 교수님 채점 셸(run_pgm_all.sh)과 같은 스타일로
# 제출본 바이너리(mnistCUDNN (1)/mnistCUDNN)를 PGM 폴더에 돌려
# 정답/추론/Inference time + SUMMARY를 출력한다.
#
# 사용법:
#   ./run_pgm_all.sh [PGM_DIR] [LABELS_FILE]
#   PGM_DIR     : 분류할 PGM 폴더 (기본 runtime/pgm)
#   LABELS_FILE : 정답 시퀀스 파일(공백 구분, 등장 순서). 없으면 정답 비교 생략.
#
# 예:
#   ./run_pgm_all.sh runtime/pgm "clips/handwriting_test.mp4.labels.txt"
set -uo pipefail
cd "$(dirname "$0")"

PGM_DIR="${1:-runtime/pgm}"
LABELS_FILE="${2:-}"
BIN="${BIN:-mnistCUDNN/mnistCUDNN}"

if [ ! -x "$BIN" ]; then
  echo "ERROR: 바이너리 없음/실행불가: $BIN  (먼저 'cd mnistCUDNN && make')"; exit 1
fi

# 교수님 바이너리는 가중치를 CWD의 ./data/ 에서 찾으므로, data/가 있는
# 바이너리 폴더에서 실행해야 한다. 그래서 모든 경로를 절대경로로 만든다.
BIN_DIR="$(cd "$(dirname "$BIN")" && pwd)"
BIN_ABS="$BIN_DIR/$(basename "$BIN")"
[ -n "$LABELS_FILE" ] && [ -f "$LABELS_FILE" ] && LABELS_FILE="$(cd "$(dirname "$LABELS_FILE")" && pwd)/$(basename "$LABELS_FILE")"

# PGM 파일을 이름순(= 등장 순서, _seq 포함)으로 정렬, 절대경로로
mapfile -t PGMS < <(ls "$PGM_DIR"/*.pgm 2>/dev/null | sort | while read -r f; do echo "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"; done)
if [ "${#PGMS[@]}" -eq 0 ]; then
  echo "ERROR: $PGM_DIR 에 PGM이 없습니다. 먼저 03_pipeline.py로 PGM을 만드세요."; exit 1
fi

# 정답 시퀀스 로드 (있으면)
LABELS=()
if [ -n "$LABELS_FILE" ] && [ -f "$LABELS_FILE" ]; then
  read -r -a LABELS < <(tr -s '[:space:]' ' ' < "$LABELS_FILE")
fi

total=0; correct=0; total_ms=0
for i in "${!PGMS[@]}"; do
  p="${PGMS[$i]}"
  out="$(cd "$BIN_DIR" && "$BIN_ABS" --image="$p" 2>/dev/null)"
  pred="$(echo "$out" | grep 'Result of classification' | grep -oE '[0-9]+' | tail -1)"
  ms="$(echo "$out"  | grep 'Inference time'         | grep -oE '[0-9]+\.?[0-9]*' | head -1)"

  echo "================================"
  echo "INPUT: $p"
  if [ "${#LABELS[@]}" -gt 0 ] && [ -n "${LABELS[$i]:-}" ]; then
    truth="${LABELS[$i]}"
    if [ "$pred" = "$truth" ]; then res="O"; correct=$((correct+1)); else res="X"; fi
    echo "정답: $truth , 추론: $pred"
    echo "Inference time: $ms ms"
    echo "결과: $res"
  else
    echo "추론: $pred"
    echo "Inference time: $ms ms"
  fi
  total=$((total+1))
  total_ms="$(echo "$total_ms + ${ms:-0}" | bc -l)"
done

echo "================================"
echo
echo "============= SUMMARY ============="
echo "Total Images              : $total"
if [ "${#LABELS[@]}" -gt 0 ]; then
  echo "Correct Predictions       : $correct"
fi
printf "Total Inference Time      : %s ms\n"  "$total_ms"
printf "Total Inference Time      : %.6f sec\n" "$(echo "$total_ms/1000" | bc -l)"
printf "Average Inference / Image : %.6f ms\n"  "$(echo "$total_ms/$total" | bc -l)"
echo "==================================="
