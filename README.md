# yolo-mnist-cudnn

손글씨 동영상을 YOLO로 감지 → 28×28 PGM 저장 → mnistCUDNN으로 분류하는
파이프라인. Jetson Orin Nano에서 **인식 정확도**와 **수행시간**을 함께 최적화하는
것을 목표로 한다.

## 구성

```
preprocess.py            MNIST식 28x28 정규화 (정확도의 핵심 레버)
scripts/01_make_dataset  사진 자동 라벨링 -> YOLO 데이터셋
scripts/02_train_yolo    YOLO 학습 + export        (데스크톱 GPU)
scripts/03_pipeline      동영상 -> YOLO -> 크롭 -> PGM   (Orin)
scripts/04_eval          분류 + 숫자별 정확도/latency + 1/3/5 회귀 검사 (Orin)
run_all.sh               03 -> 04 묶음 실행
finetune/finetune_lenet  예비용: LeNet 파인튜닝 -> 8개 .bin 재출력
mnistCUDNN/              예제 소스(소스, Makefile, .bin, 샘플 PGM) — 직접 넣어야 함
```

## 동작 원리

- **YOLO**: 숫자의 위치만 검출(단일 클래스 `digit`). 어떤 숫자인지는 판별하지 않음.
- **mnistCUDNN**: 잘라낸 28×28 PGM을 받아 실제로 0~9를 분류.
- 정확도를 가르는 핵심은 **전처리(`preprocess.py`)**. 흰 글자/검은 배경,
  획 두껍게, 20×20로 비율 유지 스케일, 무게중심 기준 28×28 중앙 배치로
  MNIST 분포에 맞춘다. 흐린 사진은 `THICKEN_FRAC`, `CLOSE_K` 노브로 보정.

## 빠른 시작

**데스크톱 (학습)**
```bash
pip install -r requirements.txt
# 사진을 data/raw/6, data/raw/8 에 넣는다
python scripts/01_make_dataset.py
python scripts/02_train_yolo.py        # -> runs/detect/digit/weights/best.pt
```

**Orin (런타임)**
```bash
# best.pt 를 복사한 뒤:
yolo export model=best.pt format=engine half=True device=0   # -> best.engine
cd mnistCUDNN && make && cd ..
./run_all.sh                            # -> runtime/results/results.csv
```

## 제약 조건

- mnistCUDNN의 **9단계 forward 구조와 순서는 변경하지 않는다.** 주변
  (알고리즘 선택, 정밀도, 컨텍스트 재사용, 타이머)만 최적화한다.
- 예제의 **1/3/5 정확도는 반드시 유지**한다. 모든 변경은 `scripts/04_eval.py`의
  1/3/5 회귀 검사로 검증한다.
- 파인튜닝 시 LeNet 구조와 8개 `.bin` 레이아웃은 예제와 동일해야 한다.

## 수행시간 측정

수행시간은 **프로그램 자체 출력으로 측정된 전체 시간(H2D/D2H 및 입출력 포함)**
기준이다. 패치한 mnistCUDNN이 `LATENCY_MS=<ms>`를 출력하면 `04_eval.py`가 이를
파싱한다. 여러 번 반복 후 중앙값을 보고한다.
