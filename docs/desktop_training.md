# 데스크탑(GPU) 학습 가이드

> 무거운 학습은 **GPU 달린 데스크탑(4090, 우분투)**에서 한다. Orin은 추론·측정·튜닝만.
> 이 문서는 "정확도를 더 끌어올리려면 데스크탑에서 무엇을 어떻게 학습시키나"를 정리한다.
> 실행 루프/제약은 `CLAUDE.md`, Orin 절차는 `RUN.md` 참고.

---

## 0. 먼저 — 어떤 "학습"이 필요한가?

채점 점수 = **인식 정확도 + latency**. 지금 상태(Orin 실측):

| 구분 | 담당 모델 | 현재 | 더 학습이 도움? |
|---|---|---|---|
| 숫자 **위치 검출** | YOLO (`best.pt`) | conf 0.93, 잘 됨 | 검출을 놓칠 때만 (Lever A) |
| 숫자 **분류(0~9)** | LeNet (`mnistCUDNN/data/*.bin`) | 8=15/15, **6=14/15** | **6/8 올리려면 이게 핵심 (Lever B)** |

→ **6이 가끔 0/5로 틀리는 건 분류기(LeNet) 문제**다. YOLO를 더 학습해도 안 고쳐진다.
   6/8 정확도를 올리는 정답은 **Lever B: LeNet 파인튜닝**이다.

### 기기 간 무엇을 주고받나 (꼭 기억)
- 데스크탑 → Orin 으로 **옮길 수 있는 것**: `best.pt`(YOLO 가중치), `*.bin` 8개(LeNet 가중치).
- **옮길 수 없는 것**: TensorRT `best.engine`. 엔진은 **기기 종속**이라 반드시 **Orin에서** `trtexec`/`yolo export`로 빌드한다. 데스크탑에서 만든 엔진은 Orin에서 안 돈다.

---

## 1. 데스크탑 환경 준비 (1회)

```bash
cd yolo-mnist-cudnn
python -m venv .venv && source .venv/bin/activate        # 선택
pip install -r requirements.txt                          # ultralytics, torch, opencv 등
pip install torchvision                                   # LeNet 파인튜닝(MNIST 로더)용

python -c "import torch; print('cuda', torch.cuda.is_available())"   # True 여야 함
```
> `torch.cuda.is_available()`가 False면 GPU용 torch 휠이 아니다.
> 데스크탑 CUDA에 맞는 휠로 다시 깔 것 (https://pytorch.org). GPU가 핵심 이유.

raw 손글씨 사진 배치 (채점과 같은 조건의 펜 글씨):
```bash
unzip backup/6.zip -d data/raw/      # -> data/raw/6/*.jpg  (약 70장)
unzip backup/8.zip -d data/raw/      # -> data/raw/8/*.jpg  (약 69장)
```

---

## Lever A. YOLO 검출기 재학습 (검출을 놓칠 때만)

검출 자체가 잘 되면 **건너뛴다.** 새 조건/각도에서 박스를 놓칠 때만.

```bash
# 1) 사진 자동 라벨링 -> YOLO 데이터셋
python scripts/01_make_dataset.py            # -> data/yolo/ + data.yaml

# 2) 학습 (데스크탑 GPU). 더 학습시키려면 epochs/데이터를 늘린다.
python scripts/02_train_yolo.py              # -> runs/detect/digit/weights/best.pt
```
더 강하게 학습하는 방법(`scripts/02_train_yolo.py` 상단 상수 수정):
- `EPOCHS` ↑ (예 80 → 150), 데이터가 적으면 과적합 주의(`patience`로 early stop).
- 사진을 더 모아 `data/raw/`에 추가 후 `01` 재실행 → 다양한 각도/조명/배경.
- 검출만 하면 되니 모델은 `yolov8n`(nano) 유지가 Orin에서 빠르다. 정확도 부족 시 `yolov8s`.

산출물 `best.pt`를 Orin으로 복사:
```bash
scp runs/detect/digit/weights/best.pt  <orin>:~/yolo-mnist-cudnn/best.pt
```
> Orin에서 엔진 빌드: `yolo export model=best.pt format=engine half=True device=0`
> (엔진은 Orin에서만! 0절 참고.)

---

## Lever B. LeNet 분류기 파인튜닝 — **6/8 정확도의 핵심**

`mnistCUDNN`이 쓰는 LeNet 가중치(`*.bin`)를, **MNIST + 우리 6/8 손글씨**로 다시 학습해
우리 필체에 적응시킨다. 구조·레이어 순서·`.bin` 레이아웃은 예제와 **반드시 동일**(채점 제약).
스크립트 `finetune/finetune_lenet.py`가 이 레이아웃을 그대로 맞춰 8개 `.bin`을 출력한다.

### B-1. 학습에 쓸 6/8 PGM 만들기 (런타임과 동일 전처리)
파인튜닝은 `preprocess.py`로 정규화된 28×28 PGM을 먹는다. **런타임과 같은 `preprocess.py`
(같은 `THICKEN_FRAC`/`CLOSE_K`)로** 만들어야 학습/추론 분포가 일치한다.
파일명 첫 글자가 정답 라벨이어야 한다(`6_*.pgm`, `8_*.pgm`).

```bash
mkdir -p finetune/our_pgm
python - <<'PY'
import glob, cv2, sys; sys.path.insert(0,".")
from preprocess import normalize_to_mnist, save_pgm
for lbl in ("6","8"):
    for i,f in enumerate(sorted(glob.glob(f"data/raw/{lbl}/*.jpg"))):
        n = normalize_to_mnist(cv2.imread(f, cv2.IMREAD_COLOR))
        if n is not None:
            save_pgm(n, f"finetune/our_pgm/{lbl}_{i:03d}.pgm")
print("done")
PY
```
> 1/3/5도 우리 필체로 모았다면 같은 방식으로 `1_*.pgm` 등 추가하면 더 좋다(없으면 생략 가능 —
> MNIST가 1/3/5를 이미 잘 한다).

### B-2. 파인튜닝 (데스크탑 GPU)
```bash
python finetune/finetune_lenet.py \
    --our-pgm-dir finetune/our_pgm \
    --out finetune/weights \
    --epochs 5 --oversample 30
```
- MNIST 전체에 **우리 6/8을 oversample(x30)로 섞어** 1/3/5를 잊지 않으면서 6/8에 적응한다.
- 6이 계속 약하면: `--oversample` ↑ (예 50), `--epochs` ↑ (예 8~10). 단 과하면 MNIST(1/3/5)를
  잊을 수 있으니 **반드시 1/3/5 게이트로 검증**(아래).
- 산출물: `finetune/weights/`에 `conv1.bin, conv1.bias.bin, conv2.bin, conv2.bias.bin,
  ip1.bin, ip1.bias.bin, ip2.bin, ip2.bias.bin` 8개.

### B-3. Orin에 적용 (원본 백업 필수)
8개 `.bin`을 Orin으로 복사한 뒤 `mnistCUDNN/data/`의 원본을 **백업하고** 덮어쓴다.
바이너리 **재빌드 불필요**(가중치는 런타임 로드).

```bash
# 데스크탑에서
scp finetune/weights/*.bin  <orin>:~/yolo-mnist-cudnn/finetune/weights/

# Orin에서
cd ~/yolo-mnist-cudnn/mnistCUDNN/data
mkdir -p orig_backup && cp *.bin orig_backup/      # ★ 원본 백업 (되돌릴 기준)
cp ~/yolo-mnist-cudnn/finetune/weights/*.bin .     # 8개 덮어쓰기
```

### B-4. 검증 (Orin) — 1/3/5 게이트 + 6/8
```bash
cd ~/yolo-mnist-cudnn
python scripts/04_eval.py --bin ./mnistCUDNN/mnistCUDNN \
    --pgm-dir runtime/pgm --samples-dir mnistCUDNN/data --runs 3
```
- **`[regression ok] 1/3/5 all still correct.`** 가 떠야 한다.
  `[REGRESSION FAIL]`이면 이 가중치는 **버리고** `orig_backup/`으로 되돌린다(채점 제약 위반).
- 6/8 정확도가 올랐는지 표로 확인. 좋으면 채택, 아니면 B-2 노브 조정 후 반복.

> 주의: 1/3/5 회귀 검사는 `mnistCUDNN/data/`의 `one/three/five_28x28.pgm`을 쓴다. 이 샘플 PGM은
> 덮어쓰지 말 것(가중치 `.bin`만 교체). 새 `.bin`으로도 이 세 장이 1/3/5로 나와야 통과.

---

## 2. 하드 제약 (어기면 채점 실패)
1. **LeNet 구조/레이어 순서/`.bin` 레이아웃**은 예제와 동일해야 한다 — `finetune_lenet.py`가
   이미 맞춰져 있으니 모델 정의를 건드리지 말 것.
2. **1/3/5 정확도 유지** — 모든 가중치 교체는 `04_eval.py` 게이트로 검증, 실패 시 롤백.
3. **원본 `.bin` 백업** 후 덮어쓰기.
4. mnistCUDNN의 **9단계 forward 순서 불변** (이건 Orin 패치 쪽 제약, `docs/mnistCUDNN_patch_guide.md`).

## 3. 한 줄 요약 흐름
```
데스크탑:  raw 사진 → (A) YOLO 학습 → best.pt
                    → (B) 6/8 PGM → LeNet 파인튜닝 → 8개 .bin
   ──(scp: best.pt, *.bin)──▶
Orin:      best.pt → best.engine 빌드(여기서만!) → 03 파이프라인
           .bin 교체(원본 백업) → 04_eval (1/3/5 게이트 + 6/8 + latency)
```
