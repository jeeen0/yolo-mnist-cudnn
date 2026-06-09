# mnistCUDNN 수정 가이드 (latency 패치 + 측정)

이 문서는 Orin의 `cudnn_samples_v7/mnistCUDNN/mnistCUDNN.cpp`를 수정해
**수행시간을 출력으로 측정**하고 **불필요한 작업을 제거**해 latency를 줄이기
위한 정확한 위치/방법을 정리한다. 줄 번호는 강의자료(lab10) 스크린샷 기준이며,
실제 파일과 ±몇 줄 차이가 날 수 있으니 **함수명/문자열로 위치를 찾는다.**

> 절대 규칙: `classify_example` 안의 **9단계 forward 호출 순서**(convolute →
> pool → convolute → pool → fc → activation(ReLU) → LRN → fc → softmax)는
> 그대로 둔다. 아래는 전부 그 *주변*만 손댄다.

확인된 사실(강의자료 + 실제 실행 출력):
- `main`은 기본 실행 시 `Testing single precision` + `Testing half precision`을
  **둘 다** 돌리고, one/three/five 3장을 분류해 `Result of classification: 1 3 5`
  와 `Test passed!`를 출력한다.
- 단일 이미지 경로: `checkCmdLineFlag(argc, argv, "image")`가 참이면
  `getCmdLineArgumentString(... "image" ...)`로 받은 한 장을 분류하고
  `Result of classification: <한 자리>`만 출력한다 (slide 866~882).
- `convoluteForward`(slide 484~)는 `convAlgorithm < 0`일 때
  `cudnnFindConvolutionForwardAlgorithm`을 호출해 **매 실행 알고리즘을 탐색**하고
  `Testing cudnnFindConvolutionForwardAlgorithm ...`를 출력한다 (slide 554~566).
- 현재 출력에는 **latency를 알려주는 값이 전혀 없다.** → 직접 심어야 한다.

---

## 패치 0. 원본 백업 (먼저!)

```bash
cd ~/CUDA_Lab/cuDNN-sample/cudnn_samples_v7/mnistCUDNN   # 실제 경로에 맞게
cp mnistCUDNN.cpp mnistCUDNN.cpp.orig
```

`.orig`는 1/3/5 회귀 비교의 기준이자, 패치가 틀렸을 때 되돌릴 원본이다.

---

## 패치 1. LATENCY_MS 타이머 (수행시간 출력) — **가장 먼저, 필수**

요구사항: "수행시간은 출력을 통해 측정된 전체 시간 — 호스트/디바이스 통신 및
입출력 포함." → `classify_example` **전체**(이미지 읽기 + H2D + 9단계 + 결과까지)를
감싸 ms로 출력한다. CUDA는 비동기이므로 **끝에서 `cudaDeviceSynchronize()`**로
GPU 작업 완료를 기다린 뒤 시간을 멈춘다.

`classify_example`(slide 766~)의 시작과 끝에 삽입:

```cpp
// 파일 상단 include 부근에 한 번만:
#include <chrono>

int classify_example(const char* fname, const Layer_t<value_type>& conv1,
                     const Layer_t<value_type>& conv2,
                     const Layer_t<value_type>& ip1,
                     const Layer_t<value_type>& ip2)
{
    auto _t0 = std::chrono::high_resolution_clock::now();   // <-- 함수 맨 처음

    int n,c,h,w;
    value_type *srcData = NULL, *dstData = NULL;
    value_type imgData_h[IMAGE_H*IMAGE_W];
    readImage(fname, imgData_h);                 // 파일 I/O 포함됨
    // ... (기존 9단계 forward 그대로) ...

    // 함수가 결과(int)를 return 하기 직전:
    cudaDeviceSynchronize();                     // GPU 작업 완료 대기 (필수)
    auto _t1 = std::chrono::high_resolution_clock::now();
    double _ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    std::cout << "LATENCY_MS=" << _ms << std::endl;   // 04_eval.py가 파싱
    return /* 기존 return 값 */;
}
```

주의: `classify_example`에 `return`이 여러 곳이면, **결과를 내보내는 정상 경로**의
return 직전에 넣는다(보통 함수 끝 1곳). `scripts/04_eval.py`는 이
`LATENCY_MS=<float>`를 정규식으로 읽고, 없으면 벽시계로 폴백하며 경고를 띄운다.

> 여러 번 실행해 **중앙값**을 보고한다(04_eval.py의 `--runs`가 처리). 첫 실행은
> 워밍업이라 느리므로 단발 측정은 신뢰하지 않는다.

---

## 패치 2. 알고리즘 탐색 제거 (고정 알고리즘) — **가장 큰 latency 절감**

`convoluteForward`(slide 484~)에서 `convAlgorithm`이 음수면 매번
`cudnnFindConvolutionForwardAlgorithm`으로 알고리즘을 새로 탐색한다(slide 554~566).
입력이 **28×28로 고정**이므로 한 번 찾은 최적 알고리즘을 고정하면 이 탐색이 사라진다.

### 2-a. 어떤 알고리즘이 최적인지 먼저 확인
패치 전, 한 번 실행해 출력에서 가장 빠른 Algo 번호를 본다. 실제 출력 예:
```
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.022528 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.033792 time requiring 0 memory
...
```
→ 위 예시는 **Algo 0**이 가장 빠르고 메모리 0. (Orin에서 직접 확인할 것.
보통 28×28·5×5에서는 `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`(=0) 또는
`IMPLICIT_PRECOMP_GEMM`(=1)이 빠르다. 메모리 0인 것을 우선.)

### 2-b. 탐색 분기를 건너뛰도록 고정
`main`에서 `convAlgorithm`을 초기화하는 곳을 찾아(보통 `int convAlgorithm = -1;`
또는 명령행으로 설정) **음수가 아닌 고정 값**으로 바꾼다. 그러면
`if (convAlgorithm < 0)` 탐색 블록(slide 554~575)을 안 타고, `else`에서
`algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;`(slide 579)로 바로 쓴다.

```cpp
// 탐색 결과 가장 빨랐던 번호로. (예: 0)
int convAlgorithm = 0;   // was -1  -> 매 실행 Find... 호출이 사라짐
```

또는 `convoluteForward` 안에서 직접 고정해도 된다(구조·순서 불변):
```cpp
void convoluteForward(...) {
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    // setTensorDesc / FilterNd / ConvolutionNd / GetForwardOutputDim 은 유지
    // (출력 크기 계산에 필요). 단 Find... 호출 블록은 건너뛴다.
    ...
}
```
이 패치 후 출력에서 `Testing cudnnFindConvolutionForwardAlgorithm ...`이
사라지면 성공.

> 회귀 확인: 패치 2 적용 후 1/3/5가 여전히 정답인지(`scripts/04_eval.py`의
> 게이트) 반드시 확인. 다른 알고리즘이라도 수치 결과는 같아야 한다.

---

## 패치 3. 단일 정밀도만 실행 (FP32 또는 FP16 택1)

`main`은 FP32와 FP16을 **둘 다** 돈다(`Testing single precision`,
`Testing half precision`). 채점 실행에선 하나만 돌리면 절반의 시간이 빠진다.

- 안전책: FP32 한 번만. `main`에서 half precision을 호출하는 블록(두 번째
  `network_t<...> ... classify_example ...` 묶음)을 주석 처리.
- 성능책: Orin 텐서코어로 **FP16(half)**이 더 빠를 수 있다. 단, 28×28은 연산이
  작아 FP16이 항상 빠르진 않으니 **반드시 before/after 측정**으로 고른다.
  FP16 채택 시 1/3/5/6/8 정확도가 유지되는지 확인.

> 구조·순서는 그대로다. "어떤 정밀도 1개를 돌릴지"만 고른다.

---

## 패치 4. (선택) 단일 이미지 경로를 채점용으로 정리

`04_eval.py`는 `mnistCUDNN --image=<path>`로 한 장씩 부른다. 기본 경로가
one/three/five를 하드코딩해 3장을 도는 것과 별개로, `--image` 분기(slide 866~882)가
한 장을 분류하고 `Result of classification: <한 자리>` + `LATENCY_MS=`를
출력하면 그대로 쓸 수 있다. 출력 문구를 바꿨다면 `04_eval.py`의
`PRED_RE`(`classification:\s*([0-9 ]+)`)도 맞춰 수정한다.

선택 최적화(있으면 좋음, 측정으로 판단):
- 핸들/디스크립터/workspace/디바이스 버퍼를 이미지마다 만들지 말고 재사용
  (영상에서 여러 장을 연속 분류할 때 고정 오버헤드 감소).
- H2D 복사를 `cudaHostAlloc`(pinned) 버퍼로.
이 두 가지는 단일 이미지 1회 실행에는 효과가 작고, 연속 분류에서 커진다.

---

## 검증 순서 (각 패치마다)

1. `make clean && make` 로 빌드.
2. `./mnistCUDNN --image=five_28x28.pgm` 로 1/3/5 한 장씩 → 정답 유지 확인.
3. `LATENCY_MS=` 가 출력되는지, 패치2 후 `Testing ...`가 사라졌는지 확인.
4. 리포지토리 루트에서 `python scripts/04_eval.py --bin <binary> --samples-dir
   <pgm들 위치>` → 1/3/5 게이트 통과 + latency 표 확인.
5. before/after latency를 기록(보고서 표에 그대로 쓴다).

never: 9단계 순서 변경, 1/3/5 정확도 하락.
