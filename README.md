## 프로젝트 소개

### 1. 프로젝트 배경
#### 1.1. 국내외 시장 현황 및 문제점
SSD(Solid-State Drive)는 빠른 입출력 속도와 낮은 지연 시간 덕분에 주요 저장 장치로 자리 잡았다. 그러나 데이터를 덮어쓸 수 없으며, 읽기와 쓰기는 페이지(page) 단위로 이루어지지만, 삭제는 이보다 훨씬 큰 블록(block) 단위로만 가능하다는 점이 문제점이다. 이로 인해 SSD는 기존의 데이터를 무효화하고 새로운 데이터를 다른 공간에 쓰는 방식으로 쓰기 작업을 수행하는데 이 과정이 반복되면 유효하지 않은 데이터가 누적되고, 결국 가비지 컬렉션(Garbage Collection, GC)이라는 정리 작업을 통해 유효한 데이터를 다른 블록으로 옮기는 추가적인 쓰기가 발생한다.
이러한 가비지 컬렉션 과정에서 발생하는 추가적인 쓰기 작업은 **쓰기 증폭** 현상을 유발하며 이로 인해 SSD의 수명을 단축시키게 되는 문제점을 발생시키는데, 기존의 FTL은 사전에 정의된 정적인 정책을 따르기 때문에 이러한 문제를 해결하기 어렵다.

#### 1.2. 필요성과 기대효과
딥러닝 기술을 도입하여 입출력 워크로드의 패턴을 분석하고, 딥러닝 모델을 활용해 데이터의 Hotness(갱신 빈도)를 예측한다. 이렇게 예측된 Hotness 레이블을 temperature로 정의해 first-touch 태깅과 온도 풀 분리, 동일 풀 우선 GC 복사를 적용하면 Hot/Cold 혼입이 줄어들고, FTL이 데이터를 블록별로 분리 저장하는 동적 정책을 구현함으로써, 가비지 컬렉션 시 유효 페이지를 옮기는 횟수를 최소화하여 **쓰기 증폭** 문제를 완화시킬 수 있다.

---

### 2. 개발 목표
#### 2.1. 목표 및 세부 내용
딥러닝을 활용하여 SSD의 Hotness 기반 FTL 정책을 제안하고 검증하는 것이 목표다. 페이지별 입출력 트레이스를 생성하는 데이터 전처리 과정을 거친 뒤, 이 정보를 통해 페이지 정보 해시 테이블을 구축하여 각 페이지의 접근 패턴에 대한 통계 정보를 누적한다. 통계 정보는 K-means 클러스터링을 통해 Hotness 레이블을 생성하는 데 사용되며, 동시에 Stacked-LSTM 모델의 입력 데이터가 되어 페이지 접근 패턴의 시계열적 특성을 학습한다. 학습이 완료된 모델은 오프라인 레이블링을 통해 전체 트레이스에 Hotness 예측 레이블을 추가하여 시뮬레이션에 사용될 Hotness 이 추가된 트레이스를 생성한다. 최종적으로, SimpleSSD-SA 시뮬레이터를 사용하여 Hotness를 고려하지 않는 기존 FTL 정책과 제안된 Hotness 기반 FTL 정책의 성능을 비교 분석한다.
#### 2.2. 기존 서비스 대비 차별성
FTL에서 블록을 첫 쓰기 시점에 온도로 태깅하고 가득 찰 때까지 동일 온도만 수용하며, GC 시에도 같은 온도 풀의 오픈 블록으로 우선 복사하여 혼입을 구조적으로 최소화한다. 이는 단순 페이지 매핑 대비 불필요한 유효 페이지 복사를 줄여 내부 WAF·지연을 낮추는 점에서 차별화된다.
#### 2.3. 사회적 가치 도입 계획
정책의 핵심 목표가 WAF와 지연의 감소이므로, 동일 자원에서의 효율 향상과 수명 연장을 기대할 수 있다.


### 3. 시스템 설계
#### 3.1. 시스템 구성도
<img width="5880" height="6307" alt="KakaoTalk_Photo_2025-09-23-20-29-50" src="https://github.com/user-attachments/assets/46baa3a5-b17f-4cfa-888a-281a57d26cc0" />


#### 3.2. 사용 기술
- **시뮬레이션**: C++17, CMake, SimpleSSD‑Standalone v2.0
- **ML**: Python, PyTorch 
- **데이터**: SNIA IOTTA YCSB 블록 트레이스

---

### 4. 개발 결과
#### 4.1. 전체 시스템 흐름도
전처리·페이지화 → 학습 데이터 생성 → 모델 학습·검증 → 오프라인 레이블링 → 시뮬레이션 환경 구성 → 결과 분석

#### 4.2. 기능 설명 및 주요 기능 명세서
- 레이블 정규화·풀 선택: 요청 temperature를 정규화해(0~3) 해당 풀에서 블록을 선택하고 태깅한다.
- FTL 배치: 첫 쓰기 순간 블록 태깅, 같은 온도만 채움
- 일관 GC: 유효 서브페이지는 같은 온도 풀의 오픈 블록으로 우선 복사하여 혼입을 억제한다.
- 메트릭: Internal WAF = `device_write_bytes / host_write_bytes_at_FTL`, GC 복사 서브페이지, 지연, IOPS 등

#### 4.3. 디렉토리 구조
```
.
├─ docs/                # 보고서/포스터/발표자료
├─ scripts/             # 전처리·학습·레이블·재현 스크립트
├─ src/                 # ICL/FTL 수정 코드
├─ models/              # 학습 모델(.pt) 및 스케일러(.pkl)
├─ traces/              # raw/processed/labeled 트레이스
├─ configs/             # 시뮬 설정 JSON
└─ results/             # 로그/표/그래프
```

#### 4.4. 산업체 멘토링 의견 및 반영 사항
- **WAF 외 성능 지표 제시**:평균 응답 시간(Latency)이 98.2% 단축되었음을 확인함으로써 쓰기 지연 시간 개선또한 명확히 입증했다.
- **GC Victim 블록 선택 방식의 차별성**:GC 횟수의 비교로 Victim 블록 선택 전략이 근본적으로 다르게 동작했음을 증명했다.
- **Wear-Leveling**:본 연구에서는 별도의 동적 Wear-Leveling 알고리즘의 구현을 포함하지 않기 때문에 장기적인 관점에서는 마모 불균형이 심화될 수 있는 한계가 있기에 향후 연구 과제로 현재의 Hot/Cold 분리 정책을 기반으로 블록 간 마모도 편차도 고려하는 알고리즘에 대한 연구가 가능할 것이다.

---

### 5. 설치 및 실행 방법
#### 5.1. 설치절차 및 실행 방법

설치
```bash
git clone git@github.com:simplessd/simplessd-standalone
cd simplessd-standalone
git submodule update --init --recursive
```

시뮬레이터 빌드 (mac 사용)
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j"$(sysctl -n hw.ncpu)"
```

실험 파라미터 설정<br>
시뮬레이션의 config 파일과 simple-ssd의 config 파라미터를 설정해 실험할 수 있다.

실행 (mac 사용)
```bash
mkdir results
./simplessd-standalone \
  ../config/sample.cfg \
  ../simplessd/config/sample.cfg \
  results
```

---

### 6. 소개 자료 및 시연 영상
#### 6.1. 프로젝트 소개 자료

#### 6.2. 시연 영상


---

### 7. 팀 구성
#### 7.1. 팀원별 소개 및 역할 분담
- 201924544 **이준형** — 데이터 전처리, 입출력 패턴 코드 분석, 모델 개발
- 201914502 **강인석** — 시뮬레이션 환경 구축 및 설정, 성능 측정(WAF), Hotness 레벨 트레이스용 FTL 정책 개선, 성능 비교
- 202155537 **김지수** — 시뮬레이션 환경 구축 및 설정, FTL 개선 정책 구현, 성능 측정, 기존 FTL VS 딥러닝 기반 FTL 비교
- 지도교수 **안성용** — 전체 연구 지도

#### 7.2. 팀원 별 참여 후기


---


### 8. 참고 문헌 및 출처
- [SimpleSSD‑Standalone(시뮬레이터)](https://docs.simplessd.org/en/v2.0.12/)
- [YCSB 트레이스](https://iotta.snia.org)
- [PyTorch](https://pytorch.org)
- [LSTM](https://developer.ibm.com/learningpaths/iot-anomaly-detection-deep-learning/intro-deep-learning-lstm-networks/)
- AGRAWAL, N., et al. "Design Tradeoffs for SSD Performance," Proc. of the 2008 USENIX Annual Technical Conference, pp. 1-14, Jun. 2008.
- YUNE, S. J.,"A Study on Improving SSD Write Amplification through Machine Learning-based Hot/Cold Page Classification," Master's Thesis, Pusan National University, Busan, 2024.
---

