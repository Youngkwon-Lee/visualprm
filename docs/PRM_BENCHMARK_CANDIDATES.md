# PhysioMM-PRM Benchmark Candidates: Selection & Comparison

**문서 버전**: 1.0
**작성일**: 2026-01-08
**목적**: PRM 벤치마크 구축을 위한 후보 접근법 선정 및 비교 분석

---

## Executive Summary

**핵심 발견**: 조사 결과, **물리치료 임상 추론 + PRM + 비디오** 조합은 존재하지 않음. 이는 명확한 연구 갭(research gap)이며, 논문 출판 가능성이 매우 높음.

**추천 접근법**: **Option 1 - PhysioKorea Native Data** (100% 새로운 데이터 수집)
- 이유: 완전한 도메인 특화성, 의료 규제 준수, PhysioKorea 통합, 최대 연구 임팩트

**대안**: Option 5 - Hybrid Approach (기존 데이터 + 새 라벨링)도 고려 가능 (비용 절감)

---

## Part 1: 후보 접근법 5개

### Option 1: PhysioKorea Native Data (처음부터 새로 수집)

**개요**:
- PhysioKorea patient-app 실제 환자 데이터 활용
- 100% 물리치료 도메인 특화 데이터
- 실제 임상 환경에서 수집된 운동 비디오 + 치료사 평가

**데이터 소스**:
```
PhysioKorea Patient-App Database
├─ Exercise Videos: 스쿼트, 런지, 플랭크, 브릿지 등
├─ Clinical Assessments: 치료사의 실제 평가 기록
├─ Compensation Patterns: 보상 동작 패턴 라벨
└─ Treatment Plans: 치료 계획 및 진행 기록
```

**PRM 구축 프로세스**:
1. **비디오 수집**: PhysioKorea DB에서 10,000개 운동 비디오 추출 (IRB 승인)
2. **질문 생성**: GPT-4V로 비디오당 임상 질문 자동 생성
   - 예: "이 환자의 스쿼트 동작에서 보상 패턴을 식별하고, 생체역학적 원인을 설명하시오."
3. **솔루션 샘플링**: InternVL3-8B로 4개 솔루션 생성 (맞는 것 + 틀린 것)
4. **RAG-Judge 라벨링**: Gemini 2.0 Flash Thinking + 물리치료 문헌 → Step-wise 라벨
5. **Selective Monte Carlo**: 애매한 케이스에만 MC 샘플링 (10%)
6. **전문가 검증**: 물리치료사 최종 검증 (5-10%)

**Pros**:
✅ 완전한 도메인 특화성 (physiotherapy clinical reasoning)
✅ 실제 임상 데이터 → 높은 실용성
✅ PhysioKorea 서비스 직접 통합 가능
✅ 의료 규제 준수 (HIPAA/GDPR 가능)
✅ 최대 연구 임팩트 (완전히 새로운 벤치마크)
✅ Med-PRM + VisualPRM 방법론 직접 적용 가능
✅ NeurIPS Datasets Track 강력한 후보

**Cons**:
❌ 데이터 수집 시간 소요 (IRB 승인 필요)
❌ 전문가 검증 비용 (물리치료사 시급 $50-100)
❌ 초기 데이터셋 규모 제한 (10,000개 현실적)
❌ 비디오 품질 불균일 (실제 환자 데이터)

**비용 추정**:
| 항목 | 방법 | 비용 |
|------|------|------|
| 질문 생성 (10,000개) | GPT-4V | $150 |
| 솔루션 샘플링 (40,000개) | InternVL3-8B (self-host) | $0 |
| RAG-Judge 라벨링 (90%) | Gemini 2.0 Flash Thinking | $2,800 |
| Monte Carlo (10%) | InternVL3-8B | $500 |
| 전문가 검증 (10%) | 물리치료사 ($80/hr × 100hrs) | $8,000 |
| **총 비용** | | **$11,450** |

**타임라인**:
- Week 1-2: IRB 승인 + 데이터 추출 계획
- Week 3-4: 10,000 비디오 추출 + 전처리
- Week 5-6: 질문 생성 + 솔루션 샘플링
- Week 7-8: RAG-Judge 라벨링 + Monte Carlo
- Week 9-10: 전문가 검증 + 품질 관리
- Week 11-12: 최종 데이터셋 구축 + 벤치마크 테스트

**PRM 변환 전략**:
```python
# PhysioKorea Video → PRM Format
{
  "video": "squat_patient_0001.mp4",
  "clinical_question": "이 환자의 스쿼트 동작을 평가하시오. 보상 패턴과 생체역학적 원인을 설명하고 치료 계획을 제시하시오.",
  "solutions": [
    {
      "solution_id": 1,
      "steps": [
        "Step 1: 환자의 무릎이 안쪽으로 collapse 되는 knee valgus 보상 패턴 관찰 ки",
        "Step 2: 이는 중둔근(gluteus medius) 약화로 인한 고관절 외전근 기능 부전이 원인 ки",
        "Step 3: 치료 계획: 중둔근 강화 운동 (side-lying hip abduction, clamshell) ки"
      ],
      "step_labels": [1, 1, 1],  # PRM labels (1=correct, 0=incorrect)
      "final_answer": "knee valgus → gluteus medius weakness → hip abduction exercise",
      "is_correct": true
    },
    {
      "solution_id": 2,
      "steps": [
        "Step 1: 환자의 무릎이 발끝보다 앞으로 나가는 패턴 관찰 ки",
        "Step 2: 이는 대퇴사두근 과활성화로 인한 문제 ки",  # ← WRONG reasoning
        "Step 3: 치료 계획: 대퇴사두근 스트레칭 ки"
      ],
      "step_labels": [1, 0, 0],  # Step 1 correct, Step 2-3 wrong
      "final_answer": "knee forward → quad overactive → quad stretching",
      "is_correct": false
    }
  ],
  "ground_truth": "knee valgus → gluteus medius weakness → hip abduction strengthening",
  "expert_verified": true
}
```

**차별성**:
| 기존 벤치마크 | PhysioMM-PRM (Option 1) |
|--------------|------------------------|
| NTU RGB+D: Action recognition | Clinical reasoning evaluation |
| FineDiving: Procedure labels | Step-wise PRM labels |
| FLEX: Fitness AQA | Physiotherapy diagnosis + treatment |
| Med-PRM: Text only | Multimodal (video + text) |
| VisualPRM: General domain | Medical domain (physiotherapy) |

**예상 연구 임팩트**:
- ✅ NeurIPS 2026 Datasets & Benchmarks Track (강력한 후보)
- ✅ 의료 AI 커뮤니티에서 큰 관심 (MICCAI, MLHC)
- ✅ PhysioKorea 서비스 품질 향상 (실용적 응용)
- ✅ 물리치료 AI 연구의 새로운 방향 제시

---

### Option 2: Adapt FineDiving (Procedure-aware → PRM)

**개요**:
- FineDiving의 procedure-aware annotations를 PRM 형식으로 변환
- 기존 3,000 비디오 + 52 action types 활용
- 스포츠 도메인 → 물리치료 도메인으로 전이

**데이터 소스**:
- FineDiving (CVPR 2022 Oral): 3,000 diving videos
- Step-level labels: 73 consecutive steps (e.g., "approach → hurdle → takeoff → flight → entry")
- Fine-grained action recognition 데이터

**PRM 구축 프로세스**:
1. **도메인 전이**: Diving steps → Rehabilitation exercise steps
   - 예: "takeoff → flight → entry" ≈ "준비 → 실행 → 착지" (재활 운동)
2. **질문-답변 생성**: GPT-4V로 각 비디오에 대한 임상 질문 생성
3. **Step-wise 라벨 재활용**: FineDiving의 step labels → PRM step labels로 매핑
4. **RAG-Judge 보완**: 물리치료 관점에서 step correctness 재평가
5. **전문가 검증**: 물리치료사가 도메인 전이 타당성 검증

**Pros**:
✅ 기존 데이터 활용 → 비용 절감
✅ 3,000 비디오 이미 확보
✅ Step-level labels 재활용 가능
✅ Fine-grained action recognition 경험 활용

**Cons**:
❌ 도메인 불일치 (diving ≠ physiotherapy)
❌ Step semantics 다름 (athletic performance vs clinical reasoning)
❌ 의료 도메인 특화성 부족
❌ 연구 임팩트 제한 (기존 데이터 재활용)
❌ 임상 적용성 낮음 (스포츠 ≠ 재활)

**비용 추정**:
| 항목 | 비용 |
|------|------|
| 질문 생성 (3,000개) | $50 |
| RAG-Judge 보완 | $800 |
| 전문가 검증 (필수) | $6,000 |
| **총 비용** | **$6,850** |

**타임라인**: 8주

**PRM 변환 전략**:
```python
# FineDiving → PhysioMM-PRM mapping (어려움)
FineDiving_step = "takeoff with armswing"
Physio_step = "스쿼트 시작 시 고관절 굴곡 개시"  # ← 도메인 갭 큼
```

**차별성**: ⚠️ 낮음 (기존 데이터 재활용)

**예상 연구 임팩트**: ⚠️ 중간 (도메인 전이 논문 가능성)

---

### Option 3: Extend FLEX (Fitness → Clinical Rehabilitation)

**개요**:
- FLEX (2024) 7,500+ 녹화 데이터 활용
- Fitness domain → Clinical rehabilitation domain 확장
- Multi-modal signals (RGB + 3D pose + sEMG) 활용

**데이터 소스**:
- FLEX: 7,500+ fitness exercise recordings
- Multi-modal: RGB video + 3D pose + sEMG + physiological signals
- Action quality assessment labels

**PRM 구축 프로세스**:
1. **데이터 필터링**: 재활 운동과 유사한 fitness exercises 선택 (스쿼트, 플랭크, 브릿지)
2. **임상 관점 재라벨링**: Fitness quality → Clinical reasoning으로 재해석
3. **Multi-modal PRM**: RGB + 3D pose → Step-wise clinical reasoning
4. **RAG-Judge**: 물리치료 문헌 기반 step correctness 평가

**Pros**:
✅ 대규모 데이터 (7,500+)
✅ Multi-modal signals 활용 가능
✅ 3D pose 정보 → 생체역학 분석 용이
✅ sEMG → 근육 활성 패턴 분석 가능

**Cons**:
❌ Fitness ≠ Rehabilitation (도메인 갭)
❌ Healthy individuals ≠ Patients (임상 적용성 낮음)
❌ 보상 패턴 라벨 없음 (fitness에서는 불필요)
❌ 의료 규제 준수 어려움 (환자 데이터 아님)
❌ 연구 임팩트 제한 (기존 데이터 활용)

**비용 추정**:
| 항목 | 비용 |
|------|------|
| 데이터 접근 (FLEX) | $0 (public) |
| 질문 생성 (7,500개) | $250 |
| RAG-Judge 라벨링 | $7,000 |
| 전문가 검증 (필수) | $15,000 |
| **총 비용** | **$22,250** |

**타임라인**: 10주

**차별성**: ⚠️ 중간 (multi-modal 강점)

**예상 연구 임팩트**: ⚠️ 중간 (fitness → clinical 전이 연구)

---

### Option 4: NTU RGB+D Physiotherapy Subset

**개요**:
- NTU RGB+D 120 (4,000+ citations) 신뢰성 활용
- 120 action classes 중 재활 관련 동작만 추출
- 대규모 데이터셋 인프라 활용

**데이터 소스**:
- NTU RGB+D 120: 114,480 video samples
- 120 action classes 중 재활 관련 동작 (~15 classes)
  - 예: "squat", "stand up", "sit down", "walking", "falling"

**PRM 구축 프로세스**:
1. **재활 동작 필터링**: 120 classes → 15 rehabilitation-relevant classes
2. **임상 컨텍스트 추가**: Action recognition → Clinical reasoning questions
3. **질문-답변 생성**: GPT-4V로 각 비디오에 대한 임상 질문 생성
4. **RAG-Judge 라벨링**: 물리치료 관점에서 step-wise 평가

**Pros**:
✅ 대규모 데이터 (114,480 samples)
✅ 높은 신뢰성 (4,000+ citations)
✅ RGB+D 센서 → 3D 정보 활용
✅ Multi-view (3 cameras) → 다각도 분석

**Cons**:
❌ Action recognition 데이터 (clinical reasoning 아님)
❌ Healthy actors (환자 아님)
❌ 재활 관련 동작 비율 낮음 (~12.5%)
❌ 임상 컨텍스트 없음 (단순 동작 녹화)
❌ 보상 패턴 라벨 없음
❌ 도메인 갭 큼 (action recognition → clinical reasoning)

**비용 추정**:
| 항목 | 비용 |
|------|------|
| 데이터 접근 (NTU) | $0 (public) |
| 재활 동작 필터링 | $100 |
| 질문 생성 (~15,000개) | $500 |
| RAG-Judge 라벨링 | $14,000 |
| 전문가 검증 (필수) | $30,000 |
| **총 비용** | **$44,600** |

**타임라인**: 12주

**차별성**: ❌ 낮음 (기존 데이터 재활용)

**예상 연구 임팩트**: ❌ 낮음 (도메인 갭 큼, 연구 기여도 제한)

---

### Option 5: Hybrid Approach (기존 데이터 + PhysioKorea)

**개요**:
- **Phase 1**: 기존 데이터셋 (FLEX, FineDiving) 활용 → Pilot (1,000 samples)
- **Phase 2**: PhysioKorea native data 수집 → Full-scale (10,000 samples)
- 점진적 확장 전략 (빠른 검증 → 대규모 구축)

**데이터 소스**:
```
Phase 1 (Pilot):
├─ FLEX: 500 fitness exercises → clinical re-labeling
├─ FineDiving: 300 diving videos → step-wise mapping
└─ PhysioKorea: 200 pilot videos
Total: 1,000 samples

Phase 2 (Full-scale):
└─ PhysioKorea: 10,000 native videos
```

**PRM 구축 프로세스**:
1. **Phase 1 (4주)**:
   - FLEX + FineDiving → 임시 PRM 구축
   - PhysioKorea pilot 200 비디오 추가
   - 방법론 검증 (RAG-Judge + Monte Carlo)
2. **Phase 2 (8주)**:
   - PhysioKorea 10,000 비디오 대규모 수집
   - Phase 1 학습 적용 → 효율적 라벨링
   - 최종 벤치마크 완성

**Pros**:
✅ 빠른 시작 (기존 데이터 활용)
✅ 점진적 확장 → 리스크 감소
✅ 방법론 조기 검증 가능
✅ Phase 1 결과 → 논문 조기 제출 가능
✅ Phase 2 → 최종 대규모 벤치마크
✅ 비용 절감 (기존 데이터 활용)

**Cons**:
⚠️ Phase 1 품질 제한 (도메인 갭)
⚠️ 두 단계 관리 복잡성
⚠️ 기존 데이터 라이선스 확인 필요

**비용 추정**:
| Phase | 항목 | 비용 |
|-------|------|------|
| Phase 1 | FLEX + FineDiving 재라벨링 | $2,000 |
| Phase 1 | PhysioKorea pilot (200) | $1,500 |
| Phase 2 | PhysioKorea full (10,000) | $11,450 |
| **총 비용** | | **$14,950** |

**타임라인**: 12주 (Phase 1: 4주 + Phase 2: 8주)

**차별성**: ✅ 높음 (최종적으로 PhysioKorea native data)

**예상 연구 임팩트**: ✅ 높음 (점진적 발표 전략 가능)

---

## Part 2: 비교 매트릭스

### 2.1 핵심 메트릭 비교

| Metric | Option 1 (PhysioKorea) | Option 2 (FineDiving) | Option 3 (FLEX) | Option 4 (NTU) | Option 5 (Hybrid) |
|--------|----------------------|-------------------|--------------|------------|---------------|
| **도메인 특화성** | ⭐⭐⭐⭐⭐ (100%) | ⭐⭐ (40%) | ⭐⭐⭐ (60%) | ⭐ (20%) | ⭐⭐⭐⭐ (80%) |
| **임상 적용성** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
| **연구 임팩트** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **비용 효율성** | ⭐⭐⭐ ($11.5K) | ⭐⭐⭐⭐ ($6.9K) | ⭐⭐ ($22.3K) | ⭐ ($44.6K) | ⭐⭐⭐⭐ ($15K) |
| **구축 속도** | ⭐⭐⭐ (12주) | ⭐⭐⭐⭐ (8주) | ⭐⭐⭐ (10주) | ⭐⭐ (12주) | ⭐⭐⭐ (12주) |
| **데이터 규모** | ⭐⭐⭐⭐ (10K) | ⭐⭐⭐ (3K) | ⭐⭐⭐⭐⭐ (7.5K) | ⭐⭐⭐⭐⭐ (114K) | ⭐⭐⭐⭐⭐ (11K) |
| **의료 규제 준수** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐ |
| **PhysioKorea 통합** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |

### 2.2 연구 기여도 비교

| Dimension | Option 1 | Option 2 | Option 3 | Option 4 | Option 5 |
|-----------|----------|----------|----------|----------|----------|
| **Novelty** (새로움) | ✅ 완전히 새로운 도메인 | ⚠️ 도메인 전이 | ⚠️ 도메인 확장 | ❌ 데이터 재활용 | ✅ 점진적 새로움 |
| **PRM Format** | ✅ Med-PRM 직접 적용 | ⚠️ 매핑 필요 | ⚠️ 매핑 필요 | ❌ 큰 갭 | ✅ 직접 적용 |
| **Clinical Reasoning** | ✅ 원래부터 설계 | ⚠️ 후처리 추가 | ⚠️ 후처리 추가 | ❌ 없음 | ✅ 설계 포함 |
| **Multimodal** | ✅ Video + Text | ✅ Video + Text | ✅ RGB+Pose+sEMG | ✅ RGB+D | ✅ Video + Text |
| **Publication Target** | NeurIPS D&B Track | MICCAI | MLHC | ❌ 약함 | NeurIPS D&B Track |

### 2.3 실용성 비교

| Dimension | Option 1 | Option 2 | Option 3 | Option 4 | Option 5 |
|-----------|----------|----------|----------|----------|----------|
| **PhysioKorea 배포** | ✅ 즉시 가능 | ❌ 도메인 갭 | ⚠️ 제한적 | ❌ 불가능 | ✅ 가능 |
| **환자 안전성** | ✅ 임상 검증됨 | ⚠️ 검증 필요 | ⚠️ 검증 필요 | ❌ 검증 어려움 | ✅ 검증 가능 |
| **치료사 채택** | ✅ 높음 | ⚠️ 중간 | ⚠️ 중간 | ❌ 낮음 | ✅ 높음 |
| **데이터 확장성** | ✅ PhysioKorea 지속 수집 | ❌ 제한적 | ❌ 제한적 | ❌ 제한적 | ✅ 지속 가능 |

---

## Part 3: 최종 추천

### 🏆 **1순위: Option 1 - PhysioKorea Native Data**

**추천 이유**:

1. **완전한 도메인 특화성**:
   - 물리치료 임상 추론 + PRM + 비디오 조합은 **세계 최초**
   - 기존 연구와 명확한 차별성 (NTU=action recognition, FineDiving=procedure, FLEX=fitness)

2. **최대 연구 임팩트**:
   - NeurIPS 2026 Datasets & Benchmarks Track 강력한 후보
   - Medical AI 커뮤니티에서 큰 관심 예상
   - Citation 잠재력: NTU RGB+D (4,000+) 수준 기대

3. **실용적 가치**:
   - PhysioKorea 서비스에 직접 통합 가능
   - 실제 환자에게 즉시 적용 가능
   - 치료사 채택률 높음 (자신들의 도메인 데이터)

4. **의료 규제 준수**:
   - IRB 승인 가능
   - HIPAA/GDPR 준수 가능
   - 환자 동의 프로세스 확립 가능

5. **비용 대비 효과**:
   - $11,450 (중간 수준)
   - 연구 임팩트 고려 시 ROI 최고

**실행 계획** (12주):

```
Week 1-2: IRB 승인 + 데이터 추출 계획
├─ IRB 신청서 작성
├─ 환자 동의서 템플릿 준비
└─ PhysioKorea DB 접근 권한 확보

Week 3-4: 데이터 수집 + 전처리
├─ 10,000 비디오 추출 (스쿼트, 런지, 플랭크, 브릿지)
├─ MediaPipe 키프레임 추출 (8 frames/video)
├─ 비디오 품질 필터링 (해상도, 프레임레이트)
└─ 익명화 처리 (얼굴 블러링)

Week 5-6: 질문-답변 생성
├─ GPT-4V로 10,000 임상 질문 자동 생성
├─ InternVL3-8B로 솔루션 샘플링 (4개/질문)
└─ 질문 품질 필터링 (치료사 샘플 검증)

Week 7-8: Hybrid Annotation
├─ RAG-Judge (Gemini 2.0 Flash Thinking) - 90%
├─ Selective Monte Carlo (InternVL3-8B) - 10%
└─ Step-wise PRM 라벨 생성

Week 9-10: 전문가 검증 + 품질 관리
├─ 물리치료사 검증 (10% = 1,000 samples)
├─ Inter-rater reliability 측정 (κ ≥ 0.85 목표)
└─ 품질 이슈 수정

Week 11-12: 최종 벤치마크 구축
├─ Train/Val/Test 분할 (70/10/20)
├─ 벤치마크 평가 스크립트 작성
├─ Baseline 모델 학습 (InternVL3-8B)
└─ 논문 초안 작성 시작
```

**예상 결과물**:

```
PhysioMM-PRM v1.0
├─ 10,000 clinical questions
├─ 40,000 step-wise solutions (4 per question)
├─ Video-based physiotherapy reasoning
├─ RAG-Judge + Monte Carlo hybrid labels
├─ Expert-verified (10%)
└─ Ready for NeurIPS 2026 submission
```

---

### 🥈 **2순위: Option 5 - Hybrid Approach**

**추천 이유** (Option 1이 리스크가 크다고 판단될 경우):

1. **빠른 검증**:
   - Phase 1 (4주)에서 방법론 검증 가능
   - 초기 결과로 conference workshop 제출 가능

2. **리스크 감소**:
   - 기존 데이터 활용 → 초기 실패 리스크 낮음
   - PhysioKorea 데이터 수집 문제 발생 시 fallback 가능

3. **비용 분산**:
   - Phase 1: $3,500 (빠른 투자)
   - Phase 2: $11,450 (검증 후 투자)

**실행 계획** (12주):

```
Phase 1 (Week 1-4): Pilot with existing data
├─ FLEX 500 samples → clinical re-labeling
├─ FineDiving 300 samples → step mapping
├─ PhysioKorea 200 pilot videos
└─ Hybrid annotation pipeline 구축

Phase 2 (Week 5-12): Full-scale PhysioKorea
└─ Option 1 실행 계획 동일
```

---

### ❌ **비추천: Option 2, 3, 4**

**공통 문제점**:
- 도메인 갭 큼 (diving/fitness ≠ physiotherapy)
- 임상 적용성 낮음 (환자 데이터 아님)
- 연구 임팩트 제한 (기존 데이터 재활용)
- PhysioKorea 통합 어려움

---

## Part 4: 실행 권고사항

### 4.1 즉시 시작 가능한 작업

**Option 1 선택 시** (추천):

1. **Week 1 (지금 즉시)**:
   ```bash
   # 1. PhysioKorea DB 접근 확인
   - patient-app 데이터베이스 스키마 분석
   - 운동 비디오 저장 위치 확인
   - 환자 동의서 확인 (비디오 연구 사용 동의 여부)

   # 2. IRB 준비
   - IRB 신청서 템플릿 다운로드
   - 연구 계획서 초안 작성
   - 환자 동의서 업데이트 (필요 시)

   # 3. Pilot 테스트 (100 비디오)
   - PhysioKorea DB에서 100개 비디오 추출
   - GPT-4V로 10개 질문 생성 테스트
   - RAG-Judge 파이프라인 테스트
   ```

2. **Week 2**:
   ```bash
   # 4. 기술 스택 구축
   cd visualprm/physiomm-prm

   # Med-PRM 복제 및 테스트
   python Med-PRM/scripts/1_sampling.sh  # 테스트

   # VisualPRM 복제 및 테스트
   python InternVL/internvl_chat/tools/reasoning_data_pipeline/visualprm_data_pipeline.py  # 테스트

   # 5. 비용 최적화
   - Gemini 2.0 Flash Thinking API 키 확보
   - InternVL3-8B self-hosting 테스트 (비용 $0)
   ```

### 4.2 리스크 관리

| 리스크 | 완화 전략 |
|--------|----------|
| IRB 승인 지연 | Pilot 100 비디오로 선행 기술 개발 |
| 비디오 품질 불균일 | 품질 필터링 기준 사전 정의 (해상도 ≥720p, FPS ≥30) |
| 전문가 검증 비용 초과 | 10% 샘플링 + 높은 confidence만 skip |
| RAG-Judge 품질 낮음 | Monte Carlo 비율 증가 (10% → 20%) |
| 데이터 규모 부족 | 초기 10,000 목표 → 최소 5,000으로 조정 가능 |

### 4.3 성공 메트릭

**Technical Metrics**:
- Model Accuracy: >80% on held-out test set
- Best-of-N Improvement: +5% vs Best-of-1
- Annotation Quality: 92% agreement with experts (Med-PRM 수준)

**Research Metrics**:
- NeurIPS 2026 Datasets Track acceptance
- 1년 내 100+ citations 목표
- Medical AI 커뮤니티 인정

**Clinical Metrics**:
- Expert Agreement: >90% on clinical correctness
- PhysioKorea 치료사 채택률: >70%
- 환자 결과 개선: 치료 준수율 +15%

---

## Part 5: 다음 단계 (Next Steps)

### 즉시 실행 (이번 주):

1. ✅ **의사결정**: Option 1 vs Option 5 선택
   - 추천: Option 1 (PhysioKorea Native)

2. ✅ **Pilot 시작**:
   ```bash
   # PhysioKorea DB에서 100개 비디오 추출 테스트
   cd visualprm/physiomm-prm
   python data/collect_physiokorea.py --limit 100

   # GPT-4V 질문 생성 테스트
   python data/generate_questions.py --num_questions 10

   # RAG-Judge 테스트
   python annotation/rag_judge_multimodal.py --test_mode
   ```

3. ✅ **기술 문서 작성**:
   - `physiomm-prm/README.md` (이미 작성됨)
   - `physiomm-prm/docs/DATA_COLLECTION.md` (수집 프로토콜)
   - `physiomm-prm/docs/ANNOTATION_GUIDE.md` (라벨링 가이드)

### 다음 주:

4. ✅ **IRB 준비**: 연구 계획서 + 환자 동의서
5. ✅ **비용 확보**: $11,450 예산 승인
6. ✅ **팀 구성**: 물리치료사 전문가 2-3명 섭외

### 1개월 후:

7. ✅ **Full-scale 데이터 수집**: 10,000 비디오
8. ✅ **Annotation 파이프라인**: RAG-Judge + Monte Carlo
9. ✅ **Baseline 모델 학습**: InternVL3-8B fine-tuning

---

## 결론

**최종 추천: Option 1 - PhysioKorea Native Data**

**핵심 근거**:
1. 세계 최초 물리치료 PRM 벤치마크 (완전한 novelty)
2. 실제 임상 데이터 → 최대 실용성
3. NeurIPS 2026 강력한 후보
4. PhysioKorea 서비스 직접 통합
5. 비용 대비 효과 최고 ($11.5K for world-first benchmark)

**Action Items**:
1. ✅ 이번 주: Pilot 100 비디오 테스트
2. ✅ 다음 주: IRB 신청 + 예산 승인
3. ✅ 1개월: Full-scale 구축 시작

**Timeline**: 12주 → NeurIPS 2026 제출 가능

---

**문서 작성**: 2026-01-08
**다음 업데이트**: Pilot 결과 후 (Week 2)
