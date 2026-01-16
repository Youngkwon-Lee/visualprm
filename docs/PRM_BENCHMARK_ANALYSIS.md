# Process Reward Model (PRM) 벤치마크 분석 종합 보고서

**작성일**: 2025-01-08
**목적**: Med-PRM 및 VisualPRM 벤치마크 구축 방법론 분석 및 의료 멀티모달 PRM 벤치마크 설계

---

## 📋 목차

1. [개요](#1-개요)
2. [VisualPRM 벤치마크 분석](#2-visualprm-벤치마크-분석)
3. [Med-PRM 벤치마크 분석](#3-med-prm-벤치마크-분석)
4. [비교 분석](#4-비교-분석)
5. [의료 멀티모달 PRM 설계안](#5-의료-멀티모달-prm-설계안)
6. [참고 자료](#6-참고-자료)

---

## 1. 개요

### 1.1 Process Reward Model (PRM)이란?

**정의**: 추론 과정의 각 단계별로 보상을 평가하는 모델

```
Outcome Reward Model (ORM)
└─ 최종 답만 평가 → 단순하지만 피드백 부족

Process Reward Model (PRM)
└─ 각 단계별 평가 → 세밀하지만 어노테이션 비용 높음
```

### 1.2 연구 배경

| 연구 | 발표 | 모달리티 | 도메인 |
|------|------|----------|--------|
| PRM800K | ICLR 2024 | 텍스트 | 수학 |
| Math-Shepherd | arXiv 2023 | 텍스트 | 수학 |
| **Med-PRM** | **EMNLP 2025** | **텍스트** | **의료** |
| **VisualPRM** | **arXiv 2025.03** | **멀티모달** | **일반** |

### 1.3 본 분석의 목표

```
Med-PRM (의료 텍스트)
    +
VisualPRM (일반 멀티모달)
    ↓
의료 멀티모달 PRM 벤치마크 설계
```

---

## 2. VisualPRM 벤치마크 분석

### 2.1 핵심 기여

1. **VisualPRM400K** 데이터셋: ~400K 샘플, 2M 단계
2. **VisualProcessBench**: 2,866 샘플, 26,950 인간 레이블
3. **성능**: InternVL2.5-78B에 +5.9점 향상 (7개 벤치마크)

### 2.2 데이터 구축 전략

#### 2.2.1 이중 데이터셋 구조

```
학습용: VisualPRM400K
├─ 목적: 대규모 PRM 학습
├─ 방법: Monte Carlo 자동 생성
├─ 규모: 400K 샘플
└─ 비용: 계산 비용만

평가용: VisualProcessBench
├─ 목적: 정확한 PRM 성능 평가
├─ 방법: 인간 전문가 어노테이션
├─ 규모: 2,866 샘플
└─ 비용: $1,443 (39 person-days)
```

#### 2.2.2 VisualPRM400K 자동 생성

**Monte Carlo 기반 Expected Accuracy**:

```python
# 핵심 알고리즘
for each step s_i in solution:
    # 16개 continuation 샘플링
    continuations = sample_continuations(
        image=I,
        question=q,
        prefix=s_[:i],
        num_samples=16
    )

    # Expected accuracy 계산
    mc_i = sum(is_correct(c) for c in continuations) / 16

    # 레이블링
    if mc_i > 0:
        label_i = "Correct (+)"
    else:
        label_i = "Incorrect (-)"
```

**데이터 통계**:
- 총 샘플: ~400K
- 총 단계: ~2M
- 평균 응답 길이: 126.9 단어
- 평균 단계 수: 5.6
- 평균 단계 길이: 22.6 단어
- 오답 단계 비율: ~10%

**소스 데이터**:
```python
source_benchmarks = {
    "MMPR v1.1": "전체",  # 멀티모달 추론
}

generation_models = [
    "InternVL2.5-8B",
    "InternVL2.5-26B",
    "InternVL2.5-78B"
]
```

#### 2.2.3 VisualProcessBench 인간 어노테이션

**데이터 수집**:

| 소스 | 샘플 수 |
|------|---------|
| MMMU | 267 |
| MathVision | 712 |
| MathVerse | 1,026 |
| DynaMath | 570 |
| WeMath | 291 |
| **총계** | **2,866** |

**솔루션 생성**:

| 모델 | 솔루션 수 |
|------|----------|
| GPT-4o | 870 |
| Claude-3.5-Sonnet | 865 |
| QvQ-72B-Preview | 825 |
| InternVL2.5-78B | 306 |

**어노테이션 프로토콜**:

```yaml
어노테이터:
  자격: 최소 대학 학위 소지자
  인원: 13명
  기간: 3일
  총 작업량: 39 person-days
  비용: ~$37/person-day

작업 단위:
  분할 수: 10개
  샘플/분할: ~300개

품질 관리:
  각 분할 검토: 10%
  검토자: 논문 저자
  재작업: 오류 발견 시 전체 분할

레이블 체계:
  - Positive (+): 단계가 정확함
  - Negative (-): 단계에 오류 있음
  - Neutral: 추론 없음/정보 추가 없음

혁신점:
  - 기존: 첫 오류만 찾기
  - VisualPRM: 모든 오류 찾기 (reflection 능력 평가)
```

**통계**:
- 총 단계: 26,950
- 정답 단계: 16,585 (61.5%)
- 오답 단계: 7,691 (28.5%)
- 중립 단계: 2,674 (10%)
- 평균 단계/솔루션: 9.4

### 2.3 평가 메트릭

**Macro F1 Score**:
```python
# 불균형 데이터 대응
F1_positive = compute_f1(positive_steps)
F1_negative = compute_f1(negative_steps)
Macro_F1 = (F1_positive + F1_negative) / 2

# VisualPRM 성능
VisualPRM_8B: 62.0
GPT-4o: 60.3
Random: 50.0
```

### 2.4 VisualPRM 모델 학습

**아키텍처**:
```
Multi-turn Chat 형식
├─ Turn 0: 이미지 + 질문 + 첫 단계
├─ Turn 1: 두 번째 단계
└─ Turn n: n번째 단계
    ↓
각 turn마다 단계 정확성 예측 (+/-)
```

**학습 설정**:
- Base Model: InternVL2.5-8B
- Optimizer: AdamW (β1=0.9, β2=0.999, weight_decay=0.05)
- Learning Rate: 1e-5 (cosine decay)
- Warmup: 5% of training steps
- Epoch: 1
- Data Packing: 활성화

**Value-based PRM vs Advantage-based PRM**:

| 타입 | 정의 | 레이블 | 성능 |
|------|------|--------|------|
| Value-based | mc_i > 0 | +/- | **더 높음** |
| Advantage-based | mc_i - mc_{i-1} > 0 | +/=/- | 낮음 |

**추론 시 점수 집계**:
```python
# Step score
step_score_i = P("+") * 1 + P("-") * 0

# Response score (여러 방법)
response_score = mean(step_scores)      # 최적
# response_score = min(step_scores)     # 보수적
# response_score = max(step_scores)     # 낙관적 (성능 낮음)
```

### 2.5 Best-of-N 평가 결과

**InternVL2.5-8B 성과** (N=8):

| 벤치마크 | Pass@1 | BoN w/ VisualPRM | 향상 |
|---------|--------|------------------|------|
| MMMU | 56.2 | 60.2 | +4.0 |
| MathVista | 64.5 | 68.5 | +4.0 |
| MathVision | 17.0 | 25.7 | +8.7 |
| MathVerse-VO | 22.8 | 35.8 | +13.0 |
| DynaMath | 9.4 | 18.0 | +8.6 |
| WeMath | 23.5 | 36.5 | +13.0 |
| LogicVista | 36.0 | 43.8 | +7.8 |
| **Overall** | **32.8** | **41.2** | **+8.4** |

**확장성 (InternVL2.5-78B)**:
- Pass@1: 46.0
- BoN w/ VisualPRM: 51.9
- 향상: +5.9 (대형 모델에도 효과적)

---

## 3. Med-PRM 벤치마크 분석

### 3.1 핵심 기여

1. **RAG-as-a-Judge**: 의학 문서 기반 자동 검증
2. **비용 효율성**: $20으로 11,678 문제 어노테이션
3. **성능**: MedQA 80.35% (8B 모델 최초 80% 돌파)

### 3.2 핵심 차별점

```
VisualPRM (일반)
└─ Monte Carlo 샘플링
    ├─ 장점: 자동화
    └─ 단점: 근거 없음

Med-PRM (의료)
└─ RAG + LLM-as-a-Judge
    ├─ 장점: 의학 근거 기반
    └─ 한계: 텍스트만 가능
```

### 3.3 데이터 구축 파이프라인

#### 3.3.1 소스 데이터

```python
training_sources = {
    "MedQA": 10178,      # 전체
    "MedMCQA": 500,      # 샘플
    "PubMedQA": 500,     # 샘플
    "MMLU-Med": 500      # 샘플
}
total_questions = 11678

evaluation_benchmarks = [
    "MedQA-4opt",
    "MedQA-5opt",
    "MedMCQA",
    "MMLU-Med",
    "DDXPlus",
    "AgentClinic-MedQA",
    "AgentClinic-NEJM"
]
```

#### 3.3.2 RAG-as-a-Judge 어노테이션

**의학 문서 검색**:

```python
medical_knowledge_sources = [
    "Clinical Guidelines",    # 임상 가이드라인
    "StatPearls",            # 의학 교과서
    "Medical Textbooks",     # 전문 교재
    "Rare Disease Corpus"    # 희귀질환 DB
]

# 토큰 할당
max_sequence_length = 4096
reserved_for_docs = 3072
reserved_for_reasoning = 1024
```

**Gemini-2.0-flash 기반 검증**:

```python
def rag_judge(question, reasoning_step, retrieved_docs):
    """의학 근거 기반 단계 검증"""

    prompt = f"""
    Clinical Question: {question}

    Medical Evidence:
    {retrieved_docs}

    Reasoning Step:
    {reasoning_step}

    Task: Is this reasoning step medically correct
    based on the evidence?

    Answer: +/- (with citation to evidence)
    """

    response = gemini_2_flash(prompt)
    return response.label, response.citation
```

**품질 필터**:

```yaml
filtering_rules:
  step_count: [3, 9]           # 너무 짧거나 긴 추론 제외
  label_balance: true           # 정답/오답 균형 유지
  degenerate_check: true        # 반복/무의미 단계 제거
  per_question_correct_limit: true  # 정답 추론 수 제한
```

#### 3.3.3 인간 검증 (샘플링)

```python
human_evaluation = {
    "annotators": {
        "physician": "4년 경력",
        "medical_student_1": "고학년",
        "medical_student_2": "고학년"
    },

    "sample_design": {
        "easy_questions": 3,
        "hard_questions": 3,
        "traces_per_question": 5
    },

    "total_annotations": 180,  # 단계 레이블

    "inter_rater_reliability": {
        "physician_vs_model": 0.71,  # Pearson
        "student_vs_model": 0.74
    }
}
```

**해석**:
- Pearson 0.71-0.74 = 강한 양의 상관관계
- 학생이 의사보다 모델과 더 일치 (흥미로운 결과)
- 샘플링 검증으로 비용 절감

### 3.4 학습 데이터 생성

```python
for question in training_set:
    # 1. 16개 후보 추론 생성
    candidate_traces = llm.generate(
        prompt=question,
        num_samples=16,
        temperature=0.7
    )

    # 2. 각 후보에 대해 의학 문서 검색
    for trace in candidate_traces:
        docs = retrieve_medical_docs(question)

        # 3. 단계별 RAG 검증
        steps = split_trace(trace, min=3, max=9)
        labels = []

        for step in steps:
            label = gemini_judge(question, step, docs)
            labels.append(label)

        # 4. 학습 샘플 저장
        training_data.append({
            "question": question,
            "trace": trace,
            "step_labels": labels,
            "evidence": docs
        })
```

### 3.5 비용 분석

```
API 비용 (Gemini-2.0-flash):
├─ 총 비용: ~$20
├─ 문제 수: 11,678
└─ 비용/문제: ~$0.0017

vs 인간 어노테이션:
├─ VisualPRM: $1,443 / 2,866 = $0.50/샘플
├─ 의료 전문가 추정: ~$5-10/샘플
└─ Med-PRM 절감: 99%+
```

### 3.6 성능 결과

**MedQA (4-option)**:

| 모델 | 크기 | 정확도 |
|------|------|--------|
| GPT-4 | - | 78.9% |
| Med-Gemini | - | 79.5% |
| **Meerkat-8B + Med-PRM** | **8B** | **80.35%** |

**특이사항**:
- 8B 모델로 최초 80% 돌파
- 기존 대형 상용 모델 초과

**전체 벤치마크 성능**:
- 7개 중 6개 벤치마크에서 SOTA
- 베이스 모델 대비 최대 +13.50% 향상

---

## 4. 비교 분석

### 4.1 종합 비교표

| 측면 | Med-PRM | VisualPRM | 의료 멀티모달 (제안) |
|------|---------|-----------|---------------------|
| **발표** | EMNLP 2025 | arXiv 2025.03 | TBD |
| **모달리티** | 텍스트 | 이미지+텍스트 | 의료영상+텍스트 |
| **도메인** | 의료 | 일반 추론 | 의료 |
| **어노테이션** | RAG+LLM | Monte Carlo | **하이브리드** |
| **의학 근거** | ✅ 필수 | ❌ | ✅ 필수 |
| **학습 데이터** | 11,678 문제 | 400K 샘플 | 50K 샘플 |
| **평가 데이터** | 180 단계 | 26,950 단계 | 5,000 단계 |
| **비용** | $20 | $1,443 | **$620** |
| **인간 검증** | 샘플링 (180) | 전체 (26,950) | 선별적 (5,000) |

### 4.2 방법론 비교

#### 4.2.1 어노테이션 전략

**Monte Carlo (VisualPRM)**:
```python
pros = [
    "완전 자동화",
    "대규모 생성 가능",
    "도메인 독립적"
]

cons = [
    "근거 없음",
    "오답 비율 낮음 (10%)",
    "의료 도메인에 부적합"
]
```

**RAG-as-a-Judge (Med-PRM)**:
```python
pros = [
    "의학 근거 제공",
    "비용 효율적 ($20)",
    "전문가 지식 반영"
]

cons = [
    "텍스트만 가능",
    "검색 품질 의존",
    "LLM API 비용"
]
```

#### 4.2.2 품질 관리

| 방법 | Med-PRM | VisualPRM |
|------|---------|-----------|
| **자동 필터** | 단계 수, 균형, 퇴화 | 없음 |
| **인간 검증** | 180 샘플 (0.015%) | 전체 (100%) |
| **검증 지표** | Pearson 상관계수 | Macro F1 |
| **재현성** | 높음 (RAG 결정적) | 중간 (MC 확률적) |

### 4.3 적용 시나리오

```
시나리오 1: 의료 텍스트 QA
├─ 최적: Med-PRM
└─ 이유: RAG로 근거 확보, 저비용

시나리오 2: 일반 멀티모달 추론
├─ 최적: VisualPRM
└─ 이유: Monte Carlo로 대규모 생성

시나리오 3: 의료 영상 진단
├─ 최적: 하이브리드 (제안)
└─ 이유: RAG(근거) + MC(영상) + 인간(선별)
```

---

## 5. 의료 멀티모달 PRM 설계안

### 5.1 설계 철학

```
Med-PRM의 장점 (의학 근거)
    +
VisualPRM의 장점 (멀티모달)
    +
비용 최적화 (선별적 인간 검증)
    =
의료 멀티모달 PRM
```

### 5.2 하이브리드 어노테이션 파이프라인

```python
def medical_multimodal_annotation(image, question, solution):
    """3단계 하이브리드 어노테이션"""

    # ===== Stage 1: RAG-Judge (Med-PRM) =====
    medical_docs = retrieve_medical_evidence(
        image=image,
        question=question,
        sources=[
            "radiology_atlas",
            "pathology_guidelines",
            "clinical_protocols",
            "case_reports"
        ],
        max_tokens=3072
    )

    rag_labels = gemini_2_flash_judge(
        image=image,
        question=question,
        solution=solution,
        evidence=medical_docs
    )

    # ===== Stage 2: Monte Carlo (VisualPRM) =====
    mc_scores = []
    for i, step in enumerate(solution.steps):
        continuations = []
        for _ in range(32):  # 의료는 더 많이
            cont = medical_mllm.generate(
                image=image,
                prefix=solution.steps[:i+1],
                temperature=0.7
            )
            continuations.append(cont)

        mc_i = sum(is_correct(c) for c in continuations) / 32
        mc_scores.append(mc_i)

    # ===== Stage 3: Hybrid Labeling =====
    final_labels = []
    uncertain_steps = []

    for idx, (rag, mc) in enumerate(zip(rag_labels, mc_scores)):
        if rag == "+" and mc > 0.7:
            label = "confident_correct"
            confidence = min(mc, 0.95)

        elif rag == "-" and mc < 0.3:
            label = "confident_incorrect"
            confidence = min(1 - mc, 0.95)

        else:
            # 불일치 케이스 → 인간 검증 필요
            label = "uncertain"
            confidence = 0.5
            uncertain_steps.append(idx)

        final_labels.append({
            "label": label,
            "confidence": confidence,
            "rag_label": rag,
            "mc_score": mc,
            "medical_evidence": medical_docs
        })

    # ===== Stage 4: Expert Verification (선별적) =====
    if uncertain_steps:
        expert_labels = medical_expert_annotation(
            image=image,
            question=question,
            solution=solution,
            uncertain_indices=uncertain_steps,
            evidence=medical_docs
        )

        # 불확실한 단계만 전문가 레이블로 대체
        for idx, expert_label in zip(uncertain_steps, expert_labels):
            final_labels[idx] = expert_label

    return final_labels, medical_docs
```

### 5.3 비용 최적화

**단계별 비용**:

| 단계 | 방법 | 처리량 | 비용 | 품질 |
|------|------|--------|------|------|
| 1 | RAG-Judge | 전체 | $20/10K | 70% |
| 2 | Monte Carlo | 전체 | $100/10K | 85% |
| 3 | 합의 확인 | 전체 | $0 | - |
| 4 | 전문가 검증 | 불일치만 (~30%) | $500/10K | 98% |
| **총합** | **하이브리드** | - | **$620/10K** | **95%** |

**vs 기존 방법**:
- 전체 전문가: $5,000/10K (87% 절감)
- VisualPRM: $1,443/2.9K → $5,000/10K (88% 절감)
- Med-PRM: $20/10K (but 텍스트만)

### 5.4 데이터셋 구성

```python
medical_multimodal_prm_dataset = {
    "training": {
        "chest_xray": {
            "samples": 20000,
            "annotation": "hybrid",
            "sources": ["MIMIC-CXR", "CheXpert"],
            "cost": "$400"
        },
        "ct_scan": {
            "samples": 10000,
            "annotation": "hybrid",
            "sources": ["RadImageNet", "LiTS"],
            "cost": "$200"
        },
        "pathology": {
            "samples": 10000,
            "annotation": "hybrid",
            "sources": ["PathVQA", "PatchCamelyon"],
            "cost": "$200"
        },
        "clinical_photos": {
            "samples": 5000,
            "annotation": "hybrid",
            "sources": ["Derm7pt", "HAM10000"],
            "cost": "$100"
        },
        "total_training": {
            "samples": 45000,
            "cost": "$900"
        }
    },

    "evaluation": {
        "radiology": {
            "samples": 500,
            "annotation": "expert_verified",
            "specialists": ["radiologist_board_certified"],
            "cost": "$2500"
        },
        "pathology": {
            "samples": 300,
            "annotation": "expert_verified",
            "specialists": ["pathologist_5yr"],
            "cost": "$1500"
        },
        "dermatology": {
            "samples": 200,
            "annotation": "expert_verified",
            "specialists": ["dermatologist"],
            "cost": "$1000"
        },
        "total_evaluation": {
            "samples": 1000,
            "cost": "$5000"
        }
    },

    "grand_total": {
        "training_samples": 45000,
        "evaluation_samples": 1000,
        "total_cost": "$5900",
        "cost_per_sample": "$0.13"
    }
}
```

### 5.5 벤치마크 스키마

```json
{
  "case_id": "medprm_cxr_0001",
  "modality": "chest_xray",
  "metadata": {
    "source": "MIMIC-CXR",
    "difficulty": "intermediate",
    "specialty": "radiology",
    "requires_specialist": true,
    "irb_approved": true,
    "phi_removed": true
  },

  "clinical_context": {
    "age_group": "60-70",
    "sex": "M",
    "symptoms": ["cough", "fever", "dyspnea"],
    "history": ["smoker_30_pack_years", "copd"],
    "vitals": {
      "temp": "38.5C",
      "spo2": "92%"
    }
  },

  "image": {
    "path": "anonymized/cxr_0001.dcm",
    "view": "PA",
    "quality": "adequate"
  },

  "question": "Describe the radiographic findings and provide a differential diagnosis.",

  "gold_standard": {
    "findings": [
      "Bilateral lower lobe infiltrates",
      "Air bronchograms present",
      "No pleural effusion"
    ],
    "diagnosis": "Community-acquired pneumonia",
    "icd10": "J18.9",
    "confidence": "high"
  },

  "solution_steps": [
    {
      "step_id": 0,
      "category": "observation",
      "content": "Bilateral patchy opacities in the lower lobes",
      "label": "correct",
      "confidence": 0.92,
      "annotation_method": "rag_mc_consensus",
      "rag_label": "+",
      "mc_score": 0.875,
      "medical_evidence": [
        {
          "source": "Fleischner Society Guidelines",
          "quote": "Ground-glass opacities may represent...",
          "relevance": 0.89
        }
      ]
    },
    {
      "step_id": 1,
      "category": "analysis",
      "content": "Pattern consistent with alveolar filling process",
      "label": "correct",
      "confidence": 0.85,
      "annotation_method": "rag_mc_consensus",
      "rag_label": "+",
      "mc_score": 0.78,
      "differential": [
        "pneumonia",
        "pulmonary_edema",
        "hemorrhage"
      ]
    },
    {
      "step_id": 2,
      "category": "integration",
      "content": "Combined with fever and elevated WBC, suggests infection",
      "label": "correct",
      "confidence": 0.95,
      "annotation_method": "expert_verified",
      "rag_label": "+",
      "mc_score": 0.65,
      "expert_id": "radiologist_001",
      "expert_note": "Correct integration of clinical and imaging findings"
    },
    {
      "step_id": 3,
      "category": "diagnosis",
      "content": "Primary diagnosis: Community-acquired pneumonia (CAP)",
      "label": "correct",
      "confidence": 0.90,
      "annotation_method": "rag_mc_consensus",
      "rag_label": "+",
      "mc_score": 0.81,
      "icd10": "J18.9",
      "supporting_guidelines": [
        "IDSA/ATS CAP Guidelines 2019"
      ]
    }
  ],

  "annotation_metadata": {
    "annotation_date": "2025-01-08",
    "rag_model": "gemini-2.0-flash",
    "mc_model": "InternVL2.5-8B",
    "mc_samples": 32,
    "expert_verified_steps": [2],
    "primary_annotator": "hybrid_system",
    "expert_reviewer": "radiologist_001",
    "total_steps": 4,
    "confident_steps": 3,
    "uncertain_steps": 1
  }
}
```

### 5.6 구현 로드맵

```
Phase 1: Prototype (Week 1-2)
├─ 50 cases from MIMIC-CXR
├─ RAG system setup
│   ├─ Radiology atlas embedding
│   └─ Gemini API integration
├─ Monte Carlo pipeline
└─ 1 radiologist validation

Phase 2: Pilot (Week 3-4)
├─ 500 cases (multi-modality)
├─ Hybrid annotation
├─ 3 specialists (rad/path/derm)
├─ IRB submission
└─ Quality metrics (Cohen's Kappa)

Phase 3: Scale-up (Week 5-8)
├─ 45,000 training cases
├─ 1,000 evaluation cases
├─ Expert verification (selective)
└─ Benchmark v1.0 release

Phase 4: Validation (Week 9-12)
├─ Baseline model evaluation
│   ├─ GPT-4V
│   ├─ Med-Gemini
│   └─ InternVL2.5
├─ PRM training
├─ Best-of-N evaluation
├─ Paper writing
└─ Dataset publication
```

### 5.7 예상 성과

**정량적 목표**:
```
학습 데이터: 45,000 케이스
평가 데이터: 1,000 케이스 (전문가 검증)
총 비용: $5,900
비용/샘플: $0.13

베이스라인 대비 예상 향상:
- MedQA: +5-10%
- PathVQA: +10-15%
- RadiologyQA: +8-12%
```

**정성적 기여**:
1. 의학 근거 기반 멀티모달 PRM (최초)
2. 비용 효율적 하이브리드 어노테이션
3. 재현 가능한 파이프라인
4. 오픈소스 공개

---

## 6. 참고 자료

### 6.1 논문

1. **VisualPRM**
   - Title: "VisualPRM: An Effective Process Reward Model for Multimodal Reasoning"
   - Authors: Weiyun Wang et al.
   - arXiv: 2503.10291
   - Date: March 2025

2. **Med-PRM**
   - Title: "Med-PRM: Medical Reasoning Models with Stepwise, Guideline-verified Process Rewards"
   - Authors: ETH Medical AI Lab
   - arXiv: 2506.11474
   - Conference: EMNLP 2025 (Oral)

3. **Math-Shepherd**
   - arXiv: 2312.08935
   - 최초 Monte Carlo PRM

4. **PRM800K**
   - OpenAI
   - ICLR 2024
   - 최초 대규모 PRM 데이터셋

### 6.2 리소스

**VisualPRM**:
- Paper: https://arxiv.org/abs/2503.10291
- 데이터셋: 논문에서 공개 예정

**Med-PRM**:
- Paper: https://arxiv.org/abs/2506.11474
- GitHub: https://github.com/eth-medical-ai-lab/Med-PRM
- Website: https://med-prm.github.io/
- Model: dmis-lab/llama-3.1-medprm-reward-v1.0 (Hugging Face)
- Dataset: dmis-lab/llama-3.1-medprm-reward-training-set (Hugging Face)

### 6.3 벤치마크

**멀티모달 추론**:
- MMMU, MathVista, MathVision, MathVerse
- DynaMath, WeMath, LogicVista

**의료 텍스트**:
- MedQA, MedMCQA, PubMedQA, MMLU-Med
- DDXPlus, AgentClinic

**의료 멀티모달 (제안)**:
- MIMIC-CXR, PathVQA, RadiologyQA
- Derm7pt, CheXpert

---

## 부록

### A. 용어 정리

- **PRM**: Process Reward Model
- **ORM**: Outcome Reward Model
- **RAG**: Retrieval-Augmented Generation
- **BoN**: Best-of-N
- **MC**: Monte Carlo
- **IRB**: Institutional Review Board
- **PHI**: Protected Health Information

### B. 비용 계산 상세

```python
# VisualPRM 비용
visualprm_cost = {
    "annotators": 13,
    "days": 3,
    "cost_per_day": 37,
    "total": 13 * 3 * 37  # $1,443
}

# Med-PRM 비용
medprm_cost = {
    "api_calls": 11678 * 16,  # questions * candidates
    "cost_per_1k": "$0.000107",  # Gemini-2.0-flash
    "total": 20  # ~$20
}

# 제안 방법 비용
hybrid_cost = {
    "rag_api": 100,
    "monte_carlo_compute": 100,
    "expert_annotation": 420,  # 30% * $1,400
    "total": 620
}
```

### C. 다음 단계 체크리스트

- [ ] Med-PRM GitHub 코드 분석
- [ ] RAG 시스템 프로토타입 구축
- [ ] MIMIC-CXR 50 케이스로 파일럿
- [ ] IRB 신청서 작성
- [ ] 전문의 3명 섭외
- [ ] 벤치마크 v0.1 릴리스

---

**문서 버전**: 1.0
**최종 수정**: 2025-01-08
**작성자**: YK Team
