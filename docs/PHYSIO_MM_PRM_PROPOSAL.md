# PhysioMM-PRM: 물리치료 멀티모달 Process Reward Model 벤치마크 제안

**작성일**: 2026-01-08
**작성자**: YK Research Team
**목적**: 물리치료/재활 도메인 특화 멀티모달 PRM 벤치마크 설계

---

## 목차

1. [배경 및 동기](#배경-및-동기)
2. [벤치마크 개요](#벤치마크-개요)
3. [세부 설계](#세부-설계)
4. [구현 전략](#구현-전략)
5. [차별점 분석](#차별점-분석)
6. [다음 단계](#다음-단계)

---

## 배경 및 동기

### 물리치료 도메인의 특수성

물리치료/재활 평가는 일반 의료 진단과 근본적으로 다른 특성을 가집니다:

| 특성 | 일반 의료 (Med-PRM) | 물리치료/재활 |
|------|---------------------|---------------|
| **판단 기준** | 진단 정확도 (병리 유무) | 동작 품질, 기능 회복 수준 |
| **시간 요소** | 단일 시점 스냅샷 | **시간적 변화** (운동 전/중/후) |
| **평가 방법** | 정적 검사 (X-ray, 혈액검사) | **동적 평가** (보행, ROM, 기능 테스트) |
| **치료 목표** | 병리 제거/완화 | **기능 향상, 통증 감소, 재발 방지** |
| **데이터 유형** | 주로 이미지 | **비디오 + 이미지 + 센서 데이터** |
| **평가자** | 의사, 영상의학과 전문의 | **물리치료사, 재활의학과 전문의** |

### Med-PRM과 VisualPRM의 한계

**Med-PRM의 한계**:
- ✅ 텍스트 기반 의료 추론 (80.35% MedQA)
- ❌ 시각적 정보 활용 불가
- ❌ 동작 평가 불가능
- ❌ 물리치료 도메인 미포함

**VisualPRM의 한계**:
- ✅ 멀티모달 추론 (이미지 + 텍스트)
- ❌ 일반 도메인 (수학, 과학 문제)
- ❌ 정적 이미지만 처리
- ❌ 의료/물리치료 특화 없음

**기존 의료 VQA 벤치마크의 한계**:
- VQA-RAD, PathVQA: 정적 의료 영상 (X-ray, 병리 슬라이드)
- ❌ 동작 평가 없음
- ❌ 물리치료 시나리오 없음
- ❌ 기능 평가 불가

### 왜 새로운 벤치마크가 필요한가?

1. **동작 평가의 중요성**:
   - 물리치료의 핵심 = 움직임(movement) 분석
   - 정적 이미지로는 보상 패턴, ROM, 안정성 평가 불가능
   - 비디오 모달리티 필수

2. **임상 적용 가능성**:
   - PhysioKorea patient-app: 홈 운동 비디오 자동 평가
   - 실시간 피드백 제공
   - 치료사 워크로드 감소

3. **연구 공백**:
   - 물리치료 AI 연구는 대부분 단일 태스크 (pose estimation, activity recognition)
   - 종합적 임상 추론 능력 평가 부재
   - Process-level 평가 벤치마크 없음

---

## 벤치마크 개요

### PhysioMM-PRM v1.0

**정의**: 물리치료/재활 도메인에 특화된 멀티모달 Process Reward Model 벤치마크

**목표**:
- 재활 운동 수행 품질 평가
- 근골격계 영상 해석 및 치료 계획
- 임상 평가 및 기능 테스트 해석
- 단계별 추론 과정 검증 (PRM 방식)

### 전체 구조

```
PhysioMM-PRM v1.0
├─ 총 10,000 질문
├─ 640,000 솔루션 (질문당 64개)
├─ 100,000 단계별 라벨 (평균 10 steps/question)
└─ 3개 서브셋

┌────────────────────────────────────────────────┐
│ 1️⃣ PhysioVideo (7,000 질문, 70%)              │
│    - 재활 운동 수행 평가                        │
│    - 보행 분석                                 │
│    - 기능 평가 테스트                          │
│    - 모달리티: 비디오 (10-30초)                │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ 2️⃣ PhysioMSK (1,500 질문, 15%)                │
│    - 근골격계 영상 (X-ray, MRI, CT)            │
│    - 정형외과 진단                             │
│    - 치료 계획 수립                            │
│    - 모달리티: 정적 이미지                     │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ 3️⃣ PhysioClinical (1,500 질문, 15%)           │
│    - 자세 평가                                 │
│    - ROM 측정                                  │
│    - 특수 검사                                 │
│    - 모달리티: 임상 사진                       │
└────────────────────────────────────────────────┘
```

### 핵심 혁신점

1. **비디오 중심 설계** (70%):
   - 세계 최초 물리치료 비디오 PRM 벤치마크
   - 시간적 변화, 동작 품질 평가 가능
   - 실제 임상 워크플로우 반영

2. **하이브리드 라벨링**:
   - RAG-Judge (Gemini Pro Vision + 물리치료 가이드라인)
   - 물리치료사 전문가 검증
   - 비용 효율성: $4,000 (vs $15,000 순수 Monte Carlo)

3. **PhysioKorea 생태계 통합**:
   - Patient-app 홈 운동 비디오 활용
   - 학습된 모델로 자동 평가 개선
   - 선순환 데이터 수집 체계

---

## 세부 설계

### 1️⃣ PhysioVideo: 재활 운동 비디오 평가

#### 개요

**목표**: 환자의 재활 운동 수행을 평가하고 보상 패턴을 식별

**데이터 규모**:
- 7,000 질문
- 5가지 핵심 운동 카테고리
- 평균 비디오 길이: 10-30초
- FPS: 30fps
- 해상도: 1080p 이상

#### 포함 운동 유형

| 카테고리 | 운동 예시 | 질문 수 | 평가 항목 | 난이도 분포 |
|---------|----------|---------|-----------|------------|
| **하지 기능 평가** | 스쿼트, 싱글레그 스쿼트, 런지 | 2,000 | FMS 패턴, 무릎/고관절 정렬, 안정성 | Easy: 30%, Medium: 50%, Hard: 20% |
| **상지 기능 평가** | 오버헤드 스쿼트, Y-balance 테스트 | 1,000 | 견갑골 안정성, 흉추 가동성 | Easy: 25%, Medium: 55%, Hard: 20% |
| **코어 안정성** | 플랭크, 사이드 플랭크, 버드독 | 1,500 | 척추 정렬, 보상 패턴 | Easy: 35%, Medium: 50%, Hard: 15% |
| **보행 분석** | 정상/병적 보행 | 1,500 | 보행 주기, 보상, 좌우 대칭 | Easy: 20%, Medium: 50%, Hard: 30% |
| **특수 검사** | Lachman, Drawer, Neer 테스트 | 1,000 | 엔드포인트, 통증 반응 | Easy: 15%, Medium: 45%, Hard: 40% |

#### 데이터 예시

```json
{
  "question_id": "physio_video_squat_001",
  "modality": "video",
  "category": "lower_extremity_assessment",
  "subcategory": "squat",
  "difficulty": "medium",

  "video_metadata": {
    "path": "videos/squat_assessment_001.mp4",
    "duration_sec": 10.5,
    "fps": 30,
    "resolution": "1920x1080",
    "total_frames": 315,
    "key_frames": [0, 150, 300],
    "view": "frontal",
    "camera_setup": "single_fixed"
  },

  "patient_context": {
    "age": 28,
    "sex": "F",
    "height_cm": 165,
    "weight_kg": 60,
    "chief_complaint": "무릎 통증",
    "injury_history": "6개월 전 우측 ACL 재건술",
    "current_status": "재활 3개월차"
  },

  "question": "환자가 보디웨이트 스쿼트를 수행하는 영상입니다. 이 환자의 움직임 패턴을 평가하고 주요 보상 패턴을 식별한 후, 적절한 교정 전략을 제시하세요.",

  "options": [
    "A) 정상적인 스쿼트 패턴 - 교정 불필요",
    "B) 무릎 내반 (knee valgus) + 발목 배굴 제한 - 고관절 외회전근 강화 + 종아리 스트레칭",
    "C) 요추 과신전 + 고관절 굴곡 제한 - 코어 안정화 + 고관절 가동성 운동",
    "D) 발 외회전 + 체중 이동 불균형 - 족부 내재근 강화 + 균형 훈련"
  ],

  "correct_answer": "B",

  "expert_annotation": {
    "primary_compensation": "knee valgus",
    "secondary_compensation": "limited ankle dorsiflexion",
    "severity": "moderate",
    "side": "bilateral, worse on right",

    "phase_analysis": {
      "descent": {
        "observation": "무릎이 점진적으로 내측으로 무너짐",
        "severity": "moderate",
        "timing": "중간 지점부터 시작"
      },
      "bottom": {
        "observation": "발목 배굴 부족으로 발뒤꿈치 살짝 들림",
        "severity": "mild",
        "compensation": "trunk forward lean to compensate"
      },
      "ascent": {
        "observation": "대퇴사두근 우세 패턴, 둔근 활성 부족",
        "severity": "moderate"
      }
    },

    "biomechanical_analysis": {
      "hip_flexion_rom": "adequate",
      "ankle_dorsiflexion_rom": "limited (10-15 degrees)",
      "knee_alignment": "valgus collapse",
      "trunk_stability": "adequate",
      "foot_position": "slight external rotation"
    },

    "treatment_recommendations": [
      "고관절 외회전근 강화 (클램쉘, 사이드 플랭크 with hip abduction)",
      "종아리 스트레칭 (gastrocnemius, soleus)",
      "발목 가동성 운동 (ankle mobilization)",
      "보정된 스쿼트 패턴 재교육 (cue: knees out)"
    ]
  },

  "related_videos": [
    {
      "video_id": "ref_knee_valgus_01",
      "path": "reference/knee_valgus_example_01.mp4",
      "similarity": 0.89,
      "description": "Similar knee valgus pattern"
    },
    {
      "video_id": "ref_squat_normal",
      "path": "reference/squat_normal.mp4",
      "similarity": 0.45,
      "description": "Normal squat for comparison"
    },
    {
      "video_id": "ref_ankle_limitation",
      "path": "reference/limited_ankle_dorsiflexion.mp4",
      "similarity": 0.78,
      "description": "Ankle dorsiflexion limitation compensation"
    }
  ],

  "related_docs": [
    "Cook et al. (2014) - Functional Movement Screen scoring criteria",
    "NASM (2018) - Overhead Squat Assessment guidelines",
    "Hewett et al. (2005) - Biomechanical measures of neuromuscular control and valgus loading",
    "Evidence on knee valgus and ACL injury risk factors"
  ],

  "solutions": [
    {
      "solution_id": 1,
      "raw_text": "Step 1: 환자의 스쿼트 패턴을 전체적으로 관찰한 결과, 하강 단계에서 양측 무릎이 내측으로 무너지는 패턴이 관찰됩니다.\n\nStep 2: 이는 무릎 내반(knee valgus)으로, 고관절 외회전근의 약화를 시사합니다.\n\nStep 3: 최저점에서 발뒤꿈치가 살짝 들리는 것으로 보아 발목 배굴 가동범위가 제한적입니다.\n\nStep 4: ACL 재건술 병력을 고려하면, 수술 후 고관절 근력 약화와 발목 가동성 감소가 흔한 합병증입니다.\n\nStep 5: 따라서 주요 교정 전략은 고관절 외회전근 강화와 발목 가동성 개선이어야 합니다.\n\nStep 6: 정답은 B입니다.",

      "prm_processed_solution": "Step 1: 환자의 스쿼트 패턴을 전체적으로 관찰한 결과, 하강 단계에서 양측 무릎이 내측으로 무너지는 패턴이 관찰됩니다. ки\n\nStep 2: 이는 무릎 내반(knee valgus)으로, 고관절 외회전근의 약화를 시사합니다. ки\n\nStep 3: 최저점에서 발뒤꿈치가 살짝 들리는 것으로 보아 발목 배굴 가동범위가 제한적입니다. ки\n\nStep 4: ACL 재건술 병력을 고려하면, 수술 후 고관절 근력 약화와 발목 가동성 감소가 흔한 합병증입니다. ки\n\nStep 5: 따라서 주요 교정 전략은 고관절 외회전근 강화와 발목 가동성 개선이어야 합니다. ки\n\nStep 6: 정답은 B입니다. ки",

      "answer": "B",
      "score": 1,

      "gemini_labels": [1, 1, 1, 1, 1, 1],
      "gemini_confidence": ["high", "high", "high", "medium", "high", "high"],

      "expert_review": {
        "reviewed": true,
        "expert_id": "pt_expert_001",
        "expert_labels": [1, 1, 1, 1, 1, 1],
        "agreement_with_gemini": true,
        "feedback": "정확한 관찰과 추론. 병력 고려도 적절함."
      }
    }
    // ... 63 more solutions
  ]
}
```

#### 데이터 수집 전략

**소스 1: PhysioKorea Patient-App** (⭐ 핵심)
- 홈 운동 비디오 (이미 수집 중)
- 물리치료사의 기존 평가 활용
- 실제 환자 데이터 → 높은 clinical validity
- 예상: 2,000-3,000 비디오

**소스 2: 공개 교육 비디오**
- YouTube (Creative Commons 라이선스)
  - FMS 공식 채널
  - NASM, NSCA 교육 비디오
  - 물리치료 대학 강의 자료
- 예상: 2,000-3,000 비디오

**소스 3: 크라우드소싱**
- 물리치료 학생/일반인 모집
- 표준 운동 수행 + 의도적 보상 패턴 유도
- 다양한 체형/연령/성별 확보
- 예상: 2,000 비디오

**소스 4: 공개 연구 데이터셋**
- Kinetics-700 (동작 인식)
- NTU RGB+D (골격 동작)
- 물리치료 관련 하위집합 추출
- 예상: 1,000 비디오

---

### 2️⃣ PhysioMSK: 근골격계 영상 진단

#### 개요

**목표**: 근골격계 영상 해석 및 물리치료 관점의 치료 계획 수립

**데이터 규모**:
- 1,500 질문
- 영상 유형: X-ray, MRI, CT
- 해부학적 부위: 척추, 어깨, 무릎, 발목

#### 포함 영상 유형

| 영상 유형 | 적응증 | 질문 수 | 평가 항목 |
|----------|--------|---------|-----------|
| **요추 MRI** | 요통, 하지 방사통 | 400 | 추간판 탈출, 신경근 압박, 척추관 협착 |
| **경추 X-ray/MRI** | 경부통, 편타손상 | 300 | 정렬, 불안정성, 퇴행성 변화 |
| **무릎 MRI** | 인대 손상, 반월판 손상 | 400 | ACL/PCL/MCL, 반월판, 연골 |
| **어깨 MRI** | 회전근개 파열, 충돌증후군 | 300 | 회전근개, 관절와순, 점액낭 |
| **발목 X-ray** | 염좌, 골절 | 100 | 인대, 골절선, 정렬 |

#### 데이터 예시

```json
{
  "question_id": "physio_msk_lumbar_001",
  "modality": "image",
  "category": "msk_imaging",
  "subcategory": "lumbar_spine",
  "difficulty": "hard",

  "image_metadata": {
    "path": "images/lumbar_mri_t2_sag_001.jpg",
    "image_type": "MRI-T2-sagittal",
    "body_part": "lumbar_spine",
    "contrast": "no",
    "slice_location": "midline"
  },

  "patient_context": {
    "age": 45,
    "sex": "M",
    "occupation": "건설 노동자",
    "chief_complaint": "요통 및 좌측 하지 방사통",
    "onset": "3주 전, 무거운 물건 들다가 급성 발생",
    "pain_pattern": "아침에 심하고 활동 시 악화, 기침/재채기 시 방사통 증가",
    "previous_treatment": "약물 치료 2주, 호전 없음",
    "neurological_signs": "좌측 족배굴 약화(4/5), L5 피부분절 감각 저하"
  },

  "question": "이 환자의 요추 MRI 소견을 분석하고, 물리치료 관점에서 가장 적절한 치료 접근법을 선택하세요.",

  "options": [
    "A) L4-5 추간판 탈출증 → McKenzie extension protocol + 신경가동술",
    "B) L5-S1 척추관 협착증 → Williams flexion exercises + 보행 훈련",
    "C) 근막성 요통 → 근력 강화 및 안정화 운동 + 수기치료",
    "D) 척추분리증 → Core stabilization + bracing"
  ],

  "correct_answer": "A",

  "imaging_findings": {
    "level": "L4-5",
    "pathology": "Posterolateral disc herniation, left side",
    "disc_degeneration": "Moderate (Pfirrmann grade 3)",
    "nerve_root_compression": "Left L5 nerve root impingement",
    "canal_stenosis": "None",
    "other_findings": "Mild facet joint hypertrophy"
  },

  "clinical_correlation": {
    "symptom_match": "Yes - L5 radiculopathy pattern",
    "neurological_level": "L5 (foot dorsiflexion weakness, L5 dermatomal sensory loss)",
    "severity": "Moderate",
    "red_flags": "None identified",
    "prognosis": "Good with conservative treatment"
  },

  "related_images": [
    {
      "path": "reference/l4_l5_disc_herniation_01.jpg",
      "similarity": 0.91,
      "description": "Similar L4-5 disc herniation"
    },
    {
      "path": "atlas/normal_lumbar_spine_t2.jpg",
      "similarity": 0.52,
      "description": "Normal lumbar spine for comparison"
    }
  ],

  "related_docs": [
    "Clinical practice guidelines for lumbar disc herniation (APTA)",
    "McKenzie method evidence for directional preference",
    "Neurodynamic techniques for radiculopathy",
    "Red flags in low back pain assessment"
  ]
}
```

#### 데이터 수집 전략

**소스 1: 공개 의료 영상 데이터셋**
- MIMIC-CXR (척추 X-ray 부분)
- Radiopaedia (Creative Commons 라이선스)
- MRNet (Stanford - 무릎 MRI)
- MURA (Stanford - 상지 X-ray)

**소스 2: PhysioKorea 협력 병원**
- 비식별화된 영상 데이터
- 물리치료 처방 기록 포함
- IRB 승인 필수

**소스 3: 의학 교육 자료**
- 대학병원 교육 케이스
- 의학 교과서 이미지

---

### 3️⃣ PhysioClinical: 임상 평가 사진

#### 개요

**목표**: 자세 평가, ROM 측정, 특수 검사 해석

**데이터 규모**:
- 1,500 질문
- 평가 유형: 자세 분석, ROM, 특수 검사
- 시점: 전면, 측면, 후면

#### 포함 평가 유형

| 평가 유형 | 측정 항목 | 질문 수 | 시각적 특징 |
|----------|-----------|---------|-------------|
| **자세 평가** | 정렬, 불균형 | 600 | Plumb line, anatomical landmarks |
| **ROM 측정** | 관절 각도 | 400 | 각도기, photo-based goniometry |
| **부종/변색** | 염증, 손상 정도 | 200 | 색상 변화, 크기 비교 |
| **특수 검사** | Lachman, Neer, Hawkins | 200 | 손 위치, 환자 반응 |
| **기능 테스트** | Y-balance, FMS | 100 | 도달 거리, 안정성 |

#### 데이터 예시

```json
{
  "question_id": "physio_clinical_posture_001",
  "modality": "image",
  "category": "postural_assessment",
  "subcategory": "lateral_view",
  "difficulty": "medium",

  "image_metadata": {
    "path": "images/posture_lateral_001.jpg",
    "view": "lateral_right",
    "landmarks_visible": ["ear", "shoulder", "hip", "knee", "ankle"],
    "plumb_line": true,
    "resolution": "2000x3000"
  },

  "patient_context": {
    "age": 32,
    "sex": "F",
    "occupation": "사무직 (컴퓨터 작업 8시간/일)",
    "chief_complaint": "만성 경부통 및 두통",
    "duration": "6개월 이상",
    "aggravating_factors": "장시간 좌식 작업, 스마트폰 사용"
  },

  "question": "이 환자의 측면 자세 사진을 분석하고, 주요 불균형 패턴과 적절한 교정 전략을 선택하세요.",

  "options": [
    "A) 전방 머리 자세 + 둥근 어깨 → 흉추 신전 운동 + 심부 경추 굴근 강화",
    "B) 과도한 요추 전만 → 골반 후방 경사 운동 + 햄스트링 스트레칭",
    "C) 평평한 등 자세 → 흉추 가동성 운동 + 고관절 굴근 강화",
    "D) 척추후만증 → Schroth method + bracing"
  ],

  "correct_answer": "A",

  "postural_analysis": {
    "head_position": "forward (3cm anterior to plumb line)",
    "cervical_spine": "increased lordosis",
    "shoulder_position": "protracted and internally rotated",
    "scapula": "abducted and downwardly rotated",
    "thoracic_spine": "increased kyphosis",
    "lumbar_spine": "normal lordosis",
    "pelvis": "neutral alignment",
    "overall_pattern": "Upper Crossed Syndrome"
  },

  "treatment_plan": {
    "tight_structures": ["pectoralis major/minor", "upper trapezius", "levator scapulae"],
    "weak_structures": ["deep neck flexors", "lower trapezius", "serratus anterior"],
    "exercises": [
      "Chin tucks (deep neck flexor strengthening)",
      "Prone Y-T-W (scapular stabilizers)",
      "Thoracic extension on foam roller",
      "Pectoralis stretching"
    ],
    "ergonomic_advice": [
      "Monitor height adjustment",
      "Regular breaks (50-10 rule)",
      "Smartphone usage modification"
    ]
  },

  "related_images": [
    {
      "path": "reference/forward_head_posture_01.jpg",
      "similarity": 0.88,
      "description": "Similar forward head posture"
    },
    {
      "path": "atlas/ideal_posture_lateral.jpg",
      "similarity": 0.45,
      "description": "Ideal posture for comparison"
    }
  ],

  "related_docs": [
    "Janda's Upper Crossed Syndrome classification",
    "Evidence for postural correction exercises",
    "Ergonomic guidelines for office workers"
  ]
}
```

---

## 구현 전략

### Phase 1: MVP (Weeks 1-8) - PhysioVideo 중심 ⭐

**목표**: 재활 운동 평가 최소 기능 벤치마크

```
PhysioVideo v1.0 (MVP)
├─ 3,000 질문
├─ 5가지 핵심 운동
│  ├─ 스쿼트 (1,000 질문)
│  ├─ 런지 (600 질문)
│  ├─ 플랭크 (500 질문)
│  ├─ 버드독 (500 질문)
│  └─ 클램쉘 (400 질문)
├─ 30,000 단계별 라벨
├─ 예상 비용: $1,500
└─ 타임라인: 8주
```

**주차별 계획**:

| 주차 | 작업 | 결과물 |
|------|------|--------|
| 1-2 | 데이터 수집 파일럿 | PhysioKorea app 비디오 100개 + YouTube 200개 |
| 3-4 | 질문 생성 및 전문가 리뷰 | 1,000 질문 초안 |
| 5 | 솔루션 생성 (vLLM) | 64,000 솔루션 |
| 6-7 | 하이브리드 라벨링 | RAG-Judge + 물리치료사 검증 |
| 8 | 품질 검증 및 공개 | MVP 벤치마크 릴리즈 |

**즉시 실행 가능 항목** (Week 1):

1. **PhysioKorea Patient-App 데이터 추출**:
   ```sql
   -- Supabase 쿼리
   SELECT video_url, exercise_type, therapist_feedback
   FROM home_exercises
   WHERE therapist_evaluated = true
   LIMIT 100;
   ```

2. **YouTube 비디오 다운로드**:
   ```bash
   # FMS 스쿼트 평가 비디오
   youtube-dl "FMS squat assessment" \
     --license=creativeCommon \
     --max-downloads=50
   ```

3. **전문가 네트워크 구축**:
   - 물리치료학과 교수진 2-3명 섭외
   - 임상 물리치료사 패널 5-10명
   - Expert review 프로토콜 수립

### Phase 2: 확장 (Weeks 9-16) - PhysioMSK 추가

```
+ PhysioMSK
├─ 2,000 질문 추가
├─ 요추/경추/무릎/어깨 MRI/X-ray
├─ 20,000 단계별 라벨
└─ 비용: $1,500
```

### Phase 3: 통합 벤치마크 (Weeks 17-24)

```
PhysioMM-PRM v1.0 (전체)
├─ PhysioVideo: 7,000
├─ PhysioMSK: 1,500
├─ PhysioClinical: 1,500
├─ 총 10,000 질문
├─ 100,000 단계별 라벨
└─ 총 비용: $4,000
```

---

## 차별점 분석

### 기존 벤치마크 대비 우위

| 특성 | Med-PRM | VisualPRM | VQA-RAD | PathVQA | **PhysioMM-PRM** |
|------|---------|-----------|---------|---------|------------------|
| **도메인** | 일반 의료 | 일반 추론 | 방사선 | 병리학 | **물리치료/재활** |
| **모달리티** | 텍스트 | 이미지 | 이미지 | 이미지 | **비디오(70%) + 이미지(30%)** |
| **시간 요소** | ❌ | ❌ | ❌ | ❌ | **✅ 동작의 시간적 변화** |
| **평가 유형** | 진단 | 추론 | 영상 해석 | 병리 해석 | **동작 품질 + 기능 평가** |
| **임상 적용** | 진단 보조 | - | 영상 판독 | 병리 진단 | **치료 계획 + 환자 교육** |
| **PRM 평가** | ✅ | ✅ | ❌ | ❌ | **✅** |
| **데이터 규모** | 11,678 Q | 2,866 Q | 315 Q | 32,799 Q | **10,000 Q** |
| **라벨 수준** | 단계별 | 단계별 | 최종 답변 | 최종 답변 | **단계별** |

### 물리치료 도메인 고유 가치

1. **동작 평가 필수성**:
   - 정적 이미지로는 불가능한 평가
   - 보상 패턴, ROM, 안정성 → 비디오 필수

2. **PhysioKorea 생태계 통합**:
   - Patient-app 홈 운동 비디오 자동 평가
   - 치료사 워크로드 감소
   - 환자 engagement 증가

3. **연구 임팩트**:
   - 세계 최초 물리치료 비디오 PRM
   - CVPR, ICCV, MICCAI 논문 가능
   - 높은 인용 잠재력

### 예상 성능

| 벤치마크 | Baseline (GPT-4V) | Med-PRM (text-only) | **PhysioMM-PRM (ours)** |
|----------|-------------------|---------------------|-------------------------|
| PhysioVideo | 65% | - | **78%** (목표) |
| PhysioMSK | 72% | 75% | **82%** (목표) |
| PhysioClinical | 68% | 70% | **80%** (목표) |
| **전체** | **67%** | **72%** | **79%** (목표) |

---

## 비용 분석

### 상세 비용 breakdown

| 항목 | 수량 | 단가 | 비용 | 비고 |
|------|------|------|------|------|
| **데이터 수집** |
| PhysioKorea 데이터 추출 | 2,000 비디오 | $0 | $0 | 기존 데이터 활용 |
| YouTube 비디오 다운로드 | 2,000 비디오 | $0 | $0 | 공개 라이선스 |
| 크라우드소싱 촬영 | 3,000 비디오 | $0.10 | $300 | 참가자 소정 사례 |
| MSK 영상 수집 | 1,500 이미지 | $0 | $0 | 공개 데이터셋 |
| **질문 생성** |
| GPT-4V 질문 생성 | 3,000 질문 | $0.02 | $60 | 비디오당 질문 생성 |
| 전문가 질문 검토 | 3,000 질문 | $0.20 | $600 | 물리치료사 검토 |
| **솔루션 생성** |
| vLLM 추론 | 640,000 솔루션 | $0 | $0 | 오픈소스 모델 사용 |
| **라벨링** |
| Gemini Pro Vision RAG-Judge | 7,000 비디오 | $0.30 | $2,100 | 비디오 처리 비용 |
| 이미지 RAG-Judge | 3,000 이미지 | $0.25 | $750 | 이미지 처리 |
| 물리치료사 expert review (10%) | 1,000 단계 | $0.50 | $500 | 품질 검증 |
| **인프라** |
| GPU 학습 (4×A100, 48시간) | 48 시간 | $10 | $480 | 모델 학습 |
| 스토리지 | 5 TB | $20/TB | $100 | S3 스토리지 |
| **총계** | | | **$4,890** | |
| **vs 순수 Monte Carlo** | | | ~~$18,000~~ | 73% 절감 |

---

## 다음 단계

### 즉시 실행 (Week 1)

1. **기존 벤치마크 조사**:
   - 물리치료/재활 관련 비디오 데이터셋 검색
   - 유사 벤치마크 존재 여부 확인
   - 차별점 재검토

2. **데이터 수집 파일럿**:
   - PhysioKorea patient-app에서 스쿼트 비디오 100개 추출
   - 물리치료사의 기존 평가 데이터 수집
   - 라벨 포맷 설계

3. **전문가 네트워크**:
   - 물리치료학과 교수 2-3명 섭외
   - Expert review 프로토콜 초안
   - 보상 구조 설계

4. **기술 검증**:
   - 비디오 PRM 아키텍처 프로토타입
   - VideoLLaMA vs Video-ChatGPT 비교
   - 10개 샘플로 PoC

### 단기 목표 (Week 2-4)

1. **기존 연구 완전 조사**:
   - arXiv, Google Scholar 검색
   - 유사 벤치마크 존재 시 차별화 전략 수정
   - 백업 플랜 준비

2. **MVP 데이터셋 구축**:
   - 300 질문 (스쿼트 중심)
   - 하이브리드 라벨링 파일럿
   - 품질 메트릭 검증

3. **논문 초안 작성**:
   - Introduction, Related Work
   - 차별점 명확화
   - 컨퍼런스 타겟 선정 (CVPR vs MICCAI)

### 장기 비전 (6-12개월)

1. **PhysioMM-PRM v1.0 공개**
2. **톱 컨퍼런스 논문 게재**
3. **PhysioKorea 제품 통합**
4. **확장 버전 개발** (센서 데이터, 3D 동작 분석)

---

## 리스크 및 대응 전략

### 주요 리스크

| 리스크 | 확률 | 영향 | 대응 전략 |
|--------|------|------|-----------|
| **유사 벤치마크 존재** | 중 | 높음 | 차별점 강화 (비디오 중심, PhysioKorea 통합) |
| **비디오 처리 기술 미성숙** | 중 | 중 | 이미지 중심으로 pivot, 비디오는 향후 확장 |
| **전문가 확보 어려움** | 낮 | 중 | 물리치료학과 협력, 크라우드소싱 활용 |
| **라벨링 품질 저하** | 중 | 높음 | 엄격한 품질 관리, expert review 비율 증가 |
| **데이터 수집 지연** | 중 | 중 | 공개 데이터로 우선 구축, 자체 수집은 병행 |

---

## 결론

**PhysioMM-PRM**은 물리치료/재활 도메인에 특화된 세계 최초의 멀티모달 Process Reward Model 벤치마크입니다.

**핵심 차별점**:
1. ✅ **비디오 중심** (70%): 동작 평가, 시간적 변화 분석
2. ✅ **물리치료 도메인**: 기존 의료 VQA와 다른 평가 기준
3. ✅ **PhysioKorea 통합**: 실제 임상 데이터 활용 및 제품 개선 선순환
4. ✅ **비용 효율성**: $4,890 (vs $18,000 순수 Monte Carlo)

**다음 단계**:
1. 기존 벤치마크 철저 조사
2. MVP 데이터 수집 파일럿
3. 기술 검증 및 프로토타입

**예상 임팩트**:
- 톱 컨퍼런스 논문 (CVPR, MICCAI)
- PhysioKorea 제품 경쟁력 강화
- 물리치료 AI 연구 표준 벤치마크

---

**문서 버전**: 1.0
**최종 업데이트**: 2026-01-08
**상태**: 제안 단계 - 기존 연구 조사 필요
