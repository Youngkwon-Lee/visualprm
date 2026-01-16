# CLAUDE.md - visualprm Project

## 🎯 Project Overview

Medical Process Reward Model (Med-PRM) evaluation and benchmarking
- **Model**: dmis-lab/llama-3.1-medprm-reward-v1.0
- **Paper**: arXiv 2506.11474v2
- **Test Set**: 5,469 medical questions × 64 solutions (Medical benchmarks: MedQA, MedMCQA, PubMedQA, MMLU-Med)
- **Metrics**: MV (Majority Voting), BoN (Best-of-N), Min P(+) (minimum step correctness)

---

## 🔄 HPC ↔ 로컬 파일 동기화 가이드

### 상황 분석

**로컬에만 있는 파일** (GitHub 미동기):
- 4_scoring_PRM_no_rag.py (현재 HPC 실행 중)
- 4_scoring_PRM_with_rag_cache.py
- CLAUDE.md (방금 생성)
- check*.py (7개)
- docs/, Med-PRM/, physiomm-prm/

**HPC에만 있는 파일** (매우 큼 ⚠️):
- input.json (~2GB)
- output/*.json (결과 파일, ~100MB+)
- model/ 폴더 (가중치, ~40GB)
- python/4_scoring_PRM.py (기존 코드)
- log_*.out (실행 로그)

### ✅ 동기화 전략

#### Step 1: GitHub에 로컬 파일 추가 (1분)

```bash
# 로컬에서
cd C:\Users\YK\triage\visualprm

# 새 코드 파일 추가
git add 4_scoring_PRM_no_rag.py 4_scoring_PRM_with_rag_cache.py
git add CLAUDE.md check*.py create_test_sample.py verify_mv_prm_logic.py
git add docs/

# Commit & Push
git commit -m "feat: Add optimized PRM scripts and HPC guidelines

- Add 4_scoring_PRM_no_rag.py for memory-efficient RAG-free evaluation
- Add 4_scoring_PRM_with_rag_cache.py for RAG document caching
- Add visualprm-specific CLAUDE.md with HPC command rules
- Add data verification scripts (check*.py)
- Add analysis documentation"

git push origin main
```

#### Step 2: HPC에서 Pull (1분)

```bash
# HPC에서
cd ~/med-prm-vl
git pull origin main
```

**확인:**
```bash
ls -l 4_scoring_PRM_no_rag.py CLAUDE.md  # 파일 존재 확인
git log --oneline -1  # 최신 commit 확인
```

---

### Step 3: HPC 결과 파일 다운로드 (WinSCP 사용)

**⚠️ 주의: 파일 크기 확인 필수!**

```bash
# HPC에서 파일 크기 확인
cd ~/med-prm-vl
du -h input.json                           # ~2GB (유지)
du -h output/                               # ~100-500MB (다운로드)
du -h model/                                # ~40GB (유지, 필요시만)
du -h python/                               # ~100MB (다운로드)
du -h log_*.out                             # ~10-100MB (다운로드)
```

#### WinSCP로 다운로드할 파일

**작은 파일 (모두 다운로드 권장):**
```
~/med-prm-vl/output/
├── medprm_scores_no_rag.json              # 실행 결과
├── medprm_scores.json                     # 원본 결과
└── FINAL_RESULTS.json                     # 최종 요약

~/med-prm-vl/
├── log_no_rag.out                         # 실행 로그
├── log_retest_skipped.out                 # 재처리 로그
└── python/4_scoring_PRM*.py              # 코드 (참고용)
```

**큰 파일 (선택사항):**
```
~/med-prm-vl/input.json                    # 2GB - HPC에만 유지
~/med-prm-vl/model/                        # 40GB - HPC에만 유지
```

#### WinSCP 사용법

**연결 설정:**
```
호스트: VM1212121914 또는 HPC IP
사용자: gun3856
포트: 22 (SSH)
인증: 비밀번호 또는 키 파일
```

**다운로드 순서:**
```
1. 로컬 폴더 생성
   C:\Users\YK\triage\visualprm\output_from_hpc\

2. WinSCP에서 다음 폴더 다운로드:
   ~/med-prm-vl/output/        → C:\Users\YK\triage\visualprm\output_from_hpc\
   ~/med-prm-vl/               → log_*.out 파일들

3. 확인:
   dir C:\Users\YK\triage\visualprm\output_from_hpc\
```

---

### Step 4: 로컬에서 결과 분석

**파일 기반 접근** (HPC 규칙 준수):

analyze_results.py:
```python
#!/usr/bin/env python3
import json

# 원본 결과
with open('output_from_hpc/medprm_scores.json') as f:
    original = json.load(f)

# 최적화 결과
with open('output_from_hpc/medprm_scores_no_rag.json') as f:
    optimized = json.load(f)

print(f"원본 (RAG 포함):  {len(original)} items")
print(f"최적화 (RAG 제외): {len(optimized)} items")

# 간단 비교
orig_skips = sum(1 for item in original for sol in item.get('solutions', [])
                 if sol.get('PRM_min_score') == float('-inf'))
opt_skips = sum(1 for item in optimized for sol in item.get('solutions', [])
                if sol.get('PRM_min_score') == float('-inf'))

print(f"\nSkip 개수:")
print(f"  원본:   {orig_skips}")
print(f"  최적화: {opt_skips} ← 감소!")
```

실행:
```bash
# 다운로드 완료 후
cd C:\Users\YK\triage\visualprm
python3 analyze_results.py
```

---

### 📋 체크리스트

- [ ] GitHub에 로컬 파일 추가 & Push
- [ ] HPC에서 `git pull origin main`
- [ ] HPC에서 파일 크기 확인 (`du -h`)
- [ ] WinSCP로 output/ 폴더 다운로드
- [ ] 로컬에서 결과 파일 존재 확인
- [ ] 결과 비교 분석 완료

---

## 🚫 HPC 명령어 규칙 (중요!)

### ❌ 금지 (절대 사용 금지)

**1. EOF 문법**
```bash
# ❌ 절대 금지 - HPC에서 작동하지 않음
python3 << 'EOF'
import json
data = json.load(open('input.json'))
print(len(data))
EOF
```

**2. 긴 python -c 명령**
```bash
# ❌ 절대 금지 - 한 줄이 길면 실행 안 됨
python3 -c "import json; data = json.load(open('input.json')); sources = Counter(d.get('data_source') for d in data); print('\n'.join(f'{src:25} {cnt:6}' for src, cnt in sources.most_common()))"
```

### ✅ 권장 (파일 기반 접근)

**모든 Python 스크립트는 .py 파일로 작성 후 실행**

**예시 1: 데이터 검증**
```bash
# ✅ 권장 방식
python3 check_data.py
```

check_data.py:
```python
#!/usr/bin/env python3
import json
from collections import Counter

data = json.load(open('input.json'))
sources = Counter(d.get('data_source') for d in data)
for src, cnt in sources.most_common():
    print(f'{src:25} {cnt:6}')
```

**예시 2: 결과 분석**
```bash
# ✅ 권장 방식
python3 analyze_results.py --input output/medprm_scores.json
```

**예시 3: HPC 배치 실행 (nohup)**
```bash
# ✅ 권장 방식 - 장시간 작업용
nohup python3 4_scoring_PRM_no_rag.py \
  --model_save_path ./model \
  --input_json_file ./input.json \
  --output_json_file ./output/medprm_scores_no_rag.json \
  --device 0 \
  --max_token_len 4096 > log_no_rag.out 2>&1 &

# 모니터링
tail -f log_no_rag.out
```

---

## 📂 프로젝트 구조

```
visualprm/
├── CLAUDE.md                          ← 이 파일
├── RETEST_GUIDE.md                    ← 스킵 항목 재처리 가이드
├── CODE_REVIEW.md                     ← MV/PRM 계산 로직 검토
├── TEAM_MEETING_SUMMARY.md            ← 진행상황 요약
│
├── input.json                         ← 원본 데이터 (5,469 항목)
├── input_test_100.json                ← 테스트용 (100 항목)
│
├── 4_scoring_PRM_no_rag.py           ← 현재 실행 중 (RAG 없음)
├── 4_scoring_PRM_with_rag_cache.py   ← RAG 캐싱 (실패)
├── step1_verify_and_extract.py       ← 스킵 항목 추출
├── step2_merge_results.py            ← 결과 병합
├── step3_run_retest.sh               ← 재처리 배치 스크립트
│
├── check.py                          ← 데이터 검증 (파일 기반)
├── output/
│   ├── medprm_scores_no_rag.json    ← RAG 없는 버전 결과
│   └── medprm_scores_final_merged.json  ← 최종 병합 결과
│
└── logs/
    ├── log_no_rag.out               ← 현재 실행 로그
    └── log_retest_skipped.out       ← 재처리 로그
```

---

## 🔧 HPC 작업 패턴

### Pattern 1: 간단한 검증 (수초)
```bash
# 파일 기반 Python 스크립트
python3 check_data.py

# Bash 네이티브 명령어
wc -l input.json
head -100 input.json | tail -10
ls -lh output/
```

### Pattern 2: 단일 실행 작업 (1-2시간)
```bash
# 파일 기반 Python 실행
python3 analyze_results.py --input output/medprm_scores.json
```

### Pattern 3: 장시간 배치 작업 (15-20시간)
```bash
# nohup + 백그라운드 + 로그 리다이렉트
nohup python3 4_scoring_PRM_no_rag.py \
  --model_save_path ./model \
  --input_json_file ./input.json \
  --output_json_file ./output/results.json \
  --device 0 > log.out 2>&1 &

# 모니터링
tail -f log.out
ps aux | grep 4_scoring_PRM
nvidia-smi
```

---

## ⚡ 현재 상태 (2026-01-16)

### Device 0: 진행 중
- **스크립트**: 4_scoring_PRM_no_rag.py
- **입력**: input.json (5,469 항목)
- **진행률**: Q11/5469 (~0.2% complete)
- **Skip 횟수**: 0 ✓ (성공)
- **예상 완료**: 15-20시간
- **모니터링**: `tail -f log_no_rag.out`

### 주요 메트릭
- **MV (Majority Voting)**: 63.6% (현재 진행 중)
- **PRM (Best-of-N)**: 54.5% (현재 진행 중)
- **Skip Ratio**: 0% ← 목표달성!

---

## 📊 Med-PRM 평가 지표 설명

| 지표 | 설명 | 계산 방식 |
|------|------|----------|
| **MV** | Majority Voting (기준선) | 64개 솔루션 중 가장 많은 답변 선택 |
| **BoN** | Best-of-N / PRM | PRM Min P(+) 가장 높은 솔루션 선택 |
| **Min P(+)** | 최소 정답 확률 | 모든 추론 단계 중 최소 P(correct) |
| **Final P(+)** | 최종 정답 확률 | 마지막 추론 단계의 P(correct) |

### Token Skip 문제 (해결됨)
- **원인**: RAG 문서 + 질문 + 솔루션 > 토큰 제한
- **결과**: 점수 계산 실패 (-inf) → PRM 정확도 붕괴
- **해결**: RAG 문서 제거 → 메모리 효율화 + skip=0 달성

---

## 🛠️ 주요 스크립트 사용법

### 데이터 검증
```bash
python3 check_data.py
# 출력: data_source별 항목 수
```

### 결과 분석 (테스트 완료 후)
```bash
python3 analyze_results.py --input output/medprm_scores_no_rag.json
# 출력: MV%, PRM%, Skip ratio, 통계
```

### 스킵 항목 추출 (필요시)
```bash
python3 step1_verify_and_extract.py
# 출력: input_skipped_items.json
```

### 결과 병합 (필요시)
```bash
python3 step2_merge_results.py
# 출력: medprm_scores_final_merged.json
```

---

## 📚 참고 링크

- **Med-PRM 논문**: arXiv 2506.11474v2
- **모델**: https://huggingface.co/dmis-lab/llama-3.1-medprm-reward-v1.0
- **모델 카드**: 11,700개 의료 QA 학습 데이터로 훈련

---

## ✅ 체크리스트

- [x] RAG 없는 버전 생성 (4_scoring_PRM_no_rag.py)
- [x] Device 0에서 전체 5,469 항목 실행 시작
- [x] Skip = 0 달성 ✓
- [ ] 실행 완료 (예상 15-20시간)
- [ ] 최종 MV/PRM 결과 분석
- [ ] 결과 비교: 원본 (MV 72.3%, PRM 22.1%) vs 최적화

---

**HPC 명령 실행 규칙**: EOF 금지 + python -c 금지 → 파일 기반 접근만 사용
