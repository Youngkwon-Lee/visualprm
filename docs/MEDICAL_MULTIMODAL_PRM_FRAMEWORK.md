# Medical Multimodal PRM Benchmark Framework

**Project**: Medical Multimodal Process Reward Model Benchmark
**Team**: YK Research Team
**Date**: 2026-01-08
**Goal**: Build medical multimodal PRM following Med-PRM → VisualPRM evolution path

---

## Executive Summary

This framework synthesizes insights from **Med-PRM** (text-only, RAG-based, $20 cost) and **VisualPRM** (multimodal, Monte Carlo, $1,443 cost) to design an optimal **Medical Multimodal PRM** that combines:

- **RAG-as-a-Judge** (Med-PRM): Cost-effective, evidence-based medical validation
- **Multimodal Architecture** (VisualPRM): Vision-language reasoning for medical images
- **Hybrid Annotation**: RAG + selective Monte Carlo + expert review

**Expected Outcome**:
- **Dataset**: 10K medical VQA questions with 100K step-wise labels
- **Cost**: $3,500 (vs $15K for pure Monte Carlo)
- **Performance**: 85-90% accuracy (vs 80.35% text-only Med-PRM)
- **Timeline**: 12 weeks

---

## Table of Contents

1. [Evolution Path Analysis](#evolution-path-analysis)
2. [Architecture Design](#architecture-design)
3. [Benchmark Construction](#benchmark-construction)
4. [Implementation Guide](#implementation-guide)
5. [Validation Strategy](#validation-strategy)
6. [Resource Planning](#resource-planning)

---

## Evolution Path Analysis

### Med-PRM → VisualPRM → Medical Multimodal PRM

```
┌─────────────────────────────────────────────────────────────────────┐
│ Med-PRM (Text-Only, Medical Domain)                                │
├─────────────────────────────────────────────────────────────────────┤
│ Innovation: RAG-as-a-Judge                                          │
│ • Use Gemini-2.0-flash + medical docs to label each step           │
│ • Cost: $20 for 11,678 questions (vs $1,443 Monte Carlo)          │
│ • Performance: 80.35% on MedQA (first 8B to exceed 80%)           │
│                                                                     │
│ Architecture:                                                       │
│ • Base: Llama-3.1-8B-Instruct                                      │
│ • Special token: " ки" (step marker from Math-Shepherd)           │
│ • Loss: Cross-entropy over +/- tokens at " ки" positions          │
│                                                                     │
│ Key Files:                                                          │
│ • 2_training.py: Custom loss, RAG document truncation              │
│ • 4_scoring_PRM.py: Best-of-N using min_plus_prob                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ VisualPRM (Multimodal, General Reasoning)                          │
├─────────────────────────────────────────────────────────────────────┤
│ Innovation: Monte Carlo Tree Search for Multimodal Steps           │
│ • Sample 16 continuations per step → compute mc_i (expected acc)  │
│ • Cost: $1,443 for 400K training samples                          │
│ • Performance: +8.4 points across 7 benchmarks                     │
│                                                                     │
│ Architecture:                                                       │
│ • Base: InternVL2.5-8B (vision-language model)                    │
│ • Value PRM: Label = 1 if mc_i > 0                                │
│ • Advantage PRM: Label = 1 if mc_i - mc_{i-1} > 0                 │
│                                                                     │
│ Datasets:                                                           │
│ • VisualPRM400K: 400K auto-generated (Monte Carlo)                │
│ • VisualProcessBench: 2,866 human-annotated ($1,443 cost)        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Medical Multimodal PRM (Our Goal)                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Innovation: Hybrid RAG-Judge + Monte Carlo + Expert Review         │
│ • RAG-Judge (primary): Medical docs + images → Gemini labels      │
│ • Monte Carlo (validation): 4 samples/step for uncertain cases    │
│ • Expert review (gold): 10% sample for quality control            │
│                                                                     │
│ Cost Breakdown:                                                     │
│ • RAG labeling: $2,500 (10K questions × $0.25/image)             │
│ • Monte Carlo: $500 (1K uncertain steps × 4 samples)             │
│ • Expert review: $500 (1K steps × $0.50/step)                    │
│ • Total: $3,500 (vs $15K pure Monte Carlo)                       │
│                                                                     │
│ Expected Performance:                                               │
│ • MedQA (text): 82.5% (+2% over Med-PRM)                          │
│ • VQA-RAD: 82.0%                                                   │
│ • PathVQA: 75.0%                                                   │
│ • MedMM-PRM Benchmark: 87.2%                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Lessons from Each System

| Aspect | Med-PRM | VisualPRM | Medical MM-PRM (Ours) |
|--------|---------|-----------|----------------------|
| **Domain** | Medical text | General multimodal | Medical multimodal |
| **Annotation** | RAG-Judge (Gemini) | Monte Carlo MCTS | Hybrid (RAG + MC + Expert) |
| **Cost** | $20 (11K Qs) | $1,443 (2.8K Qs) | $3,500 (10K Qs) |
| **Efficiency** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Medical Grounding** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **Multimodal** | ❌ | ✅ | ✅ |

**Why Hybrid is Optimal**:
1. **RAG-Judge**: Leverages medical evidence, cost-effective for most cases
2. **Monte Carlo**: Validates uncertain steps, catches edge cases
3. **Expert Review**: Quality control, handles complex diagnostics

---

## Architecture Design

### Option A: Vision-Language Model (VLM) Base

**Model**: LLaVA-Med-7B or InternVL2.5-8B

```
┌─────────────────────────────────────────────────────┐
│              Vision-Language Model                  │
│                  (LLaVA-Med-7B)                    │
├─────────────────────────────────────────────────────┤
│ Input:                                              │
│  - Image: Medical scan/photo                        │
│  - Text: [Docs] + Question + Solution              │
│                                                     │
│ Processing:                                         │
│  - Vision encoder: CLIP ViT-L/14                   │
│  - Text encoder: Llama-2-7B                        │
│  - Cross-attention fusion                          │
│                                                     │
│ Output:                                             │
│  - Logits at " ки" positions                       │
│  - Softmax over {+, -} tokens                      │
└─────────────────────────────────────────────────────┘
```

**Pros**:
- Native multimodal understanding
- Joint image-text reasoning
- Fewer architectural changes

**Cons**:
- Larger model (7B+ params)
- Higher training cost
- Fewer medical VLM checkpoints

**Implementation** (if choosing VLM):
```python
from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# Add special token " ки"
processor.tokenizer.add_special_tokens({"additional_special_tokens": [" ки"]})
model.resize_token_embeddings(len(processor.tokenizer))

# Prepare inputs
conversation = [
    {
        "role": "system",
        "content": RAG_MULTIMODAL_SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"{doc_block}Question: {question}\n\nExplanation: {solution}"}
        ]
    }
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# Forward pass
outputs = model(**inputs)
logits = outputs.logits
```

---

### Option B: Two-Tower Architecture (RECOMMENDED)

**Rationale**: Leverage Med-PRM's trained weights, modular design, lower cost

```
┌────────────────────────────────────────────────────────────────────┐
│                    Two-Tower Architecture                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────┐              ┌──────────────────┐          │
│  │  Image Tower     │              │   Text Tower     │          │
│  │                  │              │                  │          │
│  │  BiomedCLIP      │              │  Llama-3.1-8B    │          │
│  │  ViT-L/14        │              │  (Med-PRM fine-  │          │
│  │                  │              │   tuned)         │          │
│  │  Input:          │              │                  │          │
│  │  - Query image   │              │  Input:          │          │
│  │  - Ref images    │              │  - Documents     │          │
│  │    (RAG)         │              │  - Question      │          │
│  │                  │              │  - Solution      │          │
│  └────────┬─────────┘              └────────┬─────────┘          │
│           │                                 │                    │
│           │  [B, 768]                      │  [B, L, 4096]      │
│           │                                 │                    │
│           ↓                                 ↓                    │
│  ┌────────────────────────────────────────────────────┐          │
│  │          Image Projection Layer                    │          │
│  │          Linear(768 → 4096)                        │          │
│  └───────────────────────┬────────────────────────────┘          │
│                          │  [B, 4096]                            │
│                          │                                        │
│                          ↓                                        │
│  ┌────────────────────────────────────────────────────┐          │
│  │          Cross-Attention Fusion                    │          │
│  │          - Query: Text features [B, L, 4096]       │          │
│  │          - Key/Value: Image features [B, 4096]     │          │
│  │          - Output: Fused features [B, L, 4096]     │          │
│  └───────────────────────┬────────────────────────────┘          │
│                          │                                        │
│                          ↓                                        │
│  ┌────────────────────────────────────────────────────┐          │
│  │          PRM Head (Language Model Head)            │          │
│  │          - Extract logits at " ки" positions       │          │
│  │          - Compute softmax({+, -})                 │          │
│  │          - Output: [num_steps, 2]                  │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**Training Strategy**:

| Component | Training Status | Learning Rate |
|-----------|----------------|---------------|
| Text Tower (Llama-3.1-8B Med-PRM) | **Frozen** ❄️ | N/A |
| Image Tower (BiomedCLIP) | **Fine-tune** 🔥 | 1e-5 |
| Image Projector | **Train from scratch** 🔥 | 1e-4 |
| Cross-Attention | **Train from scratch** 🔥 | 1e-4 |
| PRM Head | **Frozen** ❄️ | N/A |

**Advantages**:
1. **Leverage Med-PRM**: 80.35% baseline, medical reasoning preserved
2. **Modular**: Upgrade image encoder independently (BiomedCLIP → better model)
3. **Cost-Effective**: Only ~20% of parameters trainable
4. **Flexible**: Can toggle image on/off, fallback to text-only Med-PRM

**Implementation**:

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel

class MedicalMultimodalPRM(nn.Module):
    def __init__(self, medprm_path="dmis-lab/llama-3.1-medprm-reward-v1.0"):
        super().__init__()

        # ========== Text Tower (Frozen) ==========
        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            medprm_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(medprm_path)

        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # ========== Image Tower (Fine-tunable) ==========
        self.image_encoder = CLIPVisionModel.from_pretrained(
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        # Optionally freeze lower layers, fine-tune top layers
        # for param in self.image_encoder.vision_model.encoder.layers[:8].parameters():
        #     param.requires_grad = False

        # ========== Projection Layer ==========
        image_dim = self.image_encoder.config.hidden_size  # 768
        text_dim = self.text_encoder.config.hidden_size    # 4096

        self.image_projector = nn.Sequential(
            nn.Linear(image_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

        # ========== Cross-Attention Fusion ==========
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=32,
            dropout=0.1,
            batch_first=True
        )

        # Layer norm for stability
        self.fusion_layer_norm = nn.LayerNorm(text_dim)

    def encode_image(self, images, reference_images=None):
        """
        Encode query image and optional reference images from RAG

        Args:
            images: [B, 3, 224, 224] - Query images
            reference_images: [B, K, 3, 224, 224] - K reference images per query

        Returns:
            image_features: [B, D] where D=4096
        """
        # Encode query image
        query_features = self.image_encoder(images).pooler_output  # [B, 768]

        if reference_images is not None:
            B, K, C, H, W = reference_images.shape
            ref_flat = reference_images.view(B * K, C, H, W)
            ref_features = self.image_encoder(ref_flat).pooler_output  # [B*K, 768]
            ref_features = ref_features.view(B, K, -1)  # [B, K, 768]

            # Aggregate: average pooling over reference images
            ref_pooled = ref_features.mean(dim=1)  # [B, 768]

            # Combine query + references
            combined = query_features + 0.5 * ref_pooled  # Weighted combination
        else:
            combined = query_features

        # Project to text dimension
        image_features = self.image_projector(combined)  # [B, 4096]

        return image_features

    def forward(self, input_ids, attention_mask, images=None, reference_images=None, labels=None, values=None):
        """
        Args:
            input_ids: [B, L] - Tokenized text
            attention_mask: [B, L]
            images: [B, 3, 224, 224] - Optional medical images
            reference_images: [B, K, 3, 224, 224] - Optional RAG images
            labels: [B, L] - Ground truth tokens (for training)
            values: [B, L] - Ground truth +/- labels (for training)

        Returns:
            logits: [B, L, vocab_size]
            loss (if training): scalar
        """
        # ========== Text Encoding ==========
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        text_features = outputs.hidden_states[-1]  # [B, L, 4096]
        logits = outputs.logits  # [B, L, vocab_size]

        # ========== Multimodal Fusion (if images provided) ==========
        if images is not None:
            # Encode images
            image_features = self.encode_image(images, reference_images)  # [B, 4096]

            # Expand image features to sequence length
            image_features_expanded = image_features.unsqueeze(1).expand(-1, text_features.size(1), -1)  # [B, L, 4096]

            # Cross-attention: text attends to image
            fused_features, attention_weights = self.cross_attention(
                query=text_features,
                key=image_features_expanded,
                value=image_features_expanded,
                need_weights=True
            )

            # Residual + layer norm
            fused_features = self.fusion_layer_norm(text_features + fused_features)

            # Re-project to vocabulary
            # Use text encoder's lm_head
            logits = self.text_encoder.lm_head(fused_features)

        # ========== Compute Loss (if training) ==========
        if labels is not None and values is not None:
            loss = self.compute_prm_loss(logits, labels, values)
            return logits, loss

        return logits

    def compute_prm_loss(self, logits, labels, values):
        """
        Same loss as Med-PRM: Cross-entropy over +/- tokens at labeled positions

        Args:
            logits: [B, L, vocab_size]
            labels: [B, L] - Ground truth tokens (masked with -100)
            values: [B, L] - Ground truth +/- labels (0 or 1)

        Returns:
            loss: scalar
        """
        # Get +/- token IDs
        plus_id = self.tokenizer(" +", add_special_tokens=False)["input_ids"][-1]
        minus_id = self.tokenizer(" -", add_special_tokens=False)["input_ids"][-1]

        # Extract +/- logits
        plus_logits = logits[:, :, plus_id]   # [B, L]
        minus_logits = logits[:, :, minus_id] # [B, L]

        # Shift labels and values (predict next token)
        logits_shift = torch.stack([plus_logits[:, :-1], minus_logits[:, :-1]], dim=-1)  # [B, L-1, 2]
        labels_shift = labels[:, 1:].contiguous()  # [B, L-1]
        values_shift = values[:, 1:].contiguous()  # [B, L-1]

        # Mask: only compute loss at labeled positions (labels != -100)
        mask = (labels_shift != -100)

        # Extract valid predictions and labels
        valid_logits = logits_shift[mask]  # [N, 2]
        valid_values = values_shift[mask]  # [N]

        # Ground truth: [p(+), p(-)] = [value, 1-value]
        gt_probs = torch.stack([valid_values, 1 - valid_values], dim=-1)  # [N, 2]

        # Cross-entropy loss
        loss = nn.functional.cross_entropy(valid_logits, gt_probs, reduction="mean")

        return loss

    def get_prm_scores(self, input_ids, attention_mask, images=None, reference_images=None):
        """
        Get PRM scores for evaluation (same as Med-PRM's get_prob function)

        Returns:
            {
                "plus_probs": [...],
                "min_plus_prob": float,
                "final_plus_prob": float
            }
        """
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, images, reference_images)

        # Find " ки" positions
        step_token_id = self.tokenizer(" ки", add_special_tokens=False)["input_ids"][-1]
        step_positions = (input_ids == step_token_id).nonzero(as_tuple=True)[1]

        # Get +/- token IDs
        plus_id = self.tokenizer(" +", add_special_tokens=False)["input_ids"][-1]
        minus_id = self.tokenizer(" -", add_special_tokens=False)["input_ids"][-1]

        # Compute softmax over +/- at each step
        plus_probs = []
        for pos in step_positions:
            if pos >= logits.size(1):
                continue
            two_logits = torch.stack([logits[0, pos, plus_id], logits[0, pos, minus_id]])
            probs = torch.softmax(two_logits, dim=0)
            plus_probs.append(probs[0].item())

        if plus_probs:
            min_plus = min(plus_probs)
            final_plus = plus_probs[-1]
        else:
            min_plus = None
            final_plus = None

        return {
            "plus_probs": plus_probs,
            "min_plus_prob": min_plus,
            "final_plus_prob": final_plus
        }
```

**Usage Example**:

```python
# ========== Training ==========
model = MedicalMultimodalPRM(medprm_path="dmis-lab/llama-3.1-medprm-reward-v1.0")
model = model.to("cuda").bfloat16()

# Optimizer: only train unfrozen parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW([
    {'params': model.image_encoder.parameters(), 'lr': 1e-5},
    {'params': model.image_projector.parameters(), 'lr': 1e-4},
    {'params': model.cross_attention.parameters(), 'lr': 1e-4}
])

# Training loop
for batch in dataloader:
    input_ids, attention_mask, images, ref_images, labels, values = batch

    logits, loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        reference_images=ref_images,
        labels=labels,
        values=values
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# ========== Evaluation ==========
model.eval()
question_image = load_image("chest_xray.jpg")
reference_images = [load_image(img) for img in rag_retrieve_images(question_image)]

# Prepare text input (same as Med-PRM)
messages = [
    {"role": "system", "content": RAG_MULTIMODAL_SYSTEM_PROMPT},
    {"role": "user", "content": f"{doc_block}Question: {question}\n\nExplanation: {solution}"}
]
text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = model.tokenizer(text, return_tensors="pt").to("cuda")

# Get PRM scores
scores = model.get_prm_scores(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    images=question_image.unsqueeze(0).to("cuda"),
    reference_images=torch.stack(reference_images).unsqueeze(0).to("cuda")
)

print(f"Min step probability: {scores['min_plus_prob']:.3f}")
print(f"Final step probability: {scores['final_plus_prob']:.3f}")
print(f"Per-step probabilities: {scores['plus_probs']}")
```

---

## Benchmark Construction

### Dataset Composition

**Target**: 10,000 medical VQA questions across 3 modalities

| Modality | Questions | Images | Difficulty | Data Sources |
|----------|-----------|--------|------------|-------------|
| **Radiology** | 4,000 | 4,000 | Easy: 30%<br>Medium: 50%<br>Hard: 20% | VQA-RAD, SLAKE, MedQA-Radiology, MIMIC-CXR |
| **Pathology** | 3,000 | 3,000 | Easy: 20%<br>Medium: 50%<br>Hard: 30% | PathVQA, QUILT-1M, TCGA |
| **Clinical** | 3,000 | 3,000 | Easy: 40%<br>Medium: 45%<br>Hard: 15% | DermNet, HAM10000, PMC-OA |

**Total**: 10,000 questions, 10,000 images, ~100,000 reasoning steps

### Data Collection Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Source Data Collection                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Radiology:                                                          │
│  - VQA-RAD: 315 samples (existing)                                 │
│  - SLAKE: 14,028 QA pairs (filter diagnostic questions)            │
│  - MedQA subset: 500 questions with X-ray/CT images                │
│  - MIMIC-CXR: Sample 2,000 high-quality reports + images           │
│                                                                     │
│ Pathology:                                                          │
│  - PathVQA: 32,799 samples (filter step-wise questions)            │
│  - QUILT-1M: 1M patches (sample diagnostically relevant)           │
│  - TCGA: Public cancer pathology images                            │
│                                                                     │
│ Clinical:                                                           │
│  - DermNet: 23,000 dermatology images (create QA pairs)            │
│  - HAM10000: 10,000 skin lesion images                            │
│  - PMC-OA: PubMed Central open access figures                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Question Generation                                         │
├─────────────────────────────────────────────────────────────────────┤
│ For each image without existing questions:                          │
│                                                                     │
│ Prompt GPT-4V:                                                      │
│ "Given this medical image, generate a diagnostic multiple-choice   │
│  question suitable for medical board exams. Include:               │
│  1. Clinical context (patient demographics, symptoms)              │
│  2. Question asking for diagnosis/next step/interpretation         │
│  3. 4 plausible options (one correct)                              │
│  4. Difficulty level: [easy/medium/hard]"                          │
│                                                                     │
│ Manual review:                                                       │
│  - Medical expert reviews generated questions                       │
│  - Filters out ambiguous/incorrect questions                        │
│  - Adjusts difficulty ratings                                       │
│                                                                     │
│ Cost: $0.02 × 6,000 images = $120                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Solution Generation                                         │
├─────────────────────────────────────────────────────────────────────┤
│ Use policy model to generate step-wise solutions:                   │
│                                                                     │
│ Model: Llama-3.1-8B-Instruct (or GPT-4V for multimodal)            │
│ Per question: 64 solutions (diversity via temperature=0.7)         │
│                                                                     │
│ Quality filtering:                                                   │
│  - 2 < num_steps < 10                                              │
│  - Answer extraction successful                                     │
│  - No hallucination (verify against medical literature)             │
│                                                                     │
│ Post-processing:                                                     │
│  - Add " ки" step markers                                          │
│  - Format: "Step 1: ... ки Step 2: ... ки ..."                    │
│                                                                     │
│ Cost (if using GPT-4V): $0.01 × 64 × 10,000 = $6,400              │
│ Cost (if using Llama-3.1-8B): vLLM inference ~free                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: Multimodal RAG Setup                                       │
├─────────────────────────────────────────────────────────────────────┤
│ A. Text RAG (existing from Med-PRM):                               │
│  - PubMed abstracts                                                 │
│  - Medical guidelines (UpToDate, NICE, AHA)                        │
│  - Retrieval: BM25 + Dense retriever (e.g., PubMedBERT)           │
│                                                                     │
│ B. Image RAG (new):                                                 │
│  1. Embed all images using BiomedCLIP                              │
│     embedding = biomedclip.encode_image(img)  # [768-dim]         │
│                                                                     │
│  2. Build FAISS index for fast retrieval                           │
│     import faiss                                                    │
│     index = faiss.IndexFlatIP(768)  # Inner product (cosine sim)  │
│     index.add(all_embeddings)                                      │
│                                                                     │
│  3. For each query image, retrieve top-5 similar images            │
│     query_emb = biomedclip.encode_image(query_img)                │
│     distances, indices = index.search(query_emb, k=5)             │
│                                                                     │
│  4. Attach expert annotations from similar cases                    │
│     reference_cases = [dataset[idx] for idx in indices]           │
│                                                                     │
│ Indexing cost: ~1 hour on GPU for 10K images                       │
│ Retrieval cost: ~0.01s per query                                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: Hybrid Annotation                                           │
├─────────────────────────────────────────────────────────────────────┤
│ A. Primary: RAG-Judge with Gemini Pro Vision                       │
│                                                                     │
│    For each solution step:                                          │
│    1. Retrieve relevant documents (text)                            │
│    2. Retrieve similar cases (images)                               │
│    3. Query Gemini Pro Vision:                                      │
│                                                                     │
│       Prompt:                                                        │
│       """                                                            │
│       You are an expert medical evaluator.                          │
│                                                                     │
│       Given:                                                         │
│       - Query image: [Patient's medical image]                      │
│       - Reference images: [5 similar cases with diagnoses]          │
│       - Medical guidelines: {retrieved_docs}                        │
│       - Question: {question}                                         │
│       - Reasoning step: {step_text}                                 │
│                                                                     │
│       Evaluate if this reasoning step is:                           │
│       - Logically valid: Follows from previous steps                │
│       - Medically accurate: Consistent with evidence/guidelines     │
│       - Image-grounded: Correctly interprets visual findings        │
│                                                                     │
│       Output:                                                        │
│       + if step is correct                                          │
│       - if step contains errors                                     │
│                                                                     │
│       Also provide confidence: [high/medium/low]                    │
│       """                                                            │
│                                                                     │
│    4. Extract label (+/-) and confidence                            │
│                                                                     │
│    Cost: $0.25 per image × 10,000 = $2,500                        │
│          (Assumes ~10 steps/question, batched processing)           │
│                                                                     │
│ ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│ B. Validation: Monte Carlo for Low-Confidence Steps                │
│                                                                     │
│    Trigger: When Gemini confidence = "low"                          │
│    Estimated: ~10% of steps (10,000 steps)                         │
│                                                                     │
│    Process:                                                          │
│    1. Sample 4 continuations from current step                      │
│    2. Check if continuations lead to correct answer                 │
│    3. Compute mc_i = #(correct) / 4                                │
│    4. Label = 1 if mc_i > 0.5, else 0                             │
│                                                                     │
│    Cost: GPT-4V $0.01/step × 4 samples × 10,000 = $400            │
│          Or vLLM: ~free if using open-source VLM                   │
│                                                                     │
│ ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│ C. Quality Control: Expert Review (10% Sample)                     │
│                                                                     │
│    Sample: 1,000 random steps (stratified by difficulty)           │
│    Process:                                                          │
│    - Medical expert reviews step + label                            │
│    - Corrects any errors                                            │
│    - Provides detailed feedback                                     │
│                                                                     │
│    Purpose:                                                          │
│    - Measure agreement with Gemini (Cohen's kappa)                 │
│    - Identify systematic biases                                     │
│    - Create gold standard subset                                    │
│                                                                     │
│    Cost: $0.50/step × 1,000 = $500                                │
│          (Assumes medical resident rate: $30/hour, 1 min/step)     │
│                                                                     │
│ ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│ Total Annotation Cost: $2,500 + $400 + $500 = $3,400              │
│ vs Pure Monte Carlo: ~$15,000                                      │
│ Savings: 77%                                                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 6: Quality Metrics                                             │
├─────────────────────────────────────────────────────────────────────┤
│ Compute for entire dataset:                                         │
│                                                                     │
│ 1. Inter-annotator agreement:                                       │
│    - Cohen's kappa (Gemini vs Expert): Target > 0.8               │
│    - If < 0.8: Review disagreements, retrain Gemini prompt        │
│                                                                     │
│ 2. Label distribution:                                              │
│    - % positive steps: Should be ~70-80%                           │
│    - % negative steps: Should be ~20-30%                           │
│    - If imbalanced: Adjust filtering or relabel                    │
│                                                                     │
│ 3. Difficulty calibration:                                          │
│    - Easy questions: > 90% positive steps                          │
│    - Hard questions: 60-70% positive steps                         │
│    - If not aligned: Adjust difficulty ratings                     │
│                                                                     │
│ 4. Modality coverage:                                               │
│    - Ensure balanced across radiology/pathology/clinical           │
│    - Check for domain-specific biases                              │
│                                                                     │
│ 5. Step granularity:                                                │
│    - Average steps per question: 8-12                              │
│    - Too few (<5): Questions too simple                            │
│    - Too many (>15): Solutions too verbose                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Final Dataset Format

**JSON Structure**:
```json
{
  "question_id": "medmm_001",
  "data_source": "mimic_cxr",
  "modality": "radiology",
  "difficulty": "medium",
  "question": "A 65-year-old male with dyspnea. What is the most likely diagnosis?",
  "image_path": "images/chest_xray_001.jpg",
  "image_embedding": [0.123, 0.456, ...],
  "options": [
    "Pneumonia",
    "Pneumothorax",
    "Pulmonary edema",
    "Normal chest X-ray"
  ],
  "correct_answer": "C",
  "related_docs": [
    "Pulmonary edema is characterized by...",
    "Chest X-ray findings in heart failure..."
  ],
  "related_images": [
    {
      "image_path": "reference/pulm_edema_001.jpg",
      "diagnosis": "Pulmonary edema",
      "similarity": 0.92
    },
    ...
  ],
  "solutions": [
    {
      "solution": "Step 1: The chest X-ray shows bilateral infiltrates... Step 2: ...",
      "prm_processed_solution": "Step 1: The chest X-ray shows bilateral infiltrates ки Step 2: ... ки",
      "answer": "C",
      "score": 1,
      "gemini_labels": [1, 1, 0, 1],
      "gemini_confidence": ["high", "high", "low", "medium"],
      "monte_carlo_labels": [null, null, 1, null],
      "expert_review": {
        "reviewed": false,
        "expert_label": null,
        "feedback": null
      }
    },
    // ... 63 more solutions
  ]
}
```

---

## Implementation Guide

### Week-by-Week Timeline

#### **Weeks 1-2: Replication & Setup**

**Goal**: Reproduce Med-PRM results, set up infrastructure

**Tasks**:
1. **Environment setup**
   ```bash
   # Create conda environment
   conda create -n medmm-prm python=3.10
   conda activate medmm-prm

   # Install dependencies
   pip install torch==2.1.0 torchvision transformers==4.36.0
   pip install datasets vllm accelerate wandb
   pip install flash-attn --no-build-isolation
   pip install faiss-gpu pillow scikit-learn
   ```

2. **Reproduce Med-PRM**
   ```bash
   # Clone Med-PRM
   git clone https://github.com/eth-medical-ai-lab/Med-PRM.git
   cd Med-PRM

   # Download data
   python python/0_preparing.py

   # Run evaluation (skip training, use pre-trained)
   bash scripts/4_scoring_PRM.sh
   ```

   **Success Criteria**: Achieve 80.35% ±0.5% on MedQA

3. **Set up data infrastructure**
   ```python
   # Create data directory structure
   mkdir -p data/{raw,processed,embeddings,checkpoints}
   mkdir -p data/images/{radiology,pathology,clinical}
   mkdir -p data/rag/{documents,image_index}

   # Initialize Git LFS for large files
   git lfs install
   git lfs track "*.jpg" "*.png" "*.npy"
   ```

4. **Test multimodal components**
   ```python
   # Test BiomedCLIP
   from transformers import AutoModel, AutoProcessor

   model = AutoModel.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
   processor = AutoProcessor.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

   # Test Gemini Pro Vision API
   import google.generativeai as genai

   genai.configure(api_key="YOUR_API_KEY")
   model_gemini = genai.GenerativeModel('gemini-pro-vision')
   # Test with sample medical image
   ```

**Deliverables**:
- Reproduced Med-PRM evaluation results
- Working multimodal infrastructure
- Test scripts for all components

---

#### **Weeks 3-4: Data Collection**

**Goal**: Collect 10K medical VQA questions with images

**Day-by-Day Breakdown**:

| Day | Task | Output |
|-----|------|--------|
| **Week 3, Day 1-2** | Download existing datasets (VQA-RAD, SLAKE, PathVQA) | 50K raw samples |
| **Week 3, Day 3-5** | Filter diagnostic questions, deduplicate | 6K curated questions |
| **Week 4, Day 1-3** | Generate questions for images without QA (GPT-4V) | 4K new questions |
| **Week 4, Day 4-5** | Medical expert review, adjust difficulty | 10K final questions |

**Detailed Tasks**:

**Task 3.1: Download Existing Datasets**
```python
# download_datasets.py

from datasets import load_dataset
import json

# VQA-RAD
vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
print(f"VQA-RAD: {len(vqa_rad['train'])} samples")

# SLAKE
slake = load_dataset("BoKelvin/SLAKE")
print(f"SLAKE: {len(slake['train'])} samples")

# PathVQA
pathvqa = load_dataset("flaviagiammarino/path-vqa")
print(f"PathVQA: {len(pathvqa['train'])} samples")

# Save locally
for name, dataset in [("vqa_rad", vqa_rad), ("slake", slake), ("pathvqa", pathvqa)]:
    dataset.save_to_disk(f"data/raw/{name}")
```

**Task 3.2: Filter & Deduplicate**
```python
# filter_questions.py

def is_diagnostic_question(question, answer_options):
    """
    Filter for diagnostic questions (not yes/no, not counting)
    """
    # Diagnostic keywords
    diagnostic_keywords = [
        "diagnosis", "most likely", "next step", "treatment",
        "abnormality", "findings", "interpretation"
    ]

    # Exclude yes/no questions
    if any(opt.lower() in ["yes", "no"] for opt in answer_options):
        return False

    # Exclude counting questions
    if any(word in question.lower() for word in ["how many", "count", "number of"]):
        return False

    # Include if contains diagnostic keywords
    if any(kw in question.lower() for kw in diagnostic_keywords):
        return True

    return False

def deduplicate_by_image_hash(samples):
    """
    Remove duplicate images using perceptual hashing
    """
    from imagehash import average_hash
    from PIL import Image

    seen_hashes = set()
    unique_samples = []

    for sample in samples:
        img = Image.open(sample["image_path"])
        img_hash = str(average_hash(img))

        if img_hash not in seen_hashes:
            seen_hashes.add(img_hash)
            unique_samples.append(sample)

    return unique_samples

# Apply filters
filtered = [s for s in all_samples if is_diagnostic_question(s["question"], s["options"])]
deduplicated = deduplicate_by_image_hash(filtered)

print(f"After filtering: {len(filtered)} samples")
print(f"After deduplication: {len(deduplicated)} samples")
```

**Task 3.3: Generate New Questions (GPT-4V)**
```python
# generate_questions.py

import openai
from PIL import Image
import base64
import io

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def generate_medical_question(image_path, modality):
    """
    Generate diagnostic question for medical image using GPT-4V
    """
    image_b64 = image_to_base64(image_path)

    prompt = f"""
You are a medical education expert creating questions for board exams.

Given this {modality} image, generate a diagnostic multiple-choice question.

Requirements:
1. Provide realistic clinical context (age, sex, symptoms)
2. Ask for diagnosis, next step, or image interpretation
3. Provide 4 plausible options (one correct, three distractors)
4. Distractors should be medically plausible but incorrect
5. Assign difficulty: easy/medium/hard

Output format (JSON):
{{
  "question": "A 65-year-old male presents with...",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_answer": "B",
  "difficulty": "medium",
  "explanation": "Brief explanation of correct answer"
}}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        max_tokens=500
    )

    return json.loads(response.choices[0].message.content)

# Generate for images without questions
for img_path in images_without_qa:
    question_data = generate_medical_question(img_path, modality="radiology")
    # Save to database
```

Cost: 4,000 images × $0.02 = $80

**Task 3.4: Expert Review**
```python
# expert_review_interface.py

import streamlit as st
from PIL import Image

def review_question(question_data):
    """
    Simple Streamlit interface for expert review
    """
    st.image(question_data["image_path"])
    st.write(f"**Question:** {question_data['question']}")

    for i, opt in enumerate(question_data["options"]):
        st.write(f"{chr(65+i)}) {opt}")

    st.write(f"**Generated Answer:** {question_data['correct_answer']}")
    st.write(f"**Difficulty:** {question_data['difficulty']}")

    # Expert input
    expert_answer = st.selectbox("Correct answer:", ["A", "B", "C", "D"])
    expert_difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard"])
    keep = st.checkbox("Include in dataset")
    feedback = st.text_area("Feedback (optional)")

    if st.button("Submit"):
        return {
            "keep": keep,
            "correct_answer": expert_answer,
            "difficulty": expert_difficulty,
            "feedback": feedback
        }

# Run review session
st.title("Medical Question Review")
for q in questions_to_review:
    review = review_question(q)
    # Save review results
```

**Deliverables**:
- 10,000 medical VQA questions (4K radiology, 3K pathology, 3K clinical)
- Expert-reviewed and difficulty-calibrated
- Images stored with embeddings

---

#### **Weeks 5-6: Solution Generation & RAG Setup**

**Goal**: Generate 64 solutions per question, set up multimodal RAG

**Task 5.1: Generate Solutions (vLLM)**

```python
# generate_solutions.py (adapted from Med-PRM's 3_test_dataset_sampling.py)

from vllm import LLM, SamplingParams
import json

# Load model
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.95
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_k=50,
    max_tokens=4096
)

# System prompt (multimodal version)
SYSTEM_PROMPT = """
You are a medical expert solving diagnostic questions.
Analyze the medical image and clinical context step-by-step.
Each step must start with "Step {number}:".
Provide your final answer as "the answer is (option letter)".
"""

# Generate solutions
def generate_for_question(question_data):
    # Note: This is text-only generation. For true multimodal, use GPT-4V
    # or wait for open-source VLM support in vLLM

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_data["question"]}
    ]

    # Generate 64 solutions
    outputs = llm.generate(
        prompts=[conversation] * 64,
        sampling_params=sampling_params
    )

    solutions = []
    for output in outputs:
        text = output.outputs[0].text
        # Extract answer
        answer = extract_answer(text)
        # Process for PRM
        prm_solution = add_step_markers(text)

        solutions.append({
            "solution": text,
            "prm_processed_solution": prm_solution,
            "orm_processed_solution": text + " ки",
            "answer": answer,
            "score": int(answer == question_data["correct_answer"]) if answer else 0
        })

    return solutions

# Process all questions
for question in all_questions:
    question["solutions"] = generate_for_question(question)
    # Save checkpoint every 100 questions
    if question_id % 100 == 0:
        save_checkpoint(question_id)
```

**Cost**: Using open-source Llama-3.1-8B via vLLM: **Free** (just GPU time)

**Alternative**: Use GPT-4V for true multimodal generation
- Cost: $0.01 per solution × 64 × 10,000 = $6,400
- Quality: Higher (native vision understanding)

**Task 5.2: Build Image RAG Index**

```python
# build_image_rag.py

import torch
import faiss
import numpy as np
from transformers import AutoModel, AutoProcessor
from PIL import Image
from tqdm import tqdm

# Load BiomedCLIP
model = AutoModel.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
processor = AutoProcessor.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
model = model.to("cuda").eval()

# Encode all images
all_embeddings = []
image_metadata = []

for image_path in tqdm(all_image_paths):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    all_embeddings.append(embedding.cpu().numpy()[0])
    image_metadata.append({
        "path": image_path,
        "diagnosis": get_diagnosis(image_path),
        "modality": get_modality(image_path)
    })

all_embeddings = np.array(all_embeddings).astype('float32')

# Normalize for cosine similarity
faiss.normalize_L2(all_embeddings)

# Build FAISS index
dimension = all_embeddings.shape[1]  # 768
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
index.add(all_embeddings)

# Save index
faiss.write_index(index, "data/rag/image_index/biomedclip.index")
np.save("data/rag/image_index/metadata.npy", image_metadata)

print(f"Indexed {len(all_embeddings)} images")
```

**Task 5.3: Retrieve Similar Cases**

```python
# retrieve_similar_images.py

def retrieve_similar_images(query_image_path, top_k=5):
    """
    Retrieve top-k similar medical images using FAISS
    """
    # Load index
    index = faiss.read_index("data/rag/image_index/biomedclip.index")
    metadata = np.load("data/rag/image_index/metadata.npy", allow_pickle=True)

    # Encode query
    image = Image.open(query_image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        query_embedding = model.get_image_features(**inputs).cpu().numpy()

    # Normalize
    faiss.normalize_L2(query_embedding)

    # Search
    distances, indices = index.search(query_embedding, top_k)

    # Return results
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "image_path": metadata[idx]["path"],
            "diagnosis": metadata[idx]["diagnosis"],
            "similarity": float(dist),
            "modality": metadata[idx]["modality"]
        })

    return results

# Test retrieval
test_image = "data/images/radiology/pneumonia_001.jpg"
similar = retrieve_similar_images(test_image, top_k=5)

for i, img in enumerate(similar):
    print(f"{i+1}. {img['diagnosis']} (similarity: {img['similarity']:.3f})")
```

**Deliverables**:
- 10,000 questions × 64 solutions = 640,000 solutions
- FAISS index with 10,000 image embeddings
- Retrieval pipeline validated

---

#### **Weeks 7-8: Annotation & Training**

**Goal**: Label all solutions with RAG-Judge, train multimodal PRM

**Task 7.1: RAG-Judge Annotation**

```python
# annotate_with_rag_judge.py

import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-1.5-pro-vision')

def annotate_solution_with_rag(question_data, solution_text):
    """
    Use Gemini Pro Vision + RAG to label each step
    """
    # Retrieve context
    retrieved_docs = retrieve_documents(question_data["question"])  # Text RAG
    similar_images = retrieve_similar_images(question_data["image_path"], top_k=3)  # Image RAG

    # Extract steps
    steps = extract_steps(solution_text)

    # Prepare images
    query_image = Image.open(question_data["image_path"])
    reference_images = [Image.open(img["image_path"]) for img in similar_images]

    # Annotate each step
    labels = []
    confidences = []

    for i, step_text in enumerate(steps):
        prompt = f"""
You are an expert medical evaluator.

Query Image: [See attached - patient's medical image]

Reference Cases (similar diagnoses):
{format_reference_cases(similar_images)}

Medical Guidelines:
{format_documents(retrieved_docs)}

Question: {question_data["question"]}
Options: {format_options(question_data["options"])}

Current reasoning step ({i+1}/{len(steps)}):
"{step_text}"

Previous steps:
{format_previous_steps(steps[:i])}

Evaluate this step:
1. Is it logically valid (follows from previous steps)?
2. Is it medically accurate (consistent with guidelines)?
3. Does it correctly interpret the image findings?

Output format:
Label: [+ or -]
Confidence: [high/medium/low]
Reasoning: [1-2 sentences]
"""

        # Call Gemini
        response = gemini_model.generate_content([
            query_image,
            *reference_images,
            prompt
        ])

        # Parse response
        label, confidence, reasoning = parse_gemini_response(response.text)

        labels.append(1 if label == "+" else 0)
        confidences.append(confidence)

    return {
        "gemini_labels": labels,
        "gemini_confidence": confidences,
        "num_steps": len(steps)
    }

# Annotate all solutions
for question in tqdm(all_questions, desc="Annotating"):
    for solution in question["solutions"]:
        annotation = annotate_solution_with_rag(question, solution["prm_processed_solution"])
        solution.update(annotation)

    # Save checkpoint every 50 questions
    if question["question_id"] % 50 == 0:
        save_checkpoint(question["question_id"])
```

**Cost**: $0.25 per image × 10,000 questions = $2,500

**Task 7.2: Monte Carlo Validation (Low-Confidence Steps)**

```python
# monte_carlo_validation.py

def monte_carlo_validate(step_text, question_data, num_samples=4):
    """
    Sample continuations from low-confidence step, check if they lead to correct answer
    """
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", ...)

    # Generate continuations
    prompt = f"{step_text}\n\nContinue reasoning to answer: {question_data['question']}"

    completions = llm.generate(
        prompts=[prompt] * num_samples,
        sampling_params=SamplingParams(temperature=0.8, max_tokens=512)
    )

    # Check correctness
    correct_count = 0
    for completion in completions:
        predicted_answer = extract_answer(completion.outputs[0].text)
        if predicted_answer == question_data["correct_answer"]:
            correct_count += 1

    mc_score = correct_count / num_samples
    mc_label = 1 if mc_score > 0.5 else 0

    return mc_label, mc_score

# Process low-confidence steps
low_conf_steps = [
    (q, sol, step_idx)
    for q in all_questions
    for sol in q["solutions"]
    for step_idx, conf in enumerate(sol["gemini_confidence"])
    if conf == "low"
]

print(f"Found {len(low_conf_steps)} low-confidence steps (expected ~10%)")

for question, solution, step_idx in tqdm(low_conf_steps):
    mc_label, mc_score = monte_carlo_validate(
        solution["prm_processed_solution"].split(" ки")[step_idx],
        question,
        num_samples=4
    )

    if "monte_carlo_labels" not in solution:
        solution["monte_carlo_labels"] = [None] * len(solution["gemini_labels"])

    solution["monte_carlo_labels"][step_idx] = mc_label
    solution["mc_scores"] = solution.get("mc_scores", []) + [mc_score]
```

**Cost**: 10,000 steps × 4 samples × $0.01 = $400 (if using GPT-4V)
**Alternative**: Use vLLM with open-source model: **Free**

**Task 7.3: Train Multimodal PRM**

```python
# train_multimodal_prm.py (adapted from Med-PRM's 2_training.py)

from torch.utils.data import Dataset, DataLoader
import torch

class MultimodalPRMDataset(Dataset):
    def __init__(self, questions, tokenizer, image_processor):
        self.questions = questions
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __getitem__(self, idx):
        question = self.questions[idx]

        # Load image
        image = Image.open(question["image_path"]).convert("RGB")
        image_inputs = self.image_processor(images=image, return_tensors="pt")

        # Load reference images (RAG)
        ref_images = [Image.open(img["image_path"]) for img in question["related_images"][:3]]
        ref_inputs = self.image_processor(images=ref_images, return_tensors="pt")

        # Random solution
        solution = random.choice(question["solutions"])

        # Format text
        messages = [
            {"role": "system", "content": RAG_MULTIMODAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question['question']}\n\nExplanation: {solution['prm_processed_solution']}"}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Tokenize
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=8192,
            truncation=True
        )

        # Labels: Combine Gemini + Monte Carlo
        final_labels = []
        for i, (gemini_label, mc_label) in enumerate(zip(solution["gemini_labels"], solution.get("monte_carlo_labels", [None]*len(solution["gemini_labels"])))):
            if mc_label is not None:
                final_labels.append(mc_label)  # Prefer Monte Carlo if available
            else:
                final_labels.append(gemini_label)

        # Find " ки" positions and create labels tensor
        labels, values = create_prm_labels(text_inputs["input_ids"][0], final_labels, self.tokenizer)

        return {
            "input_ids": text_inputs["input_ids"][0],
            "attention_mask": text_inputs["attention_mask"][0],
            "images": image_inputs["pixel_values"][0],
            "reference_images": ref_inputs["pixel_values"],
            "labels": labels,
            "values": values
        }

# Training loop
model = MedicalMultimodalPRM(medprm_path="dmis-lab/llama-3.1-medprm-reward-v1.0")
model = model.to("cuda").bfloat16()

optimizer = torch.optim.AdamW([
    {'params': model.image_encoder.parameters(), 'lr': 1e-5},
    {'params': model.image_projector.parameters(), 'lr': 1e-4},
    {'params': model.cross_attention.parameters(), 'lr': 1e-4}
], weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    for batch in tqdm(train_dataloader):
        logits, loss = model(**batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    # Save checkpoint
    torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")
```

**Training Config**:
- Epochs: 3-5
- Batch size: 4 per GPU (gradient accumulation 16x)
- Learning rate: 1e-5 (image), 1e-4 (fusion)
- Warmup: 5% of steps
- Total time: ~48 hours on 4× A100 GPUs

**Deliverables**:
- 100,000 step-wise labels (Gemini + Monte Carlo + Expert)
- Trained multimodal PRM model
- Training logs (W&B)

---

#### **Weeks 9-10: Evaluation & Benchmarking**

**Goal**: Evaluate on all benchmarks, create final benchmark dataset

**Task 9.1: Comprehensive Evaluation**

```python
# evaluate_multimodal_prm.py

def evaluate_prm(model, test_data, use_images=True):
    """
    Evaluate PRM on test set with Best-of-N selection
    """
    correct = 0
    total = 0

    for question in tqdm(test_data):
        # Score all solutions
        for solution in question["solutions"]:
            scores = model.get_prm_scores(
                input_ids=tokenize(solution["prm_processed_solution"]),
                attention_mask=attention_mask,
                images=load_image(question["image_path"]) if use_images else None,
                reference_images=load_images(question["related_images"]) if use_images else None
            )

            solution["PRM_min_score"] = scores["min_plus_prob"]
            solution["PRM_final_score"] = scores["final_plus_prob"]

        # Select best solution (highest min_plus_prob)
        valid_solutions = [s for s in question["solutions"] if s["PRM_min_score"] != float("-inf")]
        if not valid_solutions:
            continue

        best_solution = max(valid_solutions, key=lambda s: s["PRM_min_score"])

        if best_solution["score"] == 1:  # Correct answer
            correct += 1
        total += 1

    accuracy = correct / total * 100
    return accuracy

# Evaluate on all benchmarks
results = {}

# 1. Text-only MedQA (baseline)
results["medqa_text_only"] = evaluate_prm(model, medqa_test, use_images=False)

# 2. Multimodal benchmarks
results["vqa_rad"] = evaluate_prm(model, vqa_rad_test, use_images=True)
results["pathvqa"] = evaluate_prm(model, pathvqa_test, use_images=True)
results["slake"] = evaluate_prm(model, slake_test, use_images=True)

# 3. Our benchmark (MedMM-PRM)
results["medmm_prm_overall"] = evaluate_prm(model, medmm_prm_test, use_images=True)
results["medmm_prm_radiology"] = evaluate_prm(model, medmm_prm_test.filter(modality="radiology"), use_images=True)
results["medmm_prm_pathology"] = evaluate_prm(model, medmm_prm_test.filter(modality="pathology"), use_images=True)
results["medmm_prm_clinical"] = evaluate_prm(model, medmm_prm_test.filter(modality="clinical"), use_images=True)

# 4. Difficulty breakdown
for difficulty in ["easy", "medium", "hard"]:
    results[f"medmm_prm_{difficulty}"] = evaluate_prm(
        model,
        medmm_prm_test.filter(difficulty=difficulty),
        use_images=True
    )

print("=" * 60)
print("Evaluation Results")
print("=" * 60)
for benchmark, acc in results.items():
    print(f"{benchmark:30s}: {acc:.2f}%")
```

**Expected Results**:

| Benchmark | Text-only Med-PRM | Multimodal Med-PRM | Gain |
|-----------|-------------------|--------------------|------|
| MedQA (text) | 80.35% | 82.50% | +2.15% |
| VQA-RAD | - | 82.00% | - |
| PathVQA | - | 75.00% | - |
| SLAKE | - | 78.50% | - |
| **MedMM-PRM** | - | **87.20%** | - |
| - Radiology | - | 89.00% | - |
| - Pathology | - | 86.50% | - |
| - Clinical | - | 85.00% | - |
| - Easy | - | 94.00% | - |
| - Medium | - | 87.00% | - |
| - Hard | - | 78.50% | - |

**Task 9.2: Calibration Analysis**

```python
# calibration_analysis.py

def compute_calibration(prm_scores, ground_truth_labels):
    """
    Compute Expected Calibration Error (ECE)
    """
    from sklearn.calibration import calibration_curve

    # Bin probabilities into 10 buckets
    prob_true, prob_pred = calibration_curve(
        ground_truth_labels,
        prm_scores,
        n_bins=10,
        strategy='uniform'
    )

    # Compute ECE
    ece = np.abs(prob_true - prob_pred).mean()

    return ece, prob_true, prob_pred

# Collect all PRM scores and expert labels
all_scores = []
all_labels = []

for question in expert_reviewed_subset:  # 1K expert-reviewed questions
    for solution in question["solutions"]:
        for step_score, expert_label in zip(solution["PRM_score_list"], solution["expert_labels"]):
            all_scores.append(step_score)
            all_labels.append(expert_label)

# Compute calibration
ece, prob_true, prob_pred = compute_calibration(all_scores, all_labels)

print(f"Expected Calibration Error: {ece:.4f}")
print(f"Target: < 0.05 (well-calibrated)")

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.plot(prob_pred, prob_true, 'o-', label=f'Model (ECE={ece:.3f})')
plt.xlabel('Predicted probability')
plt.ylabel('True frequency')
plt.title('Calibration Curve - Multimodal Med-PRM')
plt.legend()
plt.savefig('calibration_curve.png')
```

**Task 9.3: Human Agreement Analysis**

```python
# human_agreement.py

from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Expert-reviewed subset (1K steps)
expert_labels = []
model_labels = []

for question in expert_reviewed_subset:
    for solution in question["solutions"]:
        if "expert_review" in solution and solution["expert_review"]["reviewed"]:
            expert_labels.extend(solution["expert_review"]["expert_labels"])
            model_labels.extend([1 if p > 0.5 else 0 for p in solution["PRM_score_list"]])

# Cohen's kappa
kappa = cohen_kappa_score(expert_labels, model_labels)
print(f"Cohen's Kappa (Model vs Expert): {kappa:.3f}")
print(f"Interpretation: {interpret_kappa(kappa)}")

# Confusion matrix
cm = confusion_matrix(expert_labels, model_labels)
print("\nConfusion Matrix:")
print("                Predicted")
print("                +      -")
print(f"Actual   +   {cm[1,1]:4d}  {cm[1,0]:4d}")
print(f"         -   {cm[0,1]:4d}  {cm[0,0]:4d}")

# Per-step error analysis
errors = [(i, e, m) for i, (e, m) in enumerate(zip(expert_labels, model_labels)) if e != m]
print(f"\nTotal disagreements: {len(errors)} / {len(expert_labels)} ({len(errors)/len(expert_labels)*100:.1f}%)")

# Sample errors for qualitative analysis
print("\nSample Errors (Model Wrong):")
for idx, expert_label, model_label in errors[:10]:
    step_text = get_step_text(idx)
    print(f"Step {idx}: {step_text}")
    print(f"  Expert: {'+ (correct)' if expert_label == 1 else '- (error)'}")
    print(f"  Model:  {'+ (correct)' if model_label == 1 else '- (error)'}")
    print()
```

**Deliverables**:
- Comprehensive benchmark results
- Calibration analysis (ECE < 0.05)
- Human agreement (Cohen's kappa > 0.80)
- Error analysis report

---

#### **Weeks 11-12: Publication & Release**

**Goal**: Write paper, release code/data, community engagement

**Task 11.1: Paper Writing**

**Title**: *Med-PRM-MM: Medical Multimodal Process Reward Models via RAG-Augmented Visual Reasoning*

**Abstract** (200 words):
```
Process reward models (PRMs) have shown promise in improving reasoning quality by
evaluating intermediate steps rather than final outcomes. However, existing work
focuses on text-only reasoning (Math-Shepherd, Med-PRM) or general multimodal tasks
(VisualPRM), leaving a gap in medical multimodal reasoning where visual evidence is
critical for diagnosis. We introduce Med-PRM-MM, the first medical multimodal PRM
that combines RAG-augmented vision-language reasoning with cost-effective annotation.

Our key contributions are: (1) A hybrid annotation strategy combining RAG-Judge
(Gemini Pro Vision + medical documents), Monte Carlo validation, and expert review,
achieving 77% cost savings over pure Monte Carlo ($3,500 vs $15,000); (2) A two-tower
architecture leveraging pre-trained Med-PRM for text reasoning and BiomedCLIP for
visual grounding, enabling modular upgrades; (3) MedMM-PRM Benchmark, a comprehensive
evaluation suite with 10K questions spanning radiology, pathology, and clinical images.

Med-PRM-MM achieves 87.2% accuracy on MedMM-PRM (+7% over text-only Med-PRM) and
82.0% on VQA-RAD, demonstrating strong multimodal reasoning. Human evaluation shows
Cohen's kappa = 0.83 agreement with expert radiologists. Code, models, and benchmark
are released at https://github.com/yourteam/med-prm-mm.
```

**Key Sections**:

1. **Introduction** (1.5 pages)
   - Motivation: Medical diagnosis requires both text and images
   - Gap: No multimodal PRM for medical domain
   - Contributions: Architecture, annotation, benchmark

2. **Related Work** (1 page)
   - Process Reward Models: Math-Shepherd, VisualPRM, Med-PRM
   - Medical VQA: LLaVA-Med, BiomedCLIP, PathVQA
   - Retrieval-Augmented Generation: RAG for medical QA

3. **Method** (3 pages)
   - 3.1 Problem Formulation
   - 3.2 Two-Tower Architecture
   - 3.3 Hybrid Annotation Pipeline
   - 3.4 Training Procedure

4. **Experiments** (3 pages)
   - 4.1 Benchmark Construction (MedMM-PRM)
   - 4.2 Baselines and Implementation
   - 4.3 Main Results
   - 4.4 Ablation Studies
     - RAG vs No-RAG
     - Text-only vs Multimodal
     - Gemini-only vs Hybrid annotation

5. **Analysis** (2 pages)
   - 5.1 Modality Analysis (radiology vs pathology vs clinical)
   - 5.2 Difficulty Breakdown (easy vs medium vs hard)
   - 5.3 Calibration Analysis
   - 5.4 Human Agreement
   - 5.5 Error Analysis (qualitative)
   - 5.6 Cost-Benefit Analysis

6. **Conclusion** (0.5 pages)
   - Summary of contributions
   - Impact on medical AI
   - Future work: Extend to video, 3D scans

**Task 11.2: Code Release**

**GitHub Repository**: `yourteam/med-prm-mm`

```bash
med-prm-mm/
├── README.md                      # Overview, setup, quick start
├── LICENSE                        # Apache 2.0
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── .gitignore
├── docs/
│   ├── installation.md
│   ├── data_format.md
│   ├── training_guide.md
│   ├── evaluation_guide.md
│   └── troubleshooting.md
├── data/
│   ├── download_benchmark.py     # Download MedMM-PRM benchmark
│   ├── process_custom_data.py    # Process your own data
│   └── examples/                  # Sample data
├── medprm_mm/
│   ├── __init__.py
│   ├── model.py                   # MedicalMultimodalPRM class
│   ├── data.py                    # Dataset, collator
│   ├── training.py                # Training loop
│   ├── evaluation.py              # Evaluation metrics
│   ├── rag/
│   │   ├── image_retrieval.py
│   │   └── document_retrieval.py
│   └── utils/
│       ├── image_processing.py
│       └── text_processing.py
├── scripts/
│   ├── 0_download_data.sh
│   ├── 1_build_rag_index.sh
│   ├── 2_train_multimodal_prm.sh
│   ├── 3_evaluate_benchmarks.sh
│   └── 4_run_demo.sh
├── notebooks/
│   ├── 01_quick_start.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_model_inference.ipynb
│   └── 04_results_analysis.ipynb
└── tests/
    ├── test_model.py
    ├── test_data.py
    └── test_rag.py
```

**README.md**:
```markdown
# Med-PRM-MM: Medical Multimodal Process Reward Models

![Banner](assets/banner.png)

[![Paper](https://img.shields.io/badge/arXiv-2501.xxxxx-red)](https://arxiv.org/abs/2501.xxxxx)
[![Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-orange)](https://huggingface.co/yourteam/med-prm-mm-8b)
[![Benchmark](https://img.shields.io/badge/🤗%20HuggingFace-Benchmark-orange)](https://huggingface.co/datasets/yourteam/medmm-prm-benchmark)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

**First medical multimodal process reward model for step-wise reasoning evaluation.**

## 🚀 Quick Start

### Installation
\`\`\`bash
pip install med-prm-mm
\`\`\`

### Inference
\`\`\`python
from medprm_mm import MedicalMultimodalPRM
from PIL import Image

# Load model
model = MedicalMultimodalPRM.from_pretrained("yourteam/med-prm-mm-8b")

# Load medical image
image = Image.open("chest_xray.jpg")

# Evaluate reasoning
question = "What is the most likely diagnosis?"
solution = "Step 1: The chest X-ray shows bilateral infiltrates..."

scores = model.evaluate(question, solution, image)
print(f"Min step confidence: {scores['min_plus_prob']:.3f}")
\`\`\`

## 📊 Benchmark Results

| Model | MedQA | VQA-RAD | PathVQA | MedMM-PRM |
|-------|-------|---------|---------|-----------|
| Med-PRM (text) | **80.35** | - | - | - |
| LLaVA-Med-7B | 75.2 | 68.4 | 54.2 | - |
| GPT-4V | 82.1 | 78.3 | 69.7 | - |
| **Med-PRM-MM (ours)** | **82.5** | **82.0** | **75.0** | **87.2** |

## 🗂️ MedMM-PRM Benchmark

Download our benchmark:
\`\`\`bash
python scripts/0_download_data.sh
\`\`\`

- **10,000 questions** across 3 modalities (radiology, pathology, clinical)
- **100,000 step-wise labels** (hybrid RAG-Judge + Monte Carlo + expert)
- **Expert-validated** subset (1,000 questions)

## 🏗️ Architecture

Two-tower design:
- **Text Tower**: Llama-3.1-8B (Med-PRM fine-tuned) [Frozen]
- **Image Tower**: BiomedCLIP [Fine-tuned]
- **Fusion**: Cross-attention + projection

## 📖 Citation

\`\`\`bibtex
@article{medprm-mm2026,
  title={Med-PRM-MM: Medical Multimodal Process Reward Models via RAG-Augmented Visual Reasoning},
  author={Your Team},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2026}
}
\`\`\`

## 📄 License

Apache 2.0
```

**Task 11.3: HuggingFace Release**

```python
# upload_to_hf.py

from huggingface_hub import HfApi, create_repo
from datasets import Dataset

api = HfApi()

# 1. Upload Model
repo_name = "yourteam/med-prm-mm-8b"
create_repo(repo_name, repo_type="model", private=False)

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"✅ Model uploaded: https://huggingface.co/{repo_name}")

# 2. Upload Benchmark
benchmark_repo = "yourteam/medmm-prm-benchmark"
create_repo(benchmark_repo, repo_type="dataset", private=False)

# Convert to HF Dataset format
dataset = Dataset.from_list(medmm_prm_questions)
dataset.push_to_hub(benchmark_repo)

print(f"✅ Benchmark uploaded: https://huggingface.co/datasets/{benchmark_repo}")
```

**Task 11.4: Gradio Demo**

```python
# demo_app.py

import gradio as gr
from medprm_mm import MedicalMultimodalPRM
from PIL import Image

model = MedicalMultimodalPRM.from_pretrained("yourteam/med-prm-mm-8b")

def evaluate_reasoning(image, question, solution):
    """
    Gradio interface for interactive evaluation
    """
    scores = model.evaluate(question, solution, image)

    # Format output
    output = f"""
### PRM Scores

- **Minimum Step Confidence**: {scores['min_plus_prob']:.3f}
- **Final Step Confidence**: {scores['final_plus_prob']:.3f}

### Per-Step Breakdown

"""
    for i, prob in enumerate(scores['plus_probs']):
        emoji = "✅" if prob > 0.7 else "⚠️" if prob > 0.4 else "❌"
        output += f"Step {i+1}: {emoji} {prob:.3f}\n"

    return output

# Gradio interface
demo = gr.Interface(
    fn=evaluate_reasoning,
    inputs=[
        gr.Image(type="pil", label="Medical Image"),
        gr.Textbox(label="Question", placeholder="What is the most likely diagnosis?"),
        gr.Textbox(label="Step-wise Reasoning", lines=10, placeholder="Step 1: ...\nStep 2: ...")
    ],
    outputs=gr.Markdown(label="PRM Evaluation"),
    title="Med-PRM-MM: Medical Multimodal Process Reward Model",
    description="Evaluate step-wise medical reasoning with visual evidence.",
    examples=[
        ["examples/pneumonia.jpg", "What abnormality is visible?", "Step 1: The chest X-ray shows bilateral infiltrates..."],
        ["examples/melanoma.jpg", "What is the diagnosis?", "Step 1: The lesion has irregular borders..."]
    ]
)

demo.launch(share=True)
```

**Task 11.5: Community Engagement**

**1. Blog Post** (Medium/Substack):
```
Title: "Building Medical Multimodal PRM: Lessons from Med-PRM and VisualPRM"

Outline:
1. Why process reward models matter for medical AI
2. The cost challenge: $15K Monte Carlo vs $3.5K hybrid
3. Architecture choices: VLM vs Two-Tower
4. Annotation quality: RAG-Judge + expert review
5. Results and future directions

Target: 2,000 views, 50 likes
```

**2. Twitter/X Thread**:
```
1/ We're releasing Med-PRM-MM: the first medical multimodal PRM! 🎉

Achieves 87.2% on our new MedMM-PRM benchmark (+7% over text-only Med-PRM)

Paper: [link]
Code: [link]
Demo: [link]

[Banner image]

2/ Key innovation: Hybrid annotation combining RAG-Judge, Monte Carlo, and expert review

Cost: $3,500 (vs $15K for pure Monte Carlo)
Quality: Cohen's kappa = 0.83 with expert radiologists

Win-win! 📊

[Cost comparison chart]

3/ Architecture: Two-tower design
- Text: Llama-3.1-8B (Med-PRM fine-tuned) [Frozen]
- Vision: BiomedCLIP [Fine-tuned]
- Fusion: Cross-attention

Modular = easy to upgrade components independently

[Architecture diagram]

4/ Results across benchmarks:
- MedQA: 82.5% (+2% over text)
- VQA-RAD: 82.0%
- PathVQA: 75.0%
- MedMM-PRM: 87.2%

Strong gains from multimodal reasoning! 📈

[Results table]

5/ MedMM-PRM Benchmark:
✅ 10K questions (radiology, pathology, clinical)
✅ 100K step-wise labels
✅ Expert-validated subset

Now available on HuggingFace! 🤗

[Benchmark card]

6/ Check out our Gradio demo!

Upload a medical image + reasoning steps → Get instant PRM evaluation

Try it: [demo link]

[Demo screenshot]

7/ Huge thanks to the Med-PRM and VisualPRM teams for inspiration!

This work shows how we can combine the best of both:
- Med-PRM's RAG efficiency
- VisualPRM's multimodal reasoning

[Comparison chart]

8/ What's next?
- Extend to videos (procedures, ultrasound)
- 3D medical imaging (CT, MRI)
- Real-time feedback during diagnostic workflow

Stay tuned! 🚀

[Roadmap image]

9/ Paper, code, models, and benchmark:

📄 arXiv: [link]
💻 GitHub: [link]
🤗 Model: [link]
📊 Benchmark: [link]
🎮 Demo: [link]

Please share if you find this useful! 🙏
```

**3. YouTube Demo** (5-10 minutes):
```
Script:

[0:00-0:30] Introduction
- Medical diagnosis requires visual evidence
- Existing PRMs are text-only (Med-PRM) or general (VisualPRM)
- We built the first medical multimodal PRM

[0:30-1:30] Problem Setup
- Show example: Chest X-ray with pneumonia
- Model must evaluate each reasoning step
- Challenge: Need both medical knowledge and image interpretation

[1:30-3:00] Our Approach
- Hybrid annotation (RAG + Monte Carlo + expert)
- Two-tower architecture (text + vision)
- Demo annotation process

[3:00-5:00] Live Demo
- Load Gradio interface
- Upload medical image
- Input reasoning steps
- Show PRM scores

[5:00-6:30] Results
- Benchmark comparison table
- Modality breakdown
- Human agreement

[6:30-7:00] Code Walkthrough
- Quick tour of GitHub repo
- How to run inference
- How to train on custom data

[7:00-8:00] Conclusion
- Impact on medical AI
- Future directions
- Call to action (try demo, cite paper)
```

**Deliverables**:
- Submitted paper to EMNLP/ACL/NeurIPS 2026
- Released code, models, benchmark on GitHub/HuggingFace
- Gradio demo with 1K+ users
- Blog post with 2K+ views
- Twitter thread with 10K+ impressions
- YouTube video with 5K+ views

---

## Validation Strategy

### Quality Assurance Checklist

**Data Quality**:
- [ ] All images have valid medical context
- [ ] Questions are clinically appropriate
- [ ] Difficulty ratings calibrated by experts
- [ ] No duplicate images (perceptual hashing)
- [ ] No HIPAA violations (all public data)

**Annotation Quality**:
- [ ] Inter-annotator agreement (Gemini vs Expert) > 0.80
- [ ] Label distribution: 70-80% positive, 20-30% negative
- [ ] Difficulty-stratified sampling for expert review
- [ ] Monte Carlo validation on low-confidence steps
- [ ] Expert feedback incorporated

**Model Quality**:
- [ ] Reproduces Med-PRM text-only results (80.35% ±0.5%)
- [ ] Multimodal gain > 5% on at least one benchmark
- [ ] Calibration error (ECE) < 0.05
- [ ] Human agreement (Cohen's kappa) > 0.80
- [ ] No catastrophic forgetting (text-only performance preserved)

**Code Quality**:
- [ ] All tests pass (pytest coverage > 90%)
- [ ] Documentation complete (README, docs/, docstrings)
- [ ] Reproducible (fixed seeds, deterministic training)
- [ ] Efficient (vLLM inference, gradient checkpointing)
- [ ] Modular (easy to swap image encoder, text encoder)

**Release Quality**:
- [ ] Paper proofread by all co-authors
- [ ] Code reviewed by 2+ team members
- [ ] HuggingFace model card complete
- [ ] Benchmark card with datasheet
- [ ] Demo tested on 10+ users
- [ ] License compliance (Apache 2.0)

---

## Resource Planning

### Budget Breakdown

| Category | Item | Cost | Notes |
|----------|------|------|-------|
| **Data Collection** | GPT-4V question generation | $120 | 6K images × $0.02 |
| | Expert review | $500 | 1K steps × $0.50 |
| **Annotation** | Gemini Pro Vision (RAG-Judge) | $2,500 | 10K questions × $0.25 |
| | Monte Carlo validation (GPT-4V) | $400 | 10K steps × 4 samples × $0.01 |
| | *Alternative: vLLM (open-source)* | *$0* | *Free (GPU time only)* |
| **Training** | GPU compute (4× A100, 48 hours) | $480 | $10/hour |
| **Evaluation** | API calls for baselines | $200 | GPT-4V, LLaVA-Med |
| **Misc** | Storage, compute for RAG index | $100 | S3, FAISS |
| **Total** | | **$4,300** | With GPT-4V |
| **Total (vLLM)** | | **$3,900** | With open-source models |

**Savings vs Pure Monte Carlo**: $15,000 - $3,900 = **$11,100 (74% reduction)**

### Compute Resources

**Development** (Weeks 1-2):
- 1× A100 GPU (40GB) for experiments
- 500GB storage

**Data Collection** (Weeks 3-4):
- CPU-only (API calls to GPT-4V, Gemini)
- 1TB storage for images

**RAG Setup** (Weeks 5-6):
- 2× A100 GPUs for embedding generation
- 1TB SSD for FAISS index
- 64 CPUs for vLLM inference

**Training** (Weeks 7-8):
- 4× A100 GPUs (80GB each) for 48 hours
- 2TB NVMe storage for checkpoints

**Evaluation** (Weeks 9-10):
- 2× A100 GPUs for inference
- CPU for metrics computation

**Total GPU-hours**: ~500 hours ($5,000 value)

### Team Structure

**Minimum Viable Team** (3 people):

1. **ML Engineer** (full-time):
   - Model architecture implementation
   - Training pipeline setup
   - Evaluation infrastructure
   - Code release preparation

2. **Medical Expert** (part-time, 20%):
   - Question quality review
   - Annotation validation
   - Expert labeling (1K sample)
   - Clinical interpretation

3. **Research Lead** (full-time):
   - Project management
   - Data collection strategy
   - Paper writing
   - Community engagement

**Optional** (larger team):
- Data Scientist: RAG optimization, ablation studies
- ML Ops: Infrastructure, deployment, monitoring
- UX Designer: Gradio demo, documentation

---

## Summary

This framework provides a complete roadmap for building a medical multimodal PRM benchmark, synthesizing insights from Med-PRM and VisualPRM:

**Key Innovations**:
1. **Hybrid Annotation**: Combines RAG-Judge ($2,500), Monte Carlo ($400), and expert review ($500) for 77% cost savings
2. **Two-Tower Architecture**: Leverages pre-trained Med-PRM (text) + BiomedCLIP (vision) for modular design
3. **MedMM-PRM Benchmark**: 10K questions, 100K labels, 3 modalities (radiology, pathology, clinical)

**Expected Impact**:
- **Performance**: 87.2% on MedMM-PRM (+7% over text-only Med-PRM)
- **Cost**: $3,900 total (vs $15K pure Monte Carlo)
- **Timeline**: 12 weeks with 3-person team

**Deliverables**:
- Published paper (EMNLP/ACL/NeurIPS 2026)
- Open-sourced code, models, benchmark
- Gradio demo with 1K+ users
- Community engagement (blog, Twitter, YouTube)

This framework balances **cost-effectiveness** (Med-PRM's RAG-Judge), **quality** (VisualPRM's multimodal reasoning), and **medical grounding** (expert validation) to create a robust benchmark that advances medical AI research.

---

**Document Version**: 1.0
**Last Updated**: 2026-01-08
**Team**: YK Research
**Status**: Ready for Implementation
