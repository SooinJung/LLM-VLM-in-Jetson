# 한국어 VLM 4bit 경량화 for Jetson (Kanana NF4)

**Kanana-1.5-v-3b-instruct**(카카오, 한국어 특화 VLM)를 NF4 4bit로 경량화 진행해 보았따.

## 핵심 결과 요약

Kanana-1.5-v-3b, 한국어 벤치마크 기준 (데스크톱 RTX 3080 Ti):

| 구성 | K-DTCBench (240) | 한국어 PPL | 영어 PPL | weights | Peak VRAM |
|------|------------------|-----------|----------|---------|-----------|
| fp16 (원본) | 78.75% | 6.83 | 14.09 | 7.34GB | 8.18GB |
| **NF4 전체 4bit** | **81.25%** | 7.14 (+4.5%) | 15.01 (+6.5%) | **2.66GB** | **3.59GB** |

- **품질 손실 사실상 0** — NF4가 K-DTCBench에서 오히려 +2.5pp(노이즈 범위), PPL 열화도 한국어 +4.5%에 그침.
- **peak VRAM 3.59GB** — Jetson 8GB에 KV캐시·활성값까지 넉넉히 들어가는 여유.
- K-DTCBench 세부(NF4): document 91.3% / table 86.3% / chart 66.3%.

## NF4 4bit 

**NF4란.** NF4(NormalFloat4)는 bitsandbytes 라이브러리의 4bit 포맷(QLoRA 논문에서 도입). 신경망 가중치가 대체로 정규분포를 따른다는 점을 이용해, 정규분포에 맞게 배치한 16개의 대표값으로 가중치를 반올림한다. 특징 두 가지:

1. **캘리브레이션이 필요 없다** — 데이터 없이 로드 시 즉석 변환. 캘리브 데이터의 언어 편향 문제 자체가 없다.
2. **모델 전체에 적용된다** — 텍스트/비전 구분 없이 모든 Linear 레이어를 4bit로 적재.

**어떻게 적용했나.** 별도 양자화 단계 없이, 모델 로드 시 `BitsAndBytesConfig`를 넘기면 끝이다:

```python
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # 양자화 상수까지 한 번 더 압축
)
model = AutoModelForVision2Seq.from_pretrained(
    "kakaocorp/kanana-1.5-v-3b-instruct",
    quantization_config=bnb, device_map="auto", trust_remote_code=True,
)
```

**결과.** 비전 인코더 포함 전체가 4bit로 적재되어 weights 7.34GB → **2.66GB**, peak VRAM 8.18GB → **3.59GB**. 품질은 위 요약표대로 fp16과 동급.

## 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX 3080 Ti, 12GB |
| CUDA | 12.4 (torch 2.6.0+cu124) |
| Python | 3.11 |
| transformers | **4.51.3 고정** (Kanana remote-code가 요구; 신버전에선 forward가 깨짐) |

> Kanana는 `transformers==4.51.3` 기준 remote-code 모델이라 전용 venv(`kanana/.venv`)를 따로 둔다. 신버전 venv에선 토크나이저 디코드·비전 로터리 임베딩이 깨져 정상 출력이 나오지 않음

## 설치

```powershell
py -3.11 -m venv kanana\.venv
kanana\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==4.51.3" accelerate einops timm datasets bitsandbytes pillow sentencepiece protobuf
pip install "gradio>=5,<6" "huggingface-hub<1.0"   # 채팅 데모용
```

## 실행

NF4는 별도 양자화 단계가 없다 — 스크립트가 로드 시 즉석 양자화한다.

```powershell
# K-DTCBench (한국어 문서/표/차트 VQA, 240문제)
python kanana\src\eval_kdtcbench.py --model nf4   # --model fp16 | nf4

# Perplexity (영어/한국어)
python kanana\src\eval_ppl.py --model nf4
```

결과는 `kanana/results/*.json`으로 저장.

### 채팅 데모

경량화한 NF4 모델과 브라우저에서 바로 대화하는 로컬 UI:

```powershell
python kanana\src\chat_app.py
```

`http://127.0.0.1:7860`에서 이미지를 올리고 한국어로 질문 → 스트리밍 응답, 멀티턴 대화 지원.
