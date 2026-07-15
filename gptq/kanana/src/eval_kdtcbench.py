"""K-DTCBench 평가 — Kanana-1.5-v-3b-instruct (fp16 vs NF4 4bit).

gptq/src/eval_kdtcbench.py 와 동일 프로토콜·출력 스키마 (11B 결과와 직접 비교용).
Kanana 는 자체 대화 포맷(processor.batch_encode_collate)을 쓰므로 입력 빌더만 다르다.

실행:
  python src/eval_kdtcbench.py --model fp16
  python src/eval_kdtcbench.py --model nf4
결과: results/kdtcbench_<model>.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch

# Windows 함정: datasets(=pyarrow) DLL 로드가 CUDA 드라이버 첫 초기화를 깨뜨린다
# (_cuda_init "no driver"). datasets import 전에 CUDA 컨텍스트를 선점해두면 이후
# .to("cuda") 가 정상 동작한다. gptq 프로젝트 mllama_compat 선-import 와 같은 계열.
if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.zeros(1, device="cuda")

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "kakaocorp/kanana-1.5-v-3b-instruct"

_PROMPT_TMPL = """\
다음 이미지를 보고 질문에 대한 올바른 답을 A, B, C, D 중 하나만 선택하세요.

질문: {question}

A. {a}
B. {b}
C. {c}
D. {d}

정답 (A/B/C/D 중 하나만):"""


def load_model(kind: str):
    """(model, processor) 반환. fp16=bf16 통짜, nf4=bnb NF4 이중양자화.

    Kanana 는 transformers 4.51.3 기준 remote-code 모델이므로 이 venv 는 그 버전 고정.
    모델 카드 표준 방식(AutoModelForVision2Seq + trust_remote_code)을 그대로 쓴다.
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    # Kanana 비전 인코더는 FA2 를 먼저 시도하고 실패 시 자체 fallback 한다.
    # Windows 엔 flash-attn 이 없으니 처음부터 sdpa 로 지정해 그 경로를 건너뛴다.
    kwargs: dict = {"trust_remote_code": True, "attn_implementation": "sdpa"}
    if kind == "fp16":
        kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **kwargs)
        model = model.to("cuda")
    else:  # nf4 — bnb 는 device_map 필요
        from transformers import BitsAndBytesConfig
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor


def _build_inputs(processor, image, question: str, choices: dict, device):
    prompt = _PROMPT_TMPL.format(
        question=question,
        a=choices["a"], b=choices["b"], c=choices["c"], d=choices["d"],
    )
    sample = {
        "image": [image.convert("RGB")],
        "conv": [
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": prompt},
        ],
    }
    inputs = processor.batch_encode_collate(
        [sample], padding_side="left", add_generation_prompt=True, max_length=8192
    )
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


def _extract_answer(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", text.strip().upper())
    return m.group(1) if m else None


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_eval(model, processor, num_samples: int | None = None,
             per_category: int | None = None) -> dict:
    ds = load_dataset("NCSOFT/K-DTCBench", split="test")
    if per_category:
        by_cat: dict[str, list[int]] = {}
        for idx, c in enumerate(ds["category"]):
            by_cat.setdefault(c, []).append(idx)
        sel = [i for c in sorted(by_cat) for i in by_cat[c][:per_category]]
        ds = ds.select(sel)
    elif num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    device = _model_device(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cats: dict[str, dict] = {}
    total_correct, total_n = 0, 0
    t_start = time.time()

    for i, row in enumerate(ds):
        cat = row["category"]
        if cat not in cats:
            cats[cat] = {"correct": 0, "total": 0, "latency_sum": 0.0}

        choices = {"a": row["choice_a"], "b": row["choice_b"],
                   "c": row["choice_c"], "d": row["choice_d"]}
        inputs = _build_inputs(processor, row["image"], row["question"], choices, device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat = time.perf_counter() - t0

        # Kanana.generate 는 새로 생성된 토큰만 반환한다(입력 미포함). 다른 규칙에도
        # 대비해 out 이 입력보다 길 때만 슬라이스한다.
        in_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0][in_len:] if out.shape[-1] > in_len else out[0]
        gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = _extract_answer(gen_text)
        correct = int(pred == row["answer"]) if pred else 0

        cats[cat]["correct"] += correct
        cats[cat]["total"] += 1
        cats[cat]["latency_sum"] += lat
        total_correct += correct
        total_n += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(ds):
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(ds)}] acc={total_correct/total_n:.3f}  "
                  f"({elapsed:.0f}s)", flush=True)

    peak_vram = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0

    by_cat = {
        c: {
            "accuracy": round(v["correct"] / v["total"], 4),
            "correct": v["correct"],
            "total": v["total"],
            "avg_latency_s": round(v["latency_sum"] / v["total"], 3),
        }
        for c, v in cats.items()
    }
    return {
        "total_accuracy": round(total_correct / total_n, 4),
        "total_correct": total_correct,
        "total_n": total_n,
        "by_category": by_cat,
        "peak_vram_gb": round(peak_vram, 2),
        "elapsed_s": round(time.time() - t_start, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fp16", "nf4"], required=True)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--per-category", type=int, default=None)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    print(f"[load] {args.model} 모델 로드 중… ({MODEL_ID})", flush=True)
    model, processor = load_model(args.model)
    if hasattr(model, "get_memory_footprint"):
        print(f"[load] weights footprint: {model.get_memory_footprint() / 1e9:.2f}GB", flush=True)
    print("[load] OK", flush=True)

    print("[eval] K-DTCBench 평가 시작…", flush=True)
    results = run_eval(model, processor, args.num_samples, args.per_category)

    tag = args.out_tag or args.model
    out = ROOT / "results" / f"kdtcbench_{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"model": args.model, "model_id": MODEL_ID, **results},
        ensure_ascii=False, indent=2
    ), encoding="utf-8")

    print(f"\n[done] {out}")
    print(f"  전체 정확도: {results['total_accuracy']:.4f} "
          f"({results['total_correct']}/{results['total_n']})")
    for cat, r in results["by_category"].items():
        print(f"  {cat}: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")
    print(f"  peak VRAM: {results['peak_vram_gb']}GB  elapsed: {results['elapsed_s']}s")


if __name__ == "__main__":
    main()
