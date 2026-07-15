"""Perplexity 비교 — Kanana-1.5-v-3b fp16 vs NF4 4bit.

gptq/src/eval_ppl.py 와 동일 프로토콜: 비겹침 1024-토큰 윈도우, 언어당 40k 토큰.
토크나이저가 다르므로 절대값은 11B 와 비교 불가 — 모델 내 fp16 대비 Δ만 본다.

실행:
  python src/eval_ppl.py --model fp16
  python src/eval_ppl.py --model nf4
결과: results/ppl_<model>.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from eval_kdtcbench import MODEL_ID, load_model

ROOT = Path(__file__).resolve().parent.parent
SEQ_LEN = 1024
TOKENS_PER_LANG = 40_000


def build_token_stream(tokenizer, lang: str, budget: int) -> torch.Tensor:
    if lang == "en":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
        ids = tokenizer(text, return_tensors="pt").input_ids[0][:budget]
    else:  # ko
        ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
        chunks, n = [], 0
        for row in ds:
            t = (row.get("text") or "").strip()
            if len(t) < 200:
                continue
            ids = tokenizer(t, return_tensors="pt").input_ids[0]
            chunks.append(ids)
            n += len(ids)
            if n >= budget:
                break
        ids = torch.cat(chunks)[:budget]
    print(f"[data] {lang}: {len(ids)} tokens")
    return ids


@torch.no_grad()
def perplexity(model, stream: torch.Tensor, device: str) -> dict:
    # Kanana forward()는 pixel_values/image_metas 가 필수라 텍스트 전용 입력이 안 된다.
    # 내부 언어모델(LlamaForCausalLM)을 직접 호출 — 양자화가 건드리는 텍스트 경로이며
    # 11B PPL(self-attn+MLP 손상 측정)과 같은 의미의 지표.
    lm = getattr(model, "language_model", model)
    total_nll, total_tok = 0.0, 0
    n_windows = len(stream) // SEQ_LEN
    t0 = time.time()
    for i in range(n_windows):
        ids = stream[i * SEQ_LEN:(i + 1) * SEQ_LEN].unsqueeze(0).to(device)
        logits = lm(input_ids=ids).logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits[0, :-1], ids[0, 1:], reduction="sum"
        )
        total_nll += nll.item()
        total_tok += SEQ_LEN - 1
        if (i + 1) % 5 == 0 or i == n_windows - 1:
            el = time.time() - t0
            print(f"  [ppl] window {i+1}/{n_windows}  "
                  f"running_ppl={torch.exp(torch.tensor(total_nll/total_tok)):.3f}  "
                  f"({el:.0f}s)", flush=True)
    return {"ppl": float(torch.exp(torch.tensor(total_nll / total_tok))),
            "tokens": total_tok, "windows": n_windows, "seq_len": SEQ_LEN}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fp16", "nf4"], required=True)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    streams = {lang: build_token_stream(tokenizer, lang, TOKENS_PER_LANG)
               for lang in ("en", "ko")}

    print(f"[load] {args.model} 모델 로드 중…", flush=True)
    model, _ = load_model(args.model)
    print("[load] OK", flush=True)

    results = {}
    for lang, stream in streams.items():
        print(f"[eval] {args.model} / {lang}", flush=True)
        results[lang] = perplexity(model, stream, "cuda")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    suffix = f"_{args.out_tag}" if args.out_tag else ""
    path = out / f"ppl_{args.model}{suffix}.json"
    json.dump({"model": args.model, "model_id": MODEL_ID, **results},
              open(path, "w"), indent=2)
    print(f"[done] {path}")
    for lang, r in results.items():
        print(f"  {lang}: PPL={r['ppl']:.4f} ({r['tokens']} tokens)")


if __name__ == "__main__":
    main()
