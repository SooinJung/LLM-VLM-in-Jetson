"""NF4 4bit 모델을 '오프라인 완결' 체크포인트로 디스크에 저장 — Jetson 이관용.

로드 시 즉석 양자화하던 것을 4bit 가중치째 저장하고, remote-code(.py)와 config를
함께 넣어 인터넷 없이 폴더만으로 로드되게 만든다. 원본 bf16(6.85GB) 불필요.

save_pretrained 만으로는 부족한 3가지를 자동 보정한다:
  1. remote-code(.py 5개) 미포함 → 허브 캐시에서 복사
  2. auto_map 에 붙는 'repo--' 접두사 → 제거(로컬 모듈을 가리키게)
  3. tokenizer_config 의 processor_class 키 → 제거(AutoProcessor 가 토크나이저로 오인)

실행 (gptq/ 에서):
  kanana\\.venv\\Scripts\\python.exe kanana\\src\\save_nf4.py
검증: HF_HUB_OFFLINE=1 로 이 폴더에서 로드 → KananaVProcessor + 정상 생성 확인됨.
"""
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from eval_kdtcbench import load_model, MODEL_ID, ROOT

OUT = ROOT / "models" / "kanana-1.5-v-3b-nf4"
CODE_FILES = ["configuration.py", "modeling.py", "processing.py",
              "processing_image.py", "tokenization.py"]
HUB_PREFIX = f"{MODEL_ID}--"


def dir_size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def _strip_prefix(v):
    if isinstance(v, list):
        return [x.replace(HUB_PREFIX, "") for x in v]
    return v.replace(HUB_PREFIX, "")


def package_offline() -> None:
    """save 이후 폴더를 오프라인 완결 상태로 보정."""
    src = Path(snapshot_download(MODEL_ID, allow_patterns=["*.py", "preprocessor_config.json"]))
    for f in CODE_FILES:
        shutil.copy(src / f, OUT / f)
    # 허브 검증본 preprocessor_config(최소) 사용 — save_pretrained 가 쓴 fuller 버전은 불필요
    shutil.copy(src / "preprocessor_config.json", OUT / "preprocessor_config.json")

    # config.json: auto_map 접두사 제거
    cfg = json.load(open(OUT / "config.json", encoding="utf-8"))
    cfg["auto_map"] = {k: _strip_prefix(v) for k, v in cfg.get("auto_map", {}).items()}
    json.dump(cfg, open(OUT / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # tokenizer_config.json: auto_map 접두사 제거 + processor_class 제거
    tc = json.load(open(OUT / "tokenizer_config.json", encoding="utf-8"))
    tc["auto_map"] = {k: _strip_prefix(v) for k, v in tc.get("auto_map", {}).items()}
    tc.pop("processor_class", None)
    json.dump(tc, open(OUT / "tokenizer_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # save_pretrained 가 남긴 special_tokens_map.json 은 AutoProcessor 오인 유발 → 제거
    stm = OUT / "special_tokens_map.json"
    if stm.exists():
        stm.unlink()


def main() -> None:
    print("[save] NF4 모델 로드 중…", flush=True)
    model, processor = load_model("nf4")
    print(f"[save] in-memory footprint: {model.get_memory_footprint()/1e9:.2f}GB", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[save] {OUT} 에 저장 중…", flush=True)
    model.save_pretrained(str(OUT), safe_serialization=True)
    processor.save_pretrained(str(OUT))

    print("[save] 오프라인 완결 패키징(remote-code 복사 + config 보정)…", flush=True)
    package_offline()

    print(f"\n[done] 오프라인 체크포인트 용량: {dir_size_gb(OUT):.2f}GB")
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"  {f.stat().st_size/1e6:8.1f} MB  {f.relative_to(OUT)}")


if __name__ == "__main__":
    main()
