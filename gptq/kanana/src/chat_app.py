"""Kanana NF4 4bit 로컬 채팅 UI — 이미지 올리고 한국어로 대화.

실행 (별도 터미널에서, gptq/ 에서):
  C:\\Users\\keti\\Pseudo\\gptq\\kanana\\.venv\\Scripts\\python.exe kanana\\src\\chat_app.py
브라우저에서 http://127.0.0.1:7860 열림. Ctrl+C 로 종료.

Windows 주의: GPU 는 샌드박스 밖에서만 잡히므로 사용자가 직접 터미널에서 실행해야 함.
"""
import threading

import gradio as gr
import torch
from transformers import TextIteratorStreamer

# load_model 은 datasets(pyarrow) import 전 CUDA 선점 init 을 포함 — 그대로 재사용
from eval_kdtcbench import load_model, MODEL_ID

print("[app] NF4 모델 로드 중…", flush=True)
model, processor = load_model("nf4")
device = next(model.parameters()).device
print(f"[app] 로드 완료 ({model.get_memory_footprint()/1e9:.2f}GB, {device})", flush=True)


def chat(message, history, image):
    if image is None:
        yield "먼저 왼쪽에 이미지를 올려주세요. 🖼️"
        return

    # Kanana 대화 포맷: 맨 앞 <image> 턴 + 히스토리(role 그대로) + 이번 질문
    conv = [{"role": "user", "content": "<image>"}]
    for h in history:
        conv.append({"role": h["role"], "content": h["content"]})
    conv.append({"role": "user", "content": message})

    inputs = processor.batch_encode_collate(
        [{"image": [image.convert("RGB")], "conv": conv}],
        padding_side="left", add_generation_prompt=True, max_length=8192,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
    kw = dict(**inputs, max_new_tokens=512, do_sample=False, streamer=streamer)
    threading.Thread(target=model.generate, kwargs=kw).start()

    acc = ""
    for chunk in streamer:
        acc += chunk
        yield acc


with gr.Blocks(title="Kanana NF4 Chat") as demo:
    gr.Markdown(f"## 🐿️ Kanana-1.5-v-3b (NF4 4bit) 로컬 채팅\n`{MODEL_ID}` · 이미지를 올리고 한국어로 질문하세요.")
    with gr.Row():
        img = gr.Image(type="pil", label="이미지", height=360)
        gr.ChatInterface(fn=chat, type="messages", additional_inputs=[img])

if __name__ == "__main__":
    demo.launch()
