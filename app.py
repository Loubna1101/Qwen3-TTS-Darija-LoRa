import os
import json
import tempfile

import gradio as gr
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
HF_MODEL_REPO = "loubna1101/Qwen3-TTS-Darija-LoRa"
DEFAULT_SPEAKER = "darija_speaker"

qwen3tts = None


def load_model():
    global qwen3tts

    if qwen3tts is not None:
        return qwen3tts

    ckpt_dir = snapshot_download(
        repo_id=HF_MODEL_REPO,
        repo_type="model"
    )

    qwen3tts = Qwen3TTSModel.from_pretrained(
        BASE_MODEL,
        attn_implementation="eager",
    )

    qwen3tts.model.talker.model = PeftModel.from_pretrained(
        qwen3tts.model.talker.model,
        os.path.join(ckpt_dir, "talker_lora")
    )

    if hasattr(qwen3tts.model.talker.model, "merge_and_unload"):
        qwen3tts.model.talker.model = qwen3tts.model.talker.model.merge_and_unload()

    spk = torch.load(os.path.join(ckpt_dir, "speaker_embedding.pt"), map_location="cpu")
    speaker_embedding = spk["embedding"]
    speaker_id = spk["speaker_id"]

    with torch.no_grad():
        emb = qwen3tts.model.talker.model.codec_embedding.weight
        emb[speaker_id] = speaker_embedding.to(emb.device).to(emb.dtype)

    with open(os.path.join(ckpt_dir, "config.json"), "r", encoding="utf-8") as f:
        saved_config = json.load(f)

    qwen3tts.model.tts_model_type = saved_config["tts_model_type"]
    qwen3tts.model.config.tts_model_type = saved_config["tts_model_type"]

    saved_talker_config = saved_config.get("talker_config", {})

    if hasattr(qwen3tts.model.config, "talker_config"):
        if "spk_id" in saved_talker_config:
            qwen3tts.model.config.talker_config.spk_id = saved_talker_config["spk_id"]
        if "spk_is_dialect" in saved_talker_config:
            qwen3tts.model.config.talker_config.spk_is_dialect = saved_talker_config["spk_is_dialect"]

    if hasattr(qwen3tts.model, "talker_config") and hasattr(qwen3tts.model.talker_config, "__dict__"):
        if "spk_id" in saved_talker_config:
            qwen3tts.model.talker_config.spk_id = saved_talker_config["spk_id"]
        if "spk_is_dialect" in saved_talker_config:
            qwen3tts.model.talker_config.spk_is_dialect = saved_talker_config["spk_is_dialect"]

    speaker_names = set(saved_talker_config.get("spk_id", {}).keys())
    qwen3tts._supported_speakers_set = lambda: speaker_names

    return qwen3tts


def synthesize(text):
    if not text or not text.strip():
        raise gr.Error("Please enter some text.")

    model = load_model()

    audio = model.generate_custom_voice(
        text=text,
        speaker=DEFAULT_SPEAKER
    )

    waveform = audio[0][0].squeeze().astype("float32")
    sr = audio[1]

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, waveform, sr)

    return tmp.name


demo = gr.Interface(
    fn=synthesize,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter Darija text here...",
        label="Text"
    ),
    outputs=gr.Audio(type="filepath", label="Generated Speech"),
    title="Darija TTS with Qwen3-TTS",
    description="Generate Darija speech from text using a LoRA fine-tuned Qwen3-TTS custom voice model.",
    examples=[
        ["سلام، كيداير؟"],
        ["واش كلشي مزيان؟"],
        ["مرحبا بكم فهاد التجربة ديال تحويل النص إلى كلام"]
    ]
)

if __name__ == "__main__":
    demo.launch()
