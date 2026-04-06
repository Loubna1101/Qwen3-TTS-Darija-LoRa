# Qwen3-TTS-Darija-LoRa

A Darija Moroccan Arabic text-to-speech project based on **Qwen3-TTS-12Hz-1.7B-Base**, fine-tuned with **LoRA** on atlasia/DODa-audio-dataset F3.

## Authors
- loubna haouach
- chaimae haddouche

## Overview
This project fine-tunes Qwen3-TTS on a Darija dataset and provides inference through a simple Gradio interface.

## Base Model
- [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)

## Fine-tuning Method
- LoRA fine-tuning
- Single-speaker custom voice
- 24kHz audio
- Speaker name used during training: `darija_speaker`

## Dataset Preparation
The training pipeline:
1. Resamples audio to 24kHz
2. Builds `train_raw_24k.jsonl`
3. Extracts codec tokens with `prepare_data.py`
4. Fine-tunes Qwen3-TTS with LoRA

## Training Setup
- Batch size: `1`
- Learning rate: `5e-5`
- Epochs: `10`

## Repository Contents
- `finetuning.ipynb`: training workflow
- `inference.ipynb`: inference workflow
- `app.py`: Gradio demo
- `requirements.txt`: dependencies

## Model Files
The trained LoRA adapter and custom speaker embedding are hosted on Hugging Face.

## Inference
The model is loaded as:
- base model from Qwen
- LoRA adapter from this project
- custom speaker embedding from this project

## Disclaimer
This project is for research and educational purposes. Please ensure you have the right to use and publish the voice data used for fine-tuning.

## Credits
- Qwen team for Qwen3-TTS
- atlasIA for the Doda dataset
- Hugging Face ecosystem
