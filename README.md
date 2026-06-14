# FunASR Speaker-Split Chinese Transcription

Local Chinese ASR transcription pipeline using:

* FunASR
* Paraformer Chinese ASR
* FSMN-VAD speech segmentation
* CT-Punc punctuation
* CAM++ speaker diarization
* NVIDIA CUDA GPU through WSL2

This setup is intended for offline transcription of downloaded interview audio/video files.

It outputs:

```text
output/file.txt        # timestamp + speaker + transcript
output/file.plain.txt  # speaker + transcript only
output/file.srt        # subtitle format
```

Large audio/video/model/cache files are intentionally ignored by Git.

---

## Tested Environment

```text
OS: WSL2 Ubuntu
GPU: NVIDIA RTX 4080 Super
Python: 3.11
Package manager: uv
PyTorch: CUDA 12.8 wheel
FunASR: 1.3.9+
```

---

## 1. Install System Dependencies

```bash
sudo apt update
sudo apt install -y ffmpeg git curl python3 python3-venv python3-pip
```

---

## 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

Check:

```bash
uv --version
```

---

## 3. Create Project Environment

From the project folder:

```bash
cd ~/asr/funasr

uv venv --python 3.11
source .venv/bin/activate
```

---

## 4. Install PyTorch CUDA

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:

```bash
uv run python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Expected:

```text
cuda: True
gpu: NVIDIA GeForce RTX 4080 SUPER
```

---

## 5. Install FunASR Dependencies

```bash
uv pip install numpy
uv pip install -U funasr modelscope huggingface_hub soundfile ffmpeg-python
```

Optional extras if needed:

```bash
uv pip install librosa scipy kaldiio
```

---

## 6. Prepare Input and Output Folders

```bash
mkdir -p input output
```

Put your source file inside:

```text
input/
```

Example:

```text
input/interview.wav
input/interview.mp4
input/interview.m4a
```

---

## 7. Convert Audio to 16 kHz Mono WAV

For best diarization stability, normalize the audio first.

For audio files:

```bash
ffmpeg -y -i input/interview.wav -ac 1 -ar 16000 output/interview_16k.wav
```

For video files:

```bash
ffmpeg -y -i input/interview.mp4 -ac 1 -ar 16000 output/interview_16k.wav
```

Check audio format:

```bash
ffprobe -v error -select_streams a:0 \
  -show_entries stream=codec_name,sample_rate,channels \
  -of default=noprint_wrappers=1 output/interview_16k.wav
```

Expected:

```text
codec_name=pcm_s16le
sample_rate=16000
channels=1
```

---

## 8. Run Transcription

```bash
uv run python transcribe_speaker.py output/interview_16k.wav
```

The first run downloads models from Hugging Face. Future runs should reuse cached models.

Outputs:

```text
output/interview_16k.txt
output/interview_16k.plain.txt
output/interview_16k.srt
```

---

## 9. Output Formats

### Timestamp TXT

```text
[00:00:01.200 - 00:00:04.850] Speaker 1: 今天我们来聊一下这个问题。
[00:00:05.100 - 00:00:09.320] Speaker 2: 对，我觉得这个背景其实很重要。
```

### Plain TXT

```text
Speaker 1: 今天我们来聊一下这个问题。

Speaker 2: 对，我觉得这个背景其实很重要。
```

### SRT

```text
1
00:00:01,200 --> 00:00:04,850
Speaker 1: 今天我们来聊一下这个问题。
```

---

## 10. Notes

Speaker labels are numeric only:

```text
Speaker 1
Speaker 2
Speaker 3
```

They are not real names. Manually map them after checking the transcript:

```text
Speaker 1 = 主持人
Speaker 2 = 嘉宾
```

For Chinese interviews, best results usually come from:

* clean mono audio
* minimal background noise
* limited speaker overlap
* two main speakers

---

## 11. Clean Up Disk Space

Delete local project files:

```bash
rm -rf ~/asr/funasr
```

This does not delete model caches.

Check cache size:

```bash
du -sh ~/.cache/huggingface ~/.cache/modelscope ~/.cache/uv 2>/dev/null
```

Delete model/package caches only if you are sure you no longer need them:

```bash
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/modelscope
uv cache clean
```

---

## 12. Files That Should Be Committed

Commit only source/config/docs:

```bash
git add transcribe_speaker.py README.md .gitignore
git status
git commit -m "Add FunASR speaker transcription tool"
```

Do not commit:

```text
.venv/
input/
output/
audio files
video files
model files
cache folders
transcript outputs
```
