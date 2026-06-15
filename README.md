# FunASR Speaker-Split Chinese Transcription

Local Chinese ASR transcription pipeline using:

* FunASR
* Paraformer Chinese ASR
* FSMN-VAD speech segmentation
* CT-Punc punctuation
* CAM++ speaker diarization
* Optional Qwen3-ASR single-speaker transcription
* Optional YouTube downloads through yt-dlp
* NVIDIA CUDA GPU through WSL2

This setup is intended for transcription of local interview audio/video files or YouTube URLs.

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

## 5. Install Dependencies

Install the Python packages:

```bash
uv pip install numpy
uv pip install -U funasr modelscope huggingface_hub soundfile "yt-dlp[default]"
uv pip install -U qwen-asr
```

After installing Qwen, force `torch` and `torchaudio` back onto the same CUDA wheel set. This avoids the common mismatch where Qwen-related installs upgrade `torch` but leave `torchaudio` on another CUDA build:

```bash
uv pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
uv run python - <<'PY'
import torch
import torchaudio
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torchaudio:", torchaudio.__version__)
PY
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

## 7. Run Transcription

```bash
uv run python transcribe_speaker.py
```

The script first asks where the source should come from:

```text
Choose transcription source:
  1. YouTube URL
  2. Local file from input/
```

For YouTube, paste the URL. The script checks `yt-dlp` for updates, downloads the media into a temporary folder, converts it to a temporary 16 kHz mono WAV with `ffmpeg`, transcribes it, and writes the outputs into `output/`.

For local files, the script lists supported audio/video files in `input/`, asks which one to use, converts it to a temporary 16 kHz mono WAV with `ffmpeg`, transcribes it, and writes the outputs into `output/`.

Then choose the transcription model:

```text
Choose transcription model:
  1. FunASR Paraformer + CAM++ speaker diarization
  2. Qwen3-ASR 0.6B, single-speaker/no diarization
  3. Qwen3-ASR 1.7B, single-speaker/no diarization
```

If you choose a Qwen model, choose whether to generate text only or also run the aligner for SRT timestamps:

```text
Choose Qwen output:
  1. Text only, fastest
  2. Text + aligned SRT, slower
```

You can also pass a file directly:

```bash
uv run python transcribe_speaker.py input/interview.mp4
```

Or pass a YouTube URL directly:

```bash
uv run python transcribe_speaker.py --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID"
```

Or choose a model directly:

```bash
uv run python transcribe_speaker.py input/talk.mp4 --model qwen3-asr-0.6b
uv run python transcribe_speaker.py input/interview.mp4 --model funasr-speaker
uv run python transcribe_speaker.py --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" --model qwen3-asr-1.7b
```

For Qwen, choose the output mode directly:

```bash
uv run python transcribe_speaker.py input/talk.mp4 --model qwen3-asr-0.6b --qwen-output text
uv run python transcribe_speaker.py input/talk.mp4 --model qwen3-asr-1.7b --qwen-output aligned-srt
```

If your `yt-dlp` install method does not support self-update, update it manually:

```bash
uv pip install -U "yt-dlp[default]"
```

The first run downloads models from Hugging Face. Future runs should reuse cached models.

Outputs:

```text
output/interview.txt
output/interview.plain.txt
output/interview.srt
```

---

## 8. Output Formats

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

## 9. Notes

Use `funasr-speaker` for interviews or multi-person recordings. It outputs speaker labels and timestamped speaker turns.

Use `qwen3-asr-0.6b` or `qwen3-asr-1.7b` for one-speaker recordings or videos where diarization is not needed. `--qwen-output text` writes only text outputs and skips the aligner. `--qwen-output aligned-srt` loads `Qwen/Qwen3-ForcedAligner-0.6B` and writes an SRT using returned timestamp units.

Qwen3-ASR is not a diarization model. Using it for diarized multi-speaker output would require a separate speaker diarization stage, splitting/merging audio segments, and reconciling timestamps. For this project, `funasr-speaker` is the practical diarized path.

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

Fun-ASR-Nano-2512 is not exposed as a diarization option here. Its current model card lists speaker diarization and timestamps as TODO items, and the observed `<|no|>` crash happens inside Nano's tokenizer/forced-alignment path rather than in this wrapper script. Use the stable `funasr-speaker` mode for diarized output for now.

---

## 10. Clean Up Disk Space

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

## 11. Files That Should Be Committed

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
