import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


MEDIA_EXTENSIONS = {
    ".aac",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}

YTDLP_INSTALL_HINT = 'Install it with: uv pip install -U "yt-dlp[default]"'
QWEN_INSTALL_HINT = "Install it with: uv pip install -U qwen-asr"

MODEL_CONFIGS = {
    "funasr-speaker": {
        "label": "FunASR Paraformer + CAM++ speaker diarization",
        "kind": "funasr-speaker",
    },
    "qwen3-asr-0.6b": {
        "label": "Qwen3-ASR 0.6B, single-speaker/no diarization",
        "kind": "qwen",
        "model_id": "Qwen/Qwen3-ASR-0.6B",
    },
    "qwen3-asr-1.7b": {
        "label": "Qwen3-ASR 1.7B, single-speaker/no diarization",
        "kind": "qwen",
        "model_id": "Qwen/Qwen3-ASR-1.7B",
    },
}


def fmt_time_ms(ms: int) -> str:
    total_seconds = ms / 1000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    millis = int(ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def fmt_srt_time(ms: int) -> str:
    return fmt_time_ms(ms).replace(".", ",")


def get_text(segment: dict) -> str:
    return (
        segment.get("text")
        or segment.get("sentence")
        or segment.get("value")
        or ""
    ).strip()


def find_media_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    )


def choose_input_file(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    files = find_media_files(input_dir)

    if not files:
        raise SystemExit(
            f"No audio/video files found in {input_dir}. "
            "Put an mp4, m4a, wav, mp3, or other supported media file there first."
        )

    print(f"Input files in {input_dir}:")
    for index, path in enumerate(files, start=1):
        print(f"  {index}. {path.name}")

    while True:
        choice = input("Choose a file number, or q to quit: ").strip()

        if choice.lower() in {"q", "quit", "exit"}:
            raise SystemExit(0)

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(files):
                return files[index - 1]

        print(f"Please enter a number from 1 to {len(files)}.")


def choose_source() -> str:
    print("Choose transcription source:")
    print("  1. YouTube URL")
    print("  2. Local file from input/")

    while True:
        choice = input("Choose 1 or 2, or q to quit: ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            raise SystemExit(0)

        if choice in {"1", "youtube", "yt"}:
            return "youtube"

        if choice in {"2", "input", "local"}:
            return "input"

        print("Please enter 1 for YouTube or 2 for input folder.")


def prompt_youtube_url() -> str:
    while True:
        url = input("Paste YouTube URL, or q to quit: ").strip()

        if url.lower() in {"q", "quit", "exit"}:
            raise SystemExit(0)

        if url:
            return url

        print("Please paste a YouTube URL.")


def choose_transcription_model() -> str:
    choices = list(MODEL_CONFIGS.items())

    print("Choose transcription model:")
    for index, (_, config) in enumerate(choices, start=1):
        print(f"  {index}. {config['label']}")

    while True:
        choice = input("Choose a model number, or q to quit: ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            raise SystemExit(0)

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(choices):
                return choices[index - 1][0]

        if choice in MODEL_CONFIGS:
            return choice

        print(f"Please enter a number from 1 to {len(choices)}.")


def resolve_input_file(audio_arg: str | None, input_dir: Path) -> Path:
    if not audio_arg:
        return choose_input_file(input_dir)

    audio_path = Path(audio_arg)
    if audio_path.exists():
        return audio_path

    input_path = input_dir / audio_arg
    if input_path.exists():
        return input_path

    raise SystemExit(f"Input file not found: {audio_arg}")


def check_ytdlp_update() -> None:
    if shutil.which("yt-dlp") is None:
        raise SystemExit(f"yt-dlp was not found. {YTDLP_INSTALL_HINT}")

    print("Checking yt-dlp for updates...")
    result = subprocess.run(
        ["yt-dlp", "-U"],
        capture_output=True,
        text=True,
    )

    for output in (result.stdout.strip(), result.stderr.strip()):
        if output:
            print(output)

    if result.returncode != 0:
        print("yt-dlp update check did not complete; continuing with installed yt-dlp.")


def download_youtube_media(url: str, temp_dir: Path) -> Path:
    check_ytdlp_update()

    output_template = temp_dir / "%(title).120B [%(id)s].%(ext)s"
    command = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-o",
        str(output_template),
        url,
    ]

    print("Downloading YouTube media with yt-dlp...")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit("yt-dlp failed while downloading the YouTube URL.") from exc

    candidates = [
        path
        for path in temp_dir.iterdir()
        if path.is_file() and path.suffix.lower() not in {".part", ".ytdl", ".temp"}
    ]

    if not candidates:
        raise SystemExit("yt-dlp finished, but no downloaded media file was found.")

    downloaded_path = max(candidates, key=lambda path: path.stat().st_mtime)
    print(f"Downloaded YouTube media: {downloaded_path.name}")
    return downloaded_path


def select_media_source(
    args: argparse.Namespace,
    input_dir: Path,
) -> tuple[str, Path | str, str]:
    if args.audio and args.youtube_url:
        raise SystemExit("Use either a local audio/video file or --youtube-url, not both.")

    if args.youtube_url:
        return "youtube", args.youtube_url, f"YouTube: {args.youtube_url}"

    if args.audio:
        input_path = resolve_input_file(args.audio, input_dir)
        return "local", input_path, str(input_path)

    source = choose_source()

    if source == "youtube":
        url = prompt_youtube_url()
        return "youtube", url, f"YouTube: {url}"

    input_path = choose_input_file(input_dir)
    return "local", input_path, str(input_path)


def materialize_media_source(source_type: str, source_value: Path | str, temp_dir: Path) -> Path:
    if source_type == "youtube":
        return download_youtube_media(str(source_value), temp_dir)

    return Path(source_value)


def convert_to_16k_wav(input_path: Path, temp_dir: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found. Install ffmpeg and try again.")

    wav_path = temp_dir / f"{input_path.stem}.16k.wav"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]

    print(f"Converting to temporary 16 kHz mono WAV: {input_path}")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffmpeg failed while converting {input_path}") from exc

    return wav_path


def get_duration_ms(media_path: Path) -> int:
    if shutil.which("ffprobe") is None:
        return 1000

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 1000

    return max(1000, int(duration * 1000))


def load_funasr_speaker_model(device: str):
    from funasr import AutoModel

    print("Loading FunASR speaker diarization model...")

    return AutoModel(
        model="funasr/paraformer-zh",
        hub="hf",
        disable_update=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        punc_model="ct-punc",
        spk_model="cam++",
        device=device,
    )


def transcribe_funasr(model, wav_path: Path) -> list[dict]:
    print(f"Transcribing temporary WAV with FunASR: {wav_path}")

    res = model.generate(
        input=str(wav_path),
        batch_size=1,
    )

    sentence_info = res[0].get("sentence_info", [])

    if not sentence_info:
        print("No sentence_info found.")
        print(res)
        return []

    return sentence_info


def load_qwen_model(model_id: str, device: str, max_new_tokens: int):
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise SystemExit(f"qwen-asr was not found. {QWEN_INSTALL_HINT}") from exc

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print(f"Loading Qwen3-ASR model: {model_id}")

    return Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=8,
        max_new_tokens=max_new_tokens,
    )


def get_result_value(result, key: str):
    if isinstance(result, dict):
        return result.get(key)

    return getattr(result, key, None)


def transcribe_qwen(model, wav_path: Path, language: str | None) -> tuple[str, str | None]:
    print(f"Transcribing temporary WAV with Qwen3-ASR: {wav_path}")

    results = model.transcribe(
        audio=str(wav_path),
        language=language,
    )

    result = results[0] if isinstance(results, list) else results
    text = (get_result_value(result, "text") or "").strip()
    detected_language = get_result_value(result, "language")

    if not text:
        print("No transcript text found.")
        print(results)

    return text, detected_language


def write_speaker_outputs(
    sentence_info: list[dict],
    outdir: Path,
    base: str,
) -> tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    txt_ts_path = outdir / f"{base}.txt"
    txt_plain_path = outdir / f"{base}.plain.txt"
    srt_path = outdir / f"{base}.srt"

    with open(txt_ts_path, "w", encoding="utf-8") as f:
        for s in sentence_info:
            start = int(s.get("start", 0))
            end = int(s.get("end", 0))
            spk = s.get("spk", "unknown")
            text = get_text(s)

            if not text:
                continue

            f.write(
                f"[{fmt_time_ms(start)} - {fmt_time_ms(end)}] "
                f"Speaker {spk}: {text}\n"
            )

    with open(txt_plain_path, "w", encoding="utf-8") as f:
        last_spk = None

        for s in sentence_info:
            spk = s.get("spk", "unknown")
            text = get_text(s)

            if not text:
                continue

            if spk != last_spk:
                if last_spk is not None:
                    f.write("\n\n")
                f.write(f"Speaker {spk}: {text}")
                last_spk = spk
            else:
                f.write(text)

        f.write("\n")

    with open(srt_path, "w", encoding="utf-8") as f:
        index = 1

        for s in sentence_info:
            start = int(s.get("start", 0))
            end = int(s.get("end", 0))
            spk = s.get("spk", "unknown")
            text = get_text(s)

            if not text:
                continue

            f.write(f"{index}\n")
            f.write(f"{fmt_srt_time(start)} --> {fmt_srt_time(end)}\n")
            f.write(f"Speaker {spk}: {text}\n\n")

            index += 1

    return txt_ts_path, txt_plain_path, srt_path


def wrap_srt_text(text: str, width: int = 42) -> str:
    clean_text = " ".join(text.split())
    if not clean_text:
        return ""

    return "\n".join(
        clean_text[index : index + width]
        for index in range(0, len(clean_text), width)
    )


def write_qwen_outputs(
    text: str,
    language: str | None,
    duration_ms: int,
    outdir: Path,
    base: str,
) -> tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    txt_ts_path = outdir / f"{base}.txt"
    txt_plain_path = outdir / f"{base}.plain.txt"
    srt_path = outdir / f"{base}.srt"

    with open(txt_ts_path, "w", encoding="utf-8") as f:
        if language:
            f.write(f"Language: {language}\n")
        f.write(f"[{fmt_time_ms(0)} - {fmt_time_ms(duration_ms)}] {text}\n")

    with open(txt_plain_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("1\n")
        f.write(f"{fmt_srt_time(0)} --> {fmt_srt_time(duration_ms)}\n")
        f.write(f"{wrap_srt_text(text)}\n")

    return txt_ts_path, txt_plain_path, srt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an input audio/video file with speaker labels."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        help="Optional local audio/video path.",
    )
    parser.add_argument("--youtube-url", help="Download a YouTube URL with yt-dlp")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS),
        help="Transcription model. If omitted, choose interactively.",
    )
    parser.add_argument(
        "--qwen-language",
        help='Optional Qwen language hint, for example "Chinese" or "English".',
    )
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens for Qwen3-ASR generation.",
    )
    parser.add_argument(
        "--input-dir",
        default="input",
        help="Folder to list in interactive mode",
    )
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="FunASR device, for example cuda:0 or cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    source_type, source_value, source_label = select_media_source(args, input_dir)
    model_choice = args.model or choose_transcription_model()
    model_config = MODEL_CONFIGS[model_choice]

    with tempfile.TemporaryDirectory(prefix="funasr-transcribe-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        media_path = materialize_media_source(source_type, source_value, temp_dir)
        output_base = media_path.stem
        wav_path = convert_to_16k_wav(media_path, temp_dir)

        if model_config["kind"] == "funasr-speaker":
            model = load_funasr_speaker_model(args.device)
            sentence_info = transcribe_funasr(model, wav_path)

            if not sentence_info:
                return

            txt_ts_path, txt_plain_path, srt_path = write_speaker_outputs(
                sentence_info,
                outdir,
                output_base,
            )
        elif model_config["kind"] == "qwen":
            model = load_qwen_model(
                model_config["model_id"],
                args.device,
                args.qwen_max_new_tokens,
            )
            text, language = transcribe_qwen(model, wav_path, args.qwen_language)

            if not text:
                return

            txt_ts_path, txt_plain_path, srt_path = write_qwen_outputs(
                text,
                language,
                get_duration_ms(wav_path),
                outdir,
                output_base,
            )

    print("Done.")
    print(f"Source             : {source_label}")
    print(f"Model              : {model_config['label']}")
    print(f"TXT with timestamp : {txt_ts_path}")
    print(f"TXT plain          : {txt_plain_path}")
    print(f"SRT                : {srt_path}")


if __name__ == "__main__":
    main()
