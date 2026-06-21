import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
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
TORCH_AUDIO_FIX_HINT = (
    "Run: uv pip install --force-reinstall torch torchaudio "
    "--index-url https://download.pytorch.org/whl/cu128"
)

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

QWEN_OUTPUT_MODES = {
    "text": "Text only, fastest",
    "aligned-srt": "Text + aligned SRT, slower",
}

SPOKEN_LANGUAGE_CHOICES = {
    "auto": {
        "label": "Auto / not sure",
        "qwen_hint": None,
    },
    "zh": {
        "label": "Chinese / Mandarin",
        "qwen_hint": "Chinese",
    },
    "en": {
        "label": "English",
        "qwen_hint": "English",
    },
    "zh-en": {
        "label": "Chinese + English mix",
        "qwen_hint": "Chinese and English",
    },
    "other": {
        "label": "Other language",
        "qwen_hint": None,
    },
}

COLOR_ENABLED = True

ANSI_CODES = {
    "reset": "0",
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
}


def set_color_enabled(no_color: bool) -> None:
    global COLOR_ENABLED
    COLOR_ENABLED = (
        not no_color
        and os.environ.get("NO_COLOR") is None
        and sys.stdout.isatty()
    )


def style(text: str, color: str | None = None, *, bold: bool = False, dim: bool = False) -> str:
    if not COLOR_ENABLED:
        return text

    codes = []
    if bold:
        codes.append(ANSI_CODES["bold"])
    if dim:
        codes.append(ANSI_CODES["dim"])
    if color:
        codes.append(ANSI_CODES[color])

    if not codes:
        return text

    return f"\033[{';'.join(codes)}m{text}\033[{ANSI_CODES['reset']}m"


def print_header(title: str) -> None:
    print()
    print(style(f"== {title} ==", "cyan", bold=True))


def print_option(index: int, label: str) -> None:
    print(f"  {style(str(index), 'green', bold=True)}. {label}")


def print_info(message: str) -> None:
    print(style("info: ", "blue", bold=True) + message)


def print_warning(message: str) -> None:
    print(style("warning: ", "yellow", bold=True) + message)


def print_success(message: str) -> None:
    print(style("done: ", "green", bold=True) + message)


def prompt_input(message: str) -> str:
    return input(style(f"? {message} ", "cyan", bold=True))


def print_summary_row(label: str, value) -> None:
    print(f"{style(label.ljust(18), 'cyan', bold=True)}: {value}")


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

    print_header(f"Input files in {input_dir}")
    for index, path in enumerate(files, start=1):
        print_option(index, path.name)

    while True:
        choice = prompt_input("Choose a file number, or q to quit:").strip()

        if choice.lower() in {"q", "quit", "exit"}:
            raise SystemExit(0)

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(files):
                return files[index - 1]

        print_warning(f"Please enter a number from 1 to {len(files)}.")


def choose_source() -> str:
    print_header("Choose transcription source")
    print_option(1, "YouTube URL")
    print_option(2, "Local file from input/")

    while True:
        choice = prompt_input("Choose 1 or 2, or q to quit:").strip().lower()

        if choice in {"q", "quit", "exit"}:
            raise SystemExit(0)

        if choice in {"1", "youtube", "yt"}:
            return "youtube"

        if choice in {"2", "input", "local"}:
            return "input"

        print_warning("Please enter 1 for YouTube or 2 for input folder.")


def prompt_youtube_url() -> str:
    while True:
        url = prompt_input("Paste YouTube URL, or q to quit:").strip()

        if url.lower() in {"q", "quit", "exit"}:
            raise SystemExit(0)

        if url:
            return url

        print_warning("Please paste a YouTube URL.")


def choose_transcription_model() -> str:
    choices = list(MODEL_CONFIGS.items())

    print_header("Choose transcription model")
    for index, (_, config) in enumerate(choices, start=1):
        print_option(index, config["label"])

    while True:
        choice = prompt_input("Choose a model number, or q to quit:").strip().lower()

        if choice in {"q", "quit", "exit"}:
            raise SystemExit(0)

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(choices):
                return choices[index - 1][0]

        if choice in MODEL_CONFIGS:
            return choice

        print_warning(f"Please enter a number from 1 to {len(choices)}.")


def choose_qwen_output_mode() -> str:
    choices = list(QWEN_OUTPUT_MODES.items())

    print_header("Choose Qwen output")
    for index, (_, label) in enumerate(choices, start=1):
        print_option(index, label)

    while True:
        choice = prompt_input("Choose output number, or q to quit:").strip().lower()

        if choice in {"q", "quit", "exit"}:
            raise SystemExit(0)

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(choices):
                return choices[index - 1][0]

        if choice in QWEN_OUTPUT_MODES:
            return choice

        print_warning(f"Please enter a number from 1 to {len(choices)}.")


def choose_spoken_language() -> tuple[str, str | None]:
    choices = list(SPOKEN_LANGUAGE_CHOICES.items())

    print_header("Choose spoken language")
    for index, (_, config) in enumerate(choices, start=1):
        print_option(index, config["label"])

    while True:
        choice = prompt_input("Choose language number, or q to quit:").strip().lower()

        if choice in {"q", "quit", "exit"}:
            raise SystemExit(0)

        language_key = None
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(choices):
                language_key = choices[index - 1][0]
        elif choice in SPOKEN_LANGUAGE_CHOICES:
            language_key = choice

        if language_key:
            if language_key == "other":
                language_name = prompt_input(
                    "Type the language name, or press Enter for Other:"
                ).strip()
                return language_key, language_name or None

            return language_key, None

        print_warning(f"Please enter a number from 1 to {len(choices)}.")


def spoken_language_label(language_key: str | None, language_name: str | None = None) -> str | None:
    if not language_key:
        return None

    if language_key == "other":
        return language_name or SPOKEN_LANGUAGE_CHOICES[language_key]["label"]

    return SPOKEN_LANGUAGE_CHOICES[language_key]["label"]


def qwen_language_hint(language_key: str | None, language_name: str | None = None) -> str | None:
    if not language_key:
        return None

    if language_key == "other":
        return language_name

    return SPOKEN_LANGUAGE_CHOICES[language_key]["qwen_hint"]


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


def resolve_ytdlp_command() -> list[str]:
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]

    ytdlp_path = shutil.which("yt-dlp")
    if ytdlp_path is None:
        raise SystemExit(f"yt-dlp was not found. {YTDLP_INSTALL_HINT}")

    return [ytdlp_path]


def check_ytdlp_update() -> list[str]:
    ytdlp_command = resolve_ytdlp_command()

    print_info("Checking yt-dlp for updates...")
    result = subprocess.run(
        [*ytdlp_command, "-U"],
        capture_output=True,
        text=True,
    )

    for output in (result.stdout.strip(), result.stderr.strip()):
        if output:
            print(output)

    if result.returncode != 0:
        print_warning("yt-dlp update check did not complete; continuing with installed yt-dlp.")

    return ytdlp_command


def download_youtube_media(url: str, temp_dir: Path) -> Path:
    ytdlp_command = check_ytdlp_update()

    output_template = temp_dir / "%(title).120B [%(id)s].%(ext)s"
    command = [
        *ytdlp_command,
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-o",
        str(output_template),
        url,
    ]

    print_info("Downloading YouTube media with yt-dlp...")

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
    print_info(f"Downloaded YouTube media: {downloaded_path.name}")
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

    print_info(f"Converting to temporary 16 kHz mono WAV: {input_path}")

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


def check_torch_audio_compat() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch was not found. "
            "Install torch and torchaudio from the same CUDA index."
        ) from exc

    try:
        import torchaudio  # noqa: F401
    except RuntimeError as exc:
        message = str(exc)
        if "different CUDA versions" in message:
            raise SystemExit(
                "PyTorch and TorchAudio were built for different CUDA versions.\n"
                f"Detected torch CUDA: {torch.version.cuda}\n"
                f"{TORCH_AUDIO_FIX_HINT}"
            ) from exc
        raise
    except ImportError as exc:
        raise SystemExit(
            "TorchAudio was not found. "
            f"{TORCH_AUDIO_FIX_HINT}"
        ) from exc


def load_funasr_speaker_model(device: str):
    check_torch_audio_compat()

    from funasr import AutoModel

    print_info("Loading FunASR speaker diarization model...")

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
    print_info(f"Transcribing temporary WAV with FunASR: {wav_path}")

    res = model.generate(
        input=str(wav_path),
        batch_size=1,
    )

    sentence_info = res[0].get("sentence_info", [])

    if not sentence_info:
        print_warning("No sentence_info found.")
        print(res)
        return []

    return sentence_info


def load_qwen_model(
    model_id: str,
    device: str,
    max_new_tokens: int,
    use_aligner: bool,
):
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise SystemExit(f"qwen-asr was not found. {QWEN_INSTALL_HINT}") from exc

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print_info(f"Loading Qwen3-ASR model: {model_id}")

    model_kwargs = {
        "dtype": dtype,
        "device_map": device,
        "max_inference_batch_size": 8,
        "max_new_tokens": max_new_tokens,
    }

    if use_aligner:
        print_info("Loading Qwen3 ForcedAligner for SRT timestamps...")
        model_kwargs["forced_aligner"] = "Qwen/Qwen3-ForcedAligner-0.6B"
        model_kwargs["forced_aligner_kwargs"] = {
            "dtype": dtype,
            "device_map": device,
        }

    return Qwen3ASRModel.from_pretrained(model_id, **model_kwargs)


def get_result_value(result, key: str):
    if isinstance(result, dict):
        return result.get(key)

    return getattr(result, key, None)


def timestamp_to_ms(value) -> int:
    return max(0, int(float(value) * 1000))


def extract_qwen_timestamp_units(result) -> list[dict]:
    time_stamps = get_result_value(result, "time_stamps") or []

    if not isinstance(time_stamps, list):
        return []

    if len(time_stamps) == 1 and isinstance(time_stamps[0], list):
        time_stamps = time_stamps[0]

    units = []
    for item in time_stamps:
        text = (get_result_value(item, "text") or "").strip()
        start_time = get_result_value(item, "start_time")
        end_time = get_result_value(item, "end_time")

        if not text or start_time is None or end_time is None:
            continue

        units.append(
            {
                "text": text,
                "start_ms": timestamp_to_ms(start_time),
                "end_ms": timestamp_to_ms(end_time),
            }
        )

    return units


def join_subtitle_text(current_text: str, next_text: str) -> str:
    if not current_text:
        return next_text

    if current_text[-1].isascii() and next_text[:1].isascii():
        return f"{current_text} {next_text}"

    return f"{current_text}{next_text}"


def build_qwen_srt_segments(units: list[dict], max_chars: int = 42) -> list[dict]:
    segments = []
    current_text = ""
    current_start = None
    current_end = None

    for unit in units:
        unit_text = unit["text"]
        candidate_text = join_subtitle_text(current_text, unit_text)

        if current_text and len(candidate_text) > max_chars:
            segments.append(
                {
                    "text": current_text,
                    "start_ms": current_start,
                    "end_ms": current_end,
                }
            )
            current_text = unit_text
            current_start = unit["start_ms"]
            current_end = unit["end_ms"]
            continue

        current_text = candidate_text
        current_start = unit["start_ms"] if current_start is None else current_start
        current_end = unit["end_ms"]

    if current_text:
        segments.append(
            {
                "text": current_text,
                "start_ms": current_start,
                "end_ms": current_end,
            }
        )

    return segments


def transcribe_qwen(
    model,
    wav_path: Path,
    language: str | None,
    return_time_stamps: bool,
) -> tuple[str, str | None, list[dict]]:
    print_info(f"Transcribing temporary WAV with Qwen3-ASR: {wav_path}")

    results = model.transcribe(
        audio=str(wav_path),
        language=language,
        return_time_stamps=return_time_stamps,
    )

    result = results[0] if isinstance(results, list) else results
    text = (get_result_value(result, "text") or "").strip()
    detected_language = get_result_value(result, "language")
    timestamp_units = extract_qwen_timestamp_units(result) if return_time_stamps else []

    if not text:
        print_warning("No transcript text found.")
        print(results)

    return text, detected_language, timestamp_units


def write_speaker_outputs(
    sentence_info: list[dict],
    outdir: Path,
    base: str,
    spoken_language: str | None,
) -> tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    txt_ts_path = outdir / f"{base}.txt"
    txt_plain_path = outdir / f"{base}.plain.txt"
    srt_path = outdir / f"{base}.srt"

    with open(txt_ts_path, "w", encoding="utf-8") as f:
        if spoken_language:
            f.write(f"Spoken language: {spoken_language}\n\n")

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
    output_mode: str,
    timestamp_units: list[dict],
) -> tuple[Path, Path, Path | None]:
    outdir.mkdir(parents=True, exist_ok=True)

    txt_ts_path = outdir / f"{base}.txt"
    txt_plain_path = outdir / f"{base}.plain.txt"
    srt_path = outdir / f"{base}.srt" if output_mode == "aligned-srt" else None
    srt_segments = build_qwen_srt_segments(timestamp_units)

    with open(txt_ts_path, "w", encoding="utf-8") as f:
        if language:
            f.write(f"Language: {language}\n")

        if srt_segments:
            for segment in srt_segments:
                f.write(
                    f"[{fmt_time_ms(segment['start_ms'])} - "
                    f"{fmt_time_ms(segment['end_ms'])}] {segment['text']}\n"
                )
        else:
            f.write(f"[{fmt_time_ms(0)} - {fmt_time_ms(duration_ms)}] {text}\n")

    with open(txt_plain_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

    if srt_path:
        if not srt_segments:
            print_warning(
                "No usable Qwen timestamps found; writing one full-duration SRT cue."
            )
            srt_segments = [
                {
                    "text": text,
                    "start_ms": 0,
                    "end_ms": duration_ms,
                }
            ]

        with open(srt_path, "w", encoding="utf-8") as f:
            for index, segment in enumerate(srt_segments, start=1):
                f.write(f"{index}\n")
                f.write(
                    f"{fmt_srt_time(segment['start_ms'])} --> "
                    f"{fmt_srt_time(segment['end_ms'])}\n"
                )
                f.write(f"{wrap_srt_text(segment['text'])}\n\n")

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
        "--spoken-language",
        choices=list(SPOKEN_LANGUAGE_CHOICES),
        help=(
            "Language spoken in the media. Used as output metadata for FunASR "
            "and as a Qwen hint when --qwen-language is not set."
        ),
    )
    parser.add_argument(
        "--spoken-language-name",
        help='Custom language name when --spoken-language other, for example "Malay".',
    )
    parser.add_argument(
        "--qwen-output",
        choices=list(QWEN_OUTPUT_MODES),
        help="Qwen output mode. If omitted, choose interactively.",
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
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored CLI output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_color_enabled(args.no_color)

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    source_type, source_value, source_label = select_media_source(args, input_dir)
    model_choice = args.model or choose_transcription_model()
    model_config = MODEL_CONFIGS[model_choice]
    qwen_output_mode = None
    spoken_language_key = args.spoken_language
    spoken_language_name = args.spoken_language_name

    if model_config["kind"] == "qwen":
        qwen_output_mode = args.qwen_output or choose_qwen_output_mode()

    needs_spoken_language = (
        model_config["kind"] == "funasr-speaker"
        or (model_config["kind"] == "qwen" and not args.qwen_language)
    )
    if not spoken_language_key and needs_spoken_language and sys.stdin.isatty():
        spoken_language_key, spoken_language_name = choose_spoken_language()

    spoken_language = spoken_language_label(
        spoken_language_key,
        spoken_language_name,
    )
    qwen_language = args.qwen_language or qwen_language_hint(
        spoken_language_key,
        spoken_language_name,
    )

    if model_config["kind"] == "funasr-speaker" and spoken_language_key == "other":
        print_warning(
            "FunASR speaker mode uses paraformer-zh, which is best for Chinese, "
            "English, and Chinese-English mixed speech. For other languages, a "
            "Qwen model may transcribe better but will not add speaker labels."
        )

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
                spoken_language,
            )
        elif model_config["kind"] == "qwen":
            use_aligner = qwen_output_mode == "aligned-srt"
            model = load_qwen_model(
                model_config["model_id"],
                args.device,
                args.qwen_max_new_tokens,
                use_aligner,
            )
            text, language, timestamp_units = transcribe_qwen(
                model,
                wav_path,
                qwen_language,
                use_aligner,
            )

            if not text:
                return

            txt_ts_path, txt_plain_path, srt_path = write_qwen_outputs(
                text,
                language or spoken_language,
                get_duration_ms(wav_path),
                outdir,
                output_base,
                qwen_output_mode,
                timestamp_units,
            )

    print()
    print_success("Transcription complete.")
    print_summary_row("Source", source_label)
    print_summary_row("Model", model_config["label"])
    if spoken_language:
        print_summary_row("Spoken language", spoken_language)
    if qwen_language:
        print_summary_row("Qwen language hint", qwen_language)
    if qwen_output_mode:
        print_summary_row("Qwen output", QWEN_OUTPUT_MODES[qwen_output_mode])
    print_summary_row("TXT with timestamp", txt_ts_path)
    print_summary_row("TXT plain", txt_plain_path)
    if srt_path:
        print_summary_row("SRT", srt_path)
    else:
        print_summary_row("SRT", "skipped")


if __name__ == "__main__":
    main()
