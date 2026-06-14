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


def load_model(device: str):
    from funasr import AutoModel

    print("Loading FunASR model...")

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


def transcribe(model, wav_path: Path) -> list[dict]:
    print(f"Transcribing temporary WAV: {wav_path}")

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


def write_outputs(sentence_info: list[dict], outdir: Path, base: str) -> tuple[Path, Path, Path]:
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

            f.write(f"[{fmt_time_ms(start)} - {fmt_time_ms(end)}] Speaker {spk}: {text}\n")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an input audio/video file with speaker labels."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        help="Optional audio/video path. If omitted, choose from the input folder.",
    )
    parser.add_argument("--input-dir", default="input", help="Folder to list in interactive mode")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="FunASR device, for example cuda:0 or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    input_path = resolve_input_file(args.audio, input_dir)
    output_base = input_path.stem

    with tempfile.TemporaryDirectory(prefix="funasr-transcribe-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        wav_path = convert_to_16k_wav(input_path, temp_dir)
        model = load_model(args.device)
        sentence_info = transcribe(model, wav_path)

    if not sentence_info:
        return

    txt_ts_path, txt_plain_path, srt_path = write_outputs(sentence_info, outdir, output_base)

    print("Done.")
    print(f"Source             : {input_path}")
    print(f"TXT with timestamp : {txt_ts_path}")
    print(f"TXT plain          : {txt_plain_path}")
    print(f"SRT                : {srt_path}")


if __name__ == "__main__":
    main()
