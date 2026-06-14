from funasr import AutoModel
from pathlib import Path
import argparse


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--outdir", default="output", help="Output directory")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = audio_path.stem

    print("Loading FunASR model...")

    model = AutoModel(
        model="funasr/paraformer-zh",
        hub="hf",
        disable_update=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        punc_model="ct-punc",
        spk_model="cam++",
        device="cuda:0",
    )

    print(f"Transcribing: {audio_path}")

    res = model.generate(
        input=str(audio_path),
        batch_size=1,
    )

    sentence_info = res[0].get("sentence_info", [])

    if not sentence_info:
        print("No sentence_info found.")
        print(res)
        return

    txt_ts_path = outdir / f"{base}.txt"
    txt_plain_path = outdir / f"{base}.plain.txt"
    srt_path = outdir / f"{base}.srt"

    # 1. TXT with timestamp
    with open(txt_ts_path, "w", encoding="utf-8") as f:
        for s in sentence_info:
            start = int(s.get("start", 0))
            end = int(s.get("end", 0))
            spk = s.get("spk", "unknown")
            text = get_text(s)

            if not text:
                continue

            f.write(f"[{fmt_time_ms(start)} - {fmt_time_ms(end)}] Speaker {spk}: {text}\n")

    # 2. TXT without timestamp, grouped by speaker turns
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

    # 3. SRT subtitle
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

    print("Done.")
    print(f"TXT with timestamp : {txt_ts_path}")
    print(f"TXT plain          : {txt_plain_path}")
    print(f"SRT                : {srt_path}")


if __name__ == "__main__":
    main()