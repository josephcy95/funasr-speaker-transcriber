from pathlib import Path
import json
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


parser = argparse.ArgumentParser()
parser.add_argument("json_file")
args = parser.parse_args()

json_path = Path(args.json_file)
outdir = json_path.parent
base = json_path.stem

with open(json_path, "r", encoding="utf-8") as f:
    res = json.load(f)

sentence_info = res[0].get("sentence_info", [])

txt_ts_path = outdir / f"{base}.txt"
txt_plain_path = outdir / f"{base}.plain.txt"
srt_path = outdir / f"{base}.srt"

with open(txt_ts_path, "w", encoding="utf-8") as f:
    for s in sentence_info:
        start = int(s.get("start", 0))
        end = int(s.get("end", 0))
        spk = s.get("spk", "unknown")
        text = (s.get("text") or s.get("sentence") or "").strip()
        if text:
            f.write(f"[{fmt_time_ms(start)} - {fmt_time_ms(end)}] Speaker {spk}: {text}\n")

with open(txt_plain_path, "w", encoding="utf-8") as f:
    last_spk = None

    for s in sentence_info:
        spk = s.get("spk", "unknown")
        text = (s.get("text") or s.get("sentence") or "").strip()

        if not text:
            continue

        if spk != last_spk:
            if last_spk is not None:
                f.write("\n")
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
        text = (s.get("text") or s.get("sentence") or "").strip()

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