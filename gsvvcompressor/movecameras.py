"""复制 cameras.json 和 cfg_args 文件从输入路径到输出路径"""

import shutil
from pathlib import Path
import argparse


def copy_configs(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["cameras.json", "cfg_args"]:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
            print(f"{src / name} -> {dst / name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--first-frame-in")
    p.add_argument("--first-frame-out")
    p.add_argument("--subsequent-format-in")
    p.add_argument("--subsequent-format-out")
    p.add_argument("--max-frames", type=int)
    args = p.parse_args()

    if args.first_frame_in:
        copy_configs(Path(args.first_frame_in), Path(args.first_frame_out))

    if args.subsequent_format_in:
        for i in range(2, args.max_frames + 1):
            copy_configs(
                Path(args.subsequent_format_in.format(i)),
                Path(args.subsequent_format_out.format(i)),
            )
