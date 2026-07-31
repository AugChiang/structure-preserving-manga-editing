"""Download the 24 font files used by the synthetic text renderer."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


NOTO_SANS = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
    "Sans/OTF/Japanese"
)
NOTO_SERIF = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
    "Serif/OTF/Japanese"
)
GOOGLE_FONTS = "https://raw.githubusercontent.com/google/fonts/main"

SOURCES = {
    "DelaGothicOne-Regular.ttf": f"{GOOGLE_FONTS}/ofl/delagothicone/DelaGothicOne-Regular.ttf",
    "KleeOne-Regular.ttf": f"{GOOGLE_FONTS}/ofl/kleeone/KleeOne-Regular.ttf",
    "KleeOne-SemiBold.ttf": f"{GOOGLE_FONTS}/ofl/kleeone/KleeOne-SemiBold.ttf",
    "Kosugi-Regular.ttf": f"{GOOGLE_FONTS}/ap" + "ache/kosugi/Kosugi-Regular.ttf",
    "KosugiMaru-Regular.ttf": f"{GOOGLE_FONTS}/ap" + "ache/kosugimaru/KosugiMaru-Regular.ttf",
    "MochiyPopOne-Regular.ttf": f"{GOOGLE_FONTS}/ofl/mochiypopone/MochiyPopOne-Regular.ttf",
    "NewTegomin-Regular.ttf": f"{GOOGLE_FONTS}/ofl/newtegomin/NewTegomin-Regular.ttf",
    "NotoSansCJKjp-Black.otf": f"{NOTO_SANS}/NotoSansCJKjp-Black.otf",
    "NotoSansCJKjp-Bold.otf": f"{NOTO_SANS}/NotoSansCJKjp-Bold.otf",
    "NotoSansCJKjp-Regular.otf": f"{NOTO_SANS}/NotoSansCJKjp-Regular.otf",
    "NotoSansJP-Black.ttf": f"{NOTO_SANS}/NotoSansCJKjp-Black.otf",
    "NotoSansJP-Bold.ttf": f"{NOTO_SANS}/NotoSansCJKjp-Bold.otf",
    "NotoSansJP-Regular.ttf": f"{NOTO_SANS}/NotoSansCJKjp-Regular.otf",
    "NotoSerifCJKjp-Bold.otf": f"{NOTO_SERIF}/NotoSerifCJKjp-Bold.otf",
    "NotoSerifCJKjp-Regular.otf": f"{NOTO_SERIF}/NotoSerifCJKjp-Regular.otf",
    "NotoSerifJP-Bold.otf": f"{NOTO_SERIF}/NotoSerifCJKjp-Bold.otf",
    "NotoSerifJP-Regular.otf": f"{NOTO_SERIF}/NotoSerifCJKjp-Regular.otf",
    "PottaOne-Regular.ttf": f"{GOOGLE_FONTS}/ofl/pottaone/PottaOne-Regular.ttf",
    "ReggaeOne-Regular.ttf": f"{GOOGLE_FONTS}/ofl/reggaeone/ReggaeOne-Regular.ttf",
    "RocknRollOne-Regular.ttf": f"{GOOGLE_FONTS}/ofl/rocknrollone/RocknRollOne-Regular.ttf",
    "ShipporiMincho-Bold.ttf": f"{GOOGLE_FONTS}/ofl/shipporimincho/ShipporiMincho-Bold.ttf",
    "ShipporiMincho-Regular.ttf": f"{GOOGLE_FONTS}/ofl/shipporimincho/ShipporiMincho-Regular.ttf",
    "YujiBoku-Regular.ttf": f"{GOOGLE_FONTS}/ofl/yujiboku/YujiBoku-Regular.ttf",
    "YujiSyuku-Regular.ttf": f"{GOOGLE_FONTS}/ofl/yujisyuku/YujiSyuku-Regular.ttf",
}


def relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("paths must be relative")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=relative_path,
        default=Path("data/fonts"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, url in SOURCES.items():
        destination = args.output_dir / name
        if destination.is_file():
            print(f"Already present: {destination}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {name}")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)


if __name__ == "__main__":
    main()
