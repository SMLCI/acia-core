"""Print the CHANGELOG.md section for a given version (used as release notes).

Usage: python .github/scripts/changelog_section.py <version>
Reads CHANGELOG.md from the repo root and prints the body of the
``## [<version>] - ...`` section (everything up to the next ``## `` heading).
Falls back to a generic line if the section is not found.
"""

import re
import sys
from pathlib import Path


def main() -> int:
    version = sys.argv[1].lstrip("v") if len(sys.argv) > 1 else ""
    changelog = Path("CHANGELOG.md")
    if not changelog.is_file() or not version:
        print(f"Release {version}".strip())
        return 0

    heading = re.compile(r"^## ")
    capturing = False
    body: list[str] = []
    for line in changelog.read_text(encoding="utf-8").splitlines():
        if heading.match(line):
            if capturing:
                break
            capturing = f"[{version}]" in line
            continue
        if capturing:
            body.append(line)

    text = "\n".join(body).strip()
    print(text if text else f"Release {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
