#!/usr/bin/env python3

import os


def main() -> None:
    base_url = os.environ.get("OPENAI_BASE_URL", "<unset>")
    print(f"paper: OPENAI_BASE_URL={base_url}")


if __name__ == "__main__":
    main()
