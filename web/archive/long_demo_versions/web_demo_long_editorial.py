"""Editorial-minimal theme for the long-video StreamErase demo.

Inference, streaming, and playback behavior are inherited from
``web_demo_long.py``. Only the template and visual theme are replaced.
"""

from pathlib import Path
import sys

from flask import send_from_directory

import web_demo_long as demo


ROOT = Path(__file__).resolve().parent
THEME_DIR = ROOT / "web" / "static_long_editorial"
TEMPLATE_DIR = ROOT / "web" / "templates_long_editorial"
demo.app.template_folder = str(TEMPLATE_DIR)
demo.app.add_url_rule(
    "/editorial_static/<path:filename>",
    endpoint="editorial_static",
    view_func=lambda filename: send_from_directory(str(THEME_DIR), filename),
)


def main():
    if not any(arg == "--port" or arg.startswith("--port=") for arg in sys.argv[1:]):
        sys.argv.extend(["--port", "5004"])
    demo.main()


if __name__ == "__main__":
    main()
