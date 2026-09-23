"""Compact light theme for recording the StreamErase long-video demo.

This entry point keeps the original dark/light themes untouched and reuses
the existing inference, Socket.IO streaming, and playback implementation.
Only the page template and CSS are replaced.
"""

from pathlib import Path

from flask import send_from_directory

import web_demo_long as demo


ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = ROOT / "web" / "templates_long_compact"
STATIC_DIR = ROOT / "web" / "static_long_compact"

demo.app.template_folder = str(TEMPLATE_DIR)
demo.app.add_url_rule(
    "/compact_static/<path:filename>",
    endpoint="compact_static",
    view_func=lambda filename: send_from_directory(str(STATIC_DIR), filename),
)


if __name__ == "__main__":
    demo.main()
