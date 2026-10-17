"""Surface-driven light theme for recording the StreamErase demo.

The inference, streaming, and playback implementation remains in
``web_demo_long.py``. This entry point only selects a new template and visual
theme; all previous themes remain available.
"""

from pathlib import Path

from flask import send_from_directory

import web_demo_long as demo


ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = ROOT / "web" / "templates_long_surface"
STATIC_DIR = ROOT / "web" / "static_long_surface"

demo.app.template_folder = str(TEMPLATE_DIR)
demo.app.add_url_rule(
    "/surface_static/<path:filename>",
    endpoint="surface_static",
    view_func=lambda filename: send_from_directory(str(STATIC_DIR), filename),
)


if __name__ == "__main__":
    demo.main()
