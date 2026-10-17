"""Independent typography test entry point for the long-video web demo.

The original ``web_demo_long.py`` remains the default dark version. This
entry point reuses its inference and streaming routes with a separate template
and stylesheet so both presentations can be run side by side.
"""

from pathlib import Path
import sys

from flask import send_from_directory

import web_demo_long as demo


ROOT = Path(__file__).resolve().parent
LIGHT_TEMPLATE_DIR = ROOT / "web" / "templates_long_light_type_test"
LIGHT_STATIC_DIR = ROOT / "web" / "static_long_light_type_test"

demo.app.template_folder = str(LIGHT_TEMPLATE_DIR)
demo.app.add_url_rule(
    "/light_type_test_static/<path:filename>",
    endpoint="light_type_test_static",
    view_func=lambda filename: send_from_directory(str(LIGHT_STATIC_DIR), filename),
)


def main():
    if not any(arg == "--port" or arg.startswith("--port=") for arg in sys.argv[1:]):
        sys.argv.extend(["--port", "5002"])
    demo.main()


if __name__ == "__main__":
    main()
