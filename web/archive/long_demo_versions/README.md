# Archived long-video demo versions

These files are historical snapshots, not runnable entry points from this
directory. Their imports and asset paths refer to their original locations.
To revisit a theme, copy its template and stylesheet into the active demo
directories after backing up the current files.

The active demo is now:

- `web_demo_long.py`
- `web/templates_long/long_index.html`
- `web/static_long/style.css`
- `web/static_long/app.js`
- `web/static_long/socket.io.min.js`

Run it from the repository root:

```bash
python web_demo_long.py --host 0.0.0.0 --port 5001
```

The active presentation comes from the final `light_type_test` version:
no hero/footer or visible numeric telemetry, integer duration labels
5s / 10s / 15s, and a default playback buffer of 3 seconds.
Duration labels are abbreviated; inference frame counts remain unchanged.

`web_demo_long_original.py`, `templates_long_original`, and
`static_long_original` preserve the original dark presentation.
Other files retain their previous theme names. Shared streaming JavaScript
and Socket.IO remain in the active `web/static_long` directory.

The separate short-video demo `web_demo.py` is unaffected.
