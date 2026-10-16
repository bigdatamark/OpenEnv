# [MODULE DOC]
# Manages a private Xvfb virtual display (801x601x24) and uses xdotool for real
# window focus and pointer clicks inside the browser sandbox.
"""An owned X display and real pointer input inside the browser VM."""

import os
import select
import subprocess
import tempfile
from pathlib import Path


class VirtualDesktop:
    # [CLASS DOC]
    # Owns an Xvfb subprocess, a temporary directory holding its log file, and an
    # environment dict with DISPLAY set for child processes.
    def __init__(self):
        # [FUNCTION DOC]
        # Creates a temp dir and log file, launches Xvfb asking it to write its display
        # number to stdout, waits up to 10 s for it, and builds env with DISPLAY=":N".
        # Raises: RuntimeError on startup timeout or failure (resources are cleaned up
        # via close() before re-raising any exception).
        # Side effects: spawns a subprocess, creates temp files.
        self.process = None
        self.directory = tempfile.TemporaryDirectory(prefix="browser-display-")
        self.log = (Path(self.directory.name) / "xvfb.log").open("wb")
        try:
            self.process = subprocess.Popen(
                ["Xvfb", "-displayfd", "1", "-screen", "0", "801x601x24", "-nolisten", "tcp"],
                stdout=subprocess.PIPE,
                stderr=self.log,
            )
            if not select.select([self.process.stdout], [], [], 10)[0]:
                raise RuntimeError("virtual display startup timed out")
            number = self.process.stdout.readline().decode().strip()
            if not number.isdecimal() or self.process.poll() is not None:
                raise RuntimeError("virtual display failed to start")
            self.env = dict(os.environ, DISPLAY=":" + number)
        except BaseException:
            self.close()
            raise

    def focus_browser(self, process_id):
        # [FUNCTION DOC]
        # Finds the first visible window of the given process id with xdotool search
        # and focuses it synchronously.
        # Raises: RuntimeError if no window is found; CalledProcessError or
        # TimeoutExpired (10 s) from xdotool.
        window_ids = subprocess.check_output(
            ["xdotool", "search", "--onlyvisible", "--pid", str(process_id)],
            env=self.env,
            text=True,
            timeout=10,
        ).splitlines()
        if not window_ids or not window_ids[0].isdecimal():
            raise RuntimeError("browser window unavailable")
        subprocess.run(
            ["xdotool", "windowfocus", "--sync", window_ids[0]],
            env=self.env,
            check=True,
            timeout=10,
        )

    def click(self, x, y):
        # [FUNCTION DOC]
        # Moves the pointer to (x, y) and left-clicks via xdotool on this display.
        # Raises: CalledProcessError or TimeoutExpired (10 s).
        subprocess.run(
            ["xdotool", "mousemove", str(x), str(y), "click", "1"],
            env=self.env,
            check=True,
            timeout=10,
        )

    def close(self):
        # [FUNCTION DOC]
        # Terminates Xvfb (kill after a 5 s wait), closes its stdout pipe, closes the
        # log file and removes the temp directory. Safe to call after a partial start.
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self.process.stdout is not None:
                self.process.stdout.close()
            self.process = None
        self.log.close()
        self.directory.cleanup()
