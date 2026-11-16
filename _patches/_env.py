"""Every environment variable the pack reads or sets goes through this module.

Settings are read when they are needed, so a change made before ComfyUI starts takes effect.
The pack sets variables in two places only: legacy setting names are copied to their new
names, and the folder holding the `ninja` build tool is put on PATH for torch's extension
builder. (prestartup_script.py runs before this package can be imported and sets the MPS
allocator watermarks itself.)
"""

import os


def get(name, default=None):
    """The variable's value, or default when it is unset."""
    return os.environ.get(name, default)


def has(name):
    return name in os.environ


def put(name, value):
    os.environ[name] = value


def prepend_path(directory):
    """Put directory first on PATH."""
    put("PATH", directory + os.pathsep + get("PATH", ""))
