"""Read settings from the environment, loading a ``.env`` file on import."""

import os
from dotenv import load_dotenv
load_dotenv()

def get_env(env_name: str, default=None, *, required: bool = True):
    """Return an environment variable.

    Args:
        env_name (str): Variable name.
        default: Value returned when the variable is unset and not required.
        required (bool): Raise when the variable is unset.

    Returns:
        str: The value, or ``default``.

    Raises:
        Exception: If the variable is unset and ``required`` is true.
    """
    if env_name in os.environ:
        return os.environ[env_name]
    if not required:
        return default
    raise Exception(f'{env_name} does not exist in environment')