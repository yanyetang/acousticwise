class InputError(Exception):
    """An uploaded or dataset file that cannot be classified; the app shows its message."""


class UnsupportedFileError(InputError):
    """Not a readable PCM WAV, or corrupt."""


class AudioTooShortError(InputError):
    """Shorter than the minimum duration (4.0 s)."""


class SilentAudioError(InputError):
    """Near-silent: RMS below the threshold."""
