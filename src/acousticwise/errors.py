class InputError(Exception):
    """An uploaded or dataset file that cannot be classified; the app shows its message."""


class UnsupportedFileError(InputError):
    """Not a readable PCM WAV, or corrupt."""


class AudioTooShortError(InputError):
    """Shorter than the minimum duration (4.0 s)."""


class SilentAudioError(InputError):
    """Near-silent: RMS below the threshold."""


class FileTooLargeError(InputError):
    """Larger than the upload size limit."""


class AudioTooLongError(InputError):
    """Longer than the duration limit."""


class InferenceError(Exception):
    """Something unexpected went wrong in the model code; not the upload's fault."""
