"""Bounded PCM WAV → MiMo's HTK magnitude log-mel; no downloads or ASR."""
from __future__ import annotations

from dataclasses import dataclass
import io
import wave

import numpy as np

from ..cancellation import check_cancelled

AUDIO_TYPES = frozenset({"audio/wav"})
MAX_AUDIO_SECONDS = 30
MAX_AUDIOS = 2
SAMPLE_RATE = 24000


@dataclass(frozen=True)
class AudioInput:
    mel: np.ndarray
    digest: str
    samples: int

    @property
    def token_count(self):
        # conv stride 2, learned pooling stride 2, groups of four RVQ frames.
        return (self.mel.shape[0] + 15) // 16


def decode_wav(data):
    from ..media import MediaError, MAX_ASSET_BYTES
    if not data or len(data) > MAX_ASSET_BYTES:
        raise MediaError("Audio must be at most 8 MiB.")
    # Require a complete little-endian RIFF envelope, not a URL, RF64 or stream.
    if (data[:4] != b"RIFF" or data[8:12] != b"WAVE"
            or int.from_bytes(data[4:8], "little") + 8 != len(data)):
        raise MediaError("Use a complete PCM WAV file.")
    position, audio_bytes, seen = 12, None, set()
    while position + 8 <= len(data):
        kind = data[position:position + 4]
        size = int.from_bytes(data[position + 4:position + 8], "little")
        if kind in (b"fmt ", b"data"):
            if kind in seen:
                raise MediaError("Duplicate WAV format or data chunk.")
            seen.add(kind)
        if kind == b"data":
            audio_bytes = size
        position += 8 + size + size % 2
        if position > len(data):
            raise MediaError("Truncated WAV chunk.")
    if position != len(data):
        raise MediaError("Invalid WAV chunk boundary.")
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            channels, width, rate, frames, compression, _ = source.getparams()
            if rate != SAMPLE_RATE or width != 2 or channels not in (1, 2) or compression != "NONE":
                raise MediaError("Use 24 kHz, 16-bit PCM WAV with one or two channels.")
            if not 480 < frames <= SAMPLE_RATE * MAX_AUDIO_SECONDS:
                raise MediaError("Audio must be longer than 20 ms and at most 30 seconds; it is not truncated.")
            raw = source.readframes(frames)
            if audio_bytes != frames * channels * width or len(raw) != audio_bytes:
                raise MediaError("Truncated WAV audio.")
    except (wave.Error, EOFError, OverflowError) as error:
        raise MediaError("Invalid PCM WAV audio.") from error
    waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels) / np.float32(32768)
    return waveform.mean(axis=1)


def log_mel(waveform):
    """torchaudio MelSpectrogram(power=1, center=True), followed by log/clamp.

    Fixed checkpoint settings: 960-point periodic Hann, hop 240, reflect
    padding, 128 HTK bands over [0, 12000] Hz, no normalization. CPU only.
    """
    check_cancelled()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or not 480 < waveform.size <= SAMPLE_RATE * MAX_AUDIO_SECONDS or not np.isfinite(waveform).all():
        raise ValueError("invalid bounded audio waveform")
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(960, dtype=np.float32) / 960)).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(waveform, (480, 480), mode="reflect"), 960)[::240]
    magnitude = np.abs(np.fft.rfft(frames * window, axis=-1)).astype(np.float32)
    frequencies = np.linspace(0, 12000, 481, dtype=np.float32)
    mel_points = np.linspace(0, 2595 * np.log10(1 + 12000 / 700), 130, dtype=np.float32)
    hz = 700 * (10 ** (mel_points / 2595) - 1)
    slopes = hz[None, :] - frequencies[:, None]
    filters = np.maximum(0, np.minimum(-slopes[:, :-2] / np.diff(hz)[:-1], slopes[:, 2:] / np.diff(hz)[1:]))
    mel = np.log(np.maximum(magnitude @ filters, np.float32(1e-7)))
    check_cancelled()
    return mel


def audio_input(part):
    waveform = decode_wav(part.data)
    return AudioInput(log_mel(waveform), part.digest, waveform.size)
