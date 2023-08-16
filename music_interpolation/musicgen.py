import typing as typ

import torch
from audiocraft.data.audio_utils import convert_audio
from audiocraft.models.musicgen import MusicGen

MelodyList = typ.List[typ.Optional[torch.Tensor]]
MelodyType = typ.Union[torch.Tensor, MelodyList]


def generate_continuation_with_chroma(
    model: MusicGen,
    prompt: torch.Tensor,
    prompt_sample_rate: int,
    descriptions: typ.Optional[typ.List[typ.Optional[str]]],
    melody_wavs: MelodyType,
    melody_sample_rate: int,
    progress: bool = False,
) -> torch.Tensor:
    if prompt.dim() == 2:
        prompt = prompt[None]
    if prompt.dim() != 3:
        raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")

    if isinstance(melody_wavs, torch.Tensor):
        if melody_wavs.dim() == 2:
            melody_wavs = melody_wavs[None]
        if melody_wavs.dim() != 3:
            raise ValueError("Melody wavs should have a shape [B, C, T].")
        melody_wavs = list(melody_wavs)
    else:
        for melody in melody_wavs:
            if melody is not None:
                assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

    prompt = convert_audio(prompt, prompt_sample_rate, model.sample_rate, model.audio_channels)
    if descriptions is None:
        descriptions = typ.cast(typ.List[str | None], [None] * len(prompt))

    melody_wavs = [
        convert_audio(wav, melody_sample_rate, model.sample_rate, model.audio_channels)
        if wav is not None
        else None
        for wav in melody_wavs
    ]
    attributes, prompt_tokens = model._prepare_tokens_and_attributes(  # type: ignore
        descriptions=descriptions, prompt=prompt, melody_wavs=melody_wavs
    )
    assert prompt_tokens is not None
    return model._generate_tokens(attributes, prompt_tokens, progress)  # type: ignore
