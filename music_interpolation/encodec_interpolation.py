from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoProcessor, EncodecConfig, EncodecModel
from transformers.models.encodec.modeling_encodec import (
    EncodecDecoder,
    EncodecEncoder,
    EncodecResidualVectorQuantizer,
)

NDFloat = npt.NDArray[np.float_]
ProcessedInput = TypedDict(
    "ProcessedInput", {"input_values": torch.Tensor, "padding_mask": torch.Tensor}
)


class EncodecInterpolation:
    """
    Interpolates between two audio clips using a trained Encodec model. This uses the SEANet model
    only to convert audio into a latent embedding (continuous vector), and then performs linear
    interpolation between the embeddings. The interpolated embeddings are then decoded back into
    audio using the SEANet model. The discrete quantization capabilities of the Encodec model are
    not used.
    """

    def __init__(self, model_name: str = "facebook/encodec_48khz", device: str = "cpu"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = cast(EncodecModel, EncodecModel.from_pretrained(model_name)).to(device)

    @property
    def sampling_rate(self) -> int:
        return cast(int, self.processor.sampling_rate)

    def interpolate(
        self, audio_a: NDFloat, audio_b: NDFloat, t_coeffs: NDFloat | None = None
    ) -> NDFloat:
        """
        Executes the interpolation between two audio clips. The audio clips must have the same
        number of channels and duration. If provided, the interpolation coefficient array 't_coeffs'
        must have the same length as the audio clips. If 't_coeffs' is not provided, a linear
        interpolation is used.
        """

        # Ensure audio_a and audio_b have the same number of channels and duration
        if len(audio_a.shape) != len(audio_b.shape):
            raise ValueError(
                f"Audio a and b must have the same number of channels, but got {audio_a.shape} "
                f"and {audio_b.shape}"
            )
        if audio_a.shape[-1] != audio_b.shape[-1]:
            raise ValueError(
                f"Audio a and b must have the same duration, but got {audio_a.shape} "
                f"and {audio_b.shape} samples"
            )

        # Preprocess the audio and encode it into per-frame latent embeddings
        inputs_a = cast(
            ProcessedInput,
            self.processor(
                raw_audio=audio_a, sampling_rate=self.sampling_rate, return_tensors="pt"
            ),
        )
        input_values_a = inputs_a["input_values"].to(self.model.device)
        padding_mask = inputs_a["padding_mask"].to(self.model.device)

        embedding_a, scale_a = encode_embedding(
            input_values_a, self.model.encoder, self.model.config, padding_mask
        )
        scale_a = scale_a.to(self.model.device)

        # padding_mask can be reused for audio_b since it has the same shape
        inputs_b = cast(
            ProcessedInput,
            self.processor(
                raw_audio=audio_b, sampling_rate=self.sampling_rate, return_tensors="pt"
            ),
        )
        input_values_b = inputs_b["input_values"].to(self.model.device)

        embedding_b, scale_b = encode_embedding(
            input_values_b, self.model.encoder, self.model.config, padding_mask
        )
        scale_b = scale_b.to(self.model.device)

        if t_coeffs is None:
            t_coeffs = np.linspace(0, 1, num=embedding_a.shape[0])
        if len(t_coeffs) != embedding_a.shape[0]:
            raise ValueError(
                f"'audio_a', 'audio_b', 't_coeffs' must all have the same length, "
                f"but got audio_a={audio_a.shape}, audio_b={audio_b.shape}, "
                f"t_coeffs={t_coeffs.shape}"
            )

        # Compute the slerp interpolation between embeddings
        interpolated_embedding = compute_interpolation_in_latent(
            embedding_a, embedding_b, t_coeffs
        )

        # Adjust the shape back to (n, 2) for decoding
        interpolated_embedding = interpolated_embedding.transpose(0, 1)

        # Linearly interpolate the scales
        t_tensor = torch.tensor(t_coeffs, device=scale_a.device, dtype=scale_a.dtype)
        interpolated_scales = torch.lerp(scale_a, scale_b, t_tensor).double()

        # Decode the interpolated embeddings into audio
        audio_tensor = decode_embeddings(
            interpolated_embedding,
            interpolated_scales,
            self.model.config,
            self.model.decoder,
            padding_mask,
        )

        # Convert audio_tensor from shape (num_seconds, num_channels, sample_rate) to
        # (num_channels, num_samples)
        audio_tensor = audio_tensor.transpose(0, 1)
        audio_tensor = audio_tensor.reshape(audio_tensor.shape[0], -1)

        # Trim the (num_channels, num_samples) audio to the same length as the input audio
        audio_tensor = audio_tensor[:, : audio_a.shape[-1]]

        audio_values = cast(NDFloat, audio_tensor.cpu().detach().numpy())
        return audio_values


# Interpolation code adapted from CRASH
# <https://github.com/simonrouard/CRASH/blob/3fc43781fbee424845b2a928f859723f605e140d/inference.py#L6-L24>
def compute_interpolation_in_latent(
    latent1: torch.Tensor, latent2: torch.Tensor, lambd: npt.ArrayLike
) -> torch.Tensor:
    """
    Implementation of Spherical Linear Interpolation (Slerp) for embeddings.
    latent1, latent2: tensors of shape (num_frames, batch_size, num_codebooks, codebook_dim)
    lambd: list of floats between 0 and 1 representing the parameter t of the Slerp
    """
    device = latent1.device
    lambd = torch.tensor(lambd, device=device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # Ensure the shapes of latent1 and latent2 match
    assert latent1.shape == latent2.shape

    # Calculate the cosine of the angle between latent1 and latent2
    norm_latent1 = cast(torch.Tensor, torch.linalg.norm(latent1, dim=-1, keepdim=True))
    norm_latent2 = cast(torch.Tensor, torch.linalg.norm(latent2, dim=-1, keepdim=True))
    cos_omega = (latent1 * latent2).sum(dim=-1, keepdim=True) / (norm_latent1 * norm_latent2)

    # Calculate the angle omega
    omega = torch.arccos(cos_omega)

    # Calculate the slerp coefficients
    a = torch.sin((1 - lambd) * omega) / torch.sin(omega)
    b = torch.sin(lambd * omega) / torch.sin(omega)

    # Perform the slerp interpolation
    interpolated_latent = a * latent1 + b * latent2

    return interpolated_latent


def encode_embedding(
    input_values: torch.Tensor,
    encoder: EncodecEncoder,
    config: EncodecConfig,
    padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes the input audio waveform into a latent embedding, a continuous-value vector.

    Args:
        input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
            Float values of the input audio waveform.
        padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
            Padding mask used to pad the `input_values`.

    Returns:
        A list of frames containing the continuous-value embedding vector for the input audio
        waveform, along with rescaling factors for each chunk when `normalize` is True. Each frames
        is a tuple `(embedding, scale)`, with `embedding` of shape
        `(batch_size, num_embeddings, frames)`.
    """
    _, channels, input_length = input_values.shape

    if channels < 1 or channels > 2:
        raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")

    chunk_length = config.chunk_length
    if chunk_length is None:
        chunk_length = input_length
        stride = input_length
    else:
        stride = config.chunk_stride or input_length

    if padding_mask is None:
        padding_mask = torch.ones_like(input_values).bool()

    frame_embeddings: list[torch.Tensor] = []
    scales: list[torch.Tensor | None] = []

    step = chunk_length - stride
    if (input_length % stride) - step != 0:
        raise ValueError(
            "The input length is not properly padded for batched chunked decoding. Make sure to "
            "pad the input correctly."
        )

    for offset in range(0, input_length - step, stride):
        mask = padding_mask[..., offset : offset + chunk_length].bool()
        frame = input_values[:, :, offset : offset + chunk_length]
        embeddings, scale = _encode_frame_embeddings(frame, mask, encoder, config)
        frame_embeddings.append(embeddings)
        scales.append(scale)

    return torch.stack(frame_embeddings), torch.tensor(scales)


def decode_embeddings(
    audio_embeddings: torch.Tensor,
    audio_scales: torch.Tensor,
    config: EncodecConfig,
    decoder: EncodecDecoder,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decodes the given audio embeddings into an output audio waveform."""
    chunk_length = config.chunk_length
    if chunk_length is None:
        if len(audio_embeddings) != 1:
            raise ValueError(f"Expected one frame, got {len(audio_embeddings)}")
        audio_values = _decode_frame_embeddings(audio_embeddings[0], decoder, audio_scales[0])
    else:
        decoded_frames = []

        for embedding, scale in zip(audio_embeddings, audio_scales):
            frames = _decode_frame_embeddings(embedding, decoder, scale)
            decoded_frames.append(frames)

        audio_values = _linear_overlap_add(decoded_frames, config.chunk_stride or 1)

    # Truncate based on padding mask
    if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
        audio_values = audio_values[..., : padding_mask.shape[-1]]

    return audio_values


def quantize_embeddings(
    audio_embeddings: torch.Tensor,
    bandwidth: float,
    quantizer: EncodecResidualVectorQuantizer,
) -> torch.Tensor:
    """
    Quantizes the given latent embedding vector using the provided quantizer.

    Args:
        embeddings (`torch.Tensor` of shape `(batch_size, num_embeddings, frames)`):
            Per-frame embedding vectors to quantize.
        bandwidth (`float`):
            Bandwidth to use for the quantization.
        quantizer (`EncodecResidualVectorQuantizer`):
            Quantizer to use, turning the continuous latent vector into discrete indices.

    Returns:
        A tensor of shape `(batch_size, num_codebooks, frames)` containing per-frame codebooks, i.e.
        the embedding space indices of the quantized latent vector.
    """
    encoded_frames: list[torch.Tensor] = [
        quantizer.encode(embedding, bandwidth).transpose(0, 1) for embedding in audio_embeddings
    ]
    return torch.stack(encoded_frames)


def _linear_overlap_add(frames: list[torch.Tensor], stride: int) -> torch.Tensor:
    # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
    # e.g., more than 2 frames per position.
    # The core idea is to use a weight function that is a triangle,
    # with a maximum value at the middle of the chunk.
    # We use this weighting when summing the frames, and divide by the sum of weights
    # for each positions at the end. Thus:
    #   - if a frame is the only one to cover a position, the weighting is a no-op.
    #   - if 2 frames cover a position:
    #          ...  ...
    #         /   \/   \
    #        /    /\    \
    #            S  T       , i.e. S offset of second frame starts, T end of first frame.
    # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
    # After the final normalization, the weight of the second frame at position `t` is
    # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
    #
    #   - if more than 2 frames overlap at a given point, we hope that by induction
    #      something sensible happens.
    if len(frames) == 0:
        raise ValueError("`frames` cannot be an empty list.")

    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    frame_length = frames[0].shape[-1]
    time_vec = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1:-1]
    weight = 0.5 - (time_vec - 0.5).abs()

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    for frame in frames:
        frame_length = frame.shape[-1]
        out[..., offset : offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset : offset + frame_length] += weight[:frame_length]
        offset += stride

    if sum_weight.min() == 0:
        raise ValueError(f"`sum_weight` minimum element must be bigger than zero: {sum_weight}`")

    return out / sum_weight


def _decode_frame_embeddings(
    embeddings: torch.Tensor,
    decoder: EncodecDecoder,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Decodes the embeddings into an output audio waveform, using the provided
    decoder. Optionally, a scale tensor can be provided to scale the output.
    """
    outputs: torch.Tensor = decoder(embeddings.float())
    if scale is not None:
        outputs = outputs * scale.view(-1, 1, 1)
    return outputs


def _encode_frame_embeddings(
    input_values: torch.Tensor,
    padding_mask: torch.Tensor,
    encoder: EncodecEncoder,
    config: EncodecConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Encodes the given input using the underlying VQVAE. If `config.normalize` is set to `True` the
    input is first normalized. The padding mask is required to compute the correct scale.
    """
    length = input_values.shape[-1]
    duration = length / config.sampling_rate

    if config.chunk_length_s is not None and duration > 1e-5 + config.chunk_length_s:
        raise RuntimeError(
            f"Duration of frame ({duration}) is longer than chunk {config.chunk_length_s}"
        )

    scale = None
    if config.normalize:
        # if the padding is non zero
        input_values = input_values * padding_mask
        mono = torch.sum(input_values, 1, keepdim=True) / input_values.shape[1]
        scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
        input_values = input_values / scale

    embeddings = encoder(input_values)
    return embeddings, scale
