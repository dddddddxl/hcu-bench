"""Bench adapter for the wheel-only recipe; the supplied archive omitted this helper."""


def load_flash_attn_extension(*, use_whl=False, extension_so=None):
    if not use_whl or extension_so is not None:
        raise ValueError("This fixed recipe supports only --whl with the prepared flash_attn wheel")
    from flash_attn.flash_attn_interface import flash_attn_cuda
    if not callable(getattr(flash_attn_cuda, "fwd", None)):
        raise RuntimeError("Installed flash_attn extension lacks fwd; use a compatible prepared image")
    return flash_attn_cuda, flash_attn_cuda.__file__
