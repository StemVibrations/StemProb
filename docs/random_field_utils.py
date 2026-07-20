"""
Helpers to create STEM `RandomFieldGenerator` across different STEM builds.

Different STEM versions have slightly different `RandomFieldGenerator.__init__`
signatures (e.g., `n_dim` vs `ndim`, positional argument order, etc.).

This helper inspects the signature at runtime and maps our standardized RF
settings to whatever your installed STEM build expects.
"""

from __future__ import annotations

import inspect
from typing import Any


def create_random_field_generator(
    *,
    dim: int,
    cov: float,
    model_name: str,
    v_scale_fluctuation: float,
    anisotropy: Any,
    angle: Any,
    seed: int,
):
    # Import here so this module is only required when RF is used.
    from stem.field_generator import RandomFieldGenerator

    sig = inspect.signature(RandomFieldGenerator.__init__)

    # Collect accepted params by kind
    pos_only = []
    accepts = {}  # name -> Parameter
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        accepts[name] = p
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            pos_only.append(p)

    # Candidate name mappings for our standardized inputs
    dim_candidates = {"n_dim", "ndim", "dim", "dimensions", "n_dimensions", "n_dimensional"}
    cov_candidates = {"cov", "COV", "coefficient_of_variation"}
    model_candidates = {"model_name", "model"}
    v_scale_candidates = {
        "v_scale_fluctuation",
        "v_scale",
        "vertical_scale_fluctuation",
        "vertical_scale",
        "v_scale_fluct",
    }
    anis_candidates = {"anisotropy", "anis"}
    angle_candidates = {"angle", "angles", "inclination", "inclinations"}
    seed_candidates = {"seed", "random_seed", "rng_seed"}

    def pick_first(candidates: set[str]) -> str | None:
        for c in candidates:
            if c in accepts:
                return c
        return None

    dim_param = pick_first(dim_candidates)
    cov_param = pick_first(cov_candidates)
    model_param = pick_first(model_candidates)
    v_scale_param = pick_first(v_scale_candidates)
    anis_param = pick_first(anis_candidates)
    angle_param = pick_first(angle_candidates)
    seed_param = pick_first(seed_candidates)

    def required_or_default(p: inspect.Parameter, name_for_msg: str, value: Any) -> Any:
        if value is None:
            if p.default is inspect._empty:
                raise TypeError(
                    f"RandomFieldGenerator signature requires parameter '{p.name}' "
                    f"({name_for_msg}), but no matching value is available."
                )
            return p.default
        return value

    pos_only_names = {p.name for p in pos_only}

    # Build positional-only args first (rare, but handle it safely).
    args: list[Any] = []
    for p in pos_only:
        if p.name == dim_param:
            args.append(required_or_default(p, "dim", dim))
        elif p.name == cov_param:
            args.append(required_or_default(p, "cov", cov))
        elif p.name == model_param:
            args.append(required_or_default(p, "model_name", model_name))
        elif p.name == v_scale_param:
            args.append(required_or_default(p, "v_scale_fluctuation", v_scale_fluctuation))
        elif p.name == anis_param:
            args.append(required_or_default(p, "anisotropy", anisotropy))
        elif p.name == angle_param:
            args.append(required_or_default(p, "angle", angle))
        elif p.name == seed_param:
            args.append(required_or_default(p, "seed", seed))
        else:
            # Unknown positional-only parameter: try default, else fail.
            if p.default is inspect._empty:
                raise TypeError(
                    f"RandomFieldGenerator has unsupported required positional-only parameter '{p.name}'."
                )
            args.append(p.default)

    kwargs: dict[str, Any] = {}
    # Important: never pass a POSITIONAL_ONLY parameter via kwargs as well.
    if dim_param and dim_param not in pos_only_names:
        kwargs[dim_param] = dim
    if cov_param and cov_param not in pos_only_names:
        kwargs[cov_param] = cov
    if model_param and model_param not in pos_only_names:
        kwargs[model_param] = model_name
    if v_scale_param and v_scale_param not in pos_only_names:
        kwargs[v_scale_param] = v_scale_fluctuation
    if anis_param and anis_param not in pos_only_names:
        kwargs[anis_param] = anisotropy
    if angle_param and angle_param not in pos_only_names:
        kwargs[angle_param] = angle
    if seed_param and seed_param not in pos_only_names:
        kwargs[seed_param] = seed

    # Sanity: if a required non-positional-only param wasn't mapped and has no default, raise.
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            continue
        if p.default is not inspect._empty:
            continue
        if p.name in kwargs:
            continue
        # Some signatures may accept **kwargs; if so we can skip this check.
        if any(pp.kind == inspect.Parameter.VAR_KEYWORD for pp in sig.parameters.values()):
            break
        raise TypeError(
            f"RandomFieldGenerator required parameter '{p.name}' was not provided by the helper."
        )

    return RandomFieldGenerator(*args, **kwargs)

