"""
Helpers to create STEM `ParameterFieldParameters` across different STEM builds.

STEM versions differ in whether the property selector is called `property_name`
or `property_names` (plural), and sometimes in other argument naming.
This helper inspects `ParameterFieldParameters.__init__` and passes only the
arguments your installed STEM build supports.
"""

from __future__ import annotations

import inspect
from typing import Any, Iterable


def create_parameter_field_parameters(
    *,
    property_name: str | None = None,
    property_names: Iterable[str] | None = None,
    function_type: str,
    field_generator: Any,
):
    from stem.additional_processes import ParameterFieldParameters

    sig = inspect.signature(ParameterFieldParameters.__init__)
    params = list(sig.parameters.values())

    # Accepted parameter names (excluding `self`)
    accepted_names = {p.name for p in params if p.name != "self"}

    def first_of(names: list[str]) -> str | None:
        for n in names:
            if n in accepted_names:
                return n
        return None

    prop_name_param = first_of(["property_name", "property"])
    prop_names_param = first_of(["property_names", "properties"])
    function_type_param = first_of(["function_type", "type"])
    field_generator_param = first_of(["field_generator", "generator", "rf_generator"])

    if prop_name_param is None and prop_names_param is None:
        raise TypeError(
            "ParameterFieldParameters signature does not contain a recognized property selector "
            "(`property_name`/`property_names`)."
        )

    kwargs: dict[str, Any] = {}

    if prop_name_param is not None:
        if property_name is None:
            # If only plural was provided, take the first
            if property_names is not None:
                property_name = next(iter(property_names), None)
            if property_name is None:
                raise ValueError("Missing RF property name(s) for ParameterFieldParameters.")
        kwargs[prop_name_param] = property_name
    elif prop_names_param is not None:
        if property_names is None:
            if property_name is None:
                raise ValueError("Missing RF property name(s) for ParameterFieldParameters.")
            property_names = [property_name]
        kwargs[prop_names_param] = list(property_names)

    if function_type_param is None:
        raise TypeError(
            "ParameterFieldParameters signature does not contain a recognized `function_type` parameter."
        )
    kwargs[function_type_param] = function_type

    if field_generator_param is None:
        raise TypeError(
            "ParameterFieldParameters signature does not contain a recognized field generator parameter."
        )
    kwargs[field_generator_param] = field_generator

    # Only pass accepted kwargs to avoid unexpected keyword errors.
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_names}
    return ParameterFieldParameters(**filtered_kwargs)

