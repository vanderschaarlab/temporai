from typing import Dict, Optional


def exclusive_args(
    values: Dict,
    arg1: str,
    arg2: str,
    arg1_friendly_name: Optional[str] = None,
    arg2_friendly_name: Optional[str] = None,
) -> None:
    arg1_value = values.get(arg1, None)
    arg2_value = values.get(arg2, None)
    arg1_name = arg1_friendly_name if arg1_friendly_name else f"`{arg1}`"
    arg2_name = arg2_friendly_name if arg2_friendly_name else f"`{arg2}`"
    if arg1_value is not None and arg2_value is not None:
        raise ValueError(f"Must provide either {arg1_name} or {arg2_name} but not both")
