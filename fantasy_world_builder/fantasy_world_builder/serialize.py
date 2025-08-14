"""Tools for serializing data."""

import json
from typing import Any

import json
import numpy as np
from typing import Any


# ---------- Custom Encoder ----------
class PrettyRoundTripJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that:
    1. Pretty-prints dicts with indentation.
    2. Keeps short lists inline (<= max_inline_list items).
    3. Serializes special types (NumPy arrays, sets) with type tags.

    # ---------- Example Usage ----------
        data = {
            "query": "Query something useful",
            "expected": np.array([0]),
            "retrieved": np.array([0, 2, 8]),
            "tags": {1, 2, 3},
            "nested": {"short": [1, 2], "long": list(range(10))}
        }

        # Serialize
        json_str = json.dumps(data, cls=PrettyRoundTripJSONEncoder, indent=2)
        print(json_str)

        # Deserialize
        restored = json.loads(json_str, object_hook=pretty_roundtrip_decoder)
        print(restored)
        print(type(restored["expected"]))  # numpy.ndarray
        print(type(restored["tags"]))      # set

    """

    def __init__(self, *args, max_inline_list: int = 5, **kwargs):
        self.max_inline_list = max_inline_list
        super().__init__(*args, **kwargs)

    def default(self, obj: Any):
        """Serialize unsupported types with type tagging."""
        if isinstance(obj, np.ndarray):
            return {"__type__": "ndarray", "dtype": str(obj.dtype), "data": obj.tolist()}
        if isinstance(obj, set):
            return {"__type__": "set", "items": list(obj)}
        return super().default(obj)

    def _encode_list(self, obj, level):
        """Pretty or inline encode lists."""
        if len(obj) <= self.max_inline_list and all(
            not isinstance(el, (list, dict)) for el in obj
        ):
            return "[ " + ", ".join(self.encode(el) for el in obj) + " ]"
        else:
            indent_str = " " * (self.indent * (level + 1))
            closing_indent = " " * (self.indent * level)
            return "[\n" + ",\n".join(
                indent_str + self.encode(el) for el in obj
            ) + "\n" + closing_indent + "]"

    def _encode_dict(self, obj, level):
        """Pretty encode dictionaries."""
        if not obj:
            return "{}"
        indent_str = " " * (self.indent * (level + 1))
        closing_indent = " " * (self.indent * level)
        items = []
        for key, value in obj.items():
            items.append(
                indent_str + json.dumps(key) + ": " + self._encode_any(value, level + 1)
            )
        return "{\n" + ",\n".join(items) + "\n" + closing_indent + "}"

    def _encode_any(self, obj, level):
        if isinstance(obj, dict):
            return self._encode_dict(obj, level)
        elif isinstance(obj, list):
            return self._encode_list(obj, level)
        else:
            return super().encode(obj)

    def encode(self, obj):
        return self._encode_any(obj, 0)


def pretty_roundtrip_decoder(obj: dict):
    """Hook to restore special types."""
    if "__type__" in obj:
        t = obj["__type__"]
        if t == "ndarray":
            return np.array(obj["data"], dtype=obj.get("dtype", None))
        if t == "set":
            return set(obj["items"])
    return obj
