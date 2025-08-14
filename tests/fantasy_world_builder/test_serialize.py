import json
import numpy as np
from fantasy_world_builder.serialize import PrettyRoundTripJSONEncoder, pretty_roundtrip_decoder

def test_pretty_print():

    data = {
        "query": "Query something useful.",
        "expected": np.array([0]),
        "retrieved": np.array([0, 2, 8]),
        "tags": {1, 2, 3},
        "nested": {"short": [1, 2], "long": list(range(10))}
    }

    json_str = json.dumps(data, cls=PrettyRoundTripJSONEncoder, indent=2)

    round_trip = json.loads(json_str, object_hook=pretty_roundtrip_decoder)
    assert round_trip.keys() == data.keys()
    np.testing.assert_array_equal(data['expected'], round_trip['expected'])
    assert round_trip["tags"] == data['tags']