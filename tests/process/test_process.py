import pytest
import numpy as np
from pyspatools.process import get_latency


@pytest.mark.parametrize('x,threshold,offset,expected', [
    (np.array([[0, 1, 2], [0, 0, 1], [0, 2, 4]]), 0, 0, [1, 2, 1]),
    (np.array([[1, -1, 2], [-3, 0, 1], [10, 2, 4]]), 0, 0, [0, 0, 0]),
    (np.array([[0, 1, 2], [2, 0, 1], [0, -2, 4]]), 1, 0, [2, 0, 1]),
    (np.array([[0, 0, 0, 0, -1, -2], [0, 0, 0, 2, 0, 1], [0, 2, 4, 4, 5, 0]]), 0, 1, [3, 2, 0]),
    (np.array([[0, 0.1, 2], [0, 0, 0.1], [0, 0.2, 0.4]]), 0, 0, [1, 2, 1]),
])
def test_get_latency(x, threshold, offset, expected):
    observed = get_latency(x, threshold=threshold, offset=offset)
    assert observed == expected
