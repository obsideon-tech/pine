import numpy as np
import pytest

from pine import Bars, BarsError


def make(**over):
    kw = dict(
        ts=np.array([0, 60, 120]),
        open=[1.0, 2.0, 3.0],
        high=[2.0, 3.0, 4.0],
        low=[0.5, 1.5, 2.5],
        close=[1.5, 2.5, 3.5],
        volume=[10.0, np.nan, 30.0],
        timeframe="1",
    )
    kw.update(over)
    return Bars(**kw)


def test_valid_bars_are_float64_int64_and_read_only():
    b = make()
    assert len(b) == 3
    assert b.ts.dtype == np.int64 and b.close.dtype == np.float64
    with pytest.raises(ValueError):
        b.close[0] = 9.0


def test_bars_copy_inputs():
    src = np.array([1.5, 2.5, 3.5])
    b = make(close=src)
    src[0] = 99.0
    assert b.close[0] == 1.5


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"ts": np.array([0, 60, 60])}, "strictly increasing"),
        ({"ts": np.array([0.0, 60.0, 120.0])}, "integer"),
        ({"close": [1.5, 2.5]}, "close has 2 values"),
        ({"close": [1.5, np.nan, 3.5]}, "close must be finite"),
        ({"high": [2.0, 2.4, 4.0]}, "index 1"),
        ({"low": [0.5, 2.1, 2.5]}, "index 1"),
        ({"volume": [1.0, -1.0, 1.0]}, "volume"),
        ({"timeframe": "1h"}, "timeframe"),
        ({"session_tz": "Mars/Olympus"}, "IANA"),
    ],
)
def test_invariants_rejected(over, msg):
    with pytest.raises(BarsError, match=msg):
        make(**over)


@pytest.mark.parametrize("tf", ["1", "15", "240", "D", "1D", "W", "1W", "M", "3M", "30S"])
def test_pine_timeframes_accepted(tf):
    assert make(timeframe=tf).timeframe == tf


def test_equals_treats_nan_volume_as_equal():
    assert make().equals(make())
    assert not make().equals(make(close=[1.5, 2.5, 3.4]))
