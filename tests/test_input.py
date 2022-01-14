import numpy as np
import pytest

from psi.controller.input import (accumulate, blocked, concatenate, coroutine,
                                  create_diff_matrix, InputData, mc_reference)


@pytest.fixture
def data():
    return []


@pytest.fixture
def pipeline(data):
    def append_data(d):
        nonlocal data
        if d is not Ellipsis:
            data.append(d)

    return blocked(10,
                    accumulate(4, 0, True, None,
                               append_data).send)


def test_input_data():
    expected_array = np.concatenate((
        np.zeros(shape=5),
        np.ones(shape=5),
        np.random.uniform(size=5),
    ))
    expected_metadata = {'t0': 0}
    expected_input = InputData(expected_array, expected_metadata)

    input_data = [
        InputData(expected_array[0:5], metadata=expected_metadata),
        InputData(expected_array[5:10], metadata=expected_metadata),
        InputData(expected_array[10:15], metadata=expected_metadata),
    ]

    result = concatenate(input_data)
    assert expected_input.metadata == result.metadata
    assert np.array_equal(expected_input, result)


def test_pipeline(data, pipeline):
    expected = []
    for i in range(8):
        d = InputData(np.random.uniform(size=5), {'n': 5})
        pipeline.send(d)
        expected.append(d[np.newaxis])
    expected = concatenate(expected, axis=-1).reshape((-1, 10))

    assert len(data) == 1
    assert data[0].shape == (4, 10)
    assert np.array_equal(expected, data[0])
    assert data[0].metadata == {'n': 5, 'block_size': 10}


def test_pipeline_flush(data, pipeline):
    expected = []
    for i in range(16):
        d = InputData(np.random.uniform(size=5), {'n': 5})
        pipeline.send(d)
        expected.append(d[np.newaxis])
        if i == 6:
            pipeline.send(Ellipsis)
            expected = []

    expected = concatenate(expected[:8], axis=-1).reshape((-1, 10))
    assert len(data) == 1
    assert data[0].shape == (4, 10)
    assert np.array_equal(expected, data[0])
    assert data[0].metadata == {'n': 5, 'block_size': 10}


def test_slice():
    r = np.random.uniform(size=100)
    d = InputData(r, {'t0_sample': 10, 'fs': 100})
    assert d[::2].metadata == {'t0_sample': 10, 'fs': 50}
    assert d[::4].metadata == {'t0_sample': 10, 'fs': 25}
    assert d[:4:].metadata == {'t0_sample': 10, 'fs': 100}
    assert d[1::2].metadata == {'t0_sample': 11, 'fs': 50}
    assert d[10::2].metadata == {'t0_sample': 20, 'fs': 50}
    assert d[10] == r[10]
    assert np.all(d[:10] == r[:10])
    assert d[-10::2].metadata == {'t0_sample': 100, 'fs': 50}
    assert d[-20::2].metadata == {'t0_sample': 90, 'fs': 50}
    assert d[-20::].metadata == {'t0_sample': 90, 'fs': 100}
    assert d[-20::-2].metadata == {'t0_sample': 90, 'fs': -50}


def test_create_diff_matrix():
    m = create_diff_matrix(4, [1, 2, 3])

    n_samples = 1000
    t = np.arange(n_samples) / n_samples
    s0 = np.sin(2 * np.pi * 10 * t)
    s1 = np.zeros_like(s0)
    s2 = np.zeros_like(s0)
    s3 = np.zeros_like(s0)
    noise = np.random.uniform(size=1000)

    data_clean = np.vstack((s0, s1, s2, s3))
    data_noisy = data_clean + noise
    data_cleaned = m @ data_noisy
    np.testing.assert_array_almost_equal(data_clean, data_cleaned)


def test_create_diff_matrix():
    m = create_diff_matrix(4, [1, 2, 3])

    n_samples = 1000
    t = np.arange(n_samples) / n_samples
    s0 = np.sin(2 * np.pi * 10 * t)
    s1 = np.zeros_like(s0)
    s2 = np.zeros_like(s0)
    s3 = np.zeros_like(s0)
    noise = np.random.uniform(size=1000)

    data_clean = np.vstack((s0, s1, s2, s3))
    data_noisy = data_clean + noise
    data_cleaned = m @ data_noisy
    np.testing.assert_array_almost_equal(data_clean, data_cleaned)
