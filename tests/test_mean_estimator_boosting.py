import numpy as np
import pytest

from aspire.reconstruction import MeanEstimator, WeightedVolumesEstimator
from aspire.source import ArrayImageSource, Simulation
from aspire.utils import Rotation, utest_tolerance
from aspire.volume import (
    AsymmetricVolume,
    CnSymmetricVolume,
    DnSymmetricVolume,
    OSymmetricVolume,
    TSymmetricVolume,
)

SEED = 23

RESOLUTION = [
    32,
    pytest.param(33, marks=pytest.mark.expensive),
]

DTYPE = [
    np.float32,
    pytest.param(np.float64, marks=pytest.mark.expensive),
]

# Symmetric volume parameters, (volume_type, symmetric_order).
VOL_PARAMS = [
    (AsymmetricVolume, None),
    (CnSymmetricVolume, 4),
    (CnSymmetricVolume, 5),
    (DnSymmetricVolume, 2),
    pytest.param((TSymmetricVolume, None), marks=pytest.mark.expensive),
    pytest.param((OSymmetricVolume, None), marks=pytest.mark.expensive),
]


# Fixtures.
@pytest.fixture(params=RESOLUTION, ids=lambda x: f"resolution={x}", scope="module")
def resolution(request):
    return request.param


@pytest.fixture(params=DTYPE, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(
    params=VOL_PARAMS, ids=lambda x: f"volume={x[0]}, order={x[1]}", scope="module"
)
def volume(request, resolution, dtype):
    Volume, order = request.param
    vol_kwargs = dict(
        L=resolution,
        C=1,
        seed=SEED,
        dtype=dtype,
    )
    if order:
        vol_kwargs["order"] = order

    return Volume(**vol_kwargs).generate()


@pytest.fixture(scope="module")
def source(volume):
    src = Simulation(
        n=200,
        vols=volume,
        offsets=0,
        amplitudes=1,
        seed=SEED,
        dtype=volume.dtype,
    )
    src = src.cache()  # precompute images

    return src


@pytest.fixture(scope="module")
def estimated_volume(source):
    estimator = MeanEstimator(source)
    estimated_volume = estimator.estimate()

    return estimated_volume


# Weighted volume fixture. Only tesing C1, C4, and C5.
@pytest.fixture(
    params=VOL_PARAMS[:3], ids=lambda x: f"volume={x[0]}, order={x[1]}", scope="module"
)
def weighted_volume(request, resolution, dtype):
    Volume, order = request.param
    vol_kwargs = dict(
        L=resolution,
        C=2,
        seed=SEED,
        dtype=dtype,
    )
    if order:
        vol_kwargs["order"] = order

    return Volume(**vol_kwargs).generate()


@pytest.fixture(scope="module")
def weighted_source(weighted_volume):
    src = Simulation(
        n=400,
        vols=weighted_volume,
        offsets=0,
        amplitudes=1,
        seed=SEED,
        dtype=weighted_volume.dtype,
    )

    return src


# MeanEstimator Tests.
def test_fsc(source, estimated_volume):
    """Compare estimated volume to source volume with FSC."""
    # Fourier Shell Correlation
    fsc_resolution, fsc = source.vols.fsc(estimated_volume, pixel_size=1, cutoff=0.5)

    # Check that resolution is less than 2.1 pixels.
    np.testing.assert_array_less(fsc_resolution, 2.1)

    # Check that second to last correlation value is high (>.90).
    np.testing.assert_array_less(0.90, fsc[0, -2])
