import copy
import logging
import numpy as np

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array, ArrayKey
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

logger = logging.getLogger(__name__)


class NumpySource(BatchProvider):

    def __init__(self, array_dict, voxel_size=None, interpolatable=None):
        def keyify(d):
            try:
                return {ArrayKey(k):v for k,v in d.items()} if d is not None else None
            except AttributeError:
                return d
        self.array_dict = keyify(array_dict)
        self.voxel_size = keyify(voxel_size)
        self.interpolatable = keyify(interpolatable)


    def _guess_interpolatable(self,array_key, dtype):
        interpolatable = dtype in [
            np.float,
            np.float32,
            np.float64,
            np.float128,
            np.uint8 # assuming this is not used for labels
        ]
        logger.warning("WARNING: You didn't set 'interpolatable' for %s. "
                      "Based on the dtype %s, it has been set to %s. "
                      "This might not be what you want.",
                      array_key, dtype, interpolatable)
        return interpolatable


    def setup(self):
        for key, data in self.array_dict.items():
            if isinstance(self.voxel_size,(list,tuple)):
                voxel_size = Coordinate(self.voxel_size)
            else:
                try:
                    voxel_size = Coordinate(self.voxel_size[key])
                except (TypeError, KeyError):
                    voxel_size = Coordinate((1,)*data.ndim)
            assert len(voxel_size) == data.ndim

            if isinstance(self.interpolatable,(bool,int)):
                interpolatable = bool(self.interpolatable)
            else:
                try:
                    interpolatable = bool(self.interpolatable[key])
                except (TypeError, KeyError):
                    interpolatable = self._guess_interpolatable(key,data.dtype)

            self.provides(
                key,
                ArraySpec(
                    roi            = Roi((0,)*data.ndim, data.shape),
                    dtype          = data.dtype,
                    voxel_size     = voxel_size,
                    interpolatable = interpolatable,
                )
            )


    def provide(self, request):
        timing = Timing(self)
        timing.start()

        batch = Batch()
        for key, request_spec in request.array_specs.items():
            spec = self.spec[key].copy()
            # not checked by BatchProvider.check_request_consistency:
            assert request_spec.dtype is None or request_spec.dtype==spec.dtype, "dtype mismatch"
            # print(f"provider spec = {spec}")
            # print(f"request  spec = {request_spec}")
            array = Array(self.array_dict[key], spec)
            if request_spec.roi is not None and request_spec.roi != spec.roi:
                array = array.crop(request_spec.roi, copy=True)
            batch.arrays[key] = array

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch