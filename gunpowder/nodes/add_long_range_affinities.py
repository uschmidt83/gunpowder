import copy
import logging
import numpy as np
import pdb

from gunpowder.volume import Volume, VolumeTypes
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class AddLongRangeAffinities(BatchFilter):


    def __init__(self, affinity_vectors, volume_type_1=None, volume_type_2=None,
        affinity_volume_type_1=None, affinity_volume_type_2=None, output_with_IDs = False):

        self.volume_type_1 = volume_type_1
        self.volume_type_2 = volume_type_2
        self.affinity_volume_type_1 = affinity_volume_type_1
        self.affinity_volume_type_2 = affinity_volume_type_2
        self.affinity_vectors = affinity_vectors
        self.output_with_IDs = output_with_IDs

        if presyn_volume is None:
            self.presyn_volume = VolumeTypes.PRESYN_BLOBS
        if postsyn_volume is None:
            self.postsyn_volume = VolumeTypes.POSTSYN_BLOBS
        if affinity_presyn_volume is None:
            self.affinity_presyn_volume = VolumeTypes.PRE_LR_AFFINITIES
        if affinity_postsyn_volume is None:
            self.affinity_postsyn_volume = VolumeTypes.POST_LR_AFFINITIES

        self.skip_next = False


    def setup(self):
        assert self.presyn_volume in self.spec, "Upstream does not provide %s needed by \
        AddGtAffinities"%self.presyn_volume
        assert self.postsyn_volume in self.spec, "Upstream does not provide %s needed by \
        AddGtAffinities"%self.postsyn_volume

        voxel_size = self.spec[self.presyn_volume].voxel_size

        self.upstream_spec = self.get_upstream_provider().spec
        self.upstream_roi = self.upstream_spec.get_total_roi()

        # get maximum offset in each dimension
        padding_arr = np.max(np.abs(self.affinity_neighborhood), axis=0)
        self.padding = tuple(round_up_to_voxel_size(padding_arr, voxel_size))

        logger.debug("padding: %s" %np.asarray(self.padding))

        # shrink provided roi according to maximum affinity vector offset
        output_spec = self.spec[self.presyn_volume].copy()
        if output_spec.roi is not None:
            output_spec.roi = output_spec.roi.grow(tuple(-padding_arr),tuple(-padding_arr))

        self.provides(self.affinity_presyn_volume, output_spec)
        self.provides(self.affinity_postsyn_volume, output_spec)

    def prepare(self, request):

        # do nothing if no gt affinities were requested
        if not (self.affinity_presyn_volume in request and self.affinity_postsyn_volume in request):
            logger.warn("no affinites requested, will do nothing")
            self.skip_next = True
            return

        requested_roi_1 = request[self.affinity_presyn_volume].roi
        requested_roi_2 = request[self.affinity_postsyn_volume].roi

        del request[self.affinity_presyn_volume]
        del request[self.affinity_postsyn_volume]

        needed_roi_volume_1 = requested_roi_1.grow(self.padding, self.padding)
        needed_roi_volume_2 = requested_roi_2.grow(self.padding, self.padding)

        if not self.presyn_volume in request:
            logger.debug("Adding downstream request for necessary volume %s"%self.presyn_volume)
            request.add(self.presyn_volume, needed_roi_volume_1, voxel_size=(40,4,4))
        else:
            logger.debug("downstream %s request: "%self.presyn_volume +\
             str(request[self.presyn_volume].roi))

            request[self.presyn_volume].roi = \
             request[self.presyn_volume].roi.union(needed_roi_volume_1)


        if not self.postsyn_volume in request:
            logger.debug("Adding downstream request for necessary volume %s"%self.postsyn_volume)
            request.add(self.postsyn_volume, needed_roi_volume_2, voxel_size=(40,4,4))
        else:
            logger.debug("downstream %s request: "%self.postsyn_volume +\
                 str(request[self.postsyn_volume].roi))

            request[self.postsyn_volume].roi = \
                request[self.postsyn_volume].roi.union(needed_roi_volume_2)


        logger.debug("upstream %s request: "%self.presyn_volume +\
             str(request[self.presyn_volume].roi))

        logger.debug("upstream %s request: "%self.postsyn_volume +\
            str(request[self.postsyn_volume].roi))

    def process(self, batch, request):

        if not self.skip_next:
            full_vol1 = batch.volumes[self.presyn_volume]
            full_vol2 = batch.volumes[self.postsyn_volume]

            # Both full_vol1 should match
            assert full_vol1.spec.dtype == full_vol2.spec.dtype,\
            "data type of volume 1(%s) and volume 2(%s) should match"%\
            (full_vol1.spec.dtype, full_vol2.spec.dtype)

            assert full_vol1.spec.voxel_size == full_vol2.spec.voxel_size,\
            "data type of volume 1(%s) and volume 2(%s) should match"%\
            (full_vol1.spec.voxel_size,full_vol2.spec.voxel_size)

            # do nothing if no gt affinities were requested
            if self.skip_next:
                self.skip_next = False
                return

            logger.debug("computing ground-truth affinities from labels")

            # Calculate affinities 1: from vol2 onto vol1

            # Initialize affinity map
            request_vol = request[self.affinity_presyn_volume]
            affinity_map = np.zeros(
                (len(self.affinity_neighborhood),) +
                tuple(request_vol.roi.get_shape()/request_vol.voxel_size),
                 dtype=full_vol1.spec.dtype)

            # calculate affinities vol 1
            vol1 = full_vol1.crop(request_vol.roi)
            for i, vector in enumerate(self.affinity_neighborhood):
                vol2 = full_vol2.crop(request_vol.roi.shift(tuple(-vector)))
                affinity_map[i,:,:,:] = np.bitwise_and(vol1.data, vol2.data)

            if not self.output_with_IDs:
                affinity_map[np.where(affinity_map != 0)] = 1

            batch.volumes[self.affinity_presyn_volume] = Volume(affinity_map,
                spec=request[self.affinity_presyn_volume].copy())

            batch.volumes[self.affinity_presyn_volume].attrs['affinity_neighborhood'] =\
             self.affinity_neighborhood

            # Calculate affinities 2: from vol1 onto vol2

            # Initialize affinity map
            request_vol = request[self.affinity_postsyn_volume]
            affinity_map = np.zeros(
                (len(self.affinity_neighborhood),) +
                tuple(request_vol.roi.get_shape()/request_vol.voxel_size),
                 dtype=full_vol1.spec.dtype)

            # calculate affinities vol 2
            vol2 = full_vol2.crop(request_vol.roi)
            for i, vector in enumerate(self.affinity_neighborhood):
                vol1 = full_vol1.crop(request_vol.roi.shift(tuple(vector)))
                affinity_map[i,:,:,:] = np.bitwise_and(vol1.data, vol2.data)

            if not self.output_with_IDs:
                affinity_map[np.where(affinity_map != 0)] = 1

            batch.volumes[self.affinity_postsyn_volume] = Volume(affinity_map,
                spec=request[self.affinity_postsyn_volume].copy())

            batch.volumes[self.affinity_postsyn_volume].attrs['affinity_neighborhood'] =\
             self.affinity_neighborhood

        # Crop all other requests
        for volume_type, volume in request.volume_specs.items():
            batch.volumes[volume_type] = batch.volumes[volume_type].crop(volume.roi)


def round_up_to_voxel_size(shape, voxel_size):
    voxel_size = np.asarray(voxel_size, dtype=float)
    shape = np.ceil(shape/voxel_size)*voxel_size
    return np.array(shape, dtype='int32')



