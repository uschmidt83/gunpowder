import copy
import logging
import numpy as np
import pdb

from gunpowder.volume import Volume, VolumeTypes
from gunpowder.points_spec import PointsSpec

from .batch_filter import BatchFilter
from scipy import ndimage
import collections

logger = logging.getLogger(__name__)


class PredictSynapticPairs(BatchFilter):

    def __init__(self, affinity_pred_volume, segmentation_volume, pred_pre_syn_points,
     pred_post_syn_points, affinity_vectors, threshold = 500):

        self.affinity_pred_volume = affinity_pred_volume
        self.segmentation_volume = segmentation_volume
        self.pred_pre_syn_points = pred_pre_syn_points
        self.pred_post_syn_points = pred_post_syn_points
        self.affinity_vectors = affinity_vectors
        self.threshold = threshold
        self.skip_next = False


    def setup(self):

        logger.debug("predict setup start" )

        assert self.affinity_pred_volume in self.spec, "Upstream does not provide %s needed by \
        AddGtAffinities"%self.affinity_pred_volume
        assert self.segmentation_volume in self.spec, "Upstream does not provide %s needed by \
        AddGtAffinities"%self.segmentation_volume

        voxel_size = self.spec[self.segmentation_volume].voxel_size

        self.upstream_spec = self.get_upstream_provider().spec
        self.upstream_roi = self.upstream_spec.get_total_roi()

        segmentation_spec = self.spec[self.segmentation_volume].copy()

        self.padding = np.max(np.abs(self.affinity_vectors), axis=0)
        self.padding = tuple(round_up_to_voxel_size(self.padding, voxel_size))

        logger.debug("seg padding: %s" %np.asarray(self.padding))

        spec = PointsSpec()
        spec.roi = segmentation_spec.roi

        self.provides(self.pred_pre_syn_points, spec)
        self.provides(self.pred_post_syn_points, spec)

    def prepare(self, request):

        # do nothing if no gt affinities were requested
        # if not (self.pred_pre_syn_points in request and self.pred_post_syn_points in request):
        #     logger.warn("no affinites requested, will do nothing")
        #     self.skip_next = True
        #     return
        logger.debug("predict prepare start" )


        points_request_roi = (request[self.pred_pre_syn_points].roi).union(
            request[self.pred_post_syn_points].roi)

        logger.debug("downstream points request: " + str(points_request_roi))

        # add required volumes with correct roi
        if self.affinity_pred_volume in request:
            request[self.affinity_pred_volume].roi = request[self.affinity_pred_volume].roi.union(
                points_request_roi)
        else:
            request.add(self.affinity_pred_volume, points_request_roi)

        # Requested extended label roi
        points_request_roi = points_request_roi.grow(self.padding, self.padding)

        # pdb.set_trace()

        if self.segmentation_volume in request:
            request[self.segmentation_volume].roi = request[self.segmentation_volume].roi.union(
                points_request_roi)
        else:
            request.add(self.segmentation_volume, points_request_roi)

        # pdb.set_trace()

        del request[self.pred_pre_syn_points]
        del request[self.pred_post_syn_points]



    def process(self, batch, request):

        # if self.skip_next:
        #         self.skip_next = False
        #         return

        pred_vol = batch.volumes[self.affinity_pred_volume].data
        seg_vol = batch.volumes[self.segmentation_volume].data
        voxel_size = self.spec[self.segmentation_volume].voxel_size

        str_3D=np.array( [[[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]],

                       [[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]],

                       [[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]]], dtype='uint8')


        # for each affinity map
        for aff_vec_i in range(pred_vol.shape[0]):
            aff_vec = self.affinity_vectors[aff_vec_i]
            aff_vec_vol = pred_vol[aff_vec_i,:,:,:]
            aff_vec_vol = (aff_vec_vol > 0.5)*1.0

            logger.debug('Vector Hits: {}'.format(np.sum(aff_vec_vol)))

            labels, label_count = ndimage.label(aff_vec_vol, str_3D)
            unique_labels, counts = np.unique(labels, return_counts=True)

            if len(unique_labels) > 1:
                unique_labels = unique_labels[1:]
                counts = counts[1:]
                selected_labels = (unique_labels[counts >= self.threshold]).tolist()

                if len(selected_labels) > 0:
                    pred_pre_syn_loc = np.asarray(ndimage.measurements.center_of_mass(aff_vec_vol, labels, selected_labels))
                    pred_post_syn_loc = pred_pre_syn_loc - aff_vec

                    potential_pairs = {}

                    for pred_pre_loc, pred_post_loc in zip(pred_pre_syn_loc, pred_post_syn_loc):

                        index = tuple((np.round(pred_pre_syn_loc[0])).astype('int'))

                        if np.all(np.array([0,0,0]) <= index) and np.all(index <= seg_vol.shape):
                            pre_neuron_id = seg_vol[index]
                        else:
                            logger.debug('Pre out of bounds: %s'%(index))
                            continue

                        index = tuple((np.round(pred_post_syn_loc[0])).astype('int'))

                        if np.all(np.array([0,0,0]) <= index) and np.all(index <= seg_vol.shape):
                            post_neuron_id = seg_vol[index]
                        else:
                            logger.debug('Post out of bounds: %s'%(index))
                            continue

                        key = str(pre_neuron_id - post_neuron_id)
                        if key != '0':
                            if key in potential_pairs.keys():
                                potential_pairs[key]['pre'].append(pred_pre_loc)
                                potential_pairs[key]['post'].append(pred_post_loc)
                            else:
                                potential_pairs[key] = {'pre': [pred_pre_loc], 'post': [pred_post_loc]}
                        else:
                            logger.debug('Pre and Post match to same neuron: %s'%(post_neuron_id))
                            continue


                    pdb.set_trace()

                # ndimage.measurements.center_of_mass(b, lbl, [1,2])

                # for label, count in label_freq.items():
                #     logger.debug('label: {}, count {}'.format(label, count))
                #     pdb.set_trace()

                #     if count >= self.threshold:




        # Both full_vol1 should match
        # assert full_vol1.spec.dtype == full_vol2.spec.dtype,\
        # "data type of volume 1(%s) and volume 2(%s) should match"%\
        # (full_vol1.spec.dtype, full_vol2.spec.dtype)

        # assert full_vol1.spec.voxel_size == full_vol2.spec.voxel_size,\
        # "data type of volume 1(%s) and volume 2(%s) should match"%\
        # (full_vol1.spec.voxel_size,full_vol2.spec.voxel_size)

        # do nothing if no gt affinities were requested


        # Calculate affinities 1: from vol2 onto vol1

        # Initialize affinity map
    #     request_vol = request[self.affinity_volume_type_1]
    #     affinity_map = np.zeros(
    #         (len(self.affinity_vectors),) +
    #         tuple(request_vol.roi.get_shape()/request_vol.voxel_size), dtype=full_vol1.spec.dtype)

    #     # calculate affinities vol 1
    #     vol1 = full_vol1.crop(request_vol.roi)
    #     for i, vector in enumerate(self.affinity_vectors):
    #         vol2 = full_vol2.crop(request_vol.roi.shift(tuple(-vector)))
    #         affinity_map[i,:,:,:] = np.bitwise_and(vol1.data, vol2.data)

    #     if not self.output_with_IDs:
    #         affinity_map[np.where(affinity_map != 0)] = 1

    #     batch.volumes[self.affinity_volume_type_1] = Volume(affinity_map,
    #         spec=request[self.affinity_volume_type_1].copy())

    #     batch.volumes[self.affinity_volume_type_1].attrs['affinity_vectors'] =\
    #      self.affinity_vectors

    #     # Calculate affinities 2: from vol1 onto vol2

    #     # Initialize affinity map
    #     request_vol = request[self.affinity_volume_type_2]
    #     affinity_map = np.zeros(
    #         (len(self.affinity_vectors),) +
    #         tuple(request_vol.roi.get_shape()/request_vol.voxel_size), dtype=full_vol1.spec.dtype)

    #     # calculate affinities vol 2
    #     vol2 = full_vol2.crop(request_vol.roi)
    #     for i, vector in enumerate(self.affinity_vectors):
    #         vol1 = full_vol1.crop(request_vol.roi.shift(tuple(vector)))
    #         affinity_map[i,:,:,:] = np.bitwise_and(vol1.data, vol2.data)

    #     if not self.output_with_IDs:
    #         affinity_map[np.where(affinity_map != 0)] = 1

    #     batch.volumes[self.affinity_volume_type_2] = Volume(affinity_map,
    #         spec=request[self.affinity_volume_type_2].copy())

    #     batch.volumes[self.affinity_volume_type_2].attrs['affinity_vectors'] =\
    #      self.affinity_vectors

    # # Crop all other requests
    # for volume_type, volume in request.volume_specs.items():
    #     batch.volumes[volume_type] = batch.volumes[volume_type].crop(volume.roi)

    # for points_type, points in request.points_specs.items():
    #     recropped = batch.points[points_type].spec.roi = points.roi
    #     batch.points[points_type] = recropped


def round_up_to_voxel_size(shape, voxel_size):
    voxel_size = np.asarray(voxel_size, dtype=float)
    shape = np.ceil(shape/voxel_size)*voxel_size
    return np.array(shape, dtype='int32')



