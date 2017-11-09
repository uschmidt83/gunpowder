import copy
import logging
import numpy as np
import pdb

from gunpowder.volume import Volume, VolumeTypes
from gunpowder.points_spec import PointsSpec
from gunpowder.points import *
from scipy.ndimage.interpolation import shift


from .batch_filter import BatchFilter
from scipy import ndimage
import collections

logger = logging.getLogger(__name__)


class PredictSynapticPairs(BatchFilter):

    def __init__(self, affinity_pre_pred_volume, affinity_post_pred_volume, segmentation_volume, pred_presyn_points_name,
     pred_postsyn_points_name, affinity_vectors, threshold = 500):

        self.affinity_pre_pred_volume = affinity_pre_pred_volume
        self.affinity_post_pred_volume = affinity_post_pred_volume
        self.segmentation_volume = segmentation_volume
        self.pred_presyn_points_name = pred_presyn_points_name
        self.pred_postsyn_points_name = pred_postsyn_points_name
        self.affinity_vectors = affinity_vectors
        self.threshold = threshold
        self.skip_next = False


    def setup(self):

        logger.debug("predict setup start" )


        assert self.affinity_pre_pred_volume  in self.spec, "Upstream does not provide %s needed by \
        AddGtAffinities"%self.affinity_pre_pred_volume

        assert self.affinity_post_pred_volume  in self.spec, "Upstream does not provide %s needed by \
        AddGtAffinities"%self.affinity_post_pred_volume

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

        self.provides(self.pred_presyn_points_name, spec)
        self.provides(self.pred_postsyn_points_name, spec)

    def prepare(self, request):

        # do nothing if no gt affinities were requested
        # if not (self.pred_presyn_points_name in request and self.pred_postsyn_points_name in request):
        #     logger.warn("no affinites requested, will do nothing")
        #     self.skip_next = True
        #     return
        logger.debug("predict prepare start" )

        output_points_request_roi = (request[self.pred_presyn_points_name].roi).union(
            request[self.pred_postsyn_points_name].roi)

        logger.debug("downstream points request: " + str(output_points_request_roi))

         # Requested extended label roi
        input_request_roi_needed = output_points_request_roi.grow(self.padding, self.padding)

        logger.debug("upstream valoumes needed: " + str(input_request_roi_needed))

        # pdb.set_trace()


        # add required volumes with correct roi
        if self.affinity_pre_pred_volume in request:
            request[self.affinity_pre_pred_volume].roi = request[self.affinity_pre_pred_volume].roi.union(
                output_points_request_roi)
        else:
            request.add(self.affinity_pre_pred_volume, output_points_request_roi)

        # add required volumes with correct roi
        if self.affinity_post_pred_volume in request:
            request[self.affinity_post_pred_volume].roi = request[self.affinity_post_pred_volume].roi.union(
                output_points_request_roi)
        else:
            request.add(self.affinity_post_pred_volume, output_points_request_roi)

        if self.segmentation_volume in request:
            request[self.segmentation_volume].roi = request[self.segmentation_volume].roi.union(
                input_request_roi_needed)
        else:
            request.add(self.segmentation_volume, input_request_roi_needed)

        del request[self.pred_presyn_points_name]
        del request[self.pred_postsyn_points_name]



    def process(self, batch, request):

        # if self.skip_next:
        #         self.skip_next = False
        #         return

        # get intersection

        pred_pre_vol = batch.volumes[self.affinity_pre_pred_volume].data
        pred_post_vol = batch.volumes[self.affinity_post_pred_volume].data
        output_points_request_roi = request[self.pred_presyn_points_name].roi

        assert pred_pre_vol.shape == pred_post_vol.shape, \
        "Pre and Post predicted affinity maps must have the same shape."

        seg_vol = batch.volumes[self.segmentation_volume].data
        voxel_size = self.spec[self.segmentation_volume].voxel_size

        str_3D=np.array( [[ [0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]],

                           [[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]],

                           [[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]]], dtype='uint8')


        # for each affinity map
        potential_pairs = {}

        # For each affinity vector
        for aff_vec_i in range(pred_pre_vol.shape[0]):
            aff_vec = self.affinity_vectors[aff_vec_i]/voxel_size

            aff_vec_pre_prob = pred_pre_vol[aff_vec_i,:,:,:]
            aff_vec_post_prob = pred_post_vol[aff_vec_i,:,:,:]

            # Calculate intersection

            ## TODO SHIFTING
            # pdb.set_trace()
            # might be +aff_vec
            shifted_aff_vec_post_prob = shift(aff_vec_post_prob, -aff_vec)
            aff_vec_vol_prob = aff_vec_pre_prob*shifted_aff_vec_post_prob

            aff_vec_vol = (aff_vec_vol_prob > 0.25)*1.0

            logger.debug('Vector Hits: {}'.format(np.sum(aff_vec_vol)))

            # Get all predicted blobs
            labels, label_count = ndimage.label(aff_vec_vol, str_3D)
            unique_labels, counts = np.unique(labels, return_counts=True)

            # If there are blobs (apart from the background)
            if len(unique_labels) > 1:
                unique_labels = unique_labels[1:]
                counts = counts[1:]

                # Get those with area larger than threhsold
                selected_labels = (unique_labels[counts >= self.threshold]).tolist()

                # If there any sufficiently large blobs
                if len(selected_labels) > 0:

                    # Get center of mass all blobs as potential pre_synaptic sites
                    pred_pre_syn_loc = np.asarray(ndimage.measurements.center_of_mass(aff_vec_vol_prob,
                        labels, selected_labels))

                    cummulative_probabilites = np.asarray(ndimage.measurements.sum(aff_vec_vol_prob,
                        labels, selected_labels))

                    # Predict Post synaptic blobs by shifting by affinity vector
                    pred_post_syn_loc = pred_pre_syn_loc - aff_vec

                    # For each potential synaptic pair
                    for pred_pre_loc, pred_post_loc, cum_prob in zip(pred_pre_syn_loc,
                        pred_post_syn_loc, cummulative_probabilites):

                        # Check in in bounds
                        index = tuple((np.round(pred_pre_syn_loc[0])).astype('int'))

                        if np.all(np.array([0,0,0]) <= index) and np.all(index <= seg_vol.shape):
                            pre_neuron_id = seg_vol[index]
                        else:
                            logger.debug('Pre out of bounds: {}'.format(index))
                            continue

                        index = tuple((np.round(pred_post_syn_loc[0])).astype('int'))

                        if np.all(np.array([0,0,0]) <= index) and np.all(index <= seg_vol.shape):
                            post_neuron_id = seg_vol[index]
                        else:
                            logger.debug('Post out of bounds: {}'.format(index))
                            continue

                        # If connecting same neuron, ignore
                        if pre_neuron_id == post_neuron_id:
                            logger.debug('Pre and Post match to same neuron: %s'%(post_neuron_id))
                            continue

                        # with key pre_neuron_id-post_neuron_id
                        key = '{}-{}'.format(pre_neuron_id, post_neuron_id)

                        if key in potential_pairs.keys():
                            # TODO remove this test after sure it's fine
                            if not (pre_neuron_id == potential_pairs[key]['pre']['neuron_id'] and \
                                post_neuron_id == potential_pairs[key]['post']['neuron_id']):
                                logger.debug('CRAZY: key:{} pre:{} post:{}'.format(key, pre_neuron_id, post_neuron_id))
                                pdb.set_trace()
                            else:
                                potential_pairs[key]['pre']['loc'].append(pred_pre_loc)
                                potential_pairs[key]['pre']['cum_prob'].append(cum_prob)

                                potential_pairs[key]['post']['loc'].append(pred_post_loc)
                                potential_pairs[key]['post']['cum_prob'].append(cum_prob)

                        else:
                            potential_pairs[key] = {
                            'pre': {
                                'loc':[pred_pre_loc],
                                'cum_prob':[cum_prob],
                                'neuron_id':pre_neuron_id
                                },
                            'post': {
                                'loc':[pred_post_loc],
                                'cum_prob':[cum_prob],
                                'neuron_id':post_neuron_id
                                },
                            }

        pred_presyn_data = {}
        pred_postsyn_data = {}

        local_synapse_id = 0
        local_location_id = 0

        pre_offset = request[self.pred_presyn_points_name].roi.get_offset()
        post_offset = request[self.pred_postsyn_points_name].roi.get_offset()


        for synapse_id, locations in potential_pairs.items():

            # For all pre synaptic locations connecting the same two neurons average their locations
            pre_loc = np.mean(locations['pre']['loc'],axis=0)
            pre_neuron_id = seg_vol[tuple(np.round(pre_loc).astype('int'))]

            # Check the average location is in the same neuron as the individual locations
            if pre_neuron_id != locations['pre']['neuron_id']:
                logger.warning('Pre average locations lies in different neuron than \
individual neuron_ids:{} average neuron_ids:{}'.format(
                    locations['pre']['neuron_id'], pre_neuron_id))
                pdb.set_trace()

            # For all post synaptic locations connecting the same two neurons average their locations
            post_loc = np.mean(locations['post']['loc'],axis=0)
            post_neuron_id = seg_vol[tuple(np.round(post_loc).astype('int'))]

            # Check the average location is in the same neuron as the individual locations
            if post_neuron_id != locations['post']['neuron_id']:
                logger.warning('Post average locations lies in different neuron than \
individual neuron_ids:{} average neuron_ids:{}'.format(
                    locations['pre']['neuron_id'], post_neuron_id))
                pdb.set_trace()

            # Calculare local ids
            local_synapse_id += 1
            synapse_id = local_synapse_id

            local_location_id += 1
            pre_loc_id = local_location_id

            local_location_id += 1
            post_loc_id = local_location_id

            # Create points
            pred_presyn_data[pre_loc_id] = PreSynPoint(location=pre_loc + pre_offset,
                location_id=pre_loc_id, synapse_id=synapse_id, partner_ids=[post_loc_id])

            pred_postsyn_data[post_loc_id] = PostSynPoint(location=post_loc + post_offset,
                location_id=post_loc_id, synapse_id=synapse_id, partner_ids=[pre_loc_id])


        pre_spec = PointsSpec(roi=request[self.pred_presyn_points_name].roi)
        batch.points[self.pred_presyn_points_name] = Points(pred_presyn_data, pre_spec)

        post_spec = PointsSpec(roi=request[self.pred_postsyn_points_name].roi)
        batch.points[self.pred_postsyn_points_name] = Points(pred_postsyn_data, post_spec)

        # pdb.set_trace()

        # Crop all other requests
        for volume_type, volume in request.volume_specs.items():
            batch.volumes[volume_type] = batch.volumes[volume_type].crop(volume.roi)

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



