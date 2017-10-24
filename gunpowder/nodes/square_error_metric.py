import copy
import logging
import numpy as np
import pdb

from gunpowder.volume import Volume, VolumeTypes
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class SquaredErrorMetric(BatchFilter):

    def __init__(self, pred_volume, GT_volume, metric_volume):
        self.pred_volume = pred_volume
        self.GT_volume = GT_volume
        self.metric_volume = metric_volume
        self.skip_next = False
        self.cummulative_mean_error = []

    def setup(self):
        assert self.pred_volume in self.spec, "Upstream does not provide prediction volume %s"%self.pred_volume
        assert self.GT_volume in self.spec, "Upstream does not provide groud truth volume %s"%self.GT_volume

        self.provides(self.metric_volume, self.spec[self.GT_volume].copy())

    def prepare(self, request):

        if not self.metric_volume in request:
            logger.warn("no metric requested, will do nothing")
            self.skip_next = True
            return

        output_roi = request[self.metric_volume].roi

        if self.pred_volume in request:
            request[self.pred_volume].roi = output_roi.union(request[self.pred_volume].roi)
        else:
            request[self.pred_volume].roi = output_roi.copy()

        if self.GT_volume in request:
            request[self.GT_volume].roi = output_roi.union(request[self.GT_volume].roi)
        else:
            request[self.GT_volume].roi = output_roi.copy()

        del request[self.metric_volume]


    def process(self, batch, request):
        # do nothing if no gt affinities were requested
        if self.skip_next:
            self.skip_next = False
            return

        pred = batch.volumes[self.pred_volume]
        GT = batch.volumes[self.GT_volume]

        sq_error_volume_data = np.square(pred.data-GT.data)
        mean_sq_error = np.mean(sq_error_volume_data)

        self.cummulative_mean_error.append(mean_sq_error)

        batch.volumes[self.metric_volume] = Volume(sq_error_volume_data, spec=request[self.metric_volume].copy())
        batch.volumes[self.metric_volume].attrs['mean_square_error'] = mean_sq_error

        logger.critical('Mean Square Error: %s'%(mean_sq_error))

        # Crop all other requests
        for volume_type, volume in request.volume_specs.items():
            batch.volumes[volume_type] = batch.volumes[volume_type].crop(volume.roi)

