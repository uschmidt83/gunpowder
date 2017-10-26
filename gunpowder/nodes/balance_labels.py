from .batch_filter import BatchFilter
from gunpowder.volume import VolumeTypes, Volume
import collections
import itertools
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BalanceLabels(BatchFilter):
    '''Creates a scale volume to balance the loss between positive and negative
    labels.

    Args:

        labels (:class:``VolumeType``): A volume containing binary labels.

        scales (:class:``VolumeType``): A volume with scales to be created. This
            new volume will have the same ROI and resolution as `labels`.

        mask (:class:``VolumeType``, optional): An optional mask (or list of
            masks) to consider for balancing. Every voxel marked with a 0 will
            not contribute to the scaling and will have a scale of 0 in
            `scales`.

        slab (tuple of int, optional): A shape specification to perform the
            balancing in slabs of this size. -1 can be used to refer to the
            actual size of the label volume. For example, a slab of::

                (2, -1, -1, -1)

            will perform the balancing for every each slice `(0:2,:)`,
            `(2:4,:)`, ... individually.

        min_frac (:class:``float``, optional): A minimum required fraction for
            both pos and neg classes in each slab. If one of the classes in a slab
            is less than or equal to this fraction, the slab will be ignored (scale=0)
            Use frac 0 to make sure there is at least one positive example.
            Use frac < 0 to never ignore a slab.

    '''

    def __init__(self, labels, scales, mask=None, slab=None, min_frac=0):

        self.labels = labels
        self.scales = scales
        self.min_frac = min_frac

        if mask is None:
            self.masks = []
        elif not isinstance(mask, collections.Iterable):
            self.masks = [mask]
        else:
            self.masks = mask

        self.slab = slab

        self.skip_next = False

    def setup(self):

        assert self.labels in self.spec, (
            "Asked to balance labels %s, which are not provided."%self.labels)

        for mask in self.masks:
            assert mask in self.spec, (
                "Asked to apply mask %s to balance labels, but mask is not "
                "provided."%mask)

        spec = self.spec[self.labels].copy()
        spec.dtype = np.float32
        self.provides(self.scales, spec)

    def prepare(self, request):

        self.skip_next = True
        if self.scales in request:
            del request[self.scales]
            self.skip_next = False

    def process(self, batch, request):

        if self.skip_next:
            self.skip_next = False
            return

        labels = batch.volumes[self.labels]

        assert len(np.unique(labels.data)) <= 2, (
            "Found more than two labels in %s."%self.labels)
        assert np.min(labels.data) in [0.0, 1.0], (
            "Labels %s are not binary."%self.labels)
        assert np.max(labels.data) in [0.0, 1.0], (
            "Labels %s are not binary."%self.labels)

        # initialize error scale with 1s
        error_scale = np.ones(labels.data.shape, dtype=np.float32)

        # set error_scale to 0 in masked-out areas
        for identifier in self.masks:
            mask = batch.volumes[identifier]
            assert labels.data.shape == mask.data.shape, (
                "Shape of mask %s %s does not match %s %s"%(
                    mask,
                    mask.data.shape,
                    self.labels,
                    labels.data.shape))
            error_scale *= mask.data

        if not self.slab:
            slab = error_scale.shape
        else:
            # slab with -1 replaced by shape
            slab = tuple(
                m if s == -1 else s
                for m, s in zip(error_scale.shape, self.slab))

        slab_ranges = (
            range(0, m, s)
            for m, s in zip(error_scale.shape, slab))

        for start in itertools.product(*slab_ranges):
            slices = tuple(
                slice(start[d], start[d] + slab[d])
                for d in range(len(slab)))
            self.__balance(
                labels.data[slices],
                error_scale[slices],
                self.min_frac)

        spec = self.spec[self.scales].copy()
        spec.roi = labels.spec.roi
        batch.volumes[self.scales] = Volume(error_scale, spec)

    def __balance(self, labels, scale, min_frac):

        # in the masked-in area, compute the fraction of positive samples
        masked_in = scale.sum()
        num_pos  = (labels*scale).sum()
        frac_pos = float(num_pos) / masked_in if masked_in > 0 else 0
        frac_pos = np.clip(frac_pos, 0.05, 0.95)
        frac_neg = 1.0 - frac_pos

        # compute the class weights for positive and negative samples
        w_pos = 1.0 / (2.0 * frac_pos)
        w_neg = 1.0 / (2.0 * frac_neg)

        if frac_pos <= min_frac or frac_neg <= min_frac:
            # If not enough pos or neg examples, ignore slab
            scale *= 0
        else:
            # scale the masked-in scale with the class weights
            scale *= (labels >= 0.5) * w_pos + (labels < 0.5) * w_neg
