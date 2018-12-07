import logging

import daisy

from .scan import Scan
from gunpowder.batch import Batch


logger = logging.getLogger(__name__)

class DaisyScan(Scan):
    '''
    '''

    def __init__(self, reference, roi_mapping, num_workers=1, cache_size=50):

        logger.info("DaisyScan initializing connection to scheduler...")
        super().__init__(reference, num_workers, cache_size)
        self.roi_mapping = roi_mapping


    def setup(self):

        super().setup()
        self.daisy_sched = daisy.Actor()


    def provide(self, request):

        scan_spec = self.spec
        daisy_batch = Batch()

        while True:

            logger.info("DaisyScan acquire_block")
            block = self.daisy_sched.acquire_block()

            if block == daisy.Actor.END_OF_BLOCK:
                logger.info("END_OF_BLOCK received from Daisy. Exiting.")
                break;

            logger.info("Got block from daisy: {}".format(block))

            # note: ``roi_mapping`` is required from the user
            # because Daisy does not know which data stream is read or write
            for key, reference_spec in self.reference.items():
                if key not in self.roi_mapping:
                    logger.error(
                        "roi_mapping does not map stream %s to either 'read_roi' "
                        "or 'write_roi'", key)
                    raise RuntimeError()

                if self.roi_mapping[key] == "read_roi":
                    scan_spec[key].roi = block.read_roi

                elif self.roi_mapping[key] == "write_roi":
                    scan_spec[key].roi = block.write_roi

            # TODO: not sure how to handle this more robustly and not discarding
            # results from the previous loop
            daisy_batch = super().provide(request=[], dummy_request=scan_spec)

            self.daisy_sched.release_block(block, ret=0)

        return daisy_batch
