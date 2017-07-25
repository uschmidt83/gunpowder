import logging
import multiprocessing
import numpy as np
import time

from gunpowder.caffe.net_io_wrapper import NetIoWrapper
from gunpowder.ext import caffe
from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.producer_pool import ProducerPool, WorkersDied
from gunpowder.roi import Roi
from gunpowder.volume import VolumeTypes, Volume, VolumeType

logger = logging.getLogger(__name__)

class TrainProcessDied(Exception):
    pass

class Train(BatchFilter):
    '''Performs one training iteration for each batch that passes through. 
    Adds the predicted affinities to the batch.

    Args:

        solver_parameters (:class:``SolverParameters``): Parameters of the 
            solver to use for training, contains the network description as 
            well.

        inputs (dict): Dictionary from :class:``VolumeType`` or batch attribute name as string
            to the names of input layers in the network.

        outputs (dict): Dictionary from :class:``VolumeType`` to the names of 
            output layers in the network. New volumes will be generated by this 
            node for each entry (if requested downstream). Set the resolution of 
            the new volume via parameter ``output_resolutions``.

        gradients (dict): Dictionary from :class:``VolumeType`` to the names of 
            output layers in the network. New volumes containing the gradient of 
            an output with respect to the loss will be generated by this node 
            for each entry (if requested downstream). Set the resolution of the 
            new volume via parameter ``output_resolutions``.

        output_resolutions (dict): Dictionary from :class:``VolumeType`` to 
            :class:``Coordinate``. This sets the resolutions of volumes created 
            by this node.

        use_gpu (int): Which GPU to use. Set to ``None`` for CPU mode.
    '''

    def __init__(self, solver_parameters, inputs, outputs, gradients, output_resolutions, use_gpu=None):

        # start training as a producer pool, so that we can gracefully exit if
        # anything goes wrong
        self.worker = ProducerPool([lambda gpu=use_gpu: self.__train(gpu)], queue_size=1)
        self.batch_in = multiprocessing.Queue(maxsize=1)

        self.solver_parameters = solver_parameters
        self.solver_initialized = False

        self.inputs    = inputs
        self.outputs   = outputs
        self.gradients = gradients
        self.output_resolutions = output_resolutions

        self.provides = self.outputs.keys() + self.gradients.keys()

    def setup(self):
        self.worker.start()

    def teardown(self):
        self.worker.stop()

    def prepare(self, request):

        # remove request parts that we provide
        for volume_type in self.provides:
            if volume_type in request.volumes:
                del request.volumes[volume_type]

    def process(self, batch, request):

        self.batch_in.put((batch,request))

        try:
            out = self.worker.get()
        except WorkersDied:
            raise TrainProcessDied()

        for volume_type in self.provides:
            if volume_type in request.volumes:
                batch.volumes[volume_type] = out.volumes[volume_type]
                batch.volumes[volume_type].roi = request.volumes[volume_type]

        batch.loss = out.loss
        batch.iteration = out.iteration

    def __train(self, use_gpu):

        start = time.time()

        if not self.solver_initialized:

            logger.info("Initializing solver...")

            if use_gpu is not None:

                logger.debug("Train process: using GPU %d"%use_gpu)
                caffe.enumerate_devices(False)
                caffe.set_devices((use_gpu,))
                caffe.set_mode_gpu()
                caffe.select_device(use_gpu, False)

            self.solver = caffe.get_solver(self.solver_parameters)
            if self.solver_parameters.resume_from is not None:
                logger.debug("Train process: restoring solver state from " + self.solver_parameters.resume_from)
                self.solver.restore(self.solver_parameters.resume_from)

            names_net_outputs = self.outputs.values() + self.gradients.values()
            self.net_io = NetIoWrapper(self.solver.net, names_net_outputs)

            self.solver_initialized = True

        batch, request = self.batch_in.get()

        data = {}
        for network_input, input_name in self.inputs.items():
            if isinstance(network_input, VolumeType):
                data[input_name] = batch.volumes[network_input].data
            elif isinstance(network_input, str):
                data[input_name] = getattr(batch, network_input)
            else:
                raise Exception("Unknown network input type {}, can't be given to network".format(network_input))
        self.net_io.set_inputs(data)

        loss = self.solver.step(1)
        # self.__consistency_check()
        output = self.net_io.get_outputs()

        for volume_type, output_name in self.outputs.items():
            batch.volumes[volume_type] = Volume(
                    data=output[output_name][0], # strip #batch dimension
                    roi=Roi(), # dummy roi, will be corrected in process()
                    resolution=self.output_resolutions[volume_type])

        if len(self.gradients) > 0:

            diffs = self.net_io.get_output_diffs()

            for volume_type, output_name in self.gradients.items():
                batch.volumes[volume_type] = Volume(
                        data=diffs[output_name][0], # strip #batch dimension
                        roi=Roi(), # dummy roi, will be corrected in process()
                        resolution=self.output_resolutions[volume_type])

        batch.loss = loss
        batch.iteration = self.solver.iter

        time_of_iteration = time.time() - start
        logger.info("Train process: iteration=%d loss=%f time=%f"%(self.solver.iter,batch.loss,time_of_iteration))

        return batch

    def __consistency_check(self):

        diffs = self.net_io.get_output_diffs()
        for k in diffs:
            assert not np.isnan(diffs[k]).any(), "Detected NaN in output diff " + k
