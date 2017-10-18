import numpy as np
import glob
import os
from gunpowder import *
from gunpowder.tensorflow import Train, Predict
import tensorflow as tf
from .provider_test import ProviderTest
import multiprocessing
import time

register_volume_type('A')
register_volume_type('B')
register_volume_type('C')

class TestTensorflowTrainSource(BatchProvider):

    def setup(self):

        spec = VolumeSpec(
            roi=Roi((0, 0), (2, 2)),
            dtype=np.float32,
            interpolatable=True,
            voxel_size=(1, 1))
        self.provides(VolumeTypes.A, spec)
        self.provides(VolumeTypes.B, spec)

    def provide(self, request):

        batch = Batch()

        spec = self.spec[VolumeTypes.A]
        spec.roi = request[VolumeTypes.A].roi

        batch.volumes[VolumeTypes.A] = Volume(
            np.array([[0, 1], [2, 3]], dtype=np.float32),
            spec)

        spec = self.spec[VolumeTypes.B]
        spec.roi = request[VolumeTypes.B].roi

        batch.volumes[VolumeTypes.B] = Volume(
            np.array([[0, 1], [2, 3]], dtype=np.float32),
            spec)

        return batch

class TestTensorflowParallel(ProviderTest):

    def create_meta_graph(self):

        # create a tf graph
        a = tf.placeholder(tf.float32, shape=(2, 2))
        b = tf.placeholder(tf.float32, shape=(2, 2))
        v = tf.Variable(1, dtype=tf.float32)
        c = a*b*v

        # dummy "loss"
        loss = tf.norm(c)

        # dummy optimizer
        opt = tf.train.AdamOptimizer()
        optimizer = opt.minimize(loss)

        tf.train.export_meta_graph(filename='tf_graph.meta')

        return [x.name for x in [a, b, c, optimizer, loss]]

    def test_output(self):

        set_verbose(True)

        # start clean
        for filename in glob.glob('tf_graph.*'):
            os.remove(filename)
        for filename in glob.glob('tf_graph_checkpoint_*'):
            os.remove(filename)
        try:
            os.remove('checkpoint')
        except:
            pass

        # create model meta graph file and get input/output names
        (a, b, c, optimizer, loss) = self.create_meta_graph()

        # train something
        source = TestTensorflowTrainSource()
        train = Train(
            'tf_graph',
            optimizer=optimizer,
            loss=loss,
            inputs={a: VolumeTypes.A, b: VolumeTypes.B},
            outputs={c: VolumeTypes.C},
            gradients={},
            save_every=1)
        pipeline = source + train

        request = BatchRequest({
            VolumeTypes.A: VolumeSpec(roi=Roi((0, 0), (2, 2))),
            VolumeTypes.B: VolumeSpec(roi=Roi((0, 0), (2, 2))),
            VolumeTypes.C: VolumeSpec(roi=Roi((0, 0), (2, 2)))
        })

        # train for a couple of iterations
        with build(pipeline):
            batch = pipeline.request_batch(request)

        # predict
        source = TestTensorflowTrainSource()
        predict = Predict(
            'tf_graph_checkpoint_1',
            inputs={a: VolumeTypes.A, b: VolumeTypes.B},
            outputs={c: VolumeTypes.C})
        pipeline = source + predict

        with build(pipeline):

            predict1 = multiprocessing.Process(target=self.predict, args=(pipeline,))
            predict2 = multiprocessing.Process(target=self.predict, args=(pipeline,))

            predict1.start()
            time.sleep(1)
            predict2.start()

            predict1.join()
            predict2.join()

    def predict(self, pipeline):

        request = BatchRequest({
            VolumeTypes.A: VolumeSpec(roi=Roi((0, 0), (2, 2))),
            VolumeTypes.B: VolumeSpec(roi=Roi((0, 0), (2, 2))),
            VolumeTypes.C: VolumeSpec(roi=Roi((0, 0), (2, 2))),
        })

        print("PredictWorker: Requesting 100 batches...")
        for i in range(100):
            batch = pipeline.request_batch(request)
        print("PredictWorker: Done.")
