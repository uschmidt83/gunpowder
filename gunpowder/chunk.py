from batch_filter import BatchFilter

class Chunk(BatchProvider):
    '''Assemble a large batch by requesting smaller chunks upstream.
    '''

    def __init__(self, chunk_spec):

        self.chunk_spec_template = chunk_spec
        self.dims = chunk_spec_template.input_roi.dims()

        assert chunk_spec.input_roi.get_offset() == (0,)*self.dims, "The chunk spec should not have an input offset, only input/output shape and optionally output offset (relative to input)."

    def request_batch(self, batch_spec):

        stride = self.chunk_spec_template.output_roi.get_shape()

        begin = batch_spec.input_roi.get_begin()
        end = batch_spec.input_roi.get_end()

        batch = Batch(batch_spec)
        batch.raw = np.zeros(batch_spec.input_roi.get_shape())
        # TODO: allocate other volumes, depending on spec
        offset = begin

        while offset < end:

            chunk_spec = BatchSpec(
                    self.chunk_spec_template.input_roi.get_shape(),
                    self.chunk_spec_template.output_roi.get_shape(),
                    offset,
                    self.chunk_spec_template.output_roi.get_offset() + offset,
            )

            chunk = self.get_upstream_provider().request_batch(chunk_spec)

            # TODO: copy chunk into batch

            for d in range(self.dims()):
                offset[d] += stride[d]
                if offset[d] >= end[d]:
                    offset[d] = begin[d]
                else:
                    break
