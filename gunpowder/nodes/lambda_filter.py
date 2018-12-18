import logging
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class Lambda(BatchFilter):

    def __init__(self, process, prepare=None, setup=None, teardown=None):
        self._process = process
        self._prepare = prepare
        self._setup = setup
        self._teardown = teardown

    def process(self, batch, request):
        self._process(batch, request)

    def prepare(self, request):
        if self._prepare is not None:
            self._prepare(request)

    def setup(self):
        if self._setup is not None:
            self._setup()

    def teardown(self):
        if self._teardown is not None:
            self._teardown()