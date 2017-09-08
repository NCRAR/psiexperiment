import logging
log = logging.getLogger(__name__)

import numpy as np

from atom.api import Unicode, Enum, Typed, Tuple, Property
from enaml.core.api import Declarative, d_

from .calibration import Calibration
from .output import ContinuousOutput, EpochOutput
from ..util import coroutine


class Channel(Declarative):

    label = d_(Unicode()).tag(metadata=True)

    # Device-specific channel identifier.
    channel = d_(Unicode()).tag(metadata=True)

    # For software-timed channels, set sampling frequency to 0.
    fs = d_(Typed(object)).tag(metadata=True)

    # Can be blank for no start trigger (i.e., acquisition begins as soon as
    # task begins)
    start_trigger = d_(Unicode()).tag(metadata=True)

    # Used to properly configure data storage.
    dtype = d_(Typed(np.dtype))

    engine = Property()

    calibration = d_(Typed(Calibration)).tag(metadata=True)

    def _get_engine(self):
        return self.parent

    def _set_engine(self, engine):
        self.set_parent(engine)

    def configure(self, plugin):
        pass


class InputChannel(Channel):

    inputs = Property().tag(transient=True)

    def _get_inputs(self):
        return self.children

    def configure(self, plugin):
        for input in self.inputs:
            log.debug('Configuring input {}'.format(input.name))
            input.configure(plugin)


class OutputChannel(Channel):

    outputs = Property().tag(transient=True)

    def _get_outputs(self):
        return self.children

    def configure(self, plugin):
        for output in self.outputs:
            log.debug('Configuring output {}'.format(output.name))
            output.configure(plugin)


class AIChannel(InputChannel):

    TERMINAL_MODES = 'pseudodifferential', 'differential', 'RSE', 'NRSE'
    expected_range = d_(Tuple()).tag(metadata=True)
    terminal_mode = d_(Enum(*TERMINAL_MODES)).tag(metadata=True)
    terminal_coupling = d_(Enum(None, 'AC', 'DC', 'ground')).tag(metadata=True)


@coroutine
def null_callback():
    offset = 0
    while True:
        event = (yield)
        with event.engine.lock:
            samples = event.engine.get_space_available(event.channel_name, offset)
            waveform = np.zeros(samples)
            event.engine.append_hw_ao(waveform)
            offset += samples


class AOChannel(OutputChannel):
    '''
    An analog output channel supports one continuous and multiple epoch
    outputs.
    '''
    TERMINAL_MODES = 'pseudodifferential', 'differential', 'RSE'

    epoch_outputs = Property()
    continuous_output = Property()

    expected_range = d_(Tuple()).tag(metadata=True)
    terminal_mode = d_(Enum(*TERMINAL_MODES)).tag(metadata=True)

    def configure(self, plugin):
        super().configure(plugin)
        if self.continuous_output is None:
            cb = null_callback()
            self.engine.register_ao_callback(cb.send, self.name)

    def _get_continuous_output(self):
        for o in self.outputs:
            if isinstance(o, ContinuousOutput):
                return o
        return None

    def _get_epoch_outputs(self):
        return [o for o in self.outputs if isinstance(o, EpochOutput)]


class DIChannel(InputChannel):
    pass


class DOChannel(Channel):
    pass
