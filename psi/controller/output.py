import logging
log = logging.getLogger(__name__)

from types import GeneratorType
from functools import partial

import numpy as np

from atom.api import (Unicode, Enum, Typed, Property, Float, Int, Bool)

from enaml.core.api import Declarative, d_
from enaml.workbench.api import Extension

from ..util import coroutine
from .queue import AbstractSignalQueue

from psi.core.enaml.api import PSIContribution

import time


class Output(PSIContribution):

    visible = d_(Bool(False))

    target_name = d_(Unicode())

    channel = Property()
    target = Property()
    engine = Property()
    fs = Property()

    # TODO: clean this up. it's sort of hackish.
    token_name = d_(Unicode())
    token = Typed(Declarative)

    def _observe_parent(self, event):
        self.target_name = event['value'].name

    def _get_target(self):
        if isinstance(self.parent, Extension):
            return None
        return self.parent

    def _set_target(self, target):
        self.set_parent(target)

    def _get_engine(self):
        return self.channel.parent

    def _get_channel(self):
        from .channel import Channel
        parent = self.parent
        while True:
            if isinstance(parent, Channel):
                return parent
            else:
                parent = parent.parent

    def _get_fs(self):
        return self.channel.fs


class AnalogOutput(Output):

    # This determines how many samples will be requested from the generator on
    # each iteration. While we can request an arbitrary number of samples, this
    # helps some caching mechanisms better cache the results for future calls.
    block_size = d_(Float(2))
    visible = d_(Bool(True))

    _generator = Typed(GeneratorType)
    _offset = Int()
    _block_samples = Int()

    def configure(self, plugin):
        pass

    def initialize_factory(self, context):
        context = context.copy()
        context['fs'] = self.channel.fs
        context['calibration'] = self.channel.calibration
        return self.token.initialize_factory(context)

    def initialize_generator(self, context):
        factory = self.initialize_factory(context)
        generator = factory()
        next(generator)
        return generator


class EpochCallback(object):

    def __init__(self, output, generator):
        self.output = output
        self.engine = output.engine
        self.channel = output.channel
        self.active = False
        self.generator = generator

    def start(self, start, delay):
        self.offset = int((start+delay)*self.channel.fs)
        self.active = True
        next(self)

    def send(self, event):
        if not self.active:
            raise StopIteration
        with self.engine.lock:
            samples = self.engine.get_buffered_samples(self.channel.name,
                                                       self.offset)
            samples = min(self.output._block_samples, samples)
            if samples == 0:
                return
            waveform, complete = self.generator.send({'samples': samples})
            self.engine.modify_hw_ao(waveform, self.offset, self.output.name)
            self.offset += len(waveform)
            if complete:
                raise StopIteration

    def clear(self, end, delay):
        with self.engine.lock:
            offset = int((end+delay)*self.channel.fs)
            samples = self.engine.get_buffered_samples(self.channel.name,
                                                       offset)
            waveform = np.zeros(samples)
            self.engine.modify_hw_ao(waveform, offset, self.output.name)
            self.active = False


class EpochOutput(AnalogOutput):

    method = d_(Enum('merge', 'replace', 'multiply'))

    _cb = Typed(object)
    _duration = Typed(object)

    def setup(self, context):
        '''
        Set up the generator in preparation for producing the signal. This
        allows the generator to cache some potentially expensive computations
        in advance rather than just before we actually want the signal played.
        '''
        generator = self.initialize_generator(context)
        cb = EpochCallback(self, generator)
        self._cb = cb
        self._duration = self.token.get_duration(context)

    def start(self, start, delay):
        '''
        Actually start the generator. It must have been initialized first.
        '''
        if self._cb is None:
            m = '{} was not initialized'.format(self.name)
            raise SystemError(m)
        self._cb.start(start, delay)
        self.engine.register_ao_callback(self._cb.send, self.channel.name)
        return self._duration

    def clear(self, end, delay):
        self._cb.clear(end, delay)

    def configure(self, plugin):
        self._block_samples = int(self.channel.fs*self.block_size)


@coroutine
def queued_epoch_callback(output, queue, auto_decrement, complete_cb=None):
    offset = 0
    engine = output.engine
    channel = output.channel
    while True:
        event = (yield)
        samples = engine.get_buffered_samples(channel.name, offset)
        log.debug('Generating %d samples at %d from queue', samples, offset)
        start = time.time()
        waveform, empty = queue.pop_buffer(samples, decrement=auto_decrement)
        engine.modify_hw_ao(waveform, offset, output.name)
        offset += len(waveform)
        if empty:
            break
    if complete_cb is not None:
        complete_cb()


class QueuedEpochOutput(EpochOutput):

    selector_name = d_(Unicode())
    queue = d_(Typed(AbstractSignalQueue))
    auto_decrement = d_(Bool(False))

    def setup(self, context, complete_cb=None):
        log.debug('Setting up queue for {}'.format(self.name))
        for setting in context:
            averages = setting['{}_averages'.format(self.name)]
            iti_duration = setting['{}_iti_duration'.format(self.name)]
            token_duration = self.token.get_duration(setting)
            delay_duration = iti_duration-token_duration
            # Somewhat surprisingly it appears to be faster to use factories in
            # the queue rather than creating the waveforms for ABR tone pips,
            # even for very short signal durations.
            factory = self.initialize_factory(setting)
            self.queue.append(factory, averages, delay_duration, token_duration,
                              setting)
        cb = queued_epoch_callback(self, self.queue, self.auto_decrement,
                                   complete_cb)
        self.engine.register_ao_callback(cb.send, self.channel.name)


@coroutine
def continuous_callback(generator):
    while True:
        event = (yield)
        with event.engine.lock:
            samples = event.engine.get_space_available(event.channel_name)
            log.debug('Generating {} samples for {}'.format(samples, event.channel_name))
            waveform = generator.send({'samples': samples})
            event.engine.append_hw_ao(waveform)
            offset += len(waveform)


class ContinuousOutput(AnalogOutput):

    def setup(self, context):
        log.debug('Configuring continuous output {}'.format(self.name))
        generator = self.initialize_generator(context)
        cb = continuous_callback(generator)
        self.engine.register_ao_callback(cb.send, self.channel.name)


class DigitalOutput(Output):

    def configure(self, plugin):
        pass


class Trigger(DigitalOutput):

    duration = d_(Float(0.1))

    def fire(self):
        self.engine.fire_sw_do(self.channel.name, duration=self.duration)


class Toggle(DigitalOutput):

    state = Bool(False)

    def _observe_state(self, event):
        try:
            # TODO: Fixme
            self.engine.set_sw_do(self.channel.name, event['value'])
        except:
            pass

    def set_high(self):
        self.state = True

    def set_low(self):
        self.state = False
