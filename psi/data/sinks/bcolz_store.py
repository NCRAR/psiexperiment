import logging
log = logging.getLogger(__name__)

import atexit
import os.path
import tempfile
import shutil

from atom.api import Unicode, Typed, List

import numpy as np
import bcolz

from psi.util import get_tagged_values
from .abstract_store import (AbstractStore, ContinuousDataChannel,
                             EpochDataChannel)


class BColzStore(AbstractStore):
    '''
    Simple class for storing acquired trial data in a HDF5 file. No analysis or
    further processing is done.
    '''
    base_path = Unicode()
    trial_log = Typed(object)
    event_log = Typed(object)
    temp_folders = List()

    def process_trials(self, results):
        names = self.trial_log.data.dtype.names
        columns = [] 
        for name in names:
            rows = [result[name] for result in results]
            columns.append(rows)
        self.trial_log.append(columns)
        self.trial_log.data.flush()

    def process_event(self, event, timestamp):
        self.event_log.append([timestamp, event])
        self.event_log.data.flush()

    def process_ai_continuous(self, name, data):
        if self._channels[name] is not None:
            self._channels[name].append(data)
            self._channels[name].data.flush()

    def process_ai_epochs(self, name, data):
        if self._channels[name] is not None:
            self._channels[name].append(data)
            self._channels[name].data.flush()

    def _get_filename(self, name, save=True):
        if save and (self.base_path != '<memory>'):
            filename = os.path.join(self.base_path, name)
        else:
            filename = tempfile.mkdtemp()
            self.temp_folders.append(filename)
            atexit.register(shutil.rmtree, filename)
        return filename

    def _create_trial_log(self, context_info):
        '''
        Create a table to hold the event log.
        '''
        filename = self._get_filename('trial_log')
        dtype = [(str(n), i.dtype) for n, i in context_info.items()]
        return bcolz.zeros(0, rootdir=filename, mode='w', dtype=dtype)

    def _create_event_log(self):
        '''
        Create a table to hold the event log.
        '''
        filename = self._get_filename('event_log')
        dtype = [('timestamp', 'float32'), ('event', 'S512')]
        return bcolz.zeros(0, rootdir=filename, mode='w', dtype=dtype)

    def create_ai_continuous(self, name, fs, dtype, save, **metadata):
        n = int(fs*60*60)
        filename = self._get_filename(name, save)
        carray = bcolz.carray([], rootdir=filename, mode='w', dtype=dtype,
                              expectedlen=n)
        carray.attrs['fs'] = fs
        for key, value in metadata.items():
            # TODO: hack alert. Need to find a way around this
            if key == 'channel_calibration':
                continue
            try:
                carray.attrs[key] = value
                print(key)
            except TypeError as e:
                print(e)
                m = 'Unable to save {} with value {} to {}'
                log.warn(m.format(key, value, name))
        self._channels[name] = ContinuousDataChannel(data=carray, fs=fs)

    def create_ai_epochs(self, name, fs, epoch_size, dtype, save, **metadata):
        filename = self._get_filename(name, save)
        epoch_samples = int(fs*epoch_size)
        base = np.empty((0, epoch_samples))
        carray = bcolz.carray(base, rootdir=filename, mode='w', dtype=dtype)
        carray.attrs[fs] = fs
        for key, value in metadata.items():
            carray.attrs[key] = value
        self._channels[name] = EpochDataChannel(data=carray, fs=fs)

    def finalize(self, workbench):
        if self.base_path != '<memory>':
            # Save the settings file
            cmd = 'psi.save_preferences'
            filename = os.path.join(self.base_path, 'final')
            params = {'filename': filename}
            core = workbench.get_plugin('enaml.workbench.core')
            core.invoke_command(cmd, params)

    def set_base_path(self, base_path):
        self.base_path = base_path
