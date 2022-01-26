import os.path
import importlib
from pathlib import Path

from scipy.interpolate import interp1d
from scipy import signal
import numpy as np
import pandas as pd

import logging
log = logging.getLogger(__name__)

from atom.api import Atom, Callable, Dict, Float, Property, Typed, Value
from enaml.core.api import Declarative, d_
from psi import SimpleState

from . import util

from psi import get_config

################################################################################
# Exceptions
################################################################################
mesg = '''
Unable to run the calibration. Please double-check that the microphone and
speaker amplifiers are powered on and that the devices are positioned properly.
If you keep receiving this message, the microphone and/or the speaker may have
gone bad and need to be replaced.

{}
'''
mesg = mesg.strip()


thd_err_mesg = 'Total harmonic distortion for {:.1f}Hz is {:.1f}%'
nf_err_mesg = 'Power at {:.1f}Hz has SNR of {:.2f}dB'


class CalibrationError(Exception):

    def __str__(self):
        return self.message


class CalibrationTHDError(CalibrationError):

    def __init__(self, frequency, thd):
        self.frequency = frequency
        self.thd = thd
        self.base_message = thd_err_mesg.format(frequency, thd)
        self.message = mesg.format(self.base_message)


class CalibrationNFError(CalibrationError):

    def __init__(self, frequency, snr):
        self.frequency = frequency
        self.snr = snr
        self.base_message = nf_err_mesg.format(frequency, snr)
        self.message = mesg.format(self.base_message)


################################################################################
# Calibration loaders
################################################################################
def from_psi_chirp(folder, output_gain=None, **kwargs):
    filename = os.path.join(folder, 'chirp_sensitivity.csv')
    sensitivity = pd.io.parsers.read_csv(filename)
    if output_gain is None:
        output_gain = sensitivity.loc[:, 'hw_ao_chirp_level'].max()
    m = sensitivity['hw_ao_chirp_level'] == output_gain
    mic_freq = sensitivity.loc[m, 'frequency'].values
    mic_sens = sensitivity.loc[m, 'sens'].values
    return InterpCalibration(mic_freq, mic_sens, **kwargs)


def from_psi_golay(folder, n_bits=None, output_gain=None, **kwargs):
    filename = os.path.join(folder, 'sensitivity.csv')
    sensitivity = pd.io.parsers.read_csv(filename)
    if n_bits is None:
        n_bits = sensitivity['n_bits'].max()
    if output_gain is None:
        m = sensitivity['n_bits'] == n_bits
        output_gain = sensitivity.loc[m, 'output_gain'].max()
    m_n_bits = sensitivity['n_bits'] == n_bits
    m_output_gain = sensitivity['output_gain'] == output_gain
    m = m_n_bits & m_output_gain
    mic_freq = sensitivity.loc[m, 'frequency'].values
    mic_sens = sensitivity.loc[m, 'sens'].values
    return InterpCalibration(mic_freq, mic_sens, **kwargs)


def from_epl(filename, **kwargs):
    calibration = pd.io.parsers.read_csv(filename, skiprows=14, delimiter='\t')
    freq = calibration['Freq(Hz)']
    spl = calibration['Mag(dB)']
    return InterpCalibration.from_spl(freq, spl, **kwargs)


def from_bcolz_store(rootdir):
    '''
    Returns the calibration used by the input
    '''
    import bcolz
    with bcolz.carray(rootdir=rootdir) as fh:
        data = fh.attrs['source']['calibration']
        class_name = data.pop('type')
        cls = globals()[class_name]
        return cls(**data)


def from_tone_sens(folder):
    filename = os.path.join(folder, 'tone_sensitivity.csv')
    df = pd.read_csv(filename)
    return PointCalibration(df['frequency'], df['sens'])


################################################################################
# Calibration routines
################################################################################
class Calibration(Atom):
    '''
    Assumes that the system is linear for a given frequency

    Parameters
    ----------
    frequency : 1D array
        Frequencies that system sensitivity was measured at.
    sensitivity : 1D array
        Sensitivity of system in dB(V/Pa).
    '''
    attrs = Dict().tag(metadata=True)

    @classmethod
    def as_attenuation(cls, vrms=1, **kwargs):
        '''
        Allows levels to be specified in dB attenuation
        '''
        return cls.from_spl([0, 100e3], [0, 0], vrms, **kwargs)

    @classmethod
    def from_magnitude(cls, frequency, magnitude, vrms=1, **kwargs):
        '''
        Generates a calibration object based on the recorded magnitude (Pa)

        Parameters
        ----------
        frequency : array-like
            List of freuquencies (in Hz)
        magnitude : array-like
            List of magnitudes (e.g., speaker output in Pa) for the specified
            RMS voltage.
        vrms : float
            RMS voltage (in Volts)

        Additional kwargs are passed to the class initialization.
        '''
        sensitivity = util.db(vrms)-util.db(magnitude)-util.db(20e-6)
        return cls(frequency, sensitivity, **kwargs)

    @classmethod
    def from_spl(cls, frequency, spl, vrms=1, **kwargs):
        '''
        Generates a calibration object based on the recorded SPL

        Parameters
        ----------
        frequency : array-like
            List of freuquencies (in Hz)
        spl : array-like
            List of magnitudes (e.g., speaker output in SPL) for the specified
            RMS voltage.
        vrms : float
            RMS voltage (in Volts)

        Additional kwargs are passed to the class initialization.
        '''
        sensitivity = util.db(vrms)-spl-util.db(20e-6)
        return cls(frequency, sensitivity, **kwargs)

    def get_spl(self, frequency, voltage):
        sensitivity = self.get_sens(frequency)
        return util.db(voltage)-sensitivity-util.db(20e-6)

    def get_sf(self, frequency, spl, attenuation=0):
        sensitivity = self.get_sens(frequency)
        vdb = sensitivity+spl+util.db(20e-6)+attenuation
        return 10**(vdb/20.0)

    def get_mean_sf(self, flb, fub, spl, attenuation=0):
        frequencies = np.arange(flb, fub)
        return self.get_sf(frequencies, spl).mean(axis=0)

    def get_attenuation(self, frequency, voltage, level):
        return self.get_spl(frequency, voltage)-level

    def get_gain(self, frequency, spl, attenuation=0):
        return util.db(self.get_sf(frequency, spl, attenuation))

    def set_fixed_gain(self, fixed_gain):
        self.fixed_gain = fixed_gain

    def get_sens(self, frequency):
        raise NotImplementedError


class FlatCalibration(Calibration):

    sensitivity = Float().tag(metadata=True)
    fixed_gain = Float(0).tag(metadata=True)
    mv_pa = Property()

    def _get_mv_pa(self):
        return util.dbi(self.sensitivity) * 1e3

    def _set_mv_pa(self, value):
        self.sensitivity = util.db(value * 1e-3)

    @classmethod
    def as_attenuation(cls, vrms=1, **kwargs):
        '''
        Allows levels to be specified in dB attenuation
        '''
        return cls.from_spl(0, vrms, **kwargs)

    @classmethod
    def from_spl(cls, spl, vrms=1, **kwargs):
        '''
        Generates a calibration object based on the recorded SPL

        Parameters
        ----------
        spl : array-like
            List of magnitudes (e.g., speaker output in SPL) for the specified
            RMS voltage.
        vrms : float
            RMS voltage (in Volts)

        Additional kwargs are passed to the class initialization.
        '''
        sensitivity = util.db(vrms)-spl-util.db(20e-6)
        return cls(sensitivity=sensitivity, **kwargs)

    @classmethod
    def from_mv_pa(cls, mv_pa, **kwargs):
        sens = util.db(mv_pa*1e-3)
        return cls(sensitivity=sens, **kwargs)

    def get_sens(self, frequency):
        return self.sensitivity-self.fixed_gain

    def get_mean_sf(self, flb, fub, spl, attenuation=0):
        return self.get_sf(flb, spl)


class UnityCalibration(FlatCalibration):

    # For unity calibration, set the property so it doesn't get saved.  This
    # value gives us unity passthrough (because the core methods assume
    # everything is in units of dB(Vrms/Pa)).
    sensitivity = Float(-util.db(20e-6)).tag(metadata=False)


class InterpCalibration(Calibration):
    '''
    Use when calibration is not flat (i.e., uniform) across frequency.

    Parameters
    ----------
    frequency : array-like, Hz
        Calibrated frequencies (in Hz)
    sensitivity : array-like, dB(V/Pa)
        Sensitivity at calibrated frequency in dB(V/Pa) assuming 1 Vrms and 0 dB
        gain.  If you have sensitivity in V/Pa, just pass it in as
        20*np.log10(sens).
    fixed_gain : float
        Fixed gain of the input or output.  The sensitivity is calculated using
        a gain of 0 dB, so if the input (e.g. a microphone preamp) or output
        (e.g. a speaker amplifier) adds a fixed gain, this needs to be factored
        into the calculation.

        For input calibrations, the gain must be negative (e.g. if the
        microphone amplifier is set to 40 dB gain, then provide -40 as the
        value).
    attrs : {None, dict}
        Extra attrs that will be saved alongside the calibration.
    '''

    frequency = Typed(np.ndarray).tag(metadata=True)
    sensitivity = Typed(np.ndarray).tag(metadata=True)
    fixed_gain = Float(0).tag(metadata=True)
    _interp = Callable()

    def __init__(self, frequency, sensitivity, fixed_gain=0, attrs=None):
        if attrs is None:
            attrs = {}
        self.frequency = np.asarray(frequency)
        self.sensitivity = np.asarray(sensitivity)
        self.fixed_gain = fixed_gain
        self.attrs = attrs
        self._interp = interp1d(frequency, sensitivity, 'linear', bounds_error=False)

    def get_sens(self, frequency):
        # Since sensitivity is in dB(V/Pa), subtracting fixed_gain from
        # sensitivity will *increase* the sensitivity of the system.
        return self._interp(frequency)-self.fixed_gain


class PointCalibration(Calibration):

    frequency = Typed(np.ndarray).tag(metadata=True)
    sensitivity = Typed(np.ndarray).tag(metadata=True)
    fixed_gain = Float(0).tag(metadata=True)

    def __init__(self, frequency, sensitivity, fixed_gain=0, attrs=None):
        if attrs is None:
            attrs = {}
        if np.isscalar(frequency):
            frequency = [frequency]
        if np.isscalar(sensitivity):
            sensitivity = [sensitivity]
        self.frequency = np.array(frequency)
        self.sensitivity = np.array(sensitivity)
        self.fixed_gain = fixed_gain
        self.attrs = attrs

    def get_sens(self, frequency):
        if np.iterable(frequency):
            return np.array([self._get_sens(f) for f in frequency])
        else:
            return self._get_sens(frequency)

    def _get_sens(self, frequency):
        try:
            i = np.flatnonzero(np.equal(self.frequency, frequency))[0]
        except IndexError:
            log.debug('Calibrated frequencies are %r', self.frequency)
            m = 'Frequency {} not calibrated'.format(frequency)
            raise CalibrationError(m)
        return self.sensitivity[i]-self.fixed_gain

    @classmethod
    def from_psi_chirp(cls, folder, output_gain=None, **kwargs):
        filename = os.path.join(folder, 'chirp_sensitivity.csv')
        sensitivity = pd.io.parsers.read_csv(filename)
        if output_gain is None:
            output_gain = sensitivity.loc[:, 'hw_ao_chirp_level'].max()
        m = sensitivity['hw_ao_chirp_level'] == output_gain
        mic_freq = sensitivity.loc[m, 'frequency'].values
        mic_sens = sensitivity.loc[m, 'sens'].values
        attrs = {
            'type': 'psi_chirp',
            'filename': str(folder),
            'output_gain': output_gain,
        }
        return cls(mic_freq, mic_sens, attrs=attrs, **kwargs)


class EPLCalibration(InterpCalibration):

    @classmethod
    def load_data(cls, filename):
        filename = Path(filename)
        calibration = pd.io.parsers.read_csv(filename, skiprows=14,
                                             delimiter='\t')
        freq = calibration['Freq(Hz)']
        spl = calibration['Mag(dB)']
        return {
            'attrs': {
                'source': filename
            },
            'frequency': freq,
            'spl': spl,
        }

    @classmethod
    def load(cls, filename, **kwargs):
        data = cls.load_data(filename)
        data.update(kwargs)
        return cls.from_spl(**data)


class CochlearCalibration(InterpCalibration):

    @classmethod
    def load_data(cls, filename):
        import tables
        with tables.open_file(filename, 'r') as fh:
            mic_freq = np.asarray(fh.get_node('/frequency').read())
            mic_sens = np.asarray(fh.get_node('/exp_mic_sens').read())
            return {
                'attrs': {
                    'source': filename
                },
                'frequency': mic_freq,
                'sensitivity': mic_sens,
            }

    @classmethod
    def load(cls, filename, **kwargs):
        data = cls.load_data(filename)
        data.update(kwargs)
        return cls(**kwargs)


class GolayCalibration(InterpCalibration):

    fs = Float().tag(metadata=True)
    phase = Typed(np.ndarray).tag(metadata=True)

    def __init__(self, frequency, sensitivity, fs=None, phase=None,
                 fixed_gain=0, **kwargs):
        super().__init__(frequency, sensitivity, fixed_gain, **kwargs)
        # fs and phase are required for the IIR stuff
        if fs is not None:
            self.fs = fs
        if phase is not None:
            self.phase = np.asarray(phase)

    @staticmethod
    def load_data(folder, n_bits=None, output_gain=None):
        from psi.data.io.calibration import CalibrationFile
        fh = CalibrationFile(folder)
        return fh._get_golay_data(n_bits, output_gain)

    @classmethod
    def load(cls, folder, n_bits=None, output_gain=None, **kwargs):
        from psi.data.io.calibration import CalibrationFile
        fh = CalibrationFile(folder)
        return fh.get_golay_calibration(n_bits, output_gain)

    def get_iir(self, fs, fl, fh, truncate=None):
        fs_ratio = self.fs/fs
        if int(fs_ratio) != fs_ratio:
            m = 'Calibration sampling rate, {}, must be an ' \
                'integer multiple of the requested sampling rate'
            raise ValueError(m.format(self.fs))

        n = (len(self.frequency)-1)/fs_ratio + 1
        if int(n) != n:
            m = 'Cannot achieve requested sampling rate ' \
                'TODO: explain why'
            raise ValueError(m)
        n = int(n)

        fc = (fl+fh)/2.0
        freq = self.frequency
        phase = self.phase
        sens = self.sensitivity - (self.get_sens(fc) + self.fixed_gain)
        sens[freq < fl] = 0
        sens[freq >= fh] = 0
        m, b = np.polyfit(freq[freq < fh], phase[freq < fh], 1)
        invphase = 2*np.pi*np.arange(len(freq))*m
        inv_csd = util.dbi(sens)*np.exp(invphase*1j)

        # Need to trim so that the data is resampled accordingly
        if fs_ratio != 1:
            inv_csd = inv_csd[:n]
        iir = np.fft.irfft(inv_csd)

        if truncate is not None:
            n = int(truncate*fs)
            iir = iir[:n]

        return iir


class ChirpCalibration(InterpCalibration):

    @staticmethod
    def load_data(folder, output_gain=None):
        folder = Path(folder)
        sensitivity = pd.io.parsers.read_csv(folder / 'chirp_summary.csv')
        if output_gain is None:
            output_gain = sensitivity['hw_ao_chirp_level'].max()

        m = sensitivity['hw_ao_chirp_level'] == output_gain
        mic_freq = sensitivity.loc[m, 'frequency'].values
        mic_sens = sensitivity.loc[m, 'sens'].values
        mic_phase = sensitivity.loc[m, 'phase'].values
        return {
            'attrs': {
                'source': folder,
                'output_gain': output_gain,
                'calibration_type': 'psi_chirp',
            },
            'frequency': mic_freq,
            'sensitivity': mic_sens,
        }

    @classmethod
    def load(cls, folder, n_bits=None, output_gain=None, **kwargs):
        data = cls.load_psi_golay(folder, n_bits, output_gain)
        data.update(kwargs)
        return cls(**data)


class CalibrationRegistry:

    def __init__(self):
        self.registry = {}

    def register(self, klass, label=None):
        calibration_type = klass.__name__
        calibration_path = f'{klass.__module__}.{calibration_type}'
        if label is None:
            label = calibration_type
        if calibration_type in self.registry:
            m = f'{label} already registered as {calibration_type}'
            raise ValueError(m)
        self.registry[calibration_path] = klass, label
        log.debug('Registered %s', calibration_path)

    def clear(self):
        self.registry.clear()

    def register_basic(self, clear=False, unity=True, fixed=True, golay=True,
                       chirp=True):
        if clear:
            self.clear()
        if unity:
            self.register(UnityCalibration, 'unity gain')
        if fixed:
            self.register(FlatCalibration, 'fixed sensitivity')
        if golay:
            self.register(GolayCalibration, 'Golay calibration')
        if chirp:
            self.register(ChirpCalibration, 'Chirp calibration')

    def get_classes(self):
        return [v[0] for v in self.registry.values()]

    def get_class(self, calibration_type):
        return self.registry[calibration_type][0]

    def get_labels(self):
        return [v[1] for v in self.registry.values()]

    def get_label(self, obj):
        name = f'{obj.__module__}.{obj.__name__}'
        return self.registry[name][1]

    def from_dict(self, calibration_type, **kw):
        if calibration_type not in self.registry:
            log.debug('Importing and registering calibration')
            # Older calibration formats may still have only the class name, not
            # the full module + class name.
            try:
                module_name, class_name = calibration_type.rsplit('.', 1)
            except ValueError:
                module_name = __name__
                class_name = calibration_type
            module = importlib.import_module(module_name)
            klass = getattr(module, class_name)
        else:
            klass = self.get_class(calibration_type)
        return klass(**kw)


calibration_registry = CalibrationRegistry()
calibration_registry.register_basic()
calibration_registry.register(EPLCalibration, 'EPL calibration')
calibration_registry.register(CochlearCalibration, 'Golay calibration (old Cochlear format)')


if __name__ == '__main__':
    import doctest
    doctest.testmod()
