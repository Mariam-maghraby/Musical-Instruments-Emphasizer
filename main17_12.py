# libraries needed for main python file
# the file we run for the app to work
import sys
import matplotlib
import numpy as np
import os
from PyQt5 import uic
import sounddevice as sd
import pyqtgraph as pg
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import *
from PyQt5 import QtCore, QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import sounddevice as sd
from scipy.io import wavfile
from scipy.fft import fftfreq, rfft, irfft, rfftfreq
from numpy.fft import fft
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import sys
import simpleaudio as sa
import sounddevice as sd
from pydub.playback import play
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from scipy import signal as sig

import logging

logging.basicConfig(level=logging.INFO, filename='log.txt',
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


matplotlib.use('QT5Agg')


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=120):
        self.fig = Figure(dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

    def returnAxes(self):
        axes = self.fig.add_subplot(111)
        return axes

########################################################################################################

###########################################<Piano Virtual Instument>########################


# sampling the data 44.1kHz, or 44,100 samples per second
sampleRate = 44100  # samples per second


def getWave(frequency, duration=0.5):
    amplitude = 4096
    time = np.linspace(0, duration, int(sampleRate*duration))
    wave = amplitude*np.sin(2*np.pi*frequency*time)
    return wave


def getPianoNotes():
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    baseFreq = 261.63  # frequency of base
    noteFreqs = {octave[i]: baseFreq*pow(2, (i/12))
                 for i in range(len(octave))}
    noteFreqs[''] = 0.0  # slient freq for the played note
    return noteFreqs


def playPiano(data):
    # Start playback
    play_obj = sa.play_buffer(data, 1, 2, sampleRate)  # 1,2
    # Wait for playback to finish before exiting
    play_obj.wait_done()


def playedPianoKey(i):
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    notesFreqs = getPianoNotes()
    sound = octave[i]
    # print(sound)
    song = [getWave(notesFreqs[note]) for note in sound.split('-')]
    song = np.concatenate(song)
    # print(song)
    data = song.astype(np.int16)
    data = amplifiedPianoSound(data)
    playPiano(data)


def amplifiedPianoSound(data):
    data = data * (16300/np.max(data))  # amplifying the wave
    data = data.astype(np.int16)
    return data

###########################################<Guitar Virtual Instument>########################


guitarSamplingFrequency = 8000

# Synthesizes a new waveform from an existing wavetable, modifies last sample by averaging.

# function for generation of the wave sound of the guitar chord notes


def karplus_strong(wavetable, n_samples):
    samples = []
    current_sample = 0
    previous_value = 0
    while len(samples) < n_samples:
        wavetable[current_sample] = 0.5 * \
            (wavetable[current_sample] + previous_value)
        samples.append(wavetable[current_sample])
        previous_value = samples[-1]
        current_sample += 1
        current_sample = current_sample % wavetable.size
    return np.array(samples)


def getStringSound(i):
    Frequencies = [55, 80, 75, 65, 44, 30]
    wavetableSize = guitarSamplingFrequency // Frequencies[i]
    # Recommend to use a wavetable made using a random signal containing either 1s or -1s.
    # Here, we generated a wavetable containing 1s.
    wavetable = (2 * np.random.randint(0, 2, wavetableSize) -
                 1).astype(np.float)
    stringSound = karplus_strong(wavetable, guitarSamplingFrequency)
    sd.play(stringSound, sampleRate)


########################################<Drums Virtual Instrument>########################################

# function for generation the sound wave of the drums
def karplus_strong_drum(wavetable, n_samples, prob):
    """Synthesizes a new waveform from an existing wavetable, modifies last sample by averaging."""
    samples = []
    current_sample = 0
    previous_value = 0
    while len(samples) < n_samples:
        r = np.random.binomial(1, prob)
        sign = float(r == 1) * 2 - 1
        wavetable[current_sample] = sign * 0.5 * \
            (wavetable[current_sample] + previous_value)
        samples.append(wavetable[current_sample])
        previous_value = samples[-1]
        current_sample += 1
        current_sample = current_sample % wavetable.size
        # print(np.array(samples))
    return np.array(samples)


#############################################################################################################

# class definition for application window components like the ui

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = uic.loadUi('ui_23_1_22.ui', self)
        self.Open.triggered.connect(self.openFile)
        self.playpauseButton.clicked.connect(self.play_pause)
        self.VolumeSlider.valueChanged.connect(self.changeVolume)
        self.SpectrogramColorPalleteComboBox.currentTextChanged.connect(
            self.changeSpectrogramColorPallete)
        self.player = QMediaPlayer()
        self.timer = QtCore.QTimer()
        self.timeInterval = 150
        self.timer.setInterval(self.timeInterval)
        self.timer.timeout.connect(self.updatePlot)
        self.spectrogramCmap = "viridis"
        self.playPauseFlag = 0
        self.playEqualizedSongFlag = 0
        self.Index = 0
        self.signalGraph = self.ui.SignalGraph.plot(
            [], [], pen=pg.mkPen(color=(0, 0, 255)))
        self.spectrogramOfMusic = MatplotlibCanvas(self)
        self.ui.SpectrogramGraph.addWidget(self.spectrogramOfMusic)
        self.spectrogramOfMusic.setHidden(True)
        self.fileIndex = 0
        self.playIcon = QtGui.QIcon()
        self.playIcon.addPixmap(QtGui.QPixmap(
            "Images\playF.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        self.whiteKey1.clicked.connect(lambda: playedPianoKey(0))
        self.blackKey1.clicked.connect(lambda: playedPianoKey(1))
        self.whiteKey2.clicked.connect(lambda: playedPianoKey(2))
        self.blackKey2.clicked.connect(lambda: playedPianoKey(3))
        self.whiteKey3.clicked.connect(lambda: playedPianoKey(4))
        self.blackKey3.clicked.connect(lambda: playedPianoKey(5))
        self.whiteKey4.clicked.connect(lambda: playedPianoKey(6))
        self.blackKey4.clicked.connect(lambda: playedPianoKey(7))
        self.whiteKey5.clicked.connect(lambda: playedPianoKey(8))
        self.blackKey5.clicked.connect(lambda: playedPianoKey(9))
        self.whiteKey6.clicked.connect(lambda: playedPianoKey(10))
        self.blackKey6.clicked.connect(lambda: playedPianoKey(11))
        self.whiteKey7.clicked.connect(lambda: playedPianoKey(0))
        self.blackKey7.clicked.connect(lambda: playedPianoKey(1))
        self.whiteKey8.clicked.connect(lambda: playedPianoKey(2))
        self.blackKey8.clicked.connect(lambda: playedPianoKey(3))
        self.whiteKey9.clicked.connect(lambda: playedPianoKey(4))
        self.blackKey9.clicked.connect(lambda: playedPianoKey(5))
        self.whiteKey10.clicked.connect(lambda: playedPianoKey(6))
        self.blackKey10.clicked.connect(lambda: playedPianoKey(7))
        self.whiteKey11.clicked.connect(lambda: playedPianoKey(8))
        self.blackKey11.clicked.connect(lambda: playedPianoKey(9))
        self.whiteKey12.clicked.connect(lambda: playedPianoKey(10))
        self.blackKey12.clicked.connect(lambda: playedPianoKey(11))
        self.whiteKey13.clicked.connect(lambda: playedPianoKey(0))

        self.guitarString1.clicked.connect(lambda: getStringSound(0))
        self.guitarString2.clicked.connect(lambda: getStringSound(1))
        self.guitarString3.clicked.connect(lambda: getStringSound(2))
        self.guitarString4.clicked.connect(lambda: getStringSound(3))
        self.guitarString5.clicked.connect(lambda: getStringSound(4))
        self.guitarString6.clicked.connect(lambda: getStringSound(5))

        self.drumsSlider.valueChanged.connect(self.strengthOfplay)
        self.drumsButton.clicked.connect(self.getdrumsSound)
        #self.drumsFrameButton.clicked.connect(lambda: getdrumsRightPartSound)

        # self.leftSideDrums.clicked.connect(getdrumsLeftPartSound)
        # self.rightSideDrums.clicked.connect(getdrumsRightPartSound)

        self.pianoWidget.setHidden(True)
        self.guitarWidget.setHidden(True)
        self.drumsWidget.setHidden(True)

        self.startPianoButton.clicked.connect(self.startPlayingPiano)
        self.startGuitarButton.clicked.connect(self.startPlayingGuitar)
        self.startDrumsButton.clicked.connect(self.startPlayingDrums)

        #################################################################################
        self.sliders = [self.PianoSlider,
                        self.GuitarSlider,
                        self.FluteSlider,
                        ]

        for slider in self.sliders:
            slider.sliderReleased.connect(self.equalizer)

        self.frequencyBands = [
            [0, 399],  # piano
            [3000, 12000],  # guitar
            [400, 2999]  # flute
        ]

    def getdrumsSound(self):
        samplingFrequency = 8000
        wavetable_size = samplingFrequency // 40
        wavetable = np.ones(wavetable_size)
        #sound = karplus_strong_drum(wavetable, 1 * samplingFrequency, 0.3)
        sound = karplus_strong_drum(
            wavetable, 1 * samplingFrequency, self.strengthOfplay())
        sd.play(sound, samplingFrequency)
        print(sound)
        #sa.play_buffer(sound, 1, 2, samplingFrequency)

    def strengthOfplay(self):
        strength = 0.1*(self.drumsSlider.value())
        print(strength)
        return float(strength)

    def startPlayingPiano(self):
        self.pianoWidget.setHidden(False)
        self.guitarWidget.setHidden(True)
        self.drumsWidget.setHidden(True)

    def startPlayingGuitar(self):
        self.guitarWidget.setHidden(False)
        self.pianoWidget.setHidden(True)
        self.drumsWidget.setHidden(True)

    def startPlayingDrums(self):
        self.drumsWidget.setHidden(False)
        self.pianoWidget.setHidden(True)
        self.guitarWidget.setHidden(True)

    def openFile(self):
        full_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, 'Open Song', QtCore.QDir.rootPath(), 'Raw Data(*.wav)')
        self.filePath = os.path.basename(full_file_path)
        logging.info(f'{self.filePath} file is opened')
        self.content = QMediaContent(QUrl.fromLocalFile(self.filePath))
        self.player.setMedia(self.content)
        self.file_ext = self.getFileExtention(self.filePath)

        if self.file_ext == 'wav':
            self.samplerate, self.data = wavfile.read(self.filePath)
            self.data = np.array(self.data).flatten()
            print(self.data)
            self.length = self.data.shape[0]
            self.duration = (self.length / self.samplerate)
            self.time = np.linspace(0., self.duration, self.length)
            self.graphPointsIncrementRate = int(
                self.length/(self.duration*((1000/self.timeInterval)-0.3)))

            if np.ndim(self.data) == 1:
                self.ui.SignalGraph.setLimits(
                    xMin=0, xMax=500000, yMin=-200000, yMax=200000)
                self.ui.SignalGraph.setYRange(
                    min(self.data), max(self.data))
                self.ui.SignalGraph.setXRange(
                    min(self.time), max(self.time))

                # self.fftArrayAbsolute = np.abs(self.fftArray)
                # self.fftArrayPhase = np.angle(self.fftArray)

            elif np.ndim(self.data) != 1:
                logging.warning(
                    "The file doesn't have the right dimensions to be played")

        else:
            logging.warning("You must select a .wav file to be played")

    def getFileExtention(self, s):
        for i in range(1, len(s)):
            if s[-i] == '.':
                return s[-(i - 1):]

    def updatePlot(self):
        time = self.time[:self.Index]
        if self.playEqualizedSongFlag:
            data = self.EqualizedSongData[:self.Index]
        else:
            data = self.data[:self.Index]

        self.Index += self.graphPointsIncrementRate
        self.signalGraph.setData(time, data)

    def equalizer(self):
        logging.info("The instruments gain is changed.")
        self.player.stop()
        self.timer.stop()
        self.Index = 0
        self.signalGraph.setData([], [])
        self.playPauseFlag = 0
        self.playEqualizedSongFlag = 1
        self.newfftArray = np.fft.rfft(self.data)
        frequencies = np.fft.rfftfreq(len(self.data), 1 / self.samplerate)

        for i in range(len(self.sliders)):
            self.newfftArray[(self.frequencyBands[i][0] <= frequencies) & (frequencies <= self.frequencyBands[i][1])] = \
                self.newfftArray[(self.frequencyBands[i][0] <= frequencies) & (
                    frequencies <= self.frequencyBands[i][1])] * self.sliders[i].value()

        self.equalizedSignal = np.fft.irfft(
            self.newfftArray).astype(self.data.dtype)
        logging.info("The equalized song is constructed successfully.")
        wavfile.write(f'new{self.fileIndex}.wav',
                      self.samplerate, self.equalizedSignal)
        logging.info("The equalized song is saved successfully.")
        self.EqualizedSong = QMediaContent(
            QUrl.fromLocalFile(f'new{self.fileIndex}.wav'))
        self.EqualizedSongSamplerate, self.EqualizedSongData = wavfile.read(
            f'new{self.fileIndex}.wav')
        self.player.setMedia(self.EqualizedSong)
        logging.info("The equalized song is successfully loaded to be played.")
        self.ui.SignalGraph.setYRange(
            min(self.EqualizedSongData), max(self.EqualizedSongData))
        self.spectrogramOfMusic.axes.cla()
        self.spectrogramOfMusic.axes.specgram(
            self.equalizedSignal, Fs=200, cmap=self.spectrogramCmap)
        self.spectrogramOfMusic.draw()
        self.fileIndex += 1

    def play_pause(self):
        self.playPauseFlag = self.playPauseFlag ^ 1

        if self.playPauseFlag:
            playIcon = QtGui.QIcon()
            playIcon.addPixmap(QtGui.QPixmap(
                "Images\playF.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.playpauseButton.setIcon(playIcon)
            self.timer.start()
            self.player.play()
        else:
            pauseIcon = QtGui.QIcon()
            pauseIcon.addPixmap(QtGui.QPixmap(
                "Images\pause.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.playpauseButton.setIcon(pauseIcon)
            self.player.pause()
            self.timer.stop()

    def changeSpectrogramColorPallete(self):
        self.spectrogramOfMusic.setHidden(False)
        self.spectrogramCmap = self.SpectrogramColorPalleteComboBox.currentText()
        self.spectrogramOfMusic.axes.cla()
        if self.playEqualizedSongFlag:
            self.spectrogramOfMusic.axes.specgram(
                self.equalizedSignal, Fs=200, cmap=self.spectrogramCmap)
        else:
            self.spectrogramOfMusic.axes.specgram(
                self.data, Fs=200, cmap=self.spectrogramCmap)
        self.spectrogramOfMusic.draw()

    def changeVolume(self):
        # self.player.setVolume(self.VolumeSlider.value())
        self.volumeValue = self.VolumeSlider.value()
        self.player.setVolume(self.volumeValue)
        logging.info(f"The volume is updated to {self.volumeValue}")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = ApplicationWindow()
    MainWindow.show()
    sys.exit(app.exec_())
