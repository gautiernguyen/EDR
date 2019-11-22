import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import datetime
import matplotlib.colors as colors
from sklearn.externals import joblib
import models
import sys
sys.path.append('../reconnection/')
sys.path.append('../identification/')
sys.path.append('../pyspace-fork2/analysis/')
import event as evt
import postProcess as pp
import mva
import crossings_util as Xing


class Jet:
    def __init__(self, event):
        self.begin = event.begin
        self.end = event.end
        self.duration = event.duration
        self.crossing = None
        self.ideal_msh = None
        self.proba = 0

    def __str__(self):
        print(self.as_event())

    def get_associated_Crossing(self, evtList):
        '''
        for a given jet, return the associated crossing
        '''
        self.crossing = evtList[np.argmax(evt.overlapWithList(self, evtList))]

    def get_ideal_msh(self, model, data):
        '''
        for a given jet, return the ideal msh of a given crossing
        detected with our region classificators
        '''
        self.ideal_msh = Xing.get_ideal_interval(self.crossing, model, data)
        self.ideal_msp = Xing.get_ideal_interval(self.crossing,
                                                 model, data, label=0)

    def get_physical_params(self, data, pos):
        self.clock_angle = _clock_angle(data['By'][self.ideal_msh.begin:self.ideal_msh.end].mean(), data['Bz'][self.ideal_msh.begin:self.ideal_msh.end].mean())
        self.velocity = (data[self.begin:self.end][['Vx', 'Vy', 'Vz']]-data[self.ideal_msh.begin:self.ideal_msh.end][['Vx', 'Vy', 'Vz']].mean()).mean()
        self.tilt = _get_tilt(self.begin)
        self.pos = pos[self.begin-datetime.timedelta(minutes=10):self.begin+datetime.timedelta(minutes=10)].mean()
        self.density = data[self.begin:self.end]['Np'].mean()

    def as_event(self):
        return evt.Event(self.begin, self.end)


def _get_tilt(date):
    if date.year % 4 == 0:
        tilts = np.zeros(367)
        tilts.fill(np.nan)

        tilts[172] = 34
        tilts[80] = 0
        tilts[264] = 0
        tilts[355] = -34

        tilts[:80] = 34/90*(np.arange(0, 80)-80)
        tilts[80:172] = 34/92*(np.arange(80, 172)-80)
        tilts[172:264] = -34/92*(np.arange(172, 264)-264)
        tilts[264:355] = -34/91*(np.arange(264, 355)-264)
        tilts[355:] = +34/89*(np.arange(355, 367)-444)
    else:
        tilts = np.zeros(366)
        tilts.fill(np.nan)

        tilts[172] = 34
        tilts[80] = 0
        tilts[264] = 0
        tilts[355] = -34

        tilts[:80] = 34/90*(np.arange(0, 80)-80)
        tilts[80:172] = 34/92*(np.arange(80, 172)-80)
        tilts[172:264] = -34/92*(np.arange(172, 264)-264)
        tilts[264:355] = -34/91*(np.arange(264, 355)-264)
        tilts[355:] = +34/89*(np.arange(355, 366)-444)
    return tilts[int(date.strftime('%j'))]


def _clock_angle(By, Bz):
    clock_serie = np.arctan(By/Bz)*180/np.pi
    if (Bz < 0) & (By < 0):
        clock_serie += -180
    if (Bz < 0) & (By > 0):
        clock_serie += 180
    if (Bz == 0) & (By > 0):
        clock_serie = 90
    if (Bz == 0) & (By < 0) > 0:
        clock_serie = -90
    return clock_serie


def list_to_csv(jet_list, filename):
    '''
    For a given event list, save it into a csv file ( filename as str)
    The csv file will then give for each event of the listbegin,
    end and physical params
    '''
    edf = pd.DataFrame(data={'begin': [x.begin for x in jet_list],
                             'end': [x.end for x in jet_list],
                             'clock_angle': [x.clock_angle for x in jet_list],
                             'tilt': [x.tilt for x in jet_list],
                             'X': [x.pos.X for x in jet_list],
                             'Y': [x.pos.Y for x in jet_list],
                             'Z': [x.pos.Z for x in jet_list],
                             'Vx': [x.velocity.Vx[0] for x in jet_list],
                             'Vy': [x.velocity.Vy[0] for x in jet_list],
                             'Vz': [x.velocity.Vz[0] for x in jet_list],
                             'Xing_begin': [x.crossing.begin for x in jet_list],
                             'Xing_end': [x.crossing.end for x in jet_list],
                             'Msh_begin': [x.ideal_msh.begin for x in jet_list],
                             'Msh_end': [x.ideal_msh.end for x in jet_list],
                             'proba': [x.proba for x in jet_list]})
    edf.to_csv(filename)
    return edf


def read_csv(filename,
             index_col=0, header=None, dateFormat="%Y/%m/%d %H:%M",
             sep=','):
    '''
    Consider a  list of events as csv file ( with at least begin and end)
    and return a list of jets
    '''
    df = pd.read_csv(filename, index_col=index_col, header=header, sep=sep)
    df['begin'] = pd.to_datetime(df['begin'], format=dateFormat)
    df['end'] = pd.to_datetime(df['end'], format=dateFormat)
    df['Xing_begin'] = pd.to_datetime(df['Xing_begin'], format=dateFormat)
    df['Xing_end'] = pd.to_datetime(df['Xing_end'], format=dateFormat)
    df['Msh_begin'] = pd.to_datetime(df['Msh_begin'], format=dateFormat)
    df['Msh_end'] = pd.to_datetime(df['Msh_end'], format=dateFormat)

    jetList = [Jet(evt.Event(df['begin'][i], df['end'][i]))
               for i in range(0, len(df))]
    for i, elt in enumerate(jetList):
        elt.clock_angle = df['clock_angle'][i]
        elt.tilt = df['tilt'][i]
        elt.velocity = pd.Series(index=['Vx', 'Vy', 'Vz'], data=[df['Vx'][i],df['Vy'][i],df['Vz'][i]])
        elt.pos = pd.Series(index=['X', 'Y', 'Z'], data=[df['X'][i],df['Y'][i],df['Z'][i]])
        elt.crossing = evt.Event(df['Xing_begin'][i], df['Xing_end'][i])
        elt.ideal_msh = evt.Event(df['Msh_begin'][i], df['Msh_end'][i])
        elt.proba = df['proba'][i]
    return jetList
