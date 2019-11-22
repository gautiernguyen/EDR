import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import datetime
import matplotlib.colors as colors
from sklearn.externals import joblib
from sklearn.cluster import KMeans
import models
import sys
sys.path.append('../reconnection/')
sys.path.append('../identification/')
sys.path.append('../pyspace-fork2/analysis/')
import event as evt
import postProcess as pp
import mva


def get_ideal_interval(event, model_region, data, label=1, method='smart',
                       feature_to_look='V', windowsize=12):
    pred = pd.Series(index=data[event.begin:event.end].resample('1T').mean().dropna().index,
                     data=medfilt(model_region.predict(data[event.begin:event.end].resample('1T').mean().dropna()),3))
    mshs = pp.makeEventList(pred, label, 1.5)
    largest_msh = mshs[np.argmax([x.duration for x in mshs])]
    if feature_to_look == 'V':
        ftl = np.sqrt(data.Vx**2+data.Vy**2+data.Vz**2)
    if feature_to_look == 'Vl':
        Vlmn = mva.MVA(data[['Bx', 'By', 'Bz']].values).vec2lmn(data[['Vx', 'Vy', 'Vz']].values)
        Vlmn = pd.DataFrame(index = data.index, data = Vlmn, columns=['Vn', 'Vm', 'Vl'])
        ftl = Vlmn.Vl
    if method == 'smart':
        stab = ftl[largest_msh.begin:largest_msh.end].rolling(windowsize, center=True,
                                                              min_periods=windowsize).std()
        cluster = KMeans(n_clusters=2)
        clustered = pd.Series(index=stab.dropna().index,
                              data=cluster.fit_predict(stab.dropna().values.reshape(-1, 1)))
        label = np.argmin(cluster.cluster_centers_)
        potential_ideals = pp.makeEventList(clustered, label, 10/60)
        ideal_msh = potential_ideals[np.argmax([x.duration for x in potential_ideals])]
        most_stab = ftl[ideal_msh.begin:ideal_msh.end].rolling(int(ideal_msh.duration.total_seconds()/10), center=True).std().idxmin()
        ideal_msh = evt.Event(most_stab-ideal_msh.duration/4,
                              most_stab+ideal_msh.duration/4)
        return ideal_msh
    if method == 'center':
        return evt.Event(largest_msh.begin+0.5*(largest_msh.duration-datetime.timedelta(minutes=duration)),
                         largest_msh.begin+0.5*(largest_msh.duration+datetime.timedelta(minutes=duration)))
    if method == 'stable':
        V = np.sqrt(data.Vx**2+data.Vy**2+data.Vz**2)
        stab = V[largest_msh.begin:largest_msh.end]
        begin = stab.rolling(int(duration*12), center=True,
                             min_periods=int(duration*12)).std().idxmin()
        return evt.Event(begin-0.5*datetime.timedelta(minutes=duration),
                         begin+0.5*datetime.timedelta(minutes=duration))


def enlarge_begin(event,  delta=datetime.timedelta(minutes=10)):
    return evt.Event(event.begin-delta, event.end)


def enlarge_end(event,  delta=datetime.timedelta(minutes=10)):
    return evt.Event(event.begin, event.end+delta)


def reaches_timecap(event, duration=datetime.timedelta(hours=3)):
    return event.duration == duration


def is_around_dipole(event, data):
    return data[event.begin:event.end].Bz.abs().max() > 100


def reaches_solar_wind(event, data, model):
    pred = pd.Series(index=data[event.begin:event.end].resample('1T').mean().dropna().index,
                     data=medfilt(model.predict(data[event.begin:event.end].resample('1T').mean().dropna()),3))
    return pred.max() == 2


def overlaps_other_crossing(event, evtlist, limit=datetime.timedelta(hours=1.5)):
    return max(evt.overlapWithList(event, [evt.Event(x.begin-limit, x.end+limit) for x in evtlist], percent=True))>0


def gets_back_into_msp(event, data, model):
    pred = pd.Series(index=data[event.begin:event.end].resample('1T').mean().dropna().index,
                     data=medfilt(model.predict(data[event.begin:event.end].resample('1T').mean().dropna()),3))
    return pred[0] != pred[-1]


def get_enlarged_crossing(event, data, model, evtlist, holesList,
                          timecap=datetime.timedelta(hours=3),
                          step=datetime.timedelta(minutes=10),
                          condition_ideal_msh=True):
    enlarged_crossings = [event]

    stop_begin = False
    stop_end = False

    counter = int(1+0.5*(timecap-event.duration)/step)

    while (stop_begin is False) | (stop_end is False):
        tmp_end = evt.Event(enlarged_crossings[-1].end,
                            enlarged_crossings[-1].end+step)
        tmp_begin = evt.Event(enlarged_crossings[-1].begin-step,
                              enlarged_crossings[-1].begin)
        ideal_msh = get_ideal_interval(enlarged_crossings[-1], model, data)
        if reaches_timecap(enlarged_crossings[-1], timecap) | len(enlarged_crossings) == counter:
            stop_begin = True
            stop_end = True

        if is_around_dipole(tmp_begin, data):
            stop_begin = True
        if is_around_dipole(tmp_end, data):
            stop_end = True

        if reaches_solar_wind(tmp_begin, data, model):
            stop_begin = True
        if reaches_solar_wind(tmp_end, data, model):
            stop_end = True

        list_to_look = [x for x in evtlist if evt.similarity(x, event) == 0]
        if overlaps_other_crossing(tmp_begin, list_to_look):
            stop_begin = True
        if overlaps_other_crossing(tmp_end, list_to_look):
            stop_end = True

        if overlaps_other_crossing(tmp_begin, holesList, datetime.timedelta(0)):
            stop_begin = True
        if overlaps_other_crossing(tmp_end, holesList, datetime.timedelta(0)):
            stop_end = True

        if gets_back_into_msp(tmp_begin, data, model):
            stop_begin = True
        if gets_back_into_msp(tmp_end, data, model):
            stop_end = True

        tmp = enlarged_crossings[-1]
        if condition_ideal_msh is True:
            if evt.similarity(ideal_msh, get_ideal_interval(evt.Event(tmp.begin-step, tmp.end+step), model, data)) > 0.5:
                stop_begin = True
                stop_end = True

        if stop_begin is False:
            tmp = enlarge_begin(tmp, delta=step)
        if stop_end is False:
            tmp = enlarge_end(tmp, delta=step)
        if evt.similarity(tmp, enlarged_crossings[-1]) < 1:
            enlarged_crossings.append(tmp)
    return enlarged_crossings[-1]


def plotJets(event, data,
             ideal_msh=None,
             spectro=None,
             pos=None,
             model=None,
             delta=3,
             pred=None,
             color='k',
             listJets=None,
             remove=False,
             remove_method='pred',
             ideal_msp=None,
             lmn=False):
    start = event.begin-datetime.timedelta(hours=delta)
    end = event.end+datetime.timedelta(hours=delta)

    n_plots = 4
    if pred is not None:
        n_plots += 1
    if ideal_msh is not None:
        n_plots += 1
    if spectro is not None:
        n_plots += 1
    if pos is not None:
        n_plots += 1

    fig = plt.figure(figsize=(10, 15))

    ax1 = plt.subplot2grid((n_plots, 9), (0, 0), colspan=9)
    ax1.plot(data[start:end]['Np'], color='black')
    ax1.set_yscale('log')
    ax1.set_ylabel('Np', fontsize=12)
    ax1.set_ylim((0.1, 50))
    ax1.axhline(1, ls='dashed', color='k')

    ax2 = plt.subplot2grid((n_plots, 9), (1, 0), colspan=9, sharex=ax1)
    B_to_plot = data[start:end][['Bx', 'By', 'Bz']]
    if lmn is True:
        Blmn = mva.MVA(data[start:end][['Bx', 'By', 'Bz']].values).vec2lmn(B_to_plot.values)
        B_to_plot = pd.DataFrame(index=B_to_plot.index,
                                 data=Blmn,
                                 columns=['Bn', 'Bm', 'Bl'])
    ax2.plot(B_to_plot[B_to_plot.columns[0]], color='black')
    ax2.plot(B_to_plot[B_to_plot.columns[1]], color='blue')
    ax2.plot(B_to_plot[B_to_plot.columns[2]], color='red')
    ax2.set_ylim(-50, 50)
    ax2.set_ylabel('Magnetic Field (nT)', fontsize=12)
    ax2.legend(B_to_plot.columns, fontsize=12,
               loc='center left', bbox_to_anchor=(1, 0.5))

    ax3 = plt.subplot2grid((n_plots, 9), (2, 0), colspan=9, sharex=ax1)
    if remove is False:
        V_to_plot = data[start:end][['Vx', 'Vy', 'Vz']]
        if lmn is True:
            Vlmn = mva.MVA(data[start:end][['Bx', 'By', 'Bz']].values).vec2lmn(V_to_plot.values)
            V_to_plot = pd.DataFrame(index=V_to_plot.index,
                                     data=Vlmn,
                                     columns=['Vn', 'Vm', 'Vl'])
        ax3.plot(V_to_plot[V_to_plot.columns[0]], color='black')
        ax3.plot(V_to_plot[V_to_plot.columns[1]], color='blue')
        ax3.plot(V_to_plot[V_to_plot.columns[2]], color='red')
    ax3.set_ylim(-400, 400)
    ax3.set_ylabel('V(km/s)', fontsize=12)
    ax3.legend(V_to_plot.columns, fontsize=12,
               loc='center left', bbox_to_anchor=(1, 0.5))

    ax4 = plt.subplot2grid((n_plots, 9), (3, 0), colspan=9, sharex=ax1)
    if remove is False:
        ax4.plot(np.sqrt(data[start:end]['Vx']**2+data[start:end]['Vy']**2+data[start:end]['Vz']**2), color='green')
    ax4.set_ylim(-50, 300)
    ax4.set_ylabel('V(km/s)', fontsize=12)

    ax = [ax1, ax2, ax3, ax4]
    i = 4
    if spectro is not None:
        ax_spectro = plt.subplot2grid((n_plots, 9), (i, 0),
                                      colspan=9, sharex=ax1)
        seum = spectro[start:end]
        t_data = seum.index
        data_tmp = np.flip(seum.values.T, 0)
        data_tmp[data_tmp < 0] = 0
        xx, yy = np.meshgrid(t_data, np.arange(0, len(seum.columns)))
        im = ax_spectro.pcolormesh(xx, yy, data_tmp,
                                   norm=colors.LogNorm(vmin=1, vmax=1e9),
                                   cmap='nipy_spectral')
        ax_spectro.set_ylabel('Ions energy', fontsize=12)
        cbaxes = fig.add_axes([0.92, 0.23, 0.03, 0.15])
        plt.colorbar(im, cax=cbaxes)
        i += 1

    if pred is not None:
        ax_pred = plt.subplot2grid((n_plots, 9), (i, 0), colspan=9, sharex=ax1)
        ax_pred.plot(pred, color=color)
        ax_pred.set_ylim(-0.01, 2.1)
        ax_pred.set_ylabel('Prediction')
        i += 1
        ax.append(ax_pred)

    if pos is not None:
        ax5 = plt.subplot2grid((n_plots, 9), (i, 1), colspan=2)
        ax5.plot(pos['X'][start:end], pos['Y'][start:end])
        ax5.set_xlim(-20, 20)
        ax5.set_ylim(-20, 20)
        ax5.set_xlabel('X(Re)')
        ax5.set_ylabel('Y(Re)')

        ax6 = plt.subplot2grid((n_plots, 9), (i, 4), colspan=2)
        ax6.plot(pos['X'][start:end], pos['Z'][start:end])
        ax6.set_xlim(-20, 20)
        ax6.set_ylim(-20, 20)
        ax6.set_xlabel('X(Re)')
        ax6.set_ylabel('Z(Re)')

        ax7 = plt.subplot2grid((n_plots, 9), (i, 7), colspan=2)
        ax7.plot(pos['Y'][start:end], pos['Z'][start:end])
        ax7.set_xlim(-20, 20)
        ax7.set_ylim(-20, 20)
        ax7.set_xlabel('Y(Re)')
        ax7.set_ylabel('Z(Re)')

        if model is not None:
            theta = np.arange(0, np.pi+0.01, 0.01)
            phi = 0
            r, Q = model(phi, theta, 2, 0.01, 0, 0)
            x = (r+Q)*np.cos(theta)
            y = (r+Q)*np.sin(theta)*np.cos(phi)
            z = (r+Q)*np.sin(theta)*np.sin(phi)
            ax5.plot(x, y, color='k')

            theta = np.arange(0, np.pi+0.01, 0.01)
            phi = np.pi
            r, Q = model(phi, theta, 2, 0.01, 0, 0)
            x = (r+Q)*np.cos(theta)
            y = (r+Q)*np.sin(theta)*np.cos(phi)
            z = (r+Q)*np.sin(theta)*np.sin(phi)
            ax5.plot(x, y, color='k')

            theta = np.arange(0, np.pi+0.01, 0.01)
            phi = np.pi/2
            r, Q = model(phi, theta, 2, 0.01, 0, 0)
            x = (r+Q)*np.cos(theta)
            y = (r+Q)*np.sin(theta)*np.cos(phi)
            z = (r+Q)*np.sin(theta)*np.sin(phi)
            ax6.plot(x, z, color='k')

            theta = np.arange(0, np.pi+0.01, 0.01)
            phi = -np.pi/2
            r, Q = model(phi, theta, 2, 0.01, 0, 0)
            x = (r+Q)*np.cos(theta)
            y = (r+Q)*np.sin(theta)*np.cos(phi)
            z = (r+Q)*np.sin(theta)*np.sin(phi)
            ax6.plot(x, z, color='k')

            phi = np.arange(0, np.pi+0.01, 0.01)
            theta = -np.pi/2
            r, Q = model(phi, theta, 2, 0.01, 0, 0)
            x = (r+Q)*np.cos(theta)
            y = (r+Q)*np.sin(theta)*np.cos(phi)
            z = (r+Q)*np.sin(theta)*np.sin(phi)
            ax7.plot(y, z, color='k')

            phi = np.arange(0, np.pi+0.01, 0.01)
            theta = np.pi/2
            r, Q = model(phi, theta, 2, 0.01, 0, 0)
            x = (r+Q)*np.cos(theta)
            y = (r+Q)*np.sin(theta)*np.cos(phi)
            z = (r+Q)*np.sin(theta)*np.sin(phi)
            ax7.plot(y, z, color='k')

    for elt in ax:
        elt.axvline(event.begin, color='k', ls='dashed')
        elt.axvline(event.end, color='k', ls='dashed')
        if ideal_msh is not None:
            elt.axvspan(ideal_msh.begin, ideal_msh.end, color='y', alpha=0.5)
        if ideal_msp is not None:
            elt.axvspan(ideal_msp.begin, ideal_msp.end,
                        color='purple', alpha=0.5)
    if listJets is not None:
        for x in listJets:
            for elt in ax:
                elt.axvspan(x.begin, x.end, color='b', alpha=0.5)

    if remove is True:
        V = data[event.begin:event.end][['Vx', 'Vy', 'Vz']].copy()
        V_msh = data[ideal_msh.begin:ideal_msh.end][['Vx', 'Vy', 'Vz']].mean()
        if remove_method == 'pred':
            V[pred == 1] -= V_msh
        if remove_method == 'all':
            V -= V_msh
        ax3.plot(V.Vx, color='black')
        ax3.plot(V.Vy, color='blue')
        ax3.plot(V.Vz, color='red')
        ax4.plot(np.sqrt(V.Vx**2+V.Vy**2+V.Vz**2))
    return fig, ax
