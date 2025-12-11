import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

def voltTOdbm(volt):
    p_w=(volt**2)/50
    dbm=10*np.log10(p_w*1000)
    return dbm
################ 250922 V2 #####################################
def dBm(full_scale_power_dbm,daps):
    v=np.sqrt((2*50*10**(full_scale_power_dbm/10))/1000)*daps*1 # 1 : twpa readout amplitude  #opx1000 documentation
    p_w=(v**2)/50
    dbm=10*np.log10(p_w*1000)-10
    return dbm
# def pump_maxgain(gain, dfps, daps):
#     avg_gain=np.mean(gain,axis=0)
#     max_gain_idx=np.unravel_index(np.argmax(avg_gain), avg_gain.shape)
#     max_gain_pump=np.array(np.array([dfps[max_gain_idx[0]],daps[max_gain_idx[1]]]))
#     return max_gain_pump, max_gain_idx
# def pump_maxdsnr(dsnr, dfps, daps):
#     avg_dsnr=np.mean(dsnr,axis=0)
#     max_dsnr_idx=np.unravel_index(np.argmax(avg_dsnr), avg_dsnr.shape)
#     max_dsnr_pump=np.array(np.array([dfps[max_dsnr_idx[0]],daps[max_dsnr_idx[1]]]))
#     return max_dsnr_pump, max_dsnr_idx
# modify to average over linear units
def pump_maxgain(gain, dfps, daps):
    linear_gain=10**(gain/20)
    avg_linear_gain=np.mean(linear_gain,axis=0)
    max_gain_idx=np.unravel_index(np.argmax(avg_linear_gain), avg_linear_gain.shape)
    max_gain_pump=np.array(np.array([dfps[max_gain_idx[0]],daps[max_gain_idx[1]]]))
    return max_gain_pump, max_gain_idx
def pump_maxdsnr(dsnr, dfps, daps):
    linear_dsnr=10**(dsnr/20)
    avg_linear_dsnr=np.mean(linear_dsnr,axis=0)
    max_dsnr_idx=np.unravel_index(np.argmax(avg_linear_dsnr), avg_linear_dsnr.shape)
    max_dsnr_pump=np.array(np.array([dfps[max_dsnr_idx[0]],daps[max_dsnr_idx[1]]]))
    return max_dsnr_pump, max_dsnr_idx
# def mvTOdbm(mv):
#     v=mv*1e-3
#     rms_v=v/np.sqrt(2)
#     p_watt=((rms_v)**2)/50
#     dbm=10*np.log10(p_watt*1000)
#     return dbm
# def snr(ds, qubits, dfps, daps):
#     noise=np.zeros((len(qubits),len(dfps),len(daps),1))
#     signal=np.zeros((len(qubits),len(dfps),len(daps),1))
#     for i in range(len(qubits)):
#         for j in range(len(dfps)):
#             for k in range(len(daps)):
#                 noise[i,j,k]=mvTOdbm(np.mean(ds.IQ_abs_noise.values[i][j][k]))
#                 signal[i,j,k]=mvTOdbm(ds.IQ_abs_signal.values[i][j][k][len(ds.IQ_abs_signal.values[i][j][k])//2])
#     return signal-noise

# def gain(ds_pumpoff,ds_pumpon, qubits, dfps, daps):
#     signal_pumpoff=np.zeros((len(qubits),len(dfps),len(daps),1))
#     signal_pumpon=np.zeros((len(qubits),len(dfps),len(daps),1))
#     for i in range(len(qubits)):
#         for j in range(len(dfps)):
#             for k in range(len(daps)):
#                 signal_pumpoff[i,j,k]=mvTOdbm(np.mean(ds_pumpoff.IQ_abs_signal.values[i][j][k]))
#                 signal_pumpon[i,j,k]=mvTOdbm(np.mean(ds_pumpon.IQ_abs_signal.values[i][j][k]))
#     return signal_pumpon-signal_pumpoff
# 251208 gain, snr from linear utit to dB (not using dBm)
def snr(ds, qubits, dfps, daps):
    noise=np.zeros((len(qubits),len(dfps),len(daps),1))
    signal=np.zeros((len(qubits),len(dfps),len(daps),1))
    for i in range(len(qubits)):
        for j in range(len(dfps)):
            for k in range(len(daps)):
                noise[i,j,k]=(np.mean(ds.IQ_abs_noise.values[i][j][k]))
                signal[i,j,k]=(ds.IQ_abs_signal.values[i][j][k][len(ds.IQ_abs_signal.values[i][j][k])//2])
    return 20*np.log10(signal/noise)
def gain(ds_pumpoff,ds_pumpon, qubits, dfps, daps):
    signal_pumpoff=np.zeros((len(qubits),len(dfps),len(daps),1))
    signal_pumpon=np.zeros((len(qubits),len(dfps),len(daps),1))
    for i in range(len(qubits)):
        for j in range(len(dfps)):
            for k in range(len(daps)):
                signal_pumpoff[i,j,k]=(np.mean(ds_pumpoff.IQ_abs_signal.values[i][j][k]))
                signal_pumpon[i,j,k]=(np.mean(ds_pumpon.IQ_abs_signal.values[i][j][k]))
    return 20*np.log10(signal_pumpon/signal_pumpoff)

################ 250928 V3 #####################################
def signal(ds):
    avg_signal=ds.IQ_abs_signal.values.mean(axis=-1, keepdims=True)
    return voltTOdbm(avg_signal)
def noise(ds, qubits, dfps, daps, n_avg):
    I=np.zeros((len(qubits),len(dfps),len(daps),n_avg))
    Q=np.zeros((len(qubits),len(dfps),len(daps),n_avg))
    for i in range(len(qubits)):
            for j in range(len(dfps)):
                for k in range(len(daps)):
                    for n in range(n_avg):
                        I[i][j][k][n]=ds.I.values[i][n][j][k]
                        Q[i][j][k][n]=ds.Q.values[i][n][j][k]
    I_noise=np.zeros((len(qubits),len(dfps),len(daps),1))
    Q_noise=np.zeros((len(qubits),len(dfps),len(daps),1))
    for i in range(len(qubits)):
            for j in range(len(dfps)):
                for k in range(len(daps)):
                    I_noise[i][j][k]=np.std(I[i][j][k])
                    Q_noise[i][j][k]=np.std(I[i][j][k])
    return (voltTOdbm(I_noise)+voltTOdbm(Q_noise))/2 #is it ok to define the noise as avg of IQ std
######################## optimizer
def lin(db_value):
        return 10 ** (db_value / 10)
def fidelity(ro,t1,snr,pgg): #snr=distance/(distribution of g,e)
    fidelity=np.exp(-ro/(2*t1))*erf(snr/(np.sqrt(2)))*pgg
    return fidelity  
def pa(qubits, dfps, daps):
    target_shape = (len(qubits), len(dfps), len(daps), 1)
    total_elements = np.prod(target_shape)
    pa = np.resize(daps, total_elements).reshape(target_shape)
    return pa
def fp(qubits, dfps, daps):
    fp = np.zeros((len(qubits), len(dfps), len(daps), 1))
    fp[:, :, :, 0] = dfps[np.newaxis, :, np.newaxis]
    return fp
def min_dsnr(qubits,dsnr_avg,snr_off,pgg, dfps, daps): 
    twpa_fct=np.sqrt(lin(dsnr_avg))
    f=fidelity(qubits[0].resonator.operations["readout"].length*1e-3,
               qubits[0].T1*1e6, 
               snr_off*twpa_fct,
               pgg)
    f_=f.reshape(len(dfps)*len(daps))
    sat_idx=np.where((np.max(f_)-f_)*100<0.1)
    sat_dsnr = dsnr_avg.reshape(len(dfps)*len(daps))[sat_idx[0]]
    return min(sat_dsnr)
def min_gain(qubits,  twpas):
    ##### 1.max a_ro_on_max s.t) #(mtpx Ro)*Pon < Psat-10 : total readout power should be smaller than saturation power
    ##### get the da , theoretically can be used 
    a_ro=np.linspace(0,1,500)
    ps_on=dBm(qubits[0].resonator.opx_output.full_scale_power_dbm,
            a_ro*qubits[0].resonator.operations["readout"].amplitude)-60-6-5
    mtpx_ps_on= ps_on + 10*np.log10(len(qubits))
    idx=np.where(mtpx_ps_on<twpas[0].p_saturation-20)[0][-1]
    a_ro_on_max=a_ro[idx]
    ###### 2. a_ro_off~a_ro_off*a_ro_on*sqrt(linG) : 
    ###### pick the Gain which compensate the lowered readout amplitude up to the readout amplitude when twpa is off
    g=np.linspace(0,25,500)
    a_ro_off=1 # value doesnt matter
    idx_=np.where(a_ro_off<a_ro_off*a_ro_on_max*np.sqrt(10**(g/10)))
    minimum_gain=g[idx_[0][0]]
    # to guarantee TWPA gain suppress HEMT noise
    if minimum_gain<10:
        minimum_gain=10
    elif minimum_gain>=10:
        minimum_gain=minimum_gain
    return minimum_gain

# def optimizer(mingain, mindsnr, gain_avg, dsnr_avg, daps, dfps, p_lo,p_if):
#     mask = gain_avg > mingain
#     masked_dsnr = np.where(mask, dsnr_avg, -np.inf)
#     flat_index = np.argmax(masked_dsnr)
#     idx = np.unravel_index(flat_index, dsnr_avg.shape)
#     print(f"Optimized ap={np.round(daps[idx[1]],5)},fp={np.round((p_lo+p_if+dfps[idx[0]])*1e-9,3)}GHz ")
#     print(f"gain_avg :{np.round(gain_avg[idx],2)}dB")
#     print(f"dsnr_avg :{np.round(dsnr_avg[idx],2)}dB")
#     return idx
################# new optimizer 2512121
def optimizer(mingain, mindsnr, Gain, dsnr,  average_dsnr, dfps, daps, p_lo,p_if):
    # this optimizer finds the pump setting which gives the highest average dSNR
    # while satisfying the min gain and dSNR for individual qubits
    gain_mask=np.all(Gain>mingain,axis=0)
    pump_idx_gain_mask=np.argwhere(gain_mask)
    idx_pump_candidate=[]
    for j in range(len(pump_idx_gain_mask)):
        x,y= pump_idx_gain_mask[j][0], pump_idx_gain_mask[j][1]
        if np.all(dsnr[:, x,y,0]>mindsnr):
            idx_pump_candidate.append(j)
    average_dsnr_=[]
    for i in range(len(idx_pump_candidate)):
        average_dsnr_.append(average_dsnr[pump_idx_gain_mask[idx_pump_candidate[i]][0]][pump_idx_gain_mask[idx_pump_candidate[i]][1]])
    optimal_pump_idx=idx_pump_candidate[np.argmax(average_dsnr_)]
    optimal_fp=dfps[pump_idx_gain_mask[optimal_pump_idx][0]]
    optimal_pp=daps[pump_idx_gain_mask[optimal_pump_idx][1]]
    print(f"Optimized ap={np.round(optimal_pp,5)},fp={np.round((p_lo+p_if+optimal_fp)*1e-9,3)}GHz ")
    return (pump_idx_gain_mask[optimal_pump_idx][0], pump_idx_gain_mask[optimal_pump_idx][1])


#---------------------------------MULTIPLXED READOUT OPTIMIZER----------------------------
def multiplexed_optimizer(qubit, Gain, dsnr, qubits): #qubit = worst snr qubit 
    idx=np.unravel_index(np.argmax(dsnr[qubit-1]),dsnr[qubit-1].shape)
    print(f"@ max dSNR for qB{qubit}")
    for i in range(len(qubits)):        
        print(f"qB{i+1}:dSNR:{np.round(dsnr[i][idx[0]][idx[1]][0],2)}dB, gain:{np.round(Gain[i][idx[0]][idx[1]][0],2)}dB")
    return idx



