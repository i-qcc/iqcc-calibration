import numpy as np
import matplotlib.pyplot as plt

# # gain, snr into db
# def res_snr(spec):#ok
#     base=spec[spec!=spec.min()]
#     signalsize=np.mean(base)-np.min(spec)
#     noise=np.std(base)
#     snr=20*np.log10((signalsize/noise))
#     return snr
def voltTOdbm(volt):
    p_w=(volt**2)/50
    dbm=10*np.log10(p_w*1000)
    return dbm
# def dBm(full_scale_power_dbm,daps):
#     v=np.sqrt((2*50*10**(full_scale_power_dbm/10))/1000)*daps*1 # 1 : twpa readout amplitude  #opx1000 documentation
#     p_w=(v**2)/50
#     dbm=10*np.log10(p_w*1000)-10
#     return dbm +5.68 #5.68 calibrated through SA on 14/09
# ## pumpon
# def pumpoon_maxgain_res_spec(IQ_abs, qubits,  dfps, daps,dfs): #ok
#     sumresult=np.mean(pump_signal_snr(IQ_abs,qubits, dfps, daps, dfs),axis=0) # get avg on all qubit result
#     signal=sumresult[:,:,0]                                             # get only the signal data
#     maxsignal=np.unravel_index(np.argmax(signal),signal.shape)          # get the pump maxarg
#     spec=[]
#     for i in range(len(qubits)):
#         val = IQ_abs.values[i][maxsignal[0]][maxsignal[1]]
#         spec.append(val)
#     specs = np.array(spec)
#     return specs 
# def pumpoon_maxdsnr_res_spec(IQ_abs, qubits,  dfps, daps,dfs): #ok
#     sumresult=np.mean(pump_signal_snr(IQ_abs,qubits, dfps, daps,dfs),axis=0) # get avg on all qubit result
#     snr=sumresult[:,:,1]                                                # get only the snr data
#     maxsnr=np.unravel_index(np.argmax(snr),snr.shape)                   # get the pump maxarg
#     spec=[]
#     for i in range(len(qubits)):
#         val = IQ_abs.values[i][maxsnr[0]][maxsnr[1]]
#         spec.append(val)
#     specs = np.array(spec)
#     return specs 
# def pump_signal_snr(IQ_abs,qubits, dfps, daps, dfs): #0909
#     pumpoff_resspec = pumpoff_res_spec_per_qubit(IQ_abs, qubits, dfs, dfps)
#     pump_s_snr=np.zeros((len(qubits),len(dfps),len(daps),2))
#     for i in range(len(qubits)):
#         for j in range(len(dfps)):
#             for k in range(len(daps)):
#                 # average signal level
#                 spec=IQ_abs.values[i][j][k] 
#                 pump_s_snr[i,j,k,0]=np.mean(voltTOdbm(spec))
#                 # snr of resonator dip
#                 spec_min=spec[spec!=spec[pumpoff_resspec[i].argmin()]]
#                 signalsize=np.mean(spec_min)-spec[pumpoff_resspec[i].argmin()]
#                 noise=np.std(spec_min)
#                 snr=20*np.log10((signalsize/noise))
#                 pump_s_snr[i,j,k,1]=snr   
#     return pump_s_snr
# ## pump off
# def pumpoff_res_spec_per_qubit(IQ_abs, qubits, dfs, dfps): #ok
#     pump0_res_spec_per_qubit = np.zeros((len(qubits), len(dfs)), dtype=float)
#     for i in range(len(qubits)):
#         sum_pump_0=np.zeros(len(dfs))
#         for j in range(len(dfps)):
#                 pump_0=IQ_abs.values[i][j][0]            
#                 sum_pump_0+=pump_0
#         pump0_res_spec_per_qubit[i]=(sum_pump_0/len(dfps))
#     return pump0_res_spec_per_qubit
# def pumpzero_signal_snr(IQ_abs, dfs, qubits, dfps, daps): #ok 0910
#     pumpoff_resspec = pumpoff_res_spec_per_qubit(IQ_abs, qubits, dfs, dfps)    
#     pumpoff_s_snr=np.zeros((len(qubits),len(dfps),len(daps),2))
#     for i in range(len(qubits)):
#         for j in range(len(dfps)):
#             for k in range(len(daps)):
#                 pumpoff_s_snr[i,j,k,0]=np.mean(voltTOdbm(pumpoff_resspec[i]))
#                 pumpoff_s_snr[i,j,k,1]=res_snr(pumpoff_resspec[i])
#     return pumpoff_s_snr
# ## get optimized pump point
# def pump_maxgain(pumpon_signal_snr,dfps,daps):
#     sumresult=np.mean(pumpon_signal_snr,axis=0)
#     signal=sumresult[:,:,0]
#     maxsignal=np.unravel_index(np.argmax(signal),signal.shape)
#     maxgain_pump=np.array(np.array([dfps[maxsignal[0]],daps[maxsignal[1]]]))
#     return maxgain_pump
# def pump_maxdsnr(pumpon_signal_snr,dfps,daps):
#     sumresult=np.mean(pumpon_signal_snr,axis=0)
#     snr=sumresult[:,:,1]
#     maxsnr=np.unravel_index(np.argmax(snr),snr.shape)
#     maxsnr_pump=np.array(np.array([dfps[maxsnr[0]],daps[maxsnr[1]]]))
#     return maxsnr_pump

################ 250922 V2 #####################################
def dBm(full_scale_power_dbm,daps):
    v=np.sqrt((2*50*10**(full_scale_power_dbm/10))/1000)*daps*1 # 1 : twpa readout amplitude  #opx1000 documentation
    p_w=(v**2)/50
    dbm=10*np.log10(p_w*1000)-10
    return dbm +5.68 #5.68 calibrated through SA on 14/09
def pump_maxgain(gain, dfps, daps):
    avg_gain=np.mean(gain,axis=0)
    max_gain_idx=np.unravel_index(np.argmax(avg_gain), avg_gain.shape)
    max_gain_pump=np.array(np.array([dfps[max_gain_idx[0]],daps[max_gain_idx[1]]]))
    return max_gain_pump, max_gain_idx
def pump_maxdsnr(dsnr, dfps, daps):
    avg_dsnr=np.mean(dsnr,axis=0)
    max_dsnr_idx=np.unravel_index(np.argmax(avg_dsnr), avg_dsnr.shape)
    max_dsnr_pump=np.array(np.array([dfps[max_dsnr_idx[0]],daps[max_dsnr_idx[1]]]))
    return max_dsnr_pump, max_dsnr_idx
def mvTOdbm(mv):
    v=mv*1e-3
    rms_v=v/np.sqrt(2)
    p_watt=((rms_v)**2)/50
    dbm=10*np.log10(p_watt*1000)
    return dbm
def snr(ds, qubits, dfps, daps):
    noise=np.zeros((len(qubits),len(dfps),len(daps),1))
    signal=np.zeros((len(qubits),len(dfps),len(daps),1))
    for i in range(len(qubits)):
        for j in range(len(dfps)):
            for k in range(len(daps)):
                noise[i,j,k]=mvTOdbm(np.mean(ds.IQ_abs_noise.values[i][j][k]))
                signal[i,j,k]=mvTOdbm(ds.IQ_abs_signal.values[i][j][k][len(ds.IQ_abs_signal.values[i][j][k])//2])
    return signal-noise
# def gain(ds_pumpoff,ds_pumpon, qubits, dfps, daps):
#     signal_pumpoff=np.zeros((len(qubits),len(dfps),len(daps),1))
#     signal_pumpon=np.zeros((len(qubits),len(dfps),len(daps),1))
#     for i in range(len(qubits)):
#         for j in range(len(dfps)):
#             for k in range(len(daps)):
#                 signal_pumpoff[i,j,k]=mvTOdbm(ds_pumpoff.IQ_abs_signal.values[i][j][k][len(ds_pumpoff.IQ_abs_signal.values[i][j][k])//2])
#                 signal_pumpon[i,j,k]=mvTOdbm(ds_pumpon.IQ_abs_signal.values[i][j][k][len(ds_pumpon.IQ_abs_signal.values[i][j][k])//2])
#     return signal_pumpon-signal_pumpoff
################ 250928 V3 #####################################
def gain(ds_pumpoff,ds_pumpon, qubits, dfps, daps):
    signal_pumpoff=np.zeros((len(qubits),len(dfps),len(daps),1))
    signal_pumpon=np.zeros((len(qubits),len(dfps),len(daps),1))
    for i in range(len(qubits)):
        for j in range(len(dfps)):
            for k in range(len(daps)):
                signal_pumpoff[i,j,k]=voltTOdbm(np.mean(ds_pumpoff.IQ_abs_signal.values[i][j][k]))
                signal_pumpon[i,j,k]=voltTOdbm(np.mean(ds_pumpon.IQ_abs_signal.values[i][j][k]))
    return signal_pumpon-signal_pumpoff
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