import numpy as np
import matplotlib.pyplot as plt

# gain, snr into db
def res_snr(spec):#ok
    base=spec[spec!=spec.min()]
    signalsize=np.mean(base)-np.min(spec)
    noise=np.std(base)
    snr=20*np.log10((signalsize/noise))
    return snr
def voltTOdbm(volt):
    p_w=(volt**2)/50
    dbm=10*np.log10(p_w*1000)
    return dbm
def dBm(full_scale_power_dbm,daps):
    v=np.sqrt((2*50*10**(full_scale_power_dbm/10))/1000)*daps*1 # 1 : twpa readout amplitude  #opx1000 documentation
    p_w=(v**2)/50
    dbm=10*np.log10(p_w*1000)-10
    return dbm +5.68 #5.68 calibrated through SA on 14/09

def mvTOdbm(mv):
    v=mv*1e-3
    rms_v=v/np.sqrt(2)
    p_watt=((rms_v)**2)/50
    dbm=10*np.log10(p_watt*1000)
    return dbm
def snr():
    noise=np.zeros((len(qubits),len(dfps),len(daps),1))
    signal=np.zeros((len(qubits),len(dfps),len(daps),1))
    for i in range(len(qubits)):
        for j in range(len(dfps)):
            for k in range(len(daps)):
                noise[i,j,k]=mvTOdbm(np.mean(ds.IQ_abs_noise.values[i][j][k]))
                signal[i,j,k]=mvTOdbm(ds.IQ_abs_signal.values[i][j][k][len(ds.IQ_abs_signal.values[i][j][k])//2])
    return signal-noise

## pumpon
def pumpoon_maxgain_res_spec(IQ_abs, qubits,  dfps, daps,dfs): #ok
    sumresult=np.mean(pump_signal_snr(IQ_abs,qubits, dfps, daps, dfs),axis=0) # get avg on all qubit result
    signal=sumresult[:,:,0]                                             # get only the signal data
    maxsignal=np.unravel_index(np.argmax(signal),signal.shape)          # get the pump maxarg
    spec=[]
    for i in range(len(qubits)):
        val = IQ_abs.values[i][maxsignal[0]][maxsignal[1]]
        spec.append(val)
    specs = np.array(spec)
    return specs 
def pumpoon_maxdsnr_res_spec(IQ_abs, qubits,  dfps, daps,dfs): #ok
    sumresult=np.mean(pump_signal_snr(IQ_abs,qubits, dfps, daps,dfs),axis=0) # get avg on all qubit result
    snr=sumresult[:,:,1]                                                # get only the snr data
    maxsnr=np.unravel_index(np.argmax(snr),snr.shape)                   # get the pump maxarg
    spec=[]
    for i in range(len(qubits)):
        val = IQ_abs.values[i][maxsnr[0]][maxsnr[1]]
        spec.append(val)
    specs = np.array(spec)
    return specs 
# def pump_signal_snr(IQ_abs,qubits, dfps, daps): #ok
#     pump_s_snr=np.zeros((len(qubits),len(dfps),len(daps),2))
#     for i in range(len(qubits)):
#         for j in range(len(dfps)):
#             for k in range(len(daps)):
#                 spec=IQ_abs.values[i][j][k]
#                 pump_s_snr[i,j,k,0]=np.mean(voltTOdbm(spec))   
#                 pump_s_snr[i,j,k,1]=res_snr(spec) 
#     return pump_s_snr
def pump_signal_snr(IQ_abs,qubits, dfps, daps, dfs): #0909
    pumpoff_resspec = pumpoff_res_spec_per_qubit(IQ_abs, qubits, dfs, dfps)
    pump_s_snr=np.zeros((len(qubits),len(dfps),len(daps),2))
    for i in range(len(qubits)):
        for j in range(len(dfps)):
            for k in range(len(daps)):
                # average signal level
                spec=IQ_abs.values[i][j][k] 
                pump_s_snr[i,j,k,0]=np.mean(voltTOdbm(spec))
                # snr of resonator dip
                spec_min=spec[spec!=spec[pumpoff_resspec[i].argmin()]]
                signalsize=np.mean(spec_min)-spec[pumpoff_resspec[i].argmin()]
                noise=np.std(spec_min)
                snr=20*np.log10((signalsize/noise))
                pump_s_snr[i,j,k,1]=snr   
    return pump_s_snr
## pump off
def pumpoff_res_spec_per_qubit(IQ_abs, qubits, dfs, dfps): #ok
    pump0_res_spec_per_qubit = np.zeros((len(qubits), len(dfs)), dtype=float)
    for i in range(len(qubits)):
        sum_pump_0=np.zeros(len(dfs))
        for j in range(len(dfps)):
                pump_0=IQ_abs.values[i][j][0]            
                sum_pump_0+=pump_0
        pump0_res_spec_per_qubit[i]=(sum_pump_0/len(dfps))
    return pump0_res_spec_per_qubit
def pumpzero_signal_snr(IQ_abs, dfs, qubits, dfps, daps): #ok 0910
    pumpoff_resspec = pumpoff_res_spec_per_qubit(IQ_abs, qubits, dfs, dfps)    
    pumpoff_s_snr=np.zeros((len(qubits),len(dfps),len(daps),2))
    for i in range(len(qubits)):
        for j in range(len(dfps)):
            for k in range(len(daps)):
                pumpoff_s_snr[i,j,k,0]=np.mean(voltTOdbm(pumpoff_resspec[i]))
                pumpoff_s_snr[i,j,k,1]=res_snr(pumpoff_resspec[i])
    return pumpoff_s_snr
### get optimized pump point
def pump_maxgain(pumpon_signal_snr,dfps,daps):
    sumresult=np.mean(pumpon_signal_snr,axis=0)
    signal=sumresult[:,:,0]
    maxsignal=np.unravel_index(np.argmax(signal),signal.shape)
    maxgain_pump=np.array(np.array([dfps[maxsignal[0]],daps[maxsignal[1]]]))
    return maxgain_pump
def pump_maxdsnr(pumpon_signal_snr,dfps,daps):
    sumresult=np.mean(pumpon_signal_snr,axis=0)
    snr=sumresult[:,:,1]
    maxsnr=np.unravel_index(np.argmax(snr),snr.shape)
    maxsnr_pump=np.array(np.array([dfps[maxsnr[0]],daps[maxsnr[1]]]))
    return maxsnr_pump


