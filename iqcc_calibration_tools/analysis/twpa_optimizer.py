import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from scipy.special import erf
import matplotlib.colors as mcolors
import math
from matplotlib.lines import Line2D
import os


def lin(db_value):
    return 10 ** (db_value / 10)
def fidelity(ro,t1,snr,prep): #snr=distance/(distribution of g,e)
    fidelity=np.exp(-ro/(2*t1))*erf(snr/(np.sqrt(2)))*prep
    return fidelity    
def dbm_to_peak_volts(dbm):
    impedance = 50
    rms_voltage = 10 ** ((dbm + 15 - 30 - 10 * np.log10(1 / impedance)) / 20)
    peak_voltage = rms_voltage * np.sqrt(2)
    return np.round(peak_voltage, 6)



def mindsnr_(ro): #241112 : this does not depend on the number and give consistent result
    f=fidelity(ro,t1,estm_snr*(ro/ro_off))
    f_max=max(f)
    saturation_index=np.where((f_max-f)*100<0.1)
    saturation_dsnr = min(dsnr[saturation_index[0]])
    return saturation_dsnr
def ftro(tro):
    estm=np.exp(-(tro/(2*t1)))*erf((snr_off*(np.max(twpa_fct)*np.sqrt(tro/ro_off))/np.sqrt(2))) 
    return np.round(estm*100,2)
def readout(kappa):
    tro=np.linspace(0.04,6,2000)
    estm=np.exp(-(tro/(2*t1)))*erf((snr_off*(np.max(twpa_fct)*np.sqrt(tro/ro_off))/np.sqrt(2)))*prep #prep added 250629
    plt.figure(figsize=(4,3),dpi=1200)
    plt.plot(tro, estm*100, label= f'Estimated fidelity')
    kappalimit_tro=(5/(kappa*2*np.pi))*1e6
    kappalimit_f=np.exp(-(kappalimit_tro/(2*t1)))*erf((snr_off*(np.max(twpa_fct)*np.sqrt(kappalimit_tro/ro_off))/np.sqrt(2))) * 100
    plt.axvline(kappalimit_tro,color='r',label=f'$\kappa$ limit: {np.round(kappalimit_f,2)}% @ {np.round(kappalimit_tro*1e3,0)}ns' )
    # plt.plot(tro[np.argmax(estm)], np.max(estm) * 100, color='black', marker='o', markersize=2, label=f'Fmax={np.round(np.max(estm)*100,2)}% @ {np.round(tro[np.argmax(estm)]*1e3,0)}ns')
    plt.legend()
    plt.ylabel('Fidelity[%]')
    # plt.ylim(80,100)
    plt.xlabel(r'$T_{\mathrm{RO}}$[us]')
    plt.title(r'$F(T_{\mathrm{RO}})$')
def fast_highfidelity(kappa):
    kappa=kappa*2*np.pi
    ro_lim=(3/kappa)*1e6
    dsnrTH0 = mindsnr(ro_off)
    f0 = fidelity(ro_off, t1, snr_off * twpa_fct *prep)
    tro = np.arange(ro_off, 0, -0.01)
    n_=[]
    fmax=[]        
    if dsnrTH0 < max(dsnr):                
            for n in range(len(tro)-1):
                # fidelity at different Tro can be estimated based on Tro_off by multiplying sqrt(Tro/Tro_off) to already known snroff(Tro_off)
                f_ = fidelity(tro[n+1], t1, snr_off * np.sqrt((tro[n+1] / ro_off)) * twpa_fct * prep)
                if (max(f_) > max(f0)):
                    n_.append(n+1)
                    fmax.append(max(f_))      
                elif max(f_)<=max(f0):
                    pass
    index=fmax.index(max(fmax))
    index=n_[index]
    plt.figure(figsize=(4,3),dpi=1200)
    n=len(n_)
    ro=np.array([ro_off,tro[n],tro[index],ro_lim])
    snroff=np.array([snr_off,snr_off*np.sqrt((tro[n]/ro_off)),snr_off*np.sqrt((tro[index]/ro_off)),snr_off*np.sqrt(ro_lim/ro_off)])
    if ro_lim>tro[n]:
        ro=np.array([ro_off,tro[index],ro_lim])
        snroff=np.array([snr_off,snr_off*np.sqrt((tro[index]/ro_off)),snr_off*np.sqrt(ro_lim/ro_off)])
    else:
            ro=ro
            snroff=snroff
    if ro_lim>tro[index]:
        ro=np.array([ro_off,ro_lim])
        snroff=np.array([snr_off,snr_off*np.sqrt(ro_lim/ro_off)])
    else:
            ro=ro
            snroff=snroff
    c=['black','purple','green','orange']
    for j in range(len(ro/t1)):                      
                    f=np.exp(-(ro/(2*t1))[j]) * erf((snroff[j]/(np.sqrt(2))*(np.sqrt(10**(dsnr/10)))))*prep
                    f_max=max(f)
                    saturation_index=np.where((f_max-f)*100<0.1)
                    saturation_dsnr = min(dsnr[saturation_index[0]])
                    plt.scatter(dsnr[dsnr>0],f[dsnr>0]*100,s=4,color=c[j], label=f'$T_{{RO}}$={ro[j]*1e3:.0f}ns,$\\Delta$SNR={saturation_dsnr:.2f}dB : F~{max(f)*100:.2f}% ')
                    plt.axvline(saturation_dsnr, color=c[j],linewidth=2, linestyle='--')#,label=f'$\\Delta$SNR={saturation_dsnr:.2f}dB')
    plt.ylabel('Fidelity[%]',fontsize=16)
    plt.xlabel('$\\Delta$SNR[dB]',fontsize=16)
    plt.tick_params('both',labelsize=16)
    plt.legend(fontsize=7,loc='lower left')
    ##### infidelity ########## 
    plt.figure(figsize=(4,3),dpi=1200)
    # ro=np.array([tro[n],tro[index]])
    # snroff=np.array([snr_off*np.sqrt((tro[n]/ro_off)),snr_off*np.sqrt((tro[index]/ro_off))])
    c=['black','purple','green','orange']
    # f=np.exp(-(ro_off/(2*t1))) * erf((snr_off/(np.sqrt(2))*(np.sqrt(10**(dsnr/10)))))*prep
    # plt.scatter(dsnr[dsnr>(max(dsnr)-2)],100-(f[dsnr>(max(dsnr)-2)]*100),s=4,color='black', label=f'$T_{{RO}}={ro_off*1e3:.0f}ns, F_{{max}}={max(f)*100:.2f}$%')
    for j in range(len(ro/t1)):                      
                    f=np.exp(-(ro/(2*t1))[j]) * erf((snroff[j]/(np.sqrt(2))*(np.sqrt(10**(dsnr/10)))))*prep
                    f_max=max(f)
                    saturation_index=np.where((f_max-f)*100<0.1)
                    saturation_dsnr = min(dsnr[saturation_index[0]])
                    plt.scatter(dsnr[dsnr>(max(dsnr)-5)],100-(f[dsnr>(max(dsnr)-5)]*100),s=4,color=c[j], label=f'$T_{{RO}}$={ro[j]*1e3:.0f}ns, $\\Delta$SNR={saturation_dsnr:.2f}dB : F~{max(f)*100:.2f}%')
                    # plt.axvline(saturation_dsnr, color=c[j],linewidth=2, linestyle='--')#,label=f'$\\Delta$SNR={saturation_dsnr:.2f}dB')
    plt.ylabel('Infidelity[%]',fontsize=16)
    plt.xlabel('$\\Delta$SNR[dB]',fontsize=16)
    plt.tick_params('both',labelsize=16)
    plt.legend(fontsize=7,loc='upper right') 
    # save_dir = r"C:\Users\wjd__\OneDrive - QM Machines LTD\바탕 화면\Fig" 
    # os.makedirs(save_dir, exist_ok=True)
    # plt.savefig(os.path.join(save_dir, "f_sat_.png"), dpi=1200,bbox_inches='tight')

def estm_f_space(self):   
    ########## SNR Gs space #############################   
    plt.figure(figsize=(5,3),dpi=1200)
    print(f"minF={np.nanmin(estm_f)*100:.2f}")
    print(f"maxF={np.nanmax(estm_f)*100:.2f}")
    print(f"offF={np.round(fid_off*100,2)}")
    plt.xlim(0,gainlimit)
    plt.ylim(0,snrlimit)   
    ff_gs=gs[gs>dsnr]#f_gs[f_gs>f_dsnr]      # only 
    ff_dsnr=dsnr[gs>dsnr]#f_dsnr[f_gs>f_dsnr]  # g > dsnr 
    ff_c=estm_f[gs>dsnr]#f_c[f_gs>f_dsnr] 
    # f_gs=gs[estm_f*100>off_f]
    # f_dsnr=dsnr[estm_f*100>off_f]
    # f_c=estm_f[estm_f*100>off_f]
    scatter=plt.scatter(ff_gs,ff_dsnr,s=4,c=ff_c*100,vmin=off_f)#,label=r'$F_{estm}=F_{RO}(SNR_{off}*\sqrt{10^{(G_s - \Delta N) / 10}})$')#, vmin=vmin, vmax=vmax,label=r'$F_{estm}=F_{RO}(SNR_{off}*\sqrt{Gs/{\Delta}N})$')      
    plt.colorbar(scatter)#, label=rf'$F_{{estm}} [\%]$')
    plt.xlabel('$G_s$[dB]',fontsize=12)
    plt.ylabel('$\\Delta$SNR=Gs-$\\Delta$N[dB]',fontsize=12)
    plt.title('Estimated Fidelity')
    # plt.legend(fontsize=6,loc='upper left')
    ########## N, Gs space #######################
    plt.figure(figsize=(5,3),dpi=1200)
    # filter dn space to give only when gs>dn (dsnr>0)
    filtered_gs=gs[gs>dn]
    filtered_dn=dn[gs>dn]
    filtered_c=estm_f[gs>dn]
    scatter_=plt.scatter(filtered_gs,filtered_dn,c=filtered_c*100, s=4)#,vmin=vmin,vmax=vmax,label=r'$F_{estm}=F_{RO}(SNR_{off}*\sqrt{Gs/{\Delta}N})$')
    plt.colorbar(scatter_, label=r'$F_{estm}=F_{RO}(SNR_{off}*\sqrt{10^{(G_s - \Delta N) / 10}})$')
    plt.legend(fontsize=6)
    plt.xlabel('$G_s$[dB]',fontsize=12)
    plt.ylabel('$\\Delta$N[dB]',fontsize=12)
    plt.xlim(0,gainlimit)
    plt.ylim(0,np.max(dn)+1)
    plt.title('Estimated Fidelity on Gs, $\\Delta$N space')
    ############### pump space ##################################
    pump=pd.read_csv(gain)['pump']
    frequency=[]
    power=[]
    for i in range(len(pump)):
        f=eval(pump[i])[0]
        p=eval(pump[i])[1]
        frequency.append(f)
        power.append(p)
    plt.figure(figsize=(5,4))
    plt.title('Estimated fidelity per Pumping Point')  
    scatter=plt.scatter(np.array(power)-62, frequency, c=estm_f*100,s=400,marker='s')#,vmin=vmin,vmax=vmax)# ,label=f'$F_{{estm}}\\%$')
    plt.colorbar(scatter,label=rf'$F_{{estm}} [\%]$')
    plt.xlabel('Pp[dBm]')
    plt.ylabel('fp[MHz]')
    plt.gca().invert_yaxis() 
def f_minsnr(write):   
    ########## SNR Gs space #############################   
    min=min_dsnr()    
    plt.figure(figsize=(5,3))
    print(f"minF={np.nanmin(estm_f)*100:.2f}")
    print(f"maxF={np.nanmax(estm_f)*100:.2f}")
    print(f"offF={np.round(fid_off*100,2)}")
    plt.xlim(0,gainlimit)
    plt.ylim(0,snrlimit)   
    f_gs=gs[estm_f*100>off_f]
    f_dsnr=dsnr[estm_f*100>off_f]
    f_c=estm_f[estm_f*100>off_f]
    # filter : g>dsnr (dn>0)
    ff_gs=f_gs[f_gs>f_dsnr]      
    ff_dsnr=f_dsnr[f_gs>f_dsnr]  
    ff_c=f_c[f_gs>f_dsnr]        
    print(np.min(ff_c))
    scatter=plt.scatter(ff_gs,ff_dsnr,s=4,c=ff_c*100,label=r'$F_{estm}=F_{RO}(SNR_{off}*\sqrt{10^{(G_s - \Delta N) / 10}})$')#, vmin=vmin, vmax=vmax,label=r'$F_{estm}=F_{RO}(SNR_{off}*\sqrt{Gs/{\Delta}N})$')
    plt.axhline(min, color='red',linestyle='--')#,label = f'$\\Delta \\mathrm{{SNR}}_{{\\text{{SAT}}}} = {min:.2f}$')
    # plt.legend(fontsize=6)
    plt.colorbar(scatter, label=rf'$F_{{estm}} [\%]$')
    plt.xlabel('Gain[dB]')
    plt.ylabel('$\\Delta$SNR=Gs-$\\Delta$N[dB]')
    plt.title('Estimated Fidelity')
    # plt.legend(fontsize=6,loc='upper left')
    ############### pump space ##################################
    pump=pd.read_csv(gain)['pump']
    frequency=[]
    power=[]
    for i in range(len(pump)):
        f=eval(pump[i])[0]
        p=eval(pump[i])[1]
        frequency.append(f)
        power.append(p)
    plt.figure(figsize=(5,4))
    plt.title('Estimated fidelity per Pumping Point')  
    scatter=plt.scatter(np.array(power)-62, frequency, c=estm_f*100,s=45,marker='s')#,vmin=off_f)#vmax=vmax)# ,label=f'$F_{{estm}}\\%$')
    indices=np.where((((dsnr))>=min) & (gs>dsnr))
    pumppoint=np.array(pump)[indices[0]] 
    p_=[]
    f_=[]
    for i in range(len(pumppoint)):
        f_.append(eval(pumppoint[i])[0])
        p_.append(eval(pumppoint[i])[1])
    plt.scatter(np.array(p_)-62,f_ ,color='red',marker='x',s=20,label = f'$\\Delta \\mathrm{{SNR}} \\geq {min:.2f}$')
    plt.legend(fontsize=6,loc='lower right')
    plt.colorbar(scatter,)
    plt.xlabel('Pp[dBm]')
    plt.ylabel('fp[MHz]')
    plt.gca().invert_yaxis() 
    ##############################
    bad=np.where(((dsnr)<min-1)&(gs>dsnr))
    badpoint= np.array(pump)[bad[0]] 
    p_bad=[]
    f_bad=[]
    for i in range(len(badpoint)):
        f_bad.append(eval(badpoint[i])[0])
        p_bad.append(eval(badpoint[i])[1])
    plt.figure(figsize=(4,4))
    frequency_bad=[]
    power_bad=[]
    for i in range(len(pump)):
        f=eval(pump[i])[0]
        p=eval(pump[i])[1]
        frequency_bad.append(f)
        power_bad.append(p)
    ##############################
    if write==True:
        for i in range(len(indices[0])):
            print(f"good={pumppoint[i],np.round(gs[indices[0][i]],1),np.round(dsnr[indices[0][i]],1)}")
        print('-----------------------------')
        for i in range(len(bad[0])):
            print(f"bad={badpoint[i],np.round(gs[bad[0][i]],1),np.round(dsnr[bad[0][i]],1)}")            
def dBm(full_scale_power_dbm,daps):
    v=np.sqrt((2*50*10**(full_scale_power_dbm/10))/1000)*daps*1 # 1 : twpa readout amplitude  #opx1000 documentation
    p_w=(v**2)/50
    dbm=10*np.log10(p_w*1000)-10
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
    ##### 1.max a_ro_on_max s.t) #(mtpx Ro)*Pon < Psat-10 
    ##### get the da , theoretically can be used 
    a_ro=np.linspace(0,1,500)
    ps_on=dBm(qubits[0].resonator.opx_output.full_scale_power_dbm,
            a_ro*qubits[0].resonator.operations["readout"].amplitude)-60-6-5
    mtpx_ps_on= ps_on + 10*np.log10(len(qubits))
    idx=np.where(mtpx_ps_on<twpas[0].p_saturation-20)[0][-1]
    a_ro_on_max=a_ro[idx]
    ###### 2. a_ro_off~a_ro_off*a_ro_on*sqrt(linG)
    ###### pick the Gain which compensate the lowered readout amplitude up to the readout amplitude when twpa is off
    g=np.linspace(0,25,500)
    a_ro_off=1 # value doesnt matter
    idx_=np.where(a_ro_off<a_ro_off*a_ro_on_max*np.sqrt(10**(g/10)))
    minimum_gain=g[idx_[0][0]]
    return minimum_gain

def optimizer(mingain, mindsnr, gain_avg, dsnr_avg, daps, dfps, p_lo,p_if):
    mask = gain_avg > mingain
    masked_dsnr = np.where(mask, dsnr_avg, -np.inf)
    flat_index = np.argmax(masked_dsnr)
    idx = np.unravel_index(flat_index, dsnr_avg.shape)
    print(f"Optimized ap={np.round(daps[idx[1]],5)},fp={np.round((p_lo+p_if+dfps[idx[0]])*1e-9,3)}GHz ")
    print(f"gain_avg :{np.round(gain_avg[idx],2)}dB")
    print(f"dsnr_avg :{np.round(dsnr_avg[idx],2)}dB")
    return idx