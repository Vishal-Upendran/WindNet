import numpy as np

def Mark_ge50(sw):
    #Mark all time points which are more than 50 km/s faster than 1 day earlier. 
    sw1=np.zeros(len(sw))
    sw1[1:]=sw[1:]-sw[:-1]
    sw1_50=np.where(sw1 > 50)[0]
    return sw1_50
def Isolated_elim(sw1_50):
    #Eliminate any isolated single data points which are marked.
    #I know this method is dumb, but it works, and I am too lazy. 
    sw2=[]
    i=0
    while i<len(sw1_50)-1:
        if sw1_50[i+1]!=sw1_50[i]+1:
            if i!=0 and sw1_50[i]!=sw1_50[i-1]+1:
                sw2.append(sw1_50[i])
            elif i==0:
                sw2.append(sw1_50[i])
            else:
                pass
        i+=1
    if sw2:
        sw2=sw1_50[np.where(sw1_50!=sw2)]
    else:
        sw2=sw1_50
    return sw2
def Group(sw2):
    #Group each contiguous block of marked points as a distinct high speed enhancement (HSE) 
    sw3=[]
    i=0
    while i<len(sw2)-1:
        start=sw2[i]
        while i<len(sw2)-1:
            if (sw2[i+1]==sw2[i]+1):
                i+=1
            else:
                break
        end=sw2[i]
        i+=1
        sw3.append([start,end])
    return sw3
def Vel_time(sw,sw3):
    #For each HSE, find the minimum speed starting 2 days ahead of the HSE till the start of the HSE, 
    #and mark it as the minimum speed (Vmin) of the HSE; 
    #find the maximum speed starting from the beginning of the HSE through 1 day after the HSE and 
    #mark it as the maximum speed (Vmax) of the HSE. 
    #For each HSE, find the last time reaching Vmin and the first time reaching Vmax 
    #and mark them as the start and end time of an SIR. 
    Velocities=[]
    Times=[]
    if sw3:
        for hse in sw3:
            #try:
            min_index=np.max([0,hse[0]-2])
            max_index=np.min([hse[1]+2,len(sw)])
            vmin_ind=min_index+np.where(sw[min_index:hse[0]+1]==np.min(sw[min_index:hse[0]+1]))[0][-1]
            vmin=sw[vmin_ind]
            vmax_ind=hse[0]+np.where(sw[hse[0]:max_index]==np.max(sw[hse[0]:max_index]))[0][0]
            vmax=sw[vmax_ind]
#             except:
#                 try:
#                     vmin_ind=np.where(sw==np.min(sw[hse[0]-1:hse[0]+1]))[0][-1]
#                     vmin=sw[vmin_ind]
#                     vmax_ind=np.where(sw==np.max(sw[hse[0]:hse[1]+2]))[0][0]
#                     vmax=sw[vmax_ind]
#                 except:
#                     vmin_ind=np.where(sw==np.min(sw[hse[0]-2:hse[0]+1]))[0][-1]
#                     vmin=sw[vmin_ind]
#                     vmax_ind=np.where(sw==np.max(sw[hse[0]:hse[1]+1]))[0][0]
#                     vmax=sw[vmax_ind]
            Times.append([vmin_ind,vmax_ind])
            Velocities.append([vmin,vmax])
    return Velocities,Times
def Final_vet(sw,Velocities,Times):
    #For the regrouped SIRs, find the Vmin and Vmax for each SIR and mark the last time of highest speed gradient as 
    #the stream interface (SI), the boundary between slow and fast wind
    #Reject any SIRs with Vmin faster than 500 km/s, or Vmax slower than 400 km, or speed increase less than 100 km/s
    SI_times=[]
    Vel_final=[]
    Time_final=[]
    if Times: #can have Velocities also
        for v,ind in zip(Velocities,Times):
            in_0=np.max([0,ind[0]])
            in_1=np.min([ind[1],len(sw)-2])
            grad_loc=np.argmax(sw[in_0+1:in_1+1]-sw[in_0:in_1])
            si=in_0+grad_loc
            if v[0]>500 or v[1]<400 or (v[1]-v[0])<100:#(sw[1:]-sw[:-1])[si]<100:
                pass
            else:
                SI_times.append(si+0.5)
                Vel_final.append(v)
                Time_final.append(ind)
        #print SI_times    
    return Vel_final,Time_final,SI_times
def HSE(sw):
    sw1_50=Mark_ge50(sw)
    sw2=Isolated_elim(sw1_50)
    sw3=Group(sw2)
    Velocities,Times=Vel_time(sw,sw3)
    Vel_final,Time_final,SI_times=Final_vet(sw,Velocities,Times)
    return Vel_final,Time_final,SI_times
def Overlap_comp(ind_sim,obs):
    ind_flag=[]
    for ind in obs:
        overlap=np.min([ind[1],ind_sim[1]])-np.max([ind[0],ind_sim[0]])
        ind_flag.append(overlap)
    return obs[np.argmax(ind_flag)],np.max(ind_flag)
def Compare_sim_obs(sim,obs):
    Vel_sim,Time_sim,SI_sim = HSE(sim)
    Vel_obs,Time_obs,SI_obs = HSE(obs)
    obs_times=[]
    obs_flag=[]
    if Time_sim:
        for times in Time_sim:
            if Time_obs:
                tm,flg=Overlap_comp(times,Time_obs)
                obs_times.append(tm)
                obs_flag.append(flg)
    return np.asarray(obs_times),np.asarray(obs_flag),SI_sim,SI_obs
def BatchwiseHSE(data,history,delay):
    TP=0
    FP=0
    FN=0
    a1=[]
    a2=[]
    a3=[]
    a4=[]
    for i in np.arange(0,data.shape[0],20-history-delay):
        obs_times,obs_flag,t_sim,t_obs=Compare_sim_obs(data[i:i+20-history-delay,1],data[i:i+20-history-delay,0])
        a1.append(obs_times)
        a2.append(obs_flag)
        a3.append(t_sim+i)
        a4.append(t_obs+i)
        tp_batch=len(np.where(obs_flag>0.0)[0])
        TP+=tp_batch
        FN+=len(t_sim)-tp_batch
        FP+=len(t_obs)-tp_batch
    return TP,FP,FN
    