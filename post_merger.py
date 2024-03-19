import bilby 
from context import tbilby
from gwpy.timeseries import TimeSeries

from scipy.interpolate import interp1d
from bilby.core.prior import Prior, Uniform, Interped
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bilby.core.likelihood import GaussianLikelihood
from bilby.core.prior import ConditionalLogUniform, LogUniform,Uniform
from scipy import interpolate
import arviz as az
import inspect
from filelock import FileLock
import tarfile
import ValidationTools
import h5py
from scipy.signal import tukey



cc = 299792458.0  # speed of light in m/s
GG = 6.67384e-11  # Newton in m^3 / (kg s^2)
Msun = 1.98855 * 10 ** 30  # solar mass in  kg
kg = 1. / Msun
metre = cc ** 2 / (GG * Msun)
secs = cc * metre
Mpc = 3.08568e+22  # Mpc in metres


def calculate_mode_waveform(f, T, phi, alpha, A, tpos, Two_pi_dt, dplus, dcross):

    Amp = np.exp(-tpos / T) * A
    angle = Two_pi_dt * f * (1 + alpha * tpos) + phi
    if dplus is not None:
        dplus = Amp * np.sin(angle)
    if dcross is not None:
        dcross = Amp * np.cos(angle)
    return dplus, dcross

def time_domain_damped_sinusoid(time, **kwargs):
    """
    Three damped sinusoidal waveforms, h+:cos, hx:sin.
    """

    #     print('****',kwargs)

    waveform_order =3 
    
    plus = np.zeros(len(time))
    cross = plus.copy()
    t_0 = kwargs['t_0'] / 1000.0
    #     print(t_0)
    dt = time - t_0
    tpos = dt[dt >= 0]
    dplus = np.zeros(len(tpos))
    dcross = dplus.copy()
    Two_pi_dt = 2 * np.pi * tpos
    dtpos = tpos[-1] - tpos[0]
    window = 1
    for waveform_number in range(waveform_order):
        if waveform_number == 2:
            A = (1 - kwargs['w_0'] - kwargs['w_1']) * 10 ** kwargs['logB']
        else:
            A = kwargs[f'w_{waveform_number}'] * 10 ** kwargs['logB']
        f = kwargs[f'f_{waveform_number}']
        T = kwargs[f'T_{waveform_number}'] / 1000.0
        phi = kwargs[f'phi_{waveform_number}']
        alpha = kwargs[f'alpha_{waveform_number}']
        Amp = np.exp(-tpos / T) * A
        # note alpha is multiplied by (t-t_0)**2
        angle = Two_pi_dt * f * (1 + alpha * tpos) + phi
        ddplus, ddcross = calculate_mode_waveform(kwargs[f'f_{waveform_number}'],
                                                    kwargs[f'T_{waveform_number}'] / 1000.0,
                                                    kwargs[f'phi_{waveform_number}'],
                                                    kwargs[f'alpha_{waveform_number}'],
                                                    A, tpos, Two_pi_dt, dplus, dcross)
        dplus += ddplus
        dcross += ddcross
    plus[dt >= 0] = dplus * window
    cross[dt >= 0] = dcross * window
    #     print(sum(plus))
    # plt.figure()
    # plt.plot(time,plus)
    # plt.xlim(0.032, 0.042)
    # plt.show()
    # 1/0
    return {'plus': plus, 'cross': cross}


def m_sol_to_geo(mm):
    # convert from solar masses to geometric units
    return mm / kg * GG / cc ** 2

def dist_Mpc_to_geo( dist):
    # convert distance from Mpc to geometric units (i.e., metres)
    return dist * Mpc

def time_geo_to_s(time):
    # convert time from seconds to geometric units
    return time / cc

def numerical_relativity_postmerger_waveform(newtime, **waveform_kwargs):

    
    #loudness = waveform_kwargs['loudness']
    #t_inspiral_ms = waveform_kwargs['t_inspiral_ms']
    #tukey_rolloff = waveform_kwargs['tukey_rolloff']

    t_0 = 0#waveform_kwargs['t_0']
    snr=50
    loudness = 2.512e-20 * snr / 50
    
    t_0=0
    
    location =r'/Users/ngut0001/Library/CloudStorage/OneDrive-MonashUniversity/HyperBilby-main (2)/HyperBilby-main/gitworkdir/post_meregr/NRtars/THC_0036-master/R01/'
    with h5py.File(location+'data.h5', 'r') as f:
        rh22 = f['rh_22']
        keys = list(rh22.keys())
        h22furthest = [key for key in keys if (
            'Rh_l2_m2' in key) and not ('Inf' in key)][-1]
        rh22data = rh22[h22furthest]
        hr_pl_msun = rh22data[:, 1] * 2 * 2 ** 0.5 # This scaling is an approximation
        hr_cr_msun = rh22data[:, 2] * 2 * 2 ** 0.5 # This scaling is an approximation
        time_msun = rh22data[:, 8]
    full_wave = (hr_pl_msun ** 2 + hr_cr_msun ** 2) ** 0.5
    postmerger_start_index = np.argmax(full_wave)
    
    # gr = GRutils()
    hscale = loudness * 1.2 / np.max([
         np.max(np.abs(hr_pl_msun[postmerger_start_index:])),
         np.max(np.abs(hr_cr_msun[postmerger_start_index:]))])  # max value should be ~max(|h+|,|hx|)
    tscale = time_geo_to_s(m_sol_to_geo(1))

    hre = hr_pl_msun * hscale
    him = hr_cr_msun * hscale
    # # th = 0 @ merger
    th = (time_msun - time_msun[postmerger_start_index]) * tscale
    tstartindex = np.argmax(th > - 0)
    hrenew = hre[tstartindex:]
    himnew = him[tstartindex:]
    thnew = th[tstartindex:] + t_0  # postmerger started @ t=0 now @ t_0

    hplus_interp_func = interp1d(thnew,
                                 hrenew,
                                 bounds_error=False, fill_value=0)

    hcross_interp_func = interp1d(thnew,
                                 himnew,
                                 bounds_error=False, fill_value=0)
    
    
    
    #time = newtime - newtime[0]
    #

    #     redefine tstartindex based on interpolated data
    tstartindex = np.argmax(newtime >= t_0 )
    tout = newtime[tstartindex:] # this is taken care of by the interpolation 
    
    hplus = np.zeros(newtime.shape)
    hcross = np.zeros(newtime.shape)
    
    
    window_dt = tout[-1]-tout[0]
    tukey_rolloff_ms=  0.2/1000    
    window = tukey(len(tout), 2 * tukey_rolloff_ms / window_dt) 
    
    
    hplus[tstartindex:]  = hplus_interp_func(tout) * window
    hcross[tstartindex:] = hcross_interp_func(tout) * window
    
        
    #time =newtime.copy()
    #time = newtime - newtime[0]
    #st_inx = np.argmax(time>=t_0)
    #hplus = np.zeros(time.shape)
    #hcross = np.zeros(time.shape)
    
    #hplus[st_inx:] = hplus_interp_func(time[st_inx:])
    #hcross[st_inx:] = hcross_interp_func(time[st_inx:])


    return {'plus': hplus, 'cross': hcross}


def Paulmodel(time,**args):
    # load the file and according to time series return the NR model 
   # h_complex = np.zeros(x.shape)
   # ret={}
   # ret['cross'] = np.real(h_complex)
   # ret['plus'] = np.imag(h_complex)
    ret = time_domain_damped_sinusoid(time=time,**args)
    
    return ret

def NRmodel(time,**args):
    # load the file and according to time series return the NR model 
   # h_complex = np.zeros(x.shape)
   # ret={}
   # ret['cross'] = np.real(h_complex)
   # ret['plus'] = np.imag(h_complex)
    ret = numerical_relativity_postmerger_waveform(newtime=time,**args)
    
    return ret


class TransdimensionalConditionalUniform_f(tbilby.core.prior.TransdimensionalConditionalUniform):   
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value
            minimum = self.minimum
            if(len(self.f_Hz)>0): # handle the first mu case
                minimum = self.f_Hz[-1] # set the minimum to be the location of the last peak 
                           
            return dict(minimum=minimum)


class TransdimensionalConditionalLogUniform_A(tbilby.core.prior.TransdimensionalConditionalLogUniform):
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value
            minimum = self.minimum
            if(len(self.A)>0): # handle the first mu case
                minimum = self.A[-1] # set the minimum to be the location of the last peak

            return dict(minimum=minimum)





def dumped_sin(x,t0,A,tau,f_Hz,alpha,phi_rad):  
    
  plus = np.zeros(len(x))
  cross = plus.copy()
    
  
  time_inx = x.copy() >= t0    
  t = x.copy()[time_inx] - t0 # start from zero offset 
      
  
  h_complex = A*np.exp(-t/tau)*np.exp(1j *( 2*np.pi*f_Hz*t*(1+alpha*t) + phi_rad))
  #h_complex = A*np.exp(-t)*np.exp(1j * 2*np.pi*f_Hz*t + phi_rad)
  
  # check of nan or overflow 
  if any(np.isnan(h_complex)):
      print('this is dead ')
  

  # indtroducing a window       
  window_dt = t[-1]-t[0]
  tukey_rolloff_ms=  0.2/1000    
  window = tukey(len(t), 2 * tukey_rolloff_ms / window_dt)    
      
  
  plus[time_inx] =  np.imag(h_complex) * window
  cross[time_inx] =  np.real(h_complex) * window
    
  ret={}
  ret['cross'] = cross
  ret['plus'] = plus
  return ret




def test_model(get_fake_data=False):
    t=np.arange(0,0.03,0.00005)
    model = dumped_sin(t,A=10**(-22),tau=0.005,f_Hz=2000,phi_rad=0*np.pi/180,alpha=2)
    plt.close('all')
    plt.plot(t,model['cross'])
    plt.plot(t,model['plus'])
    if get_fake_data:
        return t,model


plt.close('all')
n_dumped_sin=3


result = bilby.result.read_in_result(filename='paul_post_merger_result.json')
sorted_tbest=result.posterior.sort_values(by='log_likelihood',ascending=False).reset_index()
best_params = sorted_tbest.iloc[0].to_dict()
(n_dumped_sin,'A','tau','f_Hz','phi_rad','alpha')
Init_params={}
for k in np.arange(3):
    Init_params['f_Hz'+str(k)]=best_params['f_'+str(k)]
    Init_params['alpha'+str(k)]=best_params['alpha_'+str(k)]
    Init_params['tau'+str(k)]=best_params['T_'+str(k)]/1000
    Init_params['phi_rad'+str(k)]=best_params['phi_'+str(k)]
Init_params['A0']=10**best_params['logB'] *best_params['w_0']     
Init_params['A1']=10**best_params['logB'] *best_params['w_1']     
Init_params['A2']=10**best_params['logB'] *(1-best_params['w_1']-best_params['w_0'])
Init_params['n_dumped_sin']=3   
Init_params['t0']=0.00 








# temp data
#t,y = test_model(True)

#plt.plot(t,numerical_relativity_postmerger_waveform(newtime=t,t_0=0)['cross'])
#y=numerical_relativity_postmerger_waveform(newtime=t,t_0=0)
#y = numerical_relativity_postmerger_waveform(newtime=t)


#ff

componant_functions_dict={}
componant_functions_dict[dumped_sin]=(n_dumped_sin,'A','tau','f_Hz','phi_rad','alpha')
model = tbilby.core.base.create_transdimensional_model('model',  componant_functions_dict,returns_polarization=True,SaveTofile=True)

priors_t = bilby.core.prior.dict.ConditionalPriorDict()
#n,f,A,tau,phi priors 
priors_t ['n_dumped_sin'] = tbilby.core.prior.DiscreteUniform(1,n_dumped_sin,'n_dumped_sin')
priors_t = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalUniform_f,'f_Hz',nmax=n_dumped_sin,nested_conditional_transdimensional_params=['f_Hz'],conditional_params=[],prior_dict_to_add=priors_t,minimum=1000,maximum=5000)
#priors_t  = tbilby.create_plain_priors(Uniform,'f_Hz',n_dumped_sin,prior_dict_to_add=priors_t,minimum=1000,maximum=5000)
priors_t = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalLogUniform_A,'A',nmax=n_dumped_sin,nested_conditional_transdimensional_params=['A'],conditional_params=[],prior_dict_to_add=priors_t,minimum=10**(-23),maximum=10**(-17))


#priors_t  = tbilby.create_plain_priors(Uniform,'A',n_dumped_sin,prior_dict_to_add=priors_t,minimum=10**(-24), maximum=10**(-19))
priors_t  = tbilby.core.base.create_plain_priors(Uniform,'tau',n_dumped_sin,prior_dict_to_add=priors_t,minimum=10**(-5), maximum=0.1)
priors_t  = tbilby.core.base.create_plain_priors(Uniform,'phi_rad',n_dumped_sin,prior_dict_to_add=priors_t,minimum=-np.pi , maximum=np.pi )
priors_t  = tbilby.core.base.create_plain_priors(Uniform,'alpha',n_dumped_sin,prior_dict_to_add=priors_t,minimum=-6.4 , maximum=6.4 )
priors_t['t0']  = bilby.prior.Uniform(minimum=-1.5/1000,maximum =1.5/1000)
sample_for_injection = priors_t.sample(5)

# define parameters to inject.
injection_parameters = dict(    
    phase=0,
    ra=0,
    dec=0,
    psi=0,
    t_0=0,
    geocent_time=0.0,
)

constant_priors = injection_parameters.copy()
for k in constant_priors.keys():
    priors_t[k]=constant_priors[k]


for k in sample_for_injection.keys():
    if k in Init_params.keys():
        injection_parameters[k]=Init_params[k] 
    else:
        injection_parameters[k]=sample_for_injection[k]
        Init_params[k] =sample_for_injection[k]

#samples_test = priors_t.sample(500)
#plt.plot(samples_test['f_Hz0'],samples_test['f_Hz1'],'o')



duration = 240/(8192 * 2) #0.05 #(approx 0.04)

#sampling_frequency = 1024*2
sampling_frequency = 8192 * 2


t = np.arange(0,duration,1/sampling_frequency)

outdir = "outdir"
label = "NR_time_domain_source_model"
waveform_arguments={}
waveform_arguments['t_0']=0.0
# call the waveform_generator to create our waveform model.
waveform = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    time_domain_source_model=model,
    start_time=injection_parameters["geocent_time"] ,
    
)


waveform_paul = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    time_domain_source_model=Paulmodel,
    waveform_arguments=waveform_arguments,
    start_time=injection_parameters["geocent_time"] ,
    
)


waveformNR = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    time_domain_source_model=NRmodel,
    waveform_arguments=waveform_arguments,
    start_time=injection_parameters["geocent_time"],
)


fname = r'/Users/ngut0001/Library/CloudStorage/OneDrive-MonashUniversity/HyperBilby-main (2)/HyperBilby-main/gitworkdir/tbilby-sw/tbilby/examples/SensitivityCurves/ForJonas/ET_D_asd.txt' 


aLIGO_O3 = np.genfromtxt(fname)
freq=aLIGO_O3[:,0]
asd = aLIGO_O3[:,1]

ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])

for ifo in ifos:
    ifo.minimum_frequency = np.min(freq)
    ifo.maximum_frequency = np.max(freq)
    ifo.sampling_frequency = sampling_frequency
    ifo.duration = duration 

    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=freq, psd_array=asd**2
    )



    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,start_time=0)


#ifos.set_strain_data_from_power_spectral_densities(
#    sampling_frequency=sampling_frequency,
#    duration=duration,
#    start_time=injection_parameters["geocent_time"],
#)

#ifos.set_strain_data_from_zero_noise(
#    sampling_frequency=sampling_frequency, duration=duration)


# do this by hand 
#for interferometer in ifos:   
#       interferometer.set_strain_data_from_zero_noise(
#          sampling_frequency=sampling_frequency, duration=duration,start_time=0.0)


strains_model = waveform.frequency_domain_strain(injection_parameters)
strains_NR = waveformNR.frequency_domain_strain(injection_parameters)




best_params.update(injection_parameters)
#strains_paul_model =waveform_paul.frequency_domain_strain(best_params)                                    

# here you can take the NR relativity simulation instead of the model 
ifos.inject_signal(
    waveform_generator=waveformNR, parameters=injection_parameters, raise_error=False)#,injection_polarizations=True

outdir='outdir'
label='NEMO_psd'
ifos.plot_data(outdir=outdir, label=label)




const=['A','tau','phi_rad','alpha']
for c in const:
    for i in np.arange(n_dumped_sin):
        priors_t[c+str(i)] = injection_parameters[c+str(i)]

priors_t['n_dumped_sin']=3
priors_t['f_Hz1']  =injection_parameters['f_Hz1']
priors_t['f_Hz2']  =injection_parameters['f_Hz2']
  
#plt.plot(waveformNR.time_domain_strain()['cross'])

#plt.plot(t,ifos[0].strain_data.time_domain_strain,'-sr')
plt.figure(50)
ifos[0].plot_time_domain_data(bandpass_frequencies=(100,5000))



#plt.plot(ifos[0].frequency_array,np.log10(np.abs(ifos[0].frequency_domain_strain)))


x= waveformNR.frequency_array
I = x>100
plt.plot(x[I],np.log10(np.abs(strains_NR['plus'][I])))
plt.plot(x[I],np.log10(np.abs(strains_NR['cross'][I])))
x= waveform.frequency_array
I = x>100
plt.plot(x[I],np.log10(np.abs(strains_model['plus'][I])),'-k')
plt.plot(x[I],np.log10(np.abs(strains_model['cross'][I])),'.k')

#plt.plot(x[I],np.log10(np.abs(strains_paul_model['cross'][I])),'.r')


#plt.figure()
#plt.loglog(ifos[0].power_spectral_density.asd_array)
#bbb


likelihood = bilby.gw.likelihood.GravitationalWaveTransient(ifos,waveform)

best_params['t_0']=0
plt.figure()
plt.plot(t,model(t,**Init_params)['cross'],'sk')
plt.plot(t,time_domain_damped_sinusoid(t, **best_params)['cross'],'-xr')
plt.grid(True)





# launch sampler
plot_result = True
if not plot_result:
    result = bilby.core.sampler.run_sampler(
        likelihood,
        priors_t,
        sampler="dynesty",
        npoints=3,
        nlive=5,
        walks=5,
        nact=3,
        injection_parameters=injection_parameters,
        outdir=outdir,
        resume=False,
        label=label,
    )
else:    
    # pauls_script results 
    
    
    
    
    result = bilby.result.read_in_result(filename=label+'_result.json')
    
    tbilby.corner_plot_discrete_params(result,'n_post.png')
    result,cols = tbilby.preprocess_results(result_in=result, model_dict=componant_functions_dict, remove_ghost_samples= True, return_samples_of_most_freq_component_function= True    )
    tbilby.corner_plot_single_transdimenstional_param(result=result, param='A',overlay=True,filename= 'tBilby_A.png')
    tbilby.corner_plot_single_transdimenstional_param(result=result, param='f_Hz',overlay=True,filename= 'tBilby_f.png')
    tbilby.corner_plot_single_transdimenstional_param(result=result, param='tau',overlay=True,filename= 'tBilby_tau.png')
    
    df = result.posterior
    plt.figure()
    plt.hist(result.posterior['n_dumped_sin'])
    
    
    sorted_tbest = result.posterior.sort_values(by='log_likelihood',ascending=False).reset_index()
    best_params = sorted_tbest.median(axis=0).to_dict()
    print(best_params)
   
    from bilby.core.utils import infer_parameters_from_function
    needed_params = infer_parameters_from_function(model)
    
    model_parameters  ={}
    for k in needed_params:
        model_parameters[k] = 0 # just set a value 
     
    
    for k in best_params.keys():
        if k in needed_params:
            model_parameters[k] = best_params[k]
        
     
    plt.figure()
    t=np.arange(0,duration,1/sampling_frequency)
    #model_parameters['tau1']=0.0002
    #model_parameters['A1']=7*10**(-20)
    #model_parameters['A0']=2*10**(-20)
    plt.figure()
    for n in np.arange(int(model_parameters['n_dumped_sin'])):
        plt.subplot(3,4,n+1)
        plt.plot(t,dumped_sin(t,model_parameters['A'+str(n)],model_parameters['tau'+str(n)],model_parameters['f_Hz'+str(n)],model_parameters['alpha'+str(n)], model_parameters['phi_rad'+str(n)])['cross'])
    plt.subplot(3,4,12)
    


    #model_parameters['f_Hz0']=3316
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t,5*model(t,**Init_params)['cross'],'-.k',label= 'Paul\'s fit')
    plt.plot(t,model(t,**model_parameters)['cross'],'-r',label='Fit',alpha=0.6)
    plt.plot(t,NRmodel(t,t_0=0)['cross'],'b',alpha=0.6,label='NR')
    plt.legend()
    
    
    
   
        
        
    plt.subplot(2,1,2)



    from bilby.core.utils import nfft  
    
    model_freq_domain,freq_arr = nfft(model(t,**model_parameters)['cross'], sampling_frequency)
    data_freq_domain,freq_arr = nfft(NRmodel(t,t_0=0)['cross'], sampling_frequency)
    Pauls_model_freq_domain,freq_arr = nfft(model(t,**Init_params)['cross'], sampling_frequency)
        

    plt.plot(freq_arr,np.log10(data_freq_domain),'-r',label='NR simulation')                    
    plt.plot(freq_arr,np.log10(model_freq_domain),'-ko',label='fit')
    plt.plot(freq_arr,np.log10(Pauls_model_freq_domain),'-.c',label='Paul\'s fit')
    
    
    plt.legend()    
    #ax = plt.gca()
    #ax.set_yscale("log", base=10);
    #plt.ylim([10**-25,10**-21])
    plt.figure()
    
    #injection_parameters['A0']=0
    var = 'f_Hz0'
    likelihood.parameters = injection_parameters.copy()
    log_l=[]
    f_vec = np.arange(priors_t[var].minimum,priors_t[var].maximum,1)
    for f in f_vec:
        likelihood.parameters[var]=f
        log_l.append(likelihood.log_likelihood())
    
    plt.plot(f_vec,log_l,'-o')    
    plt.vlines(injection_parameters[var],np.min(log_l),np.max(log_l))
    plt.title('Inject NR model fit model ' + var)
    plt.xlabel(var)
    plt.ylabel('log likelihood')
    
    dd
        
    likelihood.parameters = injection_parameters
    print(likelihood.log_likelihood())
    likelihood.parameters = Init_params
    Init_params['phase']=0
    Init_params['ra']=0
    Init_params['dec']=0
    Init_params['psi']=0
    Init_params['t_0']=0
    Init_params['geocent_time']=0.0
    
    
    print(likelihood.log_likelihood())
    likelihood.parameters =best_params
    print(likelihood.log_likelihood())
    



    
    # import corner
    # labels = ['f_Hz0','f_Hz1','f_Hz2']
    # # cut the intresting region
    # #tbest = tbest[tbest['alpha1'] < 0]
    # samples = result.posterior[labels].values    
    # fig = corner.corner(samples, labels=labels,bins=50, quantiles=[0.025, 0.5, 0.975],
    #                    show_titles=True, title_kwargs={"fontsize": 12})


    #ValidationTools.ValidateModelandData(function=model,x=t,y=NRmodel(t,t_0=0),likelihood_func=likelihood,priors_dict=priors_t,loglog=False,polarization=True,max_iter=15000,Init_params=model_parameters)
    #ValidationTools.ValidateModelandData(function=model,x=t,y=NRmodel(t,t_0=0),likelihood_func=likelihood,priors_dict=priors_t,loglog=False,polarization=True,max_iter=15000,Init_params=Init_params)
