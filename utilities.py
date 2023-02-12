import numpy as np
from numpy.linalg import *
import os, sys
import subprocess
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import rc, rcParams
from tqdm import tqdm
###################################
rc('text',usetex=True)
# Change all fonts to 'Computer Modern'
rc('font',**{'size':14, 'family':'serif','serif':['Times New Roman']})
rcParams["axes.linewidth"]=2.5
work_dir = os.getcwd()
sys.path.append(work_dir+"/..")
plotdir=work_dir+"/plots/"
ft = 23
#######
# Follows 1805.01055
#######
arcmin_to_radian = 0.000290888
paramnames = ["H0", "omega_b", "omega_cdm", "A_s", "n_s", "tau_reio", "log10_G_eff"]
latexnames = ["$H_0\,[\mathrm{km~s^{-1}Mpc^{-1}}]$", "$\Omega_\mathrm{b}h^2$", "$\Omega_\mathrm{c}h^2$", "$A_s$", "$n_s$", r"$\tau_\mathrm{reio}$", "$\log_{10}[G_\mathrm{eff}/\mathrm{MeV^{-2}}]$"]

#fiducial = np.array([67.27, 0.02236, 0.1202, 2.101e-9, 0.9649, 0.0544, -4.5])  # set at Planck18 TTTEEE+lowE bestfit
fiducial = np.array([69.44, 0.022, 0.1205, 1.9767e-9, 0.9386, 0.0543, -1.92])  # set at SI in 3c+0f Planck18 TTTEEE+lowE bestfit from our paper
dlambda = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])*np.abs(fiducial)  # check this again !!!
#fiducial_plus = fiducial + dlambda
#fiducial_minus = fiducial - dlambda
TT, EE, TE, dd = 1, 2, 4, 5

ALPHA1 = 1.52
ALPHA2 = 2.48
ALPHA3 = 3.44

cmb_s4 = {"name":r"\textbf{CMB-S4}", "f_sky":0.47, "Delta_T":np.array([61.4, 20.8, 11.6, 2, 2, 6.7, 16.3]), "beam_size":1.4, "freq":np.array([20, 27, 39, 93, 145, 225, 280])}    # from CMB-S4 wiki

core = {"name":r"\textbf{CORE}", "f_sky":0.45, "freq":np.array([60,70,80,90, 100,115,130,145, 160,175,195,220, 255,295,340,390, 450,520,600]), "Delta_T":np.array([7.5,7.1,6.8,5.1, 5.0,5.0,3.9,3.6, 3.7,3.6,3.5,3.8, 5.6,7.4,11.1,22.0, 45.9,116.6,358.3]), "beam_size":np.array([17.87,15.39,13.52,12.08, 10.92,9.56,8.51,7.68, 7.01,6.45,5.84,5.23, 4.57,3.99,3.49,3.06, 2.65,2.29,1.98])}    # from https://arxiv.org/pdf/1706.04516.pdf

litebird = {"name":r"\textbf{LiteBIRD}", "f_sky":1, "freq":np.array([40,60,140,235,280,337,402]), "Delta_T":np.array([37.42,21.31,4.79,10.79,13.8,21.95,47.45])/np.sqrt(2), "beam_size":np.array([70.5,51.1,30.8,24.7,22.5,20.9,17.9])}      # from https://arxiv.org/pdf/2101.12449.pdf

CV_limited = {"name":r"\textbf{CV-limited}", "f_sky":1, "Delta_T":np.zeros(1), "beam_size":np.zeros(1), "freq":np.array([1])}

def noise_matrix(experiment, lx, freq_channel=6, lensing=True):
    if experiment == cmb_s4:
        if lx == 2:
            print("CMB-S4")
        beam = experiment["beam_size"]*arcmin_to_radian * 150/experiment["freq"]
        N_TT = 0
        for i in range(len(experiment["freq"])):
            N_TT += ((experiment["Delta_T"][i]*arcmin_to_radian)**2 * np.exp(lx*(lx+1)/8./np.log(2.)*beam[i]**2))
        
    else:
        if lx ==2:
            print(experiment["name"])
        beam = experiment["beam_size"]*arcmin_to_radian
        N_TT = 0.
        for i in range(len(experiment["freq"])):
            
            N_TT += (experiment["Delta_T"][i]*arcmin_to_radian)**2 * np.exp(lx*(lx+1)/8./np.log(2.)*beam[i]**2)
    N_EE = 2*N_TT
    if not lensing:
        return np.diag([N_TT, N_EE, 0.])
    else:
        return np.diag([N_TT, N_EE])

def C_matrix(l, data, lensing=True):
    if not lensing:
        return np.array([[data[TT,l-2], data[TE,l-2], 0], [data[TE,l-2], data[EE,l-2], 0], [0, 0, data[dd,l-2]]])
    else:
        return np.array([[data[TT,l-2], data[TE,l-2]], [data[TE,l-2], data[EE,l-2]]])

def the_l_sum(experiment, ix, jx, data0, data_plus1, data_minus1, data_plus2=None, data_minus2=None, l_max=2500, freq_channel=6):    
    res = 0
    if ix == jx: # diagonal element
        for l in range(2,l_max+1):
            Cl = C_matrix(l, data0) + noise_matrix(experiment, l, freq_channel=freq_channel)
            Cl_plus = C_matrix(l, data_plus1)
            CL_minus = C_matrix(l, data_minus1)
            dC = (Cl_plus - CL_minus)/2/dlambda[ix]
            tmp = np.trace(multi_dot([inv(Cl), dC, inv(Cl), dC]))
            res += (2*l+1)/2*tmp
        return experiment["f_sky"]*res
        
    else: # off-diagonal element
        for l in range(2,l_max):
            Cl = C_matrix(l, data0) + noise_matrix(experiment, l, freq_channel=freq_channel)
            Cl_plus1 = C_matrix(l, data_plus1)
            Cl_plus2 = C_matrix(l, data_plus2)
            CL_minus1 = C_matrix(l, data_minus1)
            CL_minus2 = C_matrix(l, data_minus2)
            dC1 = (Cl_plus1 - CL_minus1)/2/dlambda[ix]
            dC2 = (Cl_plus2 - CL_minus2)/2/dlambda[jx]
            tmp = np.trace(multi_dot([inv(Cl), dC1, inv(Cl), dC2]))
            res += (2*l+1)/2*tmp
        return experiment["f_sky"]*res

def compute_fisher(experiment, params=fiducial, lensing=True, l_max=2500, outfile=None, freq_channel=6):
    if lensing:
        lensing = "_lensed"
    else:
        lensing = ""
    f0 = "output/fiducial/fiducial_cl"+lensing+".dat"
    # compute C-matrix and sum
    dim = len(params)
    
    data0 = np.loadtxt(f0, unpack=True)
    if type(l_max) not in [list]:
        fisher = np.empty((dim, dim))
        for i in range(dim):
            for j in range(i,dim):
                #print(i,j)
                if i == j:
                    f_plus1 = "output/fiducial/fiducial_plus_"+paramnames[i]+"_cl.dat"
                    f_minus1 = "output/fiducial/fiducial_minus_"+paramnames[i]+"_cl.dat"
                    data_plus1 = np.loadtxt(f_plus1, unpack=True)
                    data_minus1 = np.loadtxt(f_minus1, unpack=True)
                    fisher[i,j] = the_l_sum(experiment, i, j, data0, data_plus1, data_minus1, l_max=l_max, freq_channel=freq_channel)
                else:
                    f_plus1 = "output/fiducial/fiducial_plus_"+paramnames[i]+"_cl.dat"
                    f_minus1 = "output/fiducial/fiducial_minus_"+paramnames[i]+"_cl.dat"
                    f_plus2 = "output/fiducial/fiducial_plus_"+paramnames[j]+"_cl.dat"
                    f_minus2 = "output/fiducial/fiducial_minus_"+paramnames[j]+"_cl.dat"
                    data_plus1 = np.loadtxt(f_plus1, unpack=True)
                    data_minus1 = np.loadtxt(f_minus1, unpack=True)
                    data_plus2 = np.loadtxt(f_plus2, unpack=True)
                    data_minus2 = np.loadtxt(f_minus2, unpack=True)
                    fisher[i,j] = the_l_sum(experiment, i, j, data0, data_plus1, data_minus1, data_plus2, data_minus2, l_max=l_max, freq_channel=freq_channel)
                    fisher[j,i] = fisher[i,j]      # Fisher matrix is symmetric
        print("Fisher matrix --\n", fisher)
        print("Inverse Fisher matrix diagonal --\n", np.diagonal(np.linalg.inv(fisher)))
        return
    else:
        if outfile == None:
            print("Output file needed.")
            sys.exit()
        res = []
        for lm in tqdm(l_max):
            fisher = np.empty((dim,dim))
            for i in range(dim):
                for j in range(dim):
                    #print(i,j)
                    if i == j:
                        f_plus1 = "output/fiducial/fiducial_plus_"+paramnames[i]+"_cl"+lensing+".dat"
                        f_minus1 = "output/fiducial/fiducial_minus_"+paramnames[i]+"_cl"+lensing+".dat"
                        data_plus1 = np.loadtxt(f_plus1, unpack=True)
                        data_minus1 = np.loadtxt(f_minus1, unpack=True)
                        fisher[i,j] = the_l_sum(experiment, i, j, data0, data_plus1, data_minus1, l_max=lm, freq_channel=freq_channel)
                    else:
                        f_plus1 = "output/fiducial/fiducial_plus_"+paramnames[i]+"_cl"+lensing+".dat"
                        f_minus1 = "output/fiducial/fiducial_minus_"+paramnames[i]+"_cl"+lensing+".dat"
                        f_plus2 = "output/fiducial/fiducial_plus_"+paramnames[j]+"_cl"+lensing+".dat"
                        f_minus2 = "output/fiducial/fiducial_minus_"+paramnames[j]+"_cl"+lensing+".dat"
                        data_plus1 = np.loadtxt(f_plus1, unpack=True)
                        data_minus1 = np.loadtxt(f_minus1, unpack=True)
                        data_plus2 = np.loadtxt(f_plus2, unpack=True)
                        data_minus2 = np.loadtxt(f_minus2, unpack=True)
                        fisher[i,j] = the_l_sum(experiment, i, j, data0, data_plus1, data_minus1, data_plus2, data_minus2, l_max=lm, freq_channel=freq_channel)
            res.append(fisher)
        np.save(outfile, np.array(res), fix_imports=False)
        print("Output saved in "+outfile)
        print("==================================")
        return 


def create_ini_file(params, filename, dir, N_idr=3, l_max=2500):
    with open(filename, "w") as f:
        f.write("root = output/fiducial/"+dir+"_\n")
        for i,p in enumerate(params):
            if paramnames[i] in ["log10_G_eff"]:
                for n in range(N_idr):
                    f.write("log10_G_eff_"+str(n+1)+" = "+str(p)+"\n")
            else:
                f.write(paramnames[i]+" = "+str(p)+"\n")
        f.write("output = tCl, pCl, lCl\n")
        f.write("lensing = yes\n")
        f.write("N_idr = "+str(N_idr)+"\n")
        f.write("xi_idr = 0.716489304871\n")  # Use this with N_idr=3 to get Neff=3.046
        f.write("beta_idr = 1.\n")
        f.write("idr_nature = free_streaming\n")
        f.write("stat_f_idr = 0.875\n")
        f.write("T_cmb = 2.7255\n") # write all other inputs
        f.write("N_ur = 0\n")
        f.write("Omega_k = 0.\n")
        f.write("Omega_idm_dr = 0.\n")
        f.write("l_max_scalars = "+str(l_max)+"\n")
        #f.write("gauge = newtonian\n")
        f.write("format = camb\n")
    f.close()
    #print("Input file saved as "+filename)
    #print("----------------------------------------------")
    return
    
def run_class(ini_file, pre_file=None, n_cpu=10):
    #print("Running CLASS ------")
    arg = ["./class", ini_file]
    if pre_file != None:
        arg.append(pre_file)
    subprocess.run(arg)
    return
    
def stepsize_finder(param, outfile):
    idx = paramnames.index(param)
    steps = np.array([0.05,0.025,0.01,0.005])*fiducial[idx]
    data = np.loadtxt("output/fiducial/fiducial_cl.dat", unpack=True)
    fig, ax = plt.subplots(2,4, figsize=(17,8))
    for s in tqdm(steps):
        fiducial_plus = fiducial.copy()
        fiducial_minus = fiducial.copy()
        fiducial_plus[idx] += s
        create_ini_file(fiducial_plus, "test_plus.ini", dir="test_plus", l_max=4000)
        run_class("test_plus.ini")
        fiducial_minus[idx] -= s
        create_ini_file(fiducial_minus, "test_minus.ini", dir="test_minus", l_max=4000)
        run_class("test_minus.ini")
        data_plus = np.loadtxt("output/fiducial/test_plus_cl.dat", unpack=True)
        data_minus = np.loadtxt("output/fiducial/test_minus_cl.dat", unpack=True)
        fd, sd = [], []
        for i,l in enumerate(data[0]):
            fd.append([(data_plus[sp,i]-data_minus[sp,i])/2/s for sp in [TT,EE,TE,dd]])
            sd.append([(data_plus[sp,i]-2*data[sp,i]+data_minus[sp,i])/s**2 for sp in [TT,EE,TE,dd]])
        fd = np.array(fd)
        sd = np.array(sd)
        for i in range(4):
            ax[0,i].plot(data[0], fd.T[i])
            ax[1,i].plot(data[0], sd.T[i])
            ax[1,i].set_xlabel(r"$\ell$ ", fontsize=ft-1)
    ax[0,0].set_ylabel(r"$\partial C_\ell/\partial $"+latexnames[idx], fontsize=ft-1)
    ax[1,0].set_ylabel(r"$\partial^2 C_\ell/\partial $"+latexnames[idx]+"$^2$", fontsize=ft-1)
    plt.suptitle(r"Derivative step sizes for "+latexnames[idx]+" "+str(steps/fiducial[idx]*100)+"\%")
    plt.savefig(plotdir+outfile,bbox_inches='tight')
    
    
def write_all_spectra(params, l_max=2500):
    # run class for all params
    dim = len(params)
    if dim != 7:
        print("Only 7 parameters are supported!")
        sys.exit()
    
    create_ini_file(params, filename="fiducial.ini", dir="fiducial", l_max=l_max)
    run_class("fiducial.ini")
    print("Done!")
    for i in tqdm(range(dim)):
        #print("Running CLASS for ", paramnames[i])
        params_plus = params.copy()
        params_plus[i] += dlambda[i]
        #print(params[i], params_plus[i])
        create_ini_file(params_plus, filename="fiducial_plus.ini", dir="fiducial_plus_"+paramnames[i], l_max=l_max)
        run_class("fiducial_plus.ini")
        
        params_minus = params.copy()
        params_minus[i] -= dlambda[i]
        #print(params[i], params_minus[i])
        create_ini_file(params_minus, filename="fiducial_minus.ini", dir="fiducial_minus_"+paramnames[i], l_max=l_max)
        run_class("fiducial_minus.ini")
        #print("Done!")
    return


def read_fisher(filename):
    f = np.load(filename)
    #print(f)
    return f

'''
def confidence_level_coeff(cl):
    cl_frac = norm_dist_rule[cl]
    res = np.sqrt(2) * sp.inv_erf(cl_frac)
    print(res)
    return res
'''    
def get_width(param, fisher, confidence_level):
    i = paramnames.index(param)
    #print(i)
    return confidence_level * np.sqrt(np.linalg.inv(fisher)[i,i])
    
def plot_width_lmax(experiment, lmax, filename, confidence_level, outfile):
    fisher = read_fisher(filename)
    n_lmax = len(fisher)
    fig, ax = plt.subplots(3,3, figsize=(24,13), sharex=True, tight_layout=True)
    
    for i in range(3):
        for j in range(3):
            idx = int(i*3**1 + j*3**0)
            if idx > 6:
                break
            param = paramnames[idx]  # using trinary to decimal conversion here :P
            width = np.array([get_width(param, f, confidence_level) for f in fisher])
            print(width)
            ax[i,j].plot(lmax, width, color="k", lw=3, ls="solid")
            ax[i,j].set_xlim(lmax[0], lmax[-1])
            ax[i,j].set_xlabel(r"$\ell_\mathrm{max}$", fontsize=ft+2)
            ax[i,j].set_ylabel(r"$\sigma ($"+latexnames[idx]+")", fontsize=ft+2)
            ax[i,j].tick_params('both',length=10,width=1.5,which='major',labelsize=ft+3,direction='in', top=True, right=True)
            ax[i,j].tick_params('both',length=7,width=1.2,which='minor',direction='in', top=True, right=True)
    plt.suptitle(experiment["name"], fontsize=ft+5)
    plt.savefig(plotdir+outfile,bbox_inches='tight')
    print("Figure saved in ", plotdir+outfile)
    return


def get_ellipse_parameters(param_pair, fisher):
    #fisher = read_fisher(fisher_filename)[-1]  # lmax=4000
    i, j = paramnames.index(param_pair[0]), paramnames.index(param_pair[1])
    inv_fish = np.linalg.inv(fisher)
    s2i, s2j, s2ij = inv_fish[i,i], inv_fish[j,j], inv_fish[i,j]
    a = np.sqrt(0.5*(s2i+s2j) + np.sqrt(0.25*(s2i-s2j)**2 + s2ij**2))
    b = np.sqrt(0.5*(s2i+s2j) - np.sqrt(0.25*(s2i-s2j)**2 + s2ij**2))
    phi0 = np.arctan2(2*s2ij, (s2i-s2j)) /2
    #print(param_pair[0], fiducial[i]+np.sqrt(s2i))
    #print(param_pair[1], fiducial[j]+np.sqrt(s2j))
    #print(np.sqrt(s2ij))
    return a, b, phi0
    
def plot_corner(params, outfile,
                fisher_filename="output/cmb_s4_fisher.npy",
                xlabel_kwargs={'labelpad': 20, 'fontsize':ft-2},
                ylabel_kwargs={'fontsize':ft-2},
                two_sigma=False,
                color_list=["k","royalblue","coral","turquoise"],
                positive_definite=[],
                resize_lims=True,
                tick_length=8,
                tick_width=2.5):
                
    PLOT_MULT = 4
    nparams = len(params)
    indices = np.array([paramnames.index(p) for p in params])
    fig, ax = plt.subplots(nparams, nparams, figsize=(12,12))
    plt.subplots_adjust(hspace=0, wspace=0)
    labels = np.array([latexnames[idx] for idx in indices])
    
    if not isinstance(fisher_filename, list):
        print("Not a list!")
        fisher_filename = np.array([fisher_filename])
 
    for ii in range(nparams):
        for jj in range(nparams):
            if ii == jj:
                ax[jj, ii].get_yaxis().set_visible(False)
                #ax[jj, ii].get_xaxis().set_visible(True)
                #ax[jj, ii].tick_params(direction="in", length=tick_length, width=tick_width)
                if ii < nparams-1:
                    ax[jj, ii].get_xaxis().set_ticks([])

            if ax[jj, ii] is not None:
                if ii < jj:
                    if jj < nparams-1:
                        ax[jj, ii].set_xticklabels([])
                    if ii > 0:
                        ax[jj, ii].set_yticklabels([])

                    if jj > 0:
                        # stitch axis to the one above it
                        if ax[0, ii] is not None:
                            ax[jj, ii].get_shared_x_axes().join(
                                ax[jj, ii], ax[0, ii])
                    elif ii < nparams-1:
                        if ax[jj, nparams-1] is not None:
                            ax[jj, ii].get_shared_y_axes().join(
                                ax[jj, ii], ax[jj, nparams-1])

    for ii in range(nparams):
        idx = paramnames.index(params[ii])
        for jj in range(nparams):
            idy = paramnames.index(params[jj])
            if ax[jj, ii] is not None:
                ax[jj, ii].tick_params(direction="in", length=tick_length, width=tick_width)
                if ii < jj:
                    handles=[]
                    for kk,expmt in enumerate(fisher_filename):
                        fisher = read_fisher(expmt[0])[-1]  # reads the last lmax fisher matrix
                        inv_fish = np.linalg.inv(fisher)
                        a, b, phi0 = get_ellipse_parameters([params[ii],params[jj]], fisher=fisher)
                        #print(a,b,phi0*180/np.pi, paramnames[idx],paramnames[idy])
                        # 1-sigma ellipse
                        e1 = Ellipse(xy=(fiducial[idx], fiducial[idy]),
                                 width=a * 2 * ALPHA1, height=b * 2 * ALPHA1,
                                 angle=phi0*180/np.pi,
                                 fill=False, color=color_list[kk], lw=2, label=expmt[1])
                        ax[jj,ii].add_artist(e1)
                        e1.set_alpha(1.0)

                        if two_sigma:
                            e2 = Ellipse(xy=(fiducial[idx], fiducial[idy]),
                                         width=a * 2 * ALPHA2, height=b * 2 * ALPHA2,
                                         angle=phi0*180/np.pi,
                                         fill=False, color="red", lw=1)
                            ax[jj,ii].add_artist(e2)
                            e2.set_alpha(0.5)
                        
                        if resize_lims:
                            sigma_x = np.sqrt(inv_fish[idx,idx])
                            sigma_y = np.sqrt(inv_fish[idy,idy])
                            #print(sigma_x,sigma_y)
                            if params[ii] in positive_definite:
                                ax[ii,jj].set_xlim(max(0.0, -PLOT_MULT*sigma_x),
                                                    fiducial[idx]+PLOT_MULT*sigma_x)
                            else:
                                ax[jj,ii].set_xlim(fiducial[idx] - PLOT_MULT * sigma_x,
                                                    fiducial[idx] + PLOT_MULT * sigma_x)
                            if params[jj] in positive_definite:
                                ax[jj,ii].set_ylim(max(0.0, fiducial[idy] - PLOT_MULT * sigma_y),
                                            fiducial[idy] + PLOT_MULT * sigma_y)
                            else:
                                ax[jj,ii].set_ylim(fiducial[idy] - PLOT_MULT * sigma_y,
                                            fiducial[idy] + PLOT_MULT * sigma_y)
                        
                        
                        if jj == nparams-1:
                            ax[jj, ii].set_xlabel(labels[ii], **xlabel_kwargs)
                            for tick in ax[jj, ii].get_xticklabels():
                                tick.set_rotation(20)
                                tick.set_fontsize(ft-4)
                            
                        if ii == 0:
                            ax[jj, ii].set_ylabel(labels[jj], **ylabel_kwargs)
                            for tick in ax[jj, ii].get_yticklabels():
                                tick.set_rotation(0)
                                tick.set_fontsize(ft-4)
                elif ii == jj:
                    #ax[jj, ii].get_xaxis().set_ticks([])
                    for kk,expmt in enumerate(fisher_filename):
                        fisher = read_fisher(expmt[0])[-1]  # reads the last lmax fisher matrix
                        inv_fish = np.linalg.inv(fisher)
                        # plot a gaussian if we're on the diagonal
                        sig = np.sqrt(inv_fish[idx,idx])
                        if params[ii] in positive_definite:
                            grid = np.linspace(
                                fiducial[ii],
                                fiducial[ii] + PLOT_MULT * sig, 100)
                        else:
                            grid = np.linspace(
                                fiducial[idx] - PLOT_MULT*sig,
                                fiducial[idx] + PLOT_MULT*sig, 100)
                        posmult = 1.0
                        if params[ii] in positive_definite:
                            posmult = 2.0
                        ax[jj, ii].plot(grid,
                                        posmult * np.exp(
                                            -(grid-fiducial[idx])**2 /
                                            (2 * sig**2)) / (sig * np.sqrt(2*np.pi)),
                                        '-', color=color_list[kk], lw=2, label=expmt[1])
                        #ax[jj, ii].tick_params(direction="in", length=tick_length, width=tick_width)
                        if ii == nparams-1:
                            ax[jj, ii].set_xlabel(labels[ii], **xlabel_kwargs)
                            for tick in ax[jj, ii].get_xticklabels():
                                tick.set_rotation(20)
                                tick.set_fontsize(ft-4)
                else:
                    ax[jj, ii].axis('off')
    handles, labels = ax[-1,-1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.9,0.9), loc="upper right", fontsize=ft-2, edgecolor="k")
    plt.savefig(plotdir+outfile,bbox_inches='tight')
    print("Figure saved in ", plotdir+outfile)
    return
    
    
def get_gaussian(params, fisher, confidence_level, x_limits):
    x_num = 30
    res = []
    for i in range(len(params)):
        sigma = np.sqrt(np.linalg.inv(fisher)[i,i])
        width = confidence_level_coeff(confidence_level)*sigma
        x_low, x_high = x_limits[i,0], x_limits[i,1]
        x = np.linspace(x_low, x_high, x_num)
        y = np.exp(- (x - fiducial[params[i]]*np.ones_like(x))**2 / 2/sigma**2)   # absolute value does not matter	
        res.append([x,y])
    return np.array(res)
    
def plot_1d(params, fisher, confidence_level, x_limits, outfile, color="darkblue", ls="solid", lw=3):
    fig, ax = plt.subplots(1,len(params), figsize=(8,7), sharey=True)
    data = get_gaussian(params, fisher, confidence_level, x_limits)
    for i in range(len(params)):
        ax[i].plot(data[i,0],data[i,1], color=color, ls=ls, lw=lw)
        ax[i].set_xlabel(params[i], fontsize=ft)
        ax[i].tick_params('both',length=10,width=1.5,which='major',labelsize=ft+3,direction='in', top=True, right=True)
        ax[i].tick_params('both',length=7,width=1.2,which='minor',direction='in', top=True, right=True)
    plt.savefig(plotdir+outfile,bbox_inches='tight')
    print("Figure saved in ", plotdir+outfile)
    return
    
    
    
    
