from utilities import *

#create_ini_file(fiducial, filename="fiducial.ini", dir="fiducial")
#create_ini_file(fiducial_plus, filename="fiducial_plus.ini", dir="fiducial_plus")
#create_ini_file(fiducial_minus, filename="fiducial_minus.ini", dir="fiducial_minus")

#stepsize_finder(param="log10_G_eff", outfile="derivative_log10_G_eff.pdf")


#write_all_spectra(params=fiducial, l_max=4000)
#compute_fisher(experiment=cmb_s4, lensing=True, l_max=[2500, 3000, 3500, 4000], outfile="output/cmb_s4_fisher_SI.npy")
#compute_fisher(experiment=core, lensing=True, l_max=[2500, 3000, 3500, 4000], outfile="output/core_fisher_SI.npy")
#compute_fisher(experiment=CV_limited, lensing=True, l_max=[2500, 3000, 3500, 4000], outfile="output/CV_limited_fisher_SI.npy")
#read_fisher(filename="output/trial.npy")

#plot_width_lmax(param="log10_G_eff", lmax=[2000, 2250, 2500, 2750, 3000, 3250, 3500], filename="output/trial.npy", confidence_level=1, outfile="G_eff_lmax.pdf")

#plot_width_lmax(experiment=core, lmax=[2500, 3000, 3500, 4000], filename="output/core_fisher_SI.npy", confidence_level=1, outfile="core_all_params_lmax.pdf")
#plot_width_lmax(experiment=cmb_s4, lmax=[2500, 3000, 3500, 4000], filename="output/cmb_s4_fisher_SI.npy", confidence_level=1, outfile="cmb_s4_all_params_lmax_SI.pdf")
#plot_width_lmax(experiment=CV_limited, lmax=[2500, 3000, 3500, 4000], filename="output/CV_limited_fisher_SI.npy", confidence_level=1, outfile="CV_limited_all_params_lmax.pdf")

#plot_corner(["H0", "omega_b"], outfile="H0-omega_b.pdf")
#plot_corner(["H0", "omega_b", "omega_cdm", "A_s", "n_s", "tau_reio", "log10_G_eff"], fisher_filename="output/cmb_s4_fisher_SI.npy", outfile="cmb_s4_all_corner_lmax_4000.pdf")
#plot_corner(["H0", "omega_b", "omega_cdm", "A_s", "n_s", "tau_reio", "log10_G_eff"], fisher_filename="output/core_fisher_SI.npy", outfile="core_all_corner_lmax_4000.pdf")
plot_corner(["H0", "omega_b", "omega_cdm", "A_s", "n_s", "tau_reio", "log10_G_eff"], fisher_filename=[["output/CV_limited_fisher_SI.npy","CV limited"]], outfile="CV_limited_all_corner_lmax_4000.pdf")

#plot_corner(["H0", "omega_b", "n_s", "log10_G_eff"], fisher_filename=[["output/CV_limited_fisher_SI.npy","CV limited"], ["output/cmb_s4_fisher_SI.npy","CMB-S4"], ["output/core_fisher_SI.npy","CORE"]], outfile="fisher_concise.pdf")
#plot_corner(["H0", "log10_G_eff"], outfile="H0-log10_G_eff.pdf", fisher_filename="output/cmb_s4_fisher_SI.npy")
#plot_1d(params=["x1", "x2"], fisher=np.array([[1,0],[0,4]]), confidence_level=1, x_limits=np.array([[-2,2], [0,2]]), outfile="1d.pdf")

'''
a = np.array([1,4,2,4])
b = np.exp(np.array([1,1,1,2])**2)
print(b)
'''
