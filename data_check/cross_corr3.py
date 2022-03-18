#!/usr/bin/env python3
#%% Load all
import sys
import os
sys.path.insert(1, '/home/ejagodin/codes/eric/chops')
sys.path.insert(1, '/home/ejagodin/codes/eric/chops/chopHund')
sys.path.insert(1, '/home/ejagodin/Dropbox/codes')
import csv
import numpy as np
import glob
import matplotlib.pyplot as plt
import math
import time
sys.path.insert(1, '/home/ejagodin/codes/eric/analysis/')
from ensemble_param2 import Chop, get_chopnums
import multiprocessing as mp
import csv
from tqdm import tqdm

plt.rcParams['font.size'] = '16'
plt.rcParams["font.family"] = ["serif"]

# Global Variables
globl = {'procs': 15,               # Selected number of processors
         'nbins': 101,              # Selected number of bins
         'delta_plus': 1.698e-3     # For re_tau=300
         }

# Saved locations (and other details) for referencing
loc = {'bins':'/storage/channel_re300/trainingData/data_binaries/',
       'plot_dir':'/scratch/analysis2_CNN4Burst/cross_corr/all_steps'}

# Grab available timesteps and data chop sample numbers
timesteps, data_num = get_chopnums(loc)

#%% Functions
def get_param_keys(timesteps, data_num):
    """Get the parameter keys from the chop object"""
    test_chop = Chop(timesteps[0], '000')
    param_keys = [key for key in test_chop.param.keys()]
    for i,k in enumerate(param_keys):
        print(f"{str(i).rjust(2)}: {k}")
    return param_keys

def cli_args():
    """ Read in the cli arguments and exits if not given"""
    if len(sys.argv) < 4:
        print("Needs parameters p1 & p2 and start number")
        get_param_keys(timesteps, data_num)
        sys.exit()
    else:
        p1 = sys.argv[1]
        p2 = sys.argv[2]
        start = int(sys.argv[3])
    return p1, p2, start

def make_directory_if_not_exists():
    loc['plot_dir'] = loc['plot_dir']+f"-{p1}{p2}/"
    if not os.path.exists(loc['plot_dir']):
        os.makedirs(loc['plot_dir'])
    print(loc['plot_dir'])
    
def euc_dist(pt1, pt2):
    """ Euclidean distance between two 3D indices"""
    dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)
    return dist

def wall_normal_normalize(p, param):
    """Take the input 3D data array and normalize by subtracting the mean of each
        wall-normal layer. I divide by the std dev after E(x[n]y*[n+r]) """
    param = np.array(param)
    if (p != 'Uf'): 
        wn_means = [np.mean(param[:,n,:]) for n in range(param.shape[1])]
        for n in range(param.shape[1]):
            param[:,n,:] = param[:,n,:] - wn_means[n]
    return param

def regular_normalize(p, param):
    """Take the input 3D data array and normalize by subtracting the mean of each
        wall-normal layer. I divide by the std dev after E(x[n]y*[n+r]) """
    param = np.array(param)
    if (p != 'Uf'): 
        p_mean = np.mean(param)
        param = param - p_mean
    return param

def calc_midpoints(test_chop):
    """Calculate the midpoints of x,y,z"""
    x = np.array(test_chop.X['x'])
    y = np.array(test_chop.X['y'])
    z = np.array(test_chop.X['z'])
    x,z = (x[1:] + x[:-1])/2, (z[1:] + z[:-1])/2
    return x,y,z

def reduce_domain_if_derived(p1,p2, param1,param2, xx,yy,zz):
    """Reduce the domain by taking out the '0' cells on
        the borders if p1 or p2 are a derived parameter 
        (Prod, Diss, Trans, etc.)"""
    list_der = ['TKE', 'Diss', 'Prod', 'Trans', 'Pp', 'Pn']  
    if (p1 in list_der) or (p2 in list_der):
        param1 = param1[1:-1, 1:-1, 1:-1]
        param2 = param2[1:-1, 1:-1, 1:-1]
        xx,yy,zz = xx[1:-1],yy[1:-1],zz[1:-1]
    #param1 = param1 - np.mean(param1)
    #param2 = param2 - np.mean(param2)
    return param1, param2, xx,yy,zz

def get_bins(xx,yy,zz):
    """ Using max distance of sample and number of bins, calculate bins
        and bin sizes"""
    max_dist = euc_dist([ xx[0],yy[0],zz[0] ], [ xx[-1],yy[-1],zz[-1] ])
    bin_size = max_dist/globl['nbins']
    bins = np.arange(0, max_dist, bin_size)
    return bins, bin_size

def calc_start_end(i):
    """ Start and end idx of the process (for multiprocessing)"""
    Nx = param1.shape[0]
    proc_len = math.ceil(Nx/globl['procs']) # number of x idx for each process
    start= i*proc_len
    end = min(i*proc_len + proc_len, Nx)
    return start, end

def make_blank_xcorr():
    """ Make blank xcorr array for filling
        xcorr[bin_dist, sum(p1*p2), ct]"""
    xcorr = np.zeros((len(bins),3))
    xcorr[:,0] = bins
    return xcorr

def calc_bin_array(xgrid,ygrid,zgrid, xpt0,ypt0,zpt0):
    """ Calculate distance of every other point to pt0 and place in bin"""
    dist_array = np.sqrt( np.square(xgrid-xpt0) + np.square(ygrid-ypt0) + np.square(zgrid-zpt0) )
    bin_array = dist_array / bin_size
    max_bin_array = np.full_like(bin_array, globl['nbins']-1)
    bin_array = np.minimum(bin_array.astype(int), max_bin_array).astype(int)
    return bin_array

def if_Pp_or_Pn():
    """ If p1 or p2 is P+ or P-, use the other disc_rad_xcorr_conv"""
    p_plus_minus = ['Pp', 'Pn']
    if (p1 in p_plus_minus) or (p2 in p_plus_minus):
        return True
    else:
        return False

def disc_rad_xcorr_conv(i):
    """ Calculates xcorr faster using convolution and array math
        Global Vars: xyzgrids, param12, nbins"""
    start, end = calc_start_end(i)
    xcorr = make_blank_xcorr()
    for i0 in range(start, end):
        for j0 in range(len(yy)):
            for k0 in range(len(zz)):
                # Grab pt0
                xpt0, ypt0, zpt0 = [xgrid[i0,j0,k0], ygrid[i0,j0,k0], zgrid[i0,j0,k0]]
                # Calculate distance of every other point to pt0 and place in bin
                bin_array = calc_bin_array(xgrid,ygrid,zgrid, xpt0,ypt0,zpt0)
                # Multiply every other pt of param2 to param1[pt0]
                one_conv = param1[i0,j0,k0] * param2
                # Using bin_array, add every param2[pt1] in proper bin, take ct
                for i1 in range(bin_array.shape[0]):
                    for j1 in range(bin_array.shape[1]):
                        for k1 in range(bin_array.shape[2]):
                            # Get bin and p1*p2
                            ijk_bin = bin_array[i1,j1,k1]
                            ijk_mult = one_conv[i1,j1,k1]
                            # Add to xcorr
                            #if not_Pp_or_Pn_and_nonzero(param1[i0,j0,k0]):
                            xcorr[ijk_bin,1] = xcorr[ijk_bin,1] + ijk_mult
                            xcorr[ijk_bin,2] += 1         
    return xcorr

def disc_rad_xcorr_conv_P(i):
    """ Same but doesn't use if param1==0 at current point"""
    start, end = calc_start_end(i)
    xcorr = make_blank_xcorr()
    for i0 in range(start, end):
        for j0 in range(len(yy)):
            for k0 in range(len(zz)):
                # Grab pt0
                xpt0, ypt0, zpt0 = [xgrid[i0,j0,k0], ygrid[i0,j0,k0], zgrid[i0,j0,k0]]
                # Calculate distance of every other point to pt0 and place in bin
                bin_array = calc_bin_array(xgrid,ygrid,zgrid, xpt0,ypt0,zpt0)
                # Multiply every other pt of param2 to param1[pt0]
                one_conv = param1[i0,j0,k0] * param2
                # Using bin_array, add every param2[pt1] in proper bin, take ct
                for i1 in range(bin_array.shape[0]):
                    for j1 in range(bin_array.shape[1]):
                        for k1 in range(bin_array.shape[2]):
                            # Get bin and p1*p2
                            ijk_bin = bin_array[i1,j1,k1]
                            ijk_mult = one_conv[i1,j1,k1]
                            # Add to xcorr if nonzero
                            if param1[i0,j0,k0] != 0:
                                xcorr[ijk_bin,1] = xcorr[ijk_bin,1] + ijk_mult
                                xcorr[ijk_bin,2] += 1         
    return xcorr

def sort_results(results):
    """ Summing results of x[n]y[n-m] and count of each bin"""
    assembled_results = np.zeros((len(results[0]),3))
    for n in range(len(results[0])):
        assembled_results[n,0] = results[0][n,0]
        for m in range(len(results)):
            assembled_results[n,1] = assembled_results[n,1] + results[m][n,1]
            assembled_results[n,2] = assembled_results[n,2] + results[m][n,2]
    return assembled_results

def calc_cross_corr(assembled_results):
    """Calculate the cross correlation as Rxy = E{x[n]y*[n-m]}
        where E{} = sum(x[n]y*[n-m])/count and y*=y since y is real"""
    cross_corr = np.zeros((len(assembled_results),2))
    for rbin in range(len(assembled_results)):
        cross_corr[rbin,0] = assembled_results[rbin,0] / globl['delta_plus']
        sum_xy = assembled_results[rbin,1]
        bin_ct = assembled_results[rbin,2]
        if sorted_results[rbin,2] > 0:
            cross_corr[rbin,1] = sum_xy / bin_ct
        else:
            cross_corr[rbin,1] = 0.0
    return cross_corr

def divide_by_std(p1, p2, cross_corr):
    """Divide the cross-correlations by the std dev to get the corr coeff"""
    param1, param2 = test_chop.param[p1], test_chop.param[p2]
    param1, param2 = np.array(param1), np.array(param2)
    std1, std2   = np.std(param1) , np.std(param2)
    cross_corr[:,1] = cross_corr[:,1] / (std1*std2)
    return cross_corr

def close_fig():
    try:
        plt.close()
    except:
        pass
    
def plot_fig(cross_corr, p1, p2, timestep, n):
    close_fig()
    timestep = int(float(timestep.split('p')[2])*10)
    n = str(n).zfill(3)
    if p1 == p2:
        title = f'Autocorrelation of {p1}'
        filename = f"{loc['plot_dir']}nd_autocorr_{p1}.png"
        ylim = [-1.1, 1.1]
    else:
        title = f'Cross-Correlation of {p1} and {p2}\n t={timestep}s {n}'
        filename = f"{loc['plot_dir']}xcorr_{p1}-{p2}_{timestep}_{n}.png"
        ylim = [-0.55, 0.55]
    #filename = filename.split('.')[0]+'_mag.png' if p2_mag else filename
    plt.plot(cross_corr[:,0], cross_corr[:,1])
    plt.title(title)
    plt.xlabel('$r^+$')
    plt.ylabel('$R_{ij}$')
    plt.xlim([0,250])
    plt.ylim(ylim)
    plt.tight_layout()
    plt.axhline(y=0,linestyle='--')
    plt.grid()
    plt.savefig(filename)
    close_fig()
   
def combine_all_xcorr(all_xcorr):
    """ Combine all_xcorr into one cross-correlation plot"""
    max_len = min(len(all_xcorr[0]), 101)
    add_xcorr = np.zeros_like(all_xcorr[0][:max_len])
    for n in range(len(all_xcorr)):
        #print(f"len xcorr[{n}]: {len(all_xcorr[n])}")
        add_xcorr = np.add(add_xcorr, all_xcorr[n][:max_len])
    total_xcorr = add_xcorr / len(all_xcorr)
    return total_xcorr

def save_csv(total_xcorr, p1, p2, timestep, n):
    """ Save csv for easier post-processing"""
    timestep = int(float(timestep.split('p')[2])*10)
    filename = f"{loc['plot_dir']}xcorr_{p1}-{p2}_{timestep}_{n}.csv"
    header = ['dist','prob_dens']
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        for r, pd in total_xcorr[:90]:
            writer.writerow( [r, pd] )
            
#%% Run
p1, p2, start = cli_args()
make_directory_if_not_exists()
all_xcorr = []
print('Using disc_rad_xcorr_conv_P') if if_Pp_or_Pn() else None
print(f'Working on : {start+1}-{start+25}')
for timestep in timesteps[:1]:
    for n in tqdm(np.arange(start,start+25)):
        #print(f"Working on {timestep}[{str(n).zfill(3)}]")
        # Grab data from timestep and nchop
        test_chop = Chop(timestep, str(n).zfill(3))
        xx,yy,zz = calc_midpoints(test_chop)
        # Get parameter data of 
        param1, param2 = np.array(test_chop.param[p1]), np.array(test_chop.param[p2])
        param1 = regular_normalize(p1, param1)
        param2 = regular_normalize(p2, param2)
        param1, param2, xx,yy,zz = reduce_domain_if_derived(p1,p2, param1,param2, xx,yy,zz)
        # Get bins and bin sizes
        bins, bin_size = get_bins(xx,yy,zz)
        # Create meshgrid of points
        xgrid, ygrid, zgrid = np.meshgrid(xx,yy,zz, indexing='ij')
        #st = time.time()
        pool = mp.Pool(processes=globl['procs'])
        if if_Pp_or_Pn():
            results = pool.map(disc_rad_xcorr_conv_P, [i for i in range(globl['procs'])])
        else:
            results = pool.map(disc_rad_xcorr_conv, [i for i in range(globl['procs'])])
        sorted_results = sort_results(results)
        cross_corr = calc_cross_corr(sorted_results)
        cross_corr = divide_by_std(p1, p2, cross_corr)
        all_xcorr.append(cross_corr)
        ##plot_fig(cross_corr, p1, p2, timestep, n)
        #print(f'took: {round((time.time() - st)/60, 2)}min')
        
        if ((n+1) % 25 == 0) & (n != 0):
            nstr = f"temp{str(start).zfill(3)}-{str(n).zfill(3)}"
            temp_xcorr = combine_all_xcorr(all_xcorr)
            save_csv(temp_xcorr, p1, p2, timestep, nstr)
            plot_fig(temp_xcorr, p1, p2, timestep, nstr)

#%% Combine and average temp xcorr
def combine_temp_xcorr(timestep):
    os.chdir(loc['plot_dir'])
    all_tempcsv = sorted(glob.glob('xcorr*temp*.csv'))
    timestep = timesteps[0]
    all_xcorr = []
    for tempcsv in all_tempcsv:
        tempdata = []
        with open(tempcsv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for n, row in enumerate(reader):
                if n != 0:
                    dist,pdf = row
                    tempdata.append([float(dist),float(pdf)])
        all_xcorr.append(tempdata)
    total_xcorr = combine_all_xcorr(all_xcorr) 
    save_csv(total_xcorr, p1, p2, timestep, 'all')       
    plot_fig(total_xcorr, p1, p2, timestep, 'all')

combine_temp_xcorr(timesteps[0])
        
