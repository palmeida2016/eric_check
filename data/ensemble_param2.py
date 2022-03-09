#!/usr/bin/env python3
#%% Load all
import sys
import os
sys.path.insert(1, '/home/ejagodin/codes/eric/chops')
sys.path.insert(1, '/home/ejagodin/codes/eric/chops/chopHund')
sys.path.insert(1, '/home/ejagodin/Dropbox/codes')
import csv
import numpy as np
from struct import unpack
import glob
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Constants and other global var
plt.rcParams['font.size'] = '16'
plt.rcParams["font.family"] = ["serif"]
    # Selected cutoffs of Gradient
G_contours = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    # selected parameters for analysis
analysis_param = ['TKE', 'Diss', 'Pp', 'Pn', 'Trans']
    # selected timesteps for plotting
selected = [0,1,2,3] #[56, 60, 64, 68]
    # Saved locations (and other details) for referencing
loc = {'bins':'/storage/channel_re300/trainingData/data_binaries/',
       'multi_csv':'/scratch/analysis2_CNN4Burst/',
       'gnu_csv':'_ensem.csv',
       'plt':'/scratch/analysis2_CNN4Burst/ensem2.pdf'}

def get_timesteps(loc):
    """ Get all timesteps"""
    timesteps = sorted(glob.glob(loc['bins']+'*chop_interp*'))
    timesteps = [t.split('/')[-1] for t in timesteps]
    return timesteps

def get_chopnums(loc):
    """ Get all chop numbers for timestep"""
    timesteps = get_timesteps(loc)
    data_num = sorted(glob.glob(loc['bins']+timesteps[0]+'/*int*'))
    data_num = [n.split('/')[-1].split('i')[0] for n in data_num]
    return timesteps, data_num

if __name__ == '__main__':
    timesteps, data_num = get_chopnums(loc)

#%% Chop class

class Chop:

    def __init__(self, timestep, chopnum):
        self.timestep = timestep
        self.chopnum = chopnum
        self.configfile = "%sintConfig" % chopnum
        self.datafile = "%sallDataChop" % chopnum
        self.Gbin = "%sgradChop" % chopnum
        os.chdir(loc['bins']+self.timestep)
        self.X, self.nX = self.openconfig()
        self.param = self.read_all_data()
        self.dX = self.calc_dX()
        self.param['G'] = self.read_Gbin()
        self.param['Pp'], self.param['Pn'] = self.prod_posneg()
        if __name__ == '__main__':
            self.flats = self.flatten()
            self.chop_sum = self.ensemble()

        
    def __str__(self):
        return f'Data: {self.datafile}'
    
    def check_shapes(self):
        """ Print check to make sure sizes are alright"""
        for key in self.param.keys():
            print(f"{key}: {self.param[key].shape}")
        
    def check_cond_ratios(self):
        """ Print the conditional ratios of chop"""
        params = list(self.cond_ratios.keys())
        for p in params:
            param_ratios = self.cond_ratios[p]
            print(f"         {p}:")
            tot_str = str("%05.1f" % param_ratios[0])
            for i, g_cutoff in enumerate(G_contours):
                g_str = str("%.1f" % g_cutoff)
                r_str = str("%07.3f" % param_ratios[i+1]).zfill(3)
                print(f'{g_str}: {r_str} / {tot_str}')

    def openconfig(self):
        """ Reads the config file"""
        with open(self.configfile, 'rb') as file:
            config = file.read()  
        #skip1 = unpack("<"+"c"*64, config[:64])[0:64] #skip first 64 char
        #skip2 = unpack("i"*4, config[64:80])[0:4]     #skip the next 4 int
        nnx = unpack("i", config[80:84])[0]
        nny = unpack("i", config[84:88])[0]
        nnz = unpack("i", config[88:92])[0]
        xlen = 92 + (nnx+1)*8
        ylen = xlen + (nny+1)*8
        zlen = ylen + (nnz+1)*8
        x = unpack("<"+"d"*(nnx+1), config[92:xlen])[0:nnx+1]
        y = unpack("<"+"d"*(nny+1), config[xlen:ylen])[0:nny+1]
        y = y[0:len(y)-1]
        z = unpack("<"+"d"*(nnz+1), config[ylen:zlen])[0:nnz+1]
        X = {'x':x, 'y':y, 'z':z}
        nX = {'x':nnx, 'y':nny, 'z':nnz}
        file.close()
        return X, nX
    
    def read_all_data(self):
        """ Loads all velocity data"""
        with open(self.datafile,'rb') as file:
            data = file.read()
        # Dimensions
        dims = unpack("<iiii", data[:16])[0:4]
        nx = dims[0]; ny = dims[1]; nz = dims[2]; nvar = dims[3]
        nsize = nx*ny*nz
        t, dt = unpack("<"+"d"*2, data[16:32])[:16]
        varz_start = 32; varz_end = varz_start + nvar*8
        varz = unpack("<" + "c"*nvar*8, data[varz_start:varz_end])[0:nvar*8]
        varz = [v.decode("utf-8") for v in varz]
        params = [''.join(varz[i:i+8]).strip() for i in range(0, len(varz), 8)]
        all_param = {}
        for i in range(nvar):
            p_name = params[i]
            p_start = varz_end + i*8*nsize; p_end = p_start + 8*nsize
            p_raw = unpack("<"+"d"*nsize, data[p_start:p_end])[0:nsize]
            p_array = np.asarray(p_raw)
            P_shaped = p_array.reshape((nz,ny,nx)).transpose()
            P_16 = np.float16(P_shaped)
            all_param[p_name] = P_16
        return all_param
    
    def calc_dX(self):
        """ Calculatae dx, dy, dz"""
        dX = {}
        for c in ['x', 'y', 'z']:
            dX[c] = abs(self.X[c][1] - self.X[c][0])
        return dX
    
    def prod_posneg(self):
        """ Seperates Turbulent Energy Production into >0 and <0"""
        Pp = self.param['Prod'].copy()
        Pn = Pp.copy()
        Pp[Pp < 0] = 0 # setting any negatives to zero
        Pn[Pn > 0] = 0 # setting any positives to zero
        Pn = np.abs(Pn) # absolute value for summation
        return Pp, Pn
    
    def read_Gbin(self):
        """ Read the parameter (Grad) binary"""
        filename = loc['bins']+self.timestep+'/'+self.Gbin
        with open(filename,'rb') as f:
            data = f.read()
        dims = unpack("<iiii", data[:16])[0:4]
        nx = dims[0]; ny = dims[1]; nz = dims[2]#; nvar = dims[3]
        nsize = nx*ny*nz
        #dt,t = unpack("<dd", data[16:32])[0:2]
        #varNames = unpack("<"+"c"*8*nvar, data[32:32+8*nvar])[0:8*nvar]
        #print(c,'|')
        p_start = 40; p_end = p_start + nsize*8
        p = unpack("<"+"d"*nsize, data[p_start:p_end])[0:nsize]
        p = np.array(p)
        param = p.reshape((nz,ny,nx)).transpose() 
        return param
    
    def flatten(self):
        """ Create flat arrays of all of the analysis parameters
        because easier to sort through"""
        flats = {}
        for p in self.param.keys():
            flat_param = np.ndarray.flatten(self.param[p])
            flats[p] = flat_param
        return flats

    def make_sums():
        """ create empty sums array"""
        sums = {}
        for p in analysis_param:
            sums[p] = []
        return sums
    
    def ensemble(self):
        """ Getting all the ensemble data for the chop"""
        chop_sum = Chop.make_sums()
        for g_cutoff in G_contours:
            g_grtr = list(np.where(self.flats['G'] > g_cutoff)[0])
            for p in analysis_param:
                PofG_sum = 0
                param = self.flats[p]
                for gn in g_grtr:
                    PofG_sum += param[gn]
                chop_sum[p].append(PofG_sum)
        return chop_sum
    
    def make_all_sums():
        """ Makes all_sums for the timestep"""
        all_sums = Chop.make_sums()
        for p in analysis_param:
            for g in G_contours:
                all_sums[p].append(0)
        return all_sums
    
    def add_sums_to_all(all_sums, chop_sum):
        """ Add sums from chop to all sums for timestep"""
        for p in analysis_param:
            for gn, _ in enumerate(G_contours): 
                all_sums[p][gn] += chop_sum[p][gn]
        return all_sums
     
    def compile_timestep_data(timestep):
        """Go through every chop of timestep, 
        calc params, get sums"""
        print(f"Timestep: {10*float(timestep.split('interp')[1])}s")
        all_sums = Chop.make_all_sums()
        for num in data_num: ####################################################
            print(num) if int(num) % 100 == 0 else None
            chopn = Chop(timestep, num)
            all_sums = Chop.add_sums_to_all(all_sums, chopn.chop_sum)
        return all_sums
    
    def compile_sums(selected):
        """ Compile sums of all selected timesteps"""
        multi_sums = []
        for tn in selected:
            all_sums = Chop.compile_timestep_data(timesteps[tn])
            multi_sums.append(all_sums)
        return multi_sums
    
    def legend_names(selected):
        """ Make plot legend names from selected timesteps"""
        timestep_names = ['t']
        for n, _ in enumerate(selected):
            if n+1 == 1:
                new_step = timestep_names[0]+' +  dt'
            else:
                nt = str(n+1)
                new_step = timestep_names[0]+f' + {nt}dt'
            timestep_names.append(new_step)
        return timestep_names
    
    def save_csv(multi_sums):
        """ Save multi_sums as csv"""
        timestep_names = Chop.legend_names(selected)
        headers = [str(g) for g in G_contours]
        headers.insert(0,'')
        for p in analysis_param:
            csv_name = loc['multi_csv']+p+loc['gnu_csv']
            print(f"Making: {p+loc['gnu_csv']}")
            with open(csv_name,'w',newline='') as csvfile:
                writer = csv.writer(csvfile,delimiter=',')
                writer.writerow(headers)
                for tn, all_sums in enumerate(multi_sums):
                    param = all_sums[p]
                    param_pct = [str(100*p/param[0]) for p in param]
                    param_pct.insert(0, timestep_names[tn])
                    writer.writerow(param_pct)

    def change_posneg(p):
        """ Changes Prod+ and Prod- to more readable for plot"""
        if p == 'Pp':
            return 'P+'
        elif p == 'Pn':
            return 'P-'
        else:
            return p

    def all_plot_multi(multi_sums, selected):
        """ Plot all sums of multiple timesteps overlayed"""
        plots_x = int(len(analysis_param))
        fig, axes = plt.subplots(1,plots_x,sharey=True,figsize=(16,7))
        markers = ['o', '^', 's','*']
        for i, ax in enumerate(axes.flatten()):
            p = analysis_param[i]
            times = []
            for tn, all_sums in enumerate(multi_sums):
                param = all_sums[p]
                param_pct = [100*p/param[0] for p in param]
                time_plt = ax.plot(G_contours[1:], param_pct[1:], markers[tn])
                times.append(time_plt)
            ax.set_ylim(0,100)            
            ax.set_xlim(0,3.75)
            ax.set_title(Chop.change_posneg(p))
            ax.grid()
        timestep_names = Chop.legend_names(selected)
        fig.add_subplot(111, frameon=False)
        
        #fig.delaxes(axes[-1,-1]) # deleting last, unneeded subplot space
        # hide tick and tick label of the big axis
        fig.legend(times, labels=timestep_names, title="Timesteps", borderaxespad=0.25)#, bbox_to_anchor=(0.91, 0.75)) #1.075, 0.8
        fig.subplots_adjust(right=0.5)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel("Conditional Sum Percentage")
        plt.xlabel("GradCAM Gradient")
        fig.subplots_adjust(right=0.9)
        plt.rcParams['font.size'] = '24'
        #fig.tight_layout(rect=[0,1,0,0])#rect=[0,5,0,0]) 
        plt.savefig(loc['plt'], format='pdf', bbox_inches='tight')
        plt.close()
   
    def assemble_all(selected_timesteps, data_num, given_keys=None):
        """ Go through selected timesteps and build a dict of all parameters"""
        test_chop = Chop(selected_timesteps[0], '000')
        if given_keys == None:
            param_keys = [key for key in test_chop.param.keys()]
        else:
            param_keys = given_keys
        all_param = {}
        for key in param_keys:
            all_param[key] = []
        for timestep in selected_timesteps:
            print(timestep)
            for num in data_num:
                print(num) if (int(num) % 225 == 0) and (int(num) != 0) else None
                loop_chop = Chop(timestep, num)
                for key in param_keys:
                    all_param[key].append(loop_chop.param[key])
        return all_param
    
    def flatten_all(all_param):
        """ Create a flattened dict of all param for easier analysis"""
        all_flat = {}
        for key in all_param.keys():
            print(key, end=' ')
            all_flat[key] = []
            for chop in all_param[key]:
                flat_chop = np.ndarray.flatten(chop)
                for val in flat_chop:
                    all_flat[key].append(val)
            all_flat[key] = np.array(all_flat[key])
        print('')
        return all_flat

#%% main
if __name__ == '__main__':
    #main_multi()
    #test_chop = Chop(timesteps[-1], '000')
    multi_sums = Chop.compile_sums(selected)
    #Chop.save_csv(multi_sums)
    #Chop.all_plot_multi(multi_sums, selected)
#%% TEST
#multi_sums = Chop.compile_sums(selected)
#%% PLOT
#Chop.all_plot_multi(multi_sums, selected)

#%% Test from CSV
ensem_csvs = glob.glob(loc['multi_csv']+'/*'+loc['gnu_csv'])

def grab_csv_data(ensem_csvs):
    """Grab ensemble sum data from csvs and put into dict of dicts"""
    all_ensem = {}
    for ecsv in ensem_csvs:
        param = ecsv.split('/')[-1].split('_')[0]
        all_ensem[param] = {}
        with open(ecsv,'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for n, row in enumerate(reader):
                if n == 0:
                    for m, _ in enumerate(row[1:]): 
                      all_ensem[param][str(m)] = []  
                else:
                    for m, val in enumerate(row[1:]):
                        all_ensem[param][str(m)].append(float(val))
    return all_ensem

def get_avg_ensems(all_ensem):
    """Average ensemble sum data across timesteps"""
    ensem_avg_rng = {}
    for pkey in all_ensem.keys():
        p_ensem = all_ensem[pkey]
        ensem_avg_rng[pkey] = {'g':[], 'a':[], 'r':[]}
        for gkey in p_ensem.keys():
            p_ensem[gkey] = np.array(p_ensem[gkey])
            avg_p = np.mean(p_ensem[gkey])
            range_p = np.max(p_ensem[gkey]) - np.min(p_ensem[gkey])
            ensem_avg_rng[pkey]['g'].append(gkey)
            ensem_avg_rng[pkey]['a'].append(avg_p)
            ensem_avg_rng[pkey]['r'].append(range_p)
            print(f"{pkey}[{gkey}]: avg {avg_p} rng {range_p}")
    return ensem_avg_rng

all_ensem = grab_csv_data(ensem_csvs)
avg_ensems = get_avg_ensems(all_ensem)

def plot(avg_ensems, keys):
    markers = ['o','s','d']
    for n, key in enumerate(keys):
        param = avg_ensems[key]['a'][1:]
        param = [n/100 for n in param]
        gcont = G_contours[1:]
        errbar = [n/100 for n in avg_ensems[key]['r'][1:]]
        plt.scatter(gcont, param, marker=markers[n], facecolors='none', edgecolors='k', s=75)
        plt.errorbar(gcont, param, yerr=errbar, fmt='none', color='k', capsize=1, linewidth=0.5)
        print(f"{key}: {markers[n]}")
    #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(linestyle='--',which='major')
    plt.ylabel("Conditional Fraction")
    plt.xlabel("GradCAM Value (G)")
    plt.xlim([0, 3.5])
    xticks = np.arange(0, 3.5, step=1)
    plt.xticks(xticks)
    plt.ylim([0,1])
    plt.tight_layout()
    filename = loc['multi_csv']+'cond_frac-'+'_'.join(keys)+'.pdf'
    print(filename)
    plt.savefig(filename, format='pdf')
    plt.close()
    
#plot(avg_ensems, ['Trans', 'Diss', 'TKE'])
#plot(avg_ensems, ['Pp', 'Pn'])
    
    


