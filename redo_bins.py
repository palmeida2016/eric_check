#!/usr/bin/env python3
#%% Load all
import sys
import os
sys.path.insert(1, '/home/ejagodin/codes/eric/chops')
sys.path.insert(1, '/home/ejagodin/codes/eric/chops/chopHund')
sys.path.insert(1, '/home/ejagodin/Dropbox/codes')
import csv
import numpy as np
from struct import unpack, pack
from scipy.interpolate import RegularGridInterpolator as rgi
import glob
import matplotlib.pyplot as plt
import math

# Constants and other global var
newX = {'x':30, 'y':40, 'z':30} # interpolated X
utau_ratio = 1 #0.059211 #for Re670
nu = 1e-4 # viscosity
pt = 3 # number of pts for central difference dirivative
plt.rcParams['font.size'] = '16'

# selected parameters for analysis
analysis_param = ['TKE', 'D', 'Pp', 'Pn']
# selected timesteps for plotting
selected = [56, 70] #[56, 60, 64, 68] [70]

# Saved locations (and other details) for referencing
loc = {'bins':'/storage/channel_re300/trainingData/data_binaries/',
       'means':'/scratch/analysis2_CNN4Burst/cross_corr/'}

def get_timesteps(loc):
    """ Get all timesteps"""
    timesteps = sorted(glob.glob(loc['bins']+'*chopTop*'))
    timesteps = [t.split('/')[-1] for t in timesteps]
    return timesteps

def get_chopnums(loc):
    """ Get all chop numbers for timestep"""
    timesteps = get_timesteps(loc)
    data_num = sorted(glob.glob(loc['bins']+timesteps[70]+'/*data*'))
    data_num = [n.split('/')[-1].split('d')[0] for n in data_num]
    return timesteps, data_num

timesteps, data_num = get_chopnums(loc)

#%% Chop class

class Chop:

    def __init__(self, timestep, chopnum):
        self.timestep = timestep
        self.chopnum = chopnum
        self.configfile = "%sconfigChop" % chopnum
        self.datafile = "%sdataChop" % chopnum
        self.Gbin = "%sgradChop" % chopnum
        self.intCon = "%sintConfig" % chopnum
        self.allDatafile = "%sallDataChop" % chopnum
        os.chdir(loc['bins']+self.timestep)
        self.make_interp_dir()
        self.X = self.openconfig()
        self.upper_wall = True if (self.X['y'][0]) > 0 else False
        #print(f"Is upper wall? {self.upper_wall}")
        self.U = self.read_data()
        self.nX = newX
        self.Uint, self.Xint = self.interpolate()
        self.dX = self.calc_dX()
        self.G = self.read_Gbin()
        self.Umean = self.read_means()
        self.Uf = self.calc_Uf()
        self.Pf = self.calc_Pf()
        self.Uf2, self.Vf2, self.Wf2 = self.fluct_squared()
        self.TKE = self.calc_TKE()
        self.P = self.calc_prod()
        #self.Pp, self.Pn = Chop.prod_posneg(self)
        self.D = self.calc_diss()
        self.T = self.calc_transport()
        self.varnames, self.allP = self.make_allP()
        self.all2bin()

    def __str__(self):
        return f'Data: {self.datafile}'
    
    def make_interp_dir(self):
        """ make chop interp dir if not there"""
        time = self.timestep.split('Top')[1].split('E')[0]
        new_dir = 'chop_interp'+time
        if not os.path.isdir(loc['bins']+new_dir):
            os.mkdir(loc['bins']+new_dir)
        loc['new_dir'] = loc['bins']+new_dir
        
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
        file.close()
        return X
    
    def read_data(self):
        """ Loads all velocity data"""
        with open(self.datafile,'rb') as file:
            data = file.read()
        # Dimensions
        dims = unpack("<iiii", data[:16])[0:4]
        nx = dims[0]; ny = dims[1]; nz = dims[2];
        nsize = nx*ny*nz
        # U
        u_start = 64; u_end = u_start + nsize*8
        u_vel = unpack("<"+"d"*nsize, data[u_start:u_end])[0:nsize]
        uvel = np.asarray(u_vel)
        U = uvel.reshape((nz,ny,nx)).transpose()
        U = np.float16(U)
        # V
        v_start = u_end; v_end = v_start + nsize*8 
        v_vel = unpack("<"+"d"*nsize, data[v_start:v_end])[0:nsize]
        vvel = np.asarray(v_vel)
        V = vvel.reshape((nz,ny,nx)).transpose()
        V = np.float16(V)
        # W
        w_start = v_end; w_end = w_start + nsize*8
        w_vel = unpack("<"+"d"*nsize, data[w_start:w_end])[0:nsize]
        wvel = np.asarray(w_vel)
        W = wvel.reshape((nz,ny,nx)).transpose()
        W = np.float16(W)
        # P
        p_start = w_end; p_end = p_start + nsize*8
        p = unpack("<"+"d"*nsize, data[p_start:p_end])[0:nsize]
        p = np.asarray(p)
        P = p.reshape((nz,ny,nx)).transpose()
        P = np.float16(P)
        # Combine
        allU = {'u':U, 'v':V, 'w':W, 'p':P}
        return allU

    def upper_wall_flip(upper_wall, y, z, comp_dat):
        """ Flips data from upper wall for ubiquitous interpolations"""
        if upper_wall:
            y = np.flip(y)
            z = np.flip(z)
            comp_dat = np.flip(comp_dat,1)
            comp_dat = np.flip(comp_dat,2)
        return y, z, comp_dat
    
    def upper_wall_flip_back(upper_wall, y, z, dat_new):
        """ Flips it back"""
        if upper_wall:
            dat_new = np.flip(dat_new,2)
            dat_new = np.flip(dat_new,1)
            z = np.flip(z)
            y = np.flip(y)
        return y, z, dat_new
    
    def interpolate(self):
        """ Interpolates U,V,W data to new grid size"""
        # Getting calling data
        x, y, z = self.X['x'], self.X['y'], self.X['z']
        nx1, ny1, nz1 = self.nX['x'], self.nX['y'], self.nX['z']
        Uint = {}
        for comp in ['u', 'v', 'w', 'p']:
            comp_dat = self.U[comp]
            y, z, comp_dat = Chop.upper_wall_flip(self.upper_wall, y, z, comp_dat)
            x,y,z = np.asarray(x),np.asarray(y),np.asarray(z)
            xmid,zmid = (x[1:] + x[:-1])/2, (z[1:] + z[:-1])/2
            # New Dimensions
            x_new = np.linspace(xmid[0],xmid[-1],nx1)
            y_new = np.linspace(y[0],y[-1],ny1)
            z_new = np.linspace(zmid[0],zmid[-1],nz1)
            xyz_grid = np.meshgrid(x_new, y_new, z_new, indexing='ij')
            xyz_list = np.reshape(xyz_grid, (3, -1), order='C').T
            # Interpolating
            interp = rgi((xmid,y,zmid),comp_dat,method='linear')
            dat_new = interp(xyz_list)
            dat_new = np.reshape(dat_new, (nx1,ny1,nz1))
            y, z, comp_dat = Chop.upper_wall_flip_back(self.upper_wall, y, z, dat_new)
            dat_new = dat_new/utau_ratio # if another ReTau
            Uint[comp] = dat_new
        Xint = {'x':x_new, 'y':y_new, 'z':z_new}
        return Uint, Xint
    
    def calc_dX(self):
        """ Calculatae dx, dy, dz"""
        dX = {}
        for c in ['x', 'y', 'z']:
            dX[c] = abs(self.Xint[c][1] - self.Xint[c][0])
        return dX
    
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
        g_start = 40; g_end = g_start + nsize*8
        g = unpack("<"+"d"*nsize, data[g_start:g_end])[0:nsize]
        g = np.array(g)
        G = g.reshape((nz,ny,nx)).transpose() 
        return G
    
    def read_means(self):
        """ Reads uMean generated from interpolated lower channel wall
            data """
        Umean = {}
        for p in ['u','v','w','p']:
            with open(loc['means']+f'channel_mean_{p}.csv') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                pMean = []
                for n, row in enumerate(reader):
                    if n != 0:
                        pMean.append(float(row[1]))
            Umean[p] = pMean
        return Umean
    
    def calc_Uf(self):
        """Calculate U velocity fluctuation only since mean(V,W) = 0"""
        y_int, u_int = self.Xint['y'], self.Uint['u']
        uMean = self.Umean['u']
        Uf = np.zeros_like(u_int)
        uMean = uMean[::-1] if self.upper_wall else uMean
        for n, y in enumerate(y_int):
            Uf[:,n,:] = self.Uint['u'][:,n,:] - uMean[n]
        return Uf
    
    def calc_Pf(self):
        """Calculate Pressure fluctuation"""
        pMean = self.Umean['p']
        Pf = np.zeros_like(self.Uint['p'])
        pMean = pMean[::-1] if self.upper_wall else pMean
        for n, y in enumerate(self.Xint['y']):
            Pf[:,n,:] = self.Uint['p'][:,n,:] - pMean[n]
        return Pf
    
    def fluct_squared(self):
        """ Calculates fluctuations squared"""
        # since mean(V) and mean(W) = 0
        Vf, Wf = np.array(self.Uint['v']), np.array(self.Uint['w']) 
        Uf2, Vf2, Wf2 = self.Uf**2, Vf**2, Wf**2
        return Uf2, Vf2, Wf2
    
    def calc_TKE(self):
        """ Calculate Turbulent Kinetic Energy"""
        tke = 0.5*(self.Uf2 + self.Vf2 + self.Wf2)
        return tke
    
    def startend(nX,pt):
        """Array bounds for 5-pt vs 3-pt derivative calculations"""
        nx,ny,nz = nX['x'], nX['y'], nX['z']
        s = 1 if pt == 3 else 2 # b/c 5pt central, can't calculate closest 2 to boundary
        x0,y0,z0,x1,y1,z1 = s,s,s,nx-s,ny-s,nz-s
        return x0,y0,z0,x1,y1,z1

    def dudy_only(U,dX,j,pt):
        """ du/dy only for Production"""
        # calculates U-2 through U+2 for y dimension
        if pt == 5: # 5pt central difference
            U1,U2,U3,U4 = U[j-2],U[j-1],U[j+1],U[j+2]
            deriv= (1*U1 - 8*U2 + 8*U3 -1*U4)/(12*dX['y'])
        elif pt == 3: # 3pt central difference
            U2,U3 = U[j-1],U[j+1]
            deriv = (-1*U2 + U3)/(2*dX['y'])
        return deriv    
    
    def calc_prod(self):
        """ Calculating Turbulent Energy Production"""
        Uf,Vf = self.Uf, self.Uint['v']
        #Wf,Pf = self.Uint['w'], self.Uint['p']
        nx,ny,nz = self.nX # just to reduce number of input variables
        x0,y0,z0,x1,y1,z1 = Chop.startend(self.nX,pt)
        Prod = np.zeros_like(self.Uint['v']) # empty array 
        #print("Calculating production u'v'd\u203eudy")
        for k in range(z0,z1):
            print("%.2f" %  float(100*k/z1) + '%') if k % 120 == 0 else None
            for j in range(y0,y1):
                for i in range(x0,x1):
                    # Getting every derivatave of the cell using central pt
                    dudy = Chop.dudy_only(self.Umean['u'], self.dX, j, pt)
                    # u_i u_j du_i/dx_j
                    # but since dvdX, dwdX, dudx & dudz should be 0
                    # its just -u'v'dū/dy
                    P = -1 * Uf[i,j,k] * Vf[i,j,k] * dudy 
                    Prod[i,j,k] = P
        return Prod
    
    def dUdXs3(U,dX,i,j,k):
        """ Derivatives: 3pt central"""
        deriv, comp = [0,0,0], ['x','y','z']
        for dim in range(3):
            # calculates U-2 through U+2 for each dimension
            if dim == 0: # dU/dx
                U2,U3 = U[i-1,j,k], U[i+1,j,k]
            elif dim == 1: # dU/dy
                U2,U3 = U[i,j-1,k], U[i,j+1,k]
            elif dim == 2: # dU/dz
                U2,U3 = U[i,j,k-1], U[i,j,k+1]
            deriv[dim] = (-1*U2 + U3)/(2*dX[comp[dim]])
        return deriv
    
    def dUdXs5(U,dX,i,j,k):
        """ Derivatives: 5pt central"""
        deriv, comp = [0,0,0], ['x','y','z']
        for dim in range(3):
            # calculates U-2 through U+2 for each dimension
            if dim == 0: # dU/dx
                U1,U2,U3,U4 = U[i-2,j,k],U[i-1,j,k],U[i+1,j,k],U[i+2,j,k]
            elif dim == 1: # dU/dy
                U1,U2,U3,U4 = U[i,j-2,k],U[i,j-1,k],U[i,j+1,k],U[i,j+2,k]
            elif dim == 2: # dU/dz
                U1,U2,U3,U4 = U[i,j,k-2],U[i,j,k-1],U[i,j,k+1],U[i,j,k+2]
            deriv[dim] = (1*U1 - 8*U2 + 8*U3 -1*U4)/(12*dX[comp[dim]])
        return deriv
    
    def calc_diss(self):
        """ Calculating Dissipation Rate"""
        U, V, W = self.Uf, self.Uint['v'], self.Uint['w']  #v_avg and w_avg ~ 0
        diss = np.zeros_like(self.Uint['v']) # empty array
        x0,y0,z0,x1,y1,z1 = Chop.startend(self.nX,pt)
        deriv = getattr(Chop,'dUdXs'+str(pt)) # calls either 5pt or 3pt
        for k in range(z0,z1):
            print("%.2f" %  float(100*k/z1) + '%') if k % 120 == 0 else None
            for j in range(y0,y1):
                for i in range(x0,x1):
                    # Getting every derivatave of the cell using central pt
                    dudx,dudy,dudz = deriv(U,self.dX,i,j,k)
                    dvdx,dvdy,dvdz = deriv(V,self.dX,i,j,k)
                    dwdx,dwdy,dwdz = deriv(W,self.dX,i,j,k)
                    # Finding the stress tensor componenents of the cell
                    #  Sij = (1/2)(du'i/dxj + du'j/dxi)
                    s11,s12,s13 = (dudx + dudx)/2, (dudy + dvdx)/2, (dudz + dwdx)/2
                    s21,s22,s23 = (dvdx + dudy)/2, (dvdy + dvdy)/2, (dvdz + dwdy)/2
                    s31,s32,s33 = (dwdx + dudz)/2, (dwdy + dvdz)/2, (dwdz + dwdz)/2
                    # SijSij is the double dot product, so square each component
                    # of the tensor and sum: AijBij = A11B11 + A12B12 + ...
                    s11,s12,s13 =  s11**2,s12**2,s13**2 #squaring
                    s21,s22,s23 =  s21**2,s22**2,s23**2 
                    s31,s32,s33 =  s31**2,s32**2,s33**2
                    SijSij = s11+s12+s13+s21+s22+s23+s31+s32+s33 # summing
                    eps = 2*nu*SijSij # dissipation_rate at i,j,k
                    diss[i,j,k] = eps # inserting into diss_rate array
        return diss
    
    def calc_transport(self):
        """ Calculating Turbulent Energy Production using Verma's derivations"""
        Uf,Vf = self.Uf, self.Uint['v']
        Wf,Pf = self.Uint['w'], self.Pf
        dx, dy, dz = self.dX['x'], self.dX['y'], self.dX['z']
        nx,ny,nz = self.nX # just to reduce number of input variables
        x0,y0,z0,x1,y1,z1 = Chop.startend(self.nX,pt)
        press = np.zeros_like(self.Uint['v']) # empty array 
        diff = np.zeros_like(self.Uint['v'])
        turb = np.zeros_like(self.Uint['v'])
        deriv = getattr(Chop,'dUdXs'+str(pt)) # calls either 5pt or 3pt
        PUf, PVf, PWf = np.multiply(Pf,Uf), np.multiply(Pf,Vf), np.multiply(Pf,Wf)
        for k in range(z0,z1):
            print("%.2f" %  float(100*k/z1) + '%') if k % 120 == 0 else None
            for j in range(y0,y1):
                for i in range(x0,x1):
                    # Derivatives
                    dudx, dudy, dudz = deriv(Uf,self.dX,i,j,k)
                    dvdx, dvdy, dvdz = deriv(Vf,self.dX,i,j,k)
                    dwdx, dwdy, dwdz = deriv(Wf,self.dX,i,j,k)
                    # Pressure term
                    dPUdx,_,_ = deriv(PUf,self.dX,i,j,k)
                    _,dPVdy,_ = deriv(PVf,self.dX,i,j,k)
                    _,_,dPWdz = deriv(PWf,self.dX,i,j,k)
                    press[i,j,k] = -1*dPUdx + -1*dPVdy + -1*dPWdz
                    # Diffusion Term
                    s11,s12,s13 = (dudx + dudx)/2, (dudy + dvdx)/2, (dudz + dwdx)/2
                    s21,s22,s23 = (dvdx + dudy)/2, (dvdy + dvdy)/2, (dvdz + dwdy)/2
                    s31,s32,s33 = (dwdx + dudz)/2, (dwdy + dvdz)/2, (dwdz + dwdz)/2
                    j1 = 0.5*(Uf[i,j,k]*s11 + Vf[i,j,k]*s21 + Wf[i,j,k]*s31)/dx
                    j2 = 0.5*(Uf[i,j,k]*s12 + Vf[i,j,k]*s22 + Wf[i,j,k]*s32)/dy
                    j3 = 0.5*(Uf[i,j,k]*s13 + Vf[i,j,k]*s23 + Wf[i,j,k]*s33)/dz
                    diff[i,j,k] = 2*nu*(j1 + j2 + j3)
                    # Turbulent term
                    tke2 = 2*self.TKE[i,j,k]
                    turb[i,j,k] = Uf[i,j,k]*tke2/dx + Vf[i,j,k]*tke2/dy + Wf[i,j,k]*tke2/dz
        trans = press + diff + turb
        return trans
    
    def make_allP(self):
        """ Makes the all data parameters list for writing bins"""
        varnames = ['U', 'V', 'W', 'P', 'Uf','TKE', 'Diss', 'Prod', 'Trans']
        allP = []
        for p in ['u', 'v', 'w', 'p']:
            allP.append(self.Uint[p])
        allP.append(self.Uf)
        allP.append(self.TKE)
        allP.append(self.D)
        allP.append(self.P)
        allP.append(self.T)
        return varnames, allP
          
    def all2bin(self):
        """Writes all data arrays as a binary, readable by data2ensight"""
        os.chdir(loc['new_dir'])
        nX2 = [self.nX['x'], self.nX['y'], self.nX['z']]
        nX2.append(len(self.allP)) # for nVar
        with open(self.allDatafile,'wb') as f:
            # Dimensions
            for nx in nX2: # writing nx,ny,nz and nvar
                f.write(pack("<i", nx))
            f.write(pack("<d",0)) # dt
            f.write(pack("<d",0)) # t for data2ensight
            for var in self.varnames:
                var = var.ljust(8)
                for v in var:
                    f.write(pack("c",v.encode())) # variable name
            for P in self.allP:
                P = np.array(P)
                for z in range(self.nX['z']):
                    for y in range(self.nX['y']):
                        for x in range(self.nX['x']):
                            f.write(pack("<d",P[x,y,z]))


#%% Main    
def main_multi():
    for selected_timestep in selected:
        print(timesteps[selected_timestep])
        for num in data_num:
            print(num) if int(num) % 50 == 0 else None
            timestep = timesteps[selected_timestep]
            Chop(timestep, num)


if __name__ == '__main__':
    main_multi()
