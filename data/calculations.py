# Import all necessary libraries
import numpy as np
import csv
from struct import unpack
from matplotlib import pyplot as plt

# Import Chodsp Class
class Reader:
    def __init__(self, config_file, data_file):
        self.config_file = config_file
        self.data_file = data_file
        
        # Read Data
        self.read()

    def read(self):
        X, nX = self.openConfig()
        params = self.readParams()
        return (X, nX, params)

    def openConfig(self):
        with open(self.config_file, 'rb') as file:
            config = file.read()

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

        return X, nX

    def readParams(self):
        with open(self.data_file,'rb') as file:
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

class Calculator:
    def __init__(self, X, nX, data):
        self.X = X
        self.nX = nX
        self.data = data
        self.nu = 1e-4

    def d(self, data, var):
        # Create array to store results
        out = np.zeros(data.shape)

        # Define inner indices
        x_inner_start = 1
        y_inner_start = 1
        z_inner_start = 1
        x_inner_end = data.shape[0] - 1
        y_inner_end = data.shape[1] - 1
        z_inner_end = data.shape[2] - 1

        # Calculate deltas (uniform so this works)
        dx = self.X['x'][1] - self.X['x'][0]
        dy = self.X['y'][1] - self.X['y'][0]
        dz = self.X['z'][1] - self.X['z'][0]

        if var == 'x':
            out[x_inner_start:x_inner_end, y_inner_start:y_inner_end, z_inner_start:z_inner_end] = (-data[x_inner_start-1:x_inner_end-1, y_inner_start:y_inner_end, z_inner_start:z_inner_end] + data[x_inner_start+1:x_inner_end+1, y_inner_start:y_inner_end, z_inner_start:z_inner_end]) / (2*dx)
        
        elif var == 'y':
            out[x_inner_start:x_inner_end, y_inner_start:y_inner_end, z_inner_start:z_inner_end] = (-data[x_inner_start:x_inner_end, y_inner_start-1:y_inner_end-1, z_inner_start:z_inner_end] + data[x_inner_start:x_inner_end, y_inner_start+1:y_inner_end+1, z_inner_start:z_inner_end]) / (2*dy)

        elif var == 'z':
            out[x_inner_start:x_inner_end, y_inner_start:y_inner_end, z_inner_start:z_inner_end] = (-data[x_inner_start:x_inner_end, y_inner_start:y_inner_end, z_inner_start-1:z_inner_end-1] + data[x_inner_start:x_inner_end, y_inner_start:y_inner_end, z_inner_start+1:z_inner_end+1]) / (2*dz)

        return out

    def ddx(self, data):
        # Return out
        return self.d(data, 'x')

    def ddy(self, data):
        # Return out
        return self.d(data, 'y')

    def ddz(self,data):
        # Return out
        return self.d(data, 'z')

    def sij(self):
        u = self.data['Uf']
        v = self.data['V']
        w = self.data['W']

        out = (1/2) * np.array([
            [self.ddx(u) + self.ddx(u), self.ddy(u) + self.ddx(v), + self.ddz(u) + self.ddx(w)],
            [self.ddx(v) + self.ddy(u), self.ddy(v) + self.ddy(v), + self.ddz(v) + self.ddy(w)],
            [self.ddx(w) + self.ddz(u), self.ddy(w) + self.ddz(v), + self.ddz(w) + self.ddz(w)]
            ])

        return out

    def sij2(self):
        # Allocate space for array
        out = np.zeros(self.data['Uf'].shape)

        # Get value of sij
        sij = self.sij()
        
        # Create inner indices
        x_inner_start = 0
        y_inner_start = 0
        z_inner_start = 0
        x_inner_end = out.shape[0]
        y_inner_end = out.shape[1]
        z_inner_end = out.shape[2]

        # Calculate
        out[x_inner_start:x_inner_end, y_inner_start:y_inner_end, z_inner_start:z_inner_end] = np.sum(sij[:,:,x_inner_start:x_inner_end, y_inner_start:y_inner_end, z_inner_start:z_inner_end] ** 2, axis=(0,1))

        # Return
        return out
    
    def get_u_mean(self):
        # Calculate means
        means = self.data['U'].mean(axis=(0,2))

        # Allocate space for output
        out = np.ones(self.data['U'].shape)

        # Set means to array
        for i in range(len(means)):
            out[:,i,:] = means[i]

        return out 
    def compute(self):
        self.TKE = self.TKE()
        self.prod = self.prod()
        self.diss = self.diss()
        self.trans = self.trans()
        
        plt.figure(1)
        plt.imshow(self.prod[:,5,:])
        plt.figure(2)
        plt.imshow(np.array(self.data['Prod'][:,5,:], dtype=np.float64))
        plt.show()
        print(type(self.prod[1,1,1]))
        print(type(self.data['Prod'][1,1,1]))
        

    def TKE(self):
        return (1/2) * (self.data['Uf']**2 + self.data['V']**2 + self.data['W']**2)

    def prod(self):
        u = self.get_u_mean()
        return -1 * self.data['Uf'] * self.data['V'] * self.ddy(u)

    def diss(self):
        return -2 * self.nu * self.sij2()

    def trans(self):
        # Allocate space
        out = np.zeros(self.data['Uf'].shape)

        return out

def main():
    # Read data
    reader = Reader('data004/004intConfig','data004/004allDataChop')

    calculator = Calculator(*reader.read())
    calculator.compute()

if __name__ == '__main__':
    main()
