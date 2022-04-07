from calendar import monthrange
from typing import NoReturn
from xmlrpc.client import Boolean
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import typing

L=12

trigerring = 1.4

indexer = 0


def get_random_vec():
    a = np.random.normal(0)
    b = np.random.normal(0)
    return np.array([a,b])

def apply_periodic(r):
    if r[0]<0:
        r[0]+= L
    elif r[0]>L:
        r[0]-= L
    if r[1]<0:
        r[1]+=L
    elif r[1]>L:
        r[1]-=L
    return r

def closest_image(x1,x2):
    x12 = x2 - x1
    if x12[0] > L / 2:
        x12[0] = x12[0] - L
    elif x12[0] < -L / 2:
        x12[0] = x12[0] + L
    if x12[1] > L / 2:
        x12[1] = x12[1] - L
    elif x12[1] < -L / 2:
        x12[1] = x12[1] + L
    return x12

class Cell():
    age:int = 0
    mother = None
    parent = None
    typpe = 0
    close_neighbours:int = 0

    def get_kd(self):
        return self.kd*3 if self.typpe == 1 else self.kd

    def __init__(self,parent,r ,ka, kd):
        global indexer
        self.parent = parent
        self.r = r
        self.R = np.random.normal(1.0,1e-1)
        self.ka = ka
        self.kd = kd
        indexer +=1
        self.num = indexer

    def evolve_r(self):
        force = self.parent.get_force(self)
        self.r = apply_periodic(self.r + self.parent.forces[str(self.num)] * self.parent.dt +np.sqrt(self.parent.dt*self.parent.alpha)*get_random_vec())
        self.age +=1

    def cell_divide(self):
        new_r = self.r+(1e-1)*np.random.rand(2)
        new_pos = apply_periodic(new_r)
        self.R = np.random.normal(1.0,1e-1)
        new_cell = Cell(self.parent,new_pos,self.ka,self.kd)
        if (np.random.uniform()<(1/40)):
            new_cell.typpe = 1
        new_cell.mother = self
        self.mother = new_cell
        return new_cell

    def random_choose_divide(self):
        # print(self.close_neighbours)
        prob = self.age*self.get_kd()*(1-(self.close_neighbours/6))
        rand_num = np.random.uniform()
        return rand_num <prob

    def random_choose_die(self):
        prob = self.age*self.ka
        rand_num = np.random.uniform()
        return rand_num <prob

class System():

    def dump(self, filename, plot_forces=False):
        fig, ax = plt.subplots()
        for cell in self.cells:
            ax.add_patch(plt.Circle((cell.r[0], cell.r[1]), cell.R, color='green'))
            ax.add_patch(plt.Circle((cell.r[0], cell.r[1]), cell.R*trigerring, color='red', fill=False))
        # for i in range(0, len(self.cells)):
        #     ax.add_patch(plt.Circle((r[i][0], r[i][1]), R[i], color='green'))
        #     if plot_forces:
        #         ax.quiver(r[i][0], r[i][1], force[i][0], force[i][1], angles='xy', scale_units='xy',
        #                 scale=0.8, color='black')
        plt.axis('equal')
        plt.axis('off')
        plt.xlim(0,L)
        plt.ylim(0,L)
        plt.tight_layout()

        plt.savefig(f'{filename}.png')
        plt.close()

    def __init__(self,dt, alpha, k_atr, k_rep):
        self.dt = dt
        self.alpha = alpha
        self.k_atr = k_atr
        self.k_rep = k_rep
        self.cells = [Cell(self, np.array([L/2,L/2]),10**(-2),10**(-1))]

    def get_k_rep(self,cell_a:Cell,cell_b:Cell) -> float:
        return (
            self.k_rep
            if ((cell_a.parent != None) & (cell_b.parent != None))
            else self.k_rep * 0.5
        )

    def get_single_force(self,cell_a:Cell, cell_b:Cell):
        killme = False
        addme = None
        if cell_a != cell_b:
            trigerring_distance = cell_a.R + cell_b.R
            dire = closest_image(cell_a.r,cell_b.r)
            dire_len = np.sqrt(np.sum(dire**2))
            dire_w = dire/dire_len
            if dire_len<trigerring*trigerring_distance:
                cell_a.close_neighbours += 1
                if trigerring_distance<dire_len:
                    #attractive force
                    return self.k_atr*dire_w
                else:
                    #repulsive force
                    return -self.get_k_rep(cell_a,cell_b)*(trigerring_distance-dire_len)*dire_w

        return np.array([0,0])

    def check_for_changes(self):
        new_cells=[]
        dead_cells = []
        for cell in self.cells:
            if cell.mother is None:
                if(cell.random_choose_divide()):
                    new_cell = cell.cell_divide()
                    # print(new_cell)
                    new_cells.append(new_cell)
                if (cell.random_choose_die()):
                    self.cells.remove(cell)
            else:
                dist= np.sqrt(np.sum(closest_image(cell.r, cell.mother.r)**2))
                if dist >= (cell.R+cell.mother.R)*0.98:
                    cell.mother.mother = None
                    cell.mother = None
        self.cells += new_cells

    def get_force(self, cell):
        cell.close_neighbours = 0
        temp = [self.get_single_force(cell,a) for a in self.cells]
        return np.sum(temp, axis=0)

    def evolve_force(self):
        self.forces = {str(cell.num):self.get_force(cell) for cell in self.cells}

    def move_system(self):
        {cell.evolve_r() for cell in self.cells}

Nsnaps = 200
Nrun = 50
Cell_Process_Check = 7

system = System(10**(-2),0.1,1,50)
n_of_cells = []
for snap in range(Nsnaps):
    system.dump(f't={snap}', False)
    for run in range(Nrun-1):
        system.evolve_force()
        system.move_system()
        print((snap*Nrun)+run)
        if (((snap*Nrun)+run)%Cell_Process_Check == 0):
            system.check_for_changes()
            n_of_cells.append(len(system.cells))

plt.plot(n_of_cells)
plt.show()