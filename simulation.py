import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np 
import matplotlib.colors as mcolors
PATH_FILE = 'C:/parallel_programming/shallow_water_equation/shallow_water_simulation.json'

class Simulation:
    def __init__(self):

        with open(PATH_FILE , 'r', encoding = 'utf_8') as file:
            information = json.load(file)

        self.total_time = information['time']
        self.X = information['X']
        self.Y = information['Y']
        self.H = information['H']
        self.name = information['name']
        self.total_step = information['total_step']
        self.delta_x = information['delta_x']
        self.delta_y = information['delta_y']

        self.space_X, self.space_Y = np.meshgrid(np.linspace(0, self.Y, int(float(self.Y)/self.delta_y)), 
                                                np.linspace(0, self.X, int(float(self.X)/self.delta_x)))
        self.u_list = []
        self.v_list = []
        self.h_list = []
        for i in range(self.total_step):

            M = np.array(information['step_' + str(i)])
            M = M.reshape(int(float(self.X)/self.delta_x),int(float(self.Y)/self.delta_y),3)
            h , hv, hu = M[:,:,0], M[:,:,1], M[:,:,2]
            v , u = hv/h , hu/h 

            self.v_list.append(u)
            self.u_list.append(v)
            self.h_list.append(h)      
        
    def wave_2D_simulation(self):
        fig, ax = plt.subplots(1, 1)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        vmax = np.abs(self.h_list[int(len(self.h_list) / 2)]).max()
        vmin = -0.7 * vmax

        pmesh = ax.pcolormesh(
            self.space_X,
            self.space_Y,
            self.h_list[0],
            vmin=vmin,
            vmax=vmax,
            cmap='Spectral',
            shading='auto'
        )

        cbar = fig.colorbar(pmesh, ax=ax)
        cbar.set_label("Water Height [m]")

        def update_eta(num):
            ax.set_title(f"Surface elevation $\eta$ after step = {num:.2f}")
            pmesh.set_array(self.h_list[num].flatten())
            return pmesh,

        anim = animation.FuncAnimation(
            fig, update_eta,
            frames=len(self.h_list),
            interval=10,
            blit=False
        )
        return anim

    def wave_3D_simulation(self):
        # --- Vẽ 3D ---
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(0, np.max([np.max(h) for h in self.h_list]))
        ax.set_title("Water Ripple (Single Blue Tone)")

        # Bề mặt ban đầu
        surf = [ax.plot_surface(self.space_X, self.space_Y, self.h_list[0], facecolors=np.full(self.space_X.shape, '#0055ff'), edgecolor='none')]

        # Animation update
        def update(frame):
            for coll in ax.collections:
                coll.remove()

            surf[0] = ax.plot_surface(
                self.space_X, self.space_Y, self.h_list[frame],
                facecolors=np.full(self.space_X.shape, '#0055ff'), 
                edgecolor='none'
            )
            ax.set_title(self.name  +f" - Frame {frame}")
            return surf

        # Hiển thị hoạt ảnh
        anim = animation.FuncAnimation(fig, update, frames=len(self.h_list), interval=100)
        return anim 


    def vector_field_simulation(self):
        
        fig, ax = plt.subplots(figsize=(7, 7))
        title = ax.set_title("", fontsize=16)
        ax.set_xlabel("x [km]", fontsize=12)
        ax.set_ylabel("y [km]", fontsize=12)

        Q = ax.quiver(
            self.space_X/10.0, 
            self.space_Y/10.0, 
            self.u_list[0], 
            self.v_list[0],
            scale=30.0, scale_units='inches'
        )

        def update(frame_idx):
            u = self.u_list[frame_idx]
            v = self.v_list[frame_idx]

            magnitude = np.sqrt(u**2 + v**2)
    
            max_length = 10.0
            mask = magnitude > max_length
            u[mask] = u[mask] * max_length / magnitude[mask]
            v[mask] = v[mask] * max_length / magnitude[mask]
        
            Q.set_UVC(u, v) 
            elapsed_hours = frame_idx 
            title.set_text(f"Velocity field at step = {elapsed_hours:.2f}")
            return Q, title

        anim = animation.FuncAnimation(
            fig, update, frames=len(self.u_list), interval=100, blit=False
        )
        return anim 



if __name__ == '__main__':
    S = Simulation()
    anim = S.wave_3D_simulation()
    plt.show()


