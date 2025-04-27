"""
Example case estimation of the 3D Van der Pol oscillator.

Created on 2024-03-10
Author: Assistant

References:
- [1] Van der Pol, B., 1926. On relaxation-oscillations. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from tqdm import tqdm
from jitcsde import jitcsde, y, t

class VanDerPol3D:
    """
    Simulates and visualizes the 3D Van der Pol oscillator, a non-conservative oscillator with non-linear damping.
    
    Attributes:
        mu (float): Nonlinearity parameter
        alpha (float): Damping coefficient for z
        beta (float): Coupling strength between x and z
        dt (float): Time step for the integration
        sigma_noise (float): Standard deviation of the noise
        X (numpy.ndarray): Current state of the system
        _history_X (list): History of system states
        additive (bool): Indicates if the noise is additive
        seed (int): Seed for RNG to ensure reproducibility
        noprog (bool): If True, disables the progress bar
        
    Parameters:
        mu (float): Nonlinearity parameter (default: 4.0)
        alpha (float): Damping coefficient for z (default: 0.5)
        beta (float): Coupling strength between x and z (default: 0.5)
        dt (float): Integration time step (default: 0.01)
        sigma_noise (float): Noise standard deviation (default: 0.1)
        X_init (numpy.ndarray, optional): Initial state. Random if None (default: None)
        additive (bool): If True, noise is additive (default: True)
        seed (int): RNG seed for reproducibility (default: 0)
        show_plot (bool): If True, enables plotting (default: True)
        noprog (bool): If True, disables tqdm progress bar (default: False)
    """
    def __init__(self, mu=1, alpha=0.1, beta=0.5, dt=0.005, sigma_noise=np.sqrt(0.5), X_init=None, **kwargs):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.sigma_noise = sigma_noise
        self.X0 = np.random.rand(3) if X_init is None else X_init.copy()
        self.X = [self.X0.copy()]
        self.additive = kwargs.get('additive', True)
        self.seed = kwargs.get('seed', 0)
        self.noprog = kwargs.get('noprog', False)
        self.show_plot = kwargs.get('show_plot', True)

    def _define_f(self):
        """
        Defines the deterministic component of the 3D Van der Pol oscillator.
        
        Returns:
            list: A list of symbolic differential equations representing the Van der Pol oscillator's dynamics.
        """
        f_sym = [
            y(1),  # dx/dt = y
            self.mu * (1 - y(0)**2) * y(1) - y(0) + y(2),  # dy/dt = mu(1-x^2)y - x + z
            -self.alpha * y(2) - self.beta * y(0)  # dz/dt = -alpha*z - beta*x
        ]
        return f_sym

    def _define_g(self):
        """
        Defines the stochastic component of the 3D Van der Pol oscillator.
        
        Returns:
            list: A list of symbolic equations representing the stochastic component.
        """
        return [self.sigma_noise for _ in range(3)]

    def iterate(self, time):
        """
        Advances the model state over a specified period through numerical integration.
        
        Parameters:
            time (float): The total time period over which to integrate the model.
        """
        f_sym = self._define_f()
        g_sym = self._define_g()
        SDE = jitcsde(f_sym=f_sym, g_sym=g_sym, n=3, additive=self.additive)
        SDE.set_initial_value(initial_value=self.X0, time=0.0)
        SDE.set_seed(seed=self.seed)
        
        steps = int(time / self.dt)
        for _ in tqdm(range(steps), disable=self.noprog):
            self.X0 = SDE.integrate(SDE.t + self.dt)
            self.X.append(self.X0.copy())

    @property
    def _history(self):
        """
        Returns the recorded history of the system states.
        
        Returns:
            numpy.ndarray: A 2D array of the system's states over time.
        """
        return np.array(self.X)

    def static_plot(self):
        """
        Creates a static 3D plot of the system's trajectory.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        history = self._history
        ax.plot(history[:, 0], history[:, 1], history[:, 2], lw=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Van der Pol Oscillator')
        
        plt.show()

    def animate_plot(self, total_frames=200):
        """
        Generates a 3D animation of the system's evolution.
        
        Parameters:
            total_frames (int): The total number of frames for the animation.
            
        Returns:
            matplotlib.animation.FuncAnimation: The animation object.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        history = self._history
        line, = ax.plot([], [], [], lw=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Van der Pol Oscillator')
        
        # Set the limits based on the data
        ax.set_xlim(history[:, 0].min(), history[:, 0].max())
        ax.set_ylim(history[:, 1].min(), history[:, 1].max())
        ax.set_zlim(history[:, 2].min(), history[:, 2].max())
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            return line,
        
        def animate(i):
            index = i * (len(history) // total_frames)
            line.set_data(history[:index, 0], history[:index, 1])
            line.set_3d_properties(history[:index, 2])
            return line,
        
        ani = animation.FuncAnimation(fig, animate, frames=total_frames,
                                    interval=40, blit=True, init_func=init)
        plt.close()
        return ani 