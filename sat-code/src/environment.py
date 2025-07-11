import gymnasium, numpy as np, random as rand

class SatelliteSwarmEnv(gymnasium.Env):
    def __init__(self,num_satellites,grid_size,max_timesteps,coverage_radius):
        super(SatelliteSwarmEnv, self).__init__()
        self.num_satellites = num_satellites
        self.grid_size = grid_size
        self.max_timesteps = max_timesteps
        self.coverage_radius = coverage_radius
        
        # Define action and observation spaces
        self.action_space = gymnasium.spaces.Discrete(num_satellites * 5)
        self.observation_space = gymnasium.spaces.Box(
            low=0, 
            high=grid_size, 
            shape=(num_satellites, 2), 
            dtype=np.float32
        )
    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=options)
    
class Sat:
    def __init__(self,x,y,velocity_x,velocity_y):
        self.position= set((x,y))
        self.velocity = set((velocity_x,velocity_y))

    def reset(self):
        self.position = set((rand.randint(-50,50), rand.randint(-50,50)))
        self.velocity = set((rand.randint(-10,10), rand.randint(-10,10)))

    def updateVelocity(self, velocity_x, velocity_y):
        self.velocity = set((velocity_x, velocity_y))
    
    
    

