import os 
import sys
from pathlib import Path
import itertools
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib
from sumolib import checkBinary
import gym

import numpy as np
import pandas as pd
from tlofficial import Traffic_Light


LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ

class SumoEnvironment(gym.Env):

    CONNECTION_LABEL = 0 # for traci multiclient support


    def __init__(self,  network, route_file, num_seconds , output_file = None, sumo_gui = False, begin_time = 0,  max_depart_delay = 100000, time_to_teleport = -1, delta_time = 5, yellow_time = 3, min_green = 5, max_green = 50, sumo_seed='random', fixed_ts= False):

        self.net = network
        self.routes = route_file
        self.sumo_gui = sumo_gui
        self.begin_time = begin_time
        
        if self.sumo_gui:
            self.sumoBinary = sumolib.checkBinary('sumo-gui')
        else:
            self.sumoBinary = sumolib.checkBinary('sumo')
        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."

        self.sim_max_time = num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None
        

        if LIBSUMO:
            traci.start([sumolib.checkBinary('sumo'), '-n', self.net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary('sumo'), '-n', self.net], label='init_connection'+self.label)
            conn = traci.getConnection('init_connection'+self.label)
        
        
        self.tl_id = "1"
        self.Traffic_Light = Traffic_Light(self, self.tl_id, self.delta_time, self.yellow_time, self.min_green, self.max_green)

        conn.close()
    
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}

        self.run = 0
        self.metrics = []
        self.output_file = output_file
        
        self.vehicles = dict()
 
    @property #sim_step is not callable
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()

    
    def start_simulation(self):
        sumo_cmd = [self.sumoBinary,
                     '-n', self.net,
                     '-r', self.routes, 
                     '--waiting-time-memory', '50000',
                     '--max-depart-delay', str(self.max_depart_delay), 
                     '--time-to-teleport', str(self.time_to_teleport)]
        if self.begin_time > 0:
            sumo_cmd.append('-b {}'.format(self.begin_time))
        
        if self.sumo_seed == 'random':
            sumo_cmd.append('--random')
        else:
            sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        
        if self.sumo_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
        
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)


    def reset(self):
        #reset simulation
        if self.run != 0:
            traci.close()
            self.save_csv(self.output_file) #create .csv in reset :/
        self.run += 1
        self.metrics = []


        #initialize simulation and start SUMO client
        
        self.start_simulation()

        self.Traffic_Light = Traffic_Light(self, self.tl_id,  self.delta_time, self.yellow_time, self.min_green, self.max_green) 
    
        observation = self.Traffic_Light.compute_observation()

        return observation  


    def render(self):
        # set sumo_gui variable to true 
        self.sumo_gui = True
        print("rendering the environment")

    def close(self):
        # close sumo simulation
        traci.close()

    
    def step(self, action):
      
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self.sumo_step()
                if self.sim_step % 5 == 0:
                    info = self.compute_step_info()
                    self.metrics.append(info)
                  
        else:
            self.apply_action(action)

            time_to_act = False
            while not time_to_act:
                self.sumo_step()
                

                self.Traffic_Light.update()
                if self.Traffic_Light.time_to_act:
                    time_to_act = True
                    

                if self.sim_step % 5 == 0:
                    info = self.compute_step_info()
                    self.metrics.append(info)
                    

        observation = self.Traffic_Light.compute_observation()
        reward = self.Traffic_Light.compute_reward()
        done = self.compute_done()
    
        
        return observation, reward, done, {}
    
    def compute_done(self):
        done = self.sim_step > self.sim_max_time
        return done
        
    def apply_action(self, action):
        #set the next green phase for the traffic light 
        self.Traffic_Light.set_next_phase(action)

    @property
    def observation_space(self):
        return self.Traffic_Light.observation_space
    
    @property
    def action_space(self):
        return self.Traffic_Light.action_space

    def sumo_step(self):

        traci.simulationStep()


    def save_csv(self, output_file):
         if output_file is not None:
             df = pd.DataFrame(self.metrics)
             Path(Path(output_file).parent).mkdir(parents=True, exist_ok=True)
             df.to_csv(output_file  + 'metrics' + '.csv', index=False)
         return


    def compute_step_info(self):
     
        return {
            'step_time': self.sim_step,
            'reward': self.Traffic_Light.last_reward,
            'total_stopped': self.Traffic_Light.get_total_queued() ,
            'total_wait_time': sum(self.Traffic_Light.get_waiting_time_per_lane()),
            'fuel_consumption': self.Traffic_Light.get_fuel_consumption(),
            'noise_emission': self.Traffic_Light.noiseEmission(),
            'waitingreward' : self.Traffic_Light.waiting_time_reward()
             }


    ## Below functions are for discrete state space

    def encode(self, state):
        density_queue = [self._discretize_density(d) for d in state[0:4]]

        
        # tuples are hashable and can be used as key in python dictionary

        return tuple(density_queue) #4 items
    
    
    def _discretize_density(self, density):

        return min(int(density*10), 9)
    








