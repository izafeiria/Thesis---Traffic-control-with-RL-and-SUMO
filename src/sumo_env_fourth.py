########## sumo env 4th
## main changes on the compute_observations() and encode() methods
import os
import sys
from pathlib import Path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
import sumolib
from sumolib import checkBinary
import gym

import numpy as np
import pandas as pd
from tlofficial import Traffic_Light

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ

class SumoEnvironments(gym.Env):
    """
    SUMO Environment for Traffic Signal Control
    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases definition
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param num_seconds: (int) Number of simulated seconds on SUMO
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    """
    CONNECTION_LABEL = 0
    
    def __init__(self,  network, route_file, num_seconds , output_file = None, sumo_gui = False, begin_time = 0,  max_depart_delay = 100000, time_to_teleport = -1, delta_time = 5, yellow_time = 3, min_green = 5, max_green = 50, sumo_seed='random', fixed_ts= False):

        self.net = network
        self.routes = route_file
        self.sumo_gui = sumo_gui
        self.begin_time = begin_time
        
        if self.sumo_gui:
            self.sumoBinary = sumolib.checkBinary('sumo-gui')
        else:
            self.sumoBinary = sumolib.checkBinary('sumo')

        self.sim_max_time = num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.label = str(SumoEnvironments.CONNECTION_LABEL)
        SumoEnvironments.CONNECTION_LABEL += 1
        self.sumo = None
        
        if LIBSUMO:
            traci.start([sumolib.checkBinary('sumo'), '-n', self.net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary('sumo'), '-n', self.net], label='init_connection'+self.label)
            conn = traci.getConnection('init_connection'+self.label)
        
        self.ts_ids = traci.trafficlight.getIDList()
        self.traffic_lights = {ts: Traffic_Light(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green) for ts in self.ts_ids}
        self.vehicles = dict()
        conn.close()
    
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}

        self.run = 0
        self.metrics = []
        self.output_file = output_file
    
    def start_simulation(self):
        sumo_cmd = [self.sumoBinary,
                      '-n', self.net,
                      '-r', self.routes, 
                      '--waiting-time-memory', '10000',
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
            self.save_csv(self.output_file)
        self.run += 1
        self.metrics = []


        #initialize simulation and start SUMO client
        
        self.start_simulation()
        self.traffic_lights = {ts: Traffic_Light(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green) for ts in self.ts_ids}
        self.vehicles = dict()

        observation = self.compute_observations()

        return observation 

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()
    
    def step(self, action):
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self.sumo_step()
                if self.sim_step % 5 == 0:
                    info = self.compute_step_info()
                    self.metrics.append(info)
        else:
            self.apply_actions(action)

            time_to_act = False
            while not time_to_act:
                self.sumo_step()
                
                for ts in self.ts_ids:
                    self.traffic_lights[ts].update()
                    if self.traffic_lights[ts].time_to_act:
                        time_to_act = True


        observations = self.compute_observations()
        rewards = self.compute_rewards()
        done = {'__all__': self.sim_step > self.sim_max_time}
        done.update({ts_id: False for ts_id in self.ts_ids})
        info = self.compute_step_info()
        self.metrics.append(info)
        
        return observations, rewards, done, {}
        
    def render(self):
        # set sumo_gui variab;e to true 
        self.sumo_gui = True
        print("rendering the environment")

    def close(self):
        # close sumo simulation
        traci.close()

    def apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """
        for ts, action in actions.items():
            self.traffic_lights[ts].set_next_phase(action)
    
    def compute_observations(self):
        phase = dict
        phase = {ts: self.traffic_lights[ts].phase if self.traffic_lights[ts].phase == 0 else 1 for ts in self.ts_ids }

        obs = dict
        obs = {ts: self.traffic_lights[ts].compute_observation() for ts in self.ts_ids if self.traffic_lights[ts].time_to_act}

        if '1' in obs.keys():
            obs['1'].append(phase['gneJ5']) 
        if 'gneJ5' in obs.keys():
            obs['gneJ5'].append(phase['1'] )
        if 'gneJ10' in obs.keys():
            obs['gneJ10'].append(phase['gneJ5'])
        if 'gneJ0' in obs.keys():
            obs['gneJ0'].append(phase['gneJ10'])

        return obs
    def compute_rewards(self):
        return {ts: self.traffic_lights[ts].compute_reward() for ts in self.ts_ids if self.traffic_lights[ts].time_to_act}
    
    def observation_spaces(self, ts_id):
        return self.traffic_lights[ts_id].observation_space
    
    def action_spaces(self, ts_id):
        return self.traffic_lights[ts_id].action_space
    
    @property
    def observation_space(self):
        return self.traffic_lights[self.ts_ids[0]].observation_space
    
    @property
    def action_space(self):
        return self.traffic_lights[self.ts_ids[0]].action_space
    
    def sumo_step(self):
        traci.simulationStep()

    def compute_step_info(self):
        reward = {ts : self.traffic_lights[ts].last_reward for ts in self.ts_ids}

        return {
            'step_time': self.sim_step,
            'reward': (sum(reward.values()))/len(reward),
            'total_stopped':  sum(self.traffic_lights[ts].get_total_queued() for ts in self.ts_ids) ,
            'total_wait_time': sum(sum(self.traffic_lights[ts].get_waiting_time_per_lane()) for ts in self.ts_ids),
            'fuel_consumption': sum(self.traffic_lights[ts].get_fuel_consumption() for ts in self.ts_ids),
            'noise_emission': sum(self.traffic_lights[ts].noiseEmission() for ts in self.ts_ids)
              }
    
    def save_csv(self, output_file):
          if output_file is not None:
              df = pd.DataFrame(self.metrics)
              Path(Path(output_file).parent).mkdir(parents=True, exist_ok=True)
              df.to_csv(output_file + '.csv', index=False)
          return output_file + '.csv'
     
    def close(self):
          traci.close()
         
    # Below functions are for discrete state space

    def encode(self, state):
        phase = state[4:]
        density_queue = [self._discretize_density(d) for d in state[0:4]]
        return tuple(phase + density_queue)
    
    
    def _discretize_density(self, density):
        return min(int(density*10), 9)
    
