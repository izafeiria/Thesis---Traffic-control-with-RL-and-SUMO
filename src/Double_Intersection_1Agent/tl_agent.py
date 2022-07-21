#traffic light as an agent for 1 agent of the double intersections network
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import numpy as np
from gym import spaces

class Traffic_Light:
    
    """
    This class represents a Traffic Signal/Light of our intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """
    
    def __init__(self, env, tl_id, delta_time, yellow_time, min_green, max_green):
           self.id = tl_id
           self.env = env
           self.delta_time = delta_time
           self.yellow_time = yellow_time
           self.min_green = min_green
           self.max_green = max_green
           self.green_phase = 0
           self.is_yellow = False
           self.time_since_last_phase_change = 0
           self.next_action_time = 0
           self.last_wt = 0.0
           
           self.fuel_coeff = 1
           self.cars_coeff = 1
           self.time_coeff = 1
           self.noise_coeff = 1
           
           self.rewards = []
           self.features = []
           
           self.last_passed_cars = 0.0
           self.last_q_measure = 0.0
           self.last_noise = 0.0
           self.last_CO = 0.0
           self.last_CO2 = 0.0
           self.last_fuel = 0.0
           self.last_occ = None
           self.last_reward = None
           
           self.phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0].phases
           self.num_green_phases = len(self.phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
           self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id))) # Remove duplicates and keep order
           # outgoing lanes used for below functions, link[0][1] is the out lane
           self.out_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(self.id) if link]
           self.out_lanes = list(set(self.out_lanes)) # with set() remove duplicates

           self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases + len(self.lanes), dtype=np.float32), high=np.ones(self.num_green_phases + len(self.lanes)), dtype=np.float32)

           self.action_space = spaces.Discrete(self.num_green_phases)
           
           #initialize logic for the traffic light
           programs = traci.trafficlight.getAllProgramLogics(self.id)
           logic = programs[0]
           logic.type = 0
           logic.phases = self.phases
           traci.trafficlight.setProgramLogic(self.id, logic)
           
           self.cars_rew = []
           self.noise_rew = []
           self.fuel_rew = []
           self.time_rew = []


    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step
    
    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            traci.trafficlight.setPhase(self.id, int(self.green_phase))
            self.is_yellow = False

    def set_next_phase(self, new_phase):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current
        :param new_phase: (int) Number between [0..num_green_phases] 
        """
        new_phase *= 2 
        if self.phase == new_phase or self.time_since_last_phase_change < self.yellow_time:
            self.green_phase = self.phase
            traci.trafficlight.setPhase(self.id, self.green_phase)
            self.next_action_time = self.env.sim_step + self.delta_time
    
        else:
            self.green_phase = new_phase
            traci.trafficlight.setPhase(self.id, self.phase + 1)  # turns yellow
            self.next_action_time = self.env.sim_step + self.delta_time + self.yellow_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0
                
    def compute_observation(self):
        phase_id = [1 if self.phase/2 == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding  
        phase = [1 if phase_id == [1,0] else 0]
        density = self.get_lanes_density()
        
        observation = tuple(phase + density)
        return observation
            
    def compute_reward(self):
        self.last_reward = self.waiting_time_reward() + self.fuel_reward() + self.passed_cars_reward() + self.noiseEmission_reward()
        return self.last_reward
    
    ### FEATURES ##
    def passed_cars(self): 
        MIN = 0
        MAX = 9
        cars = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes)
        norm = (cars - MIN)/ (MAX - MIN)
        return norm


    def noiseEmission(self):

        MIN = 68.94
        MAX = 292.56
        noise = sum(traci.lane.getNoiseEmission(lane) for lane in self.lanes)
        norm = (noise - MIN)/ (MAX - MIN)
        return norm

    def get_fuel_consumption(self):
        MIN = 2.15
        MAX = 37.27
        fuel = sum(traci.lane.getFuelConsumption(lane) for lane in self.lanes)
        norm = (fuel - MIN)/ (MAX - MIN)
        return norm
        
    def waitingTime(self):
        MIN = 0
        MAX = 4.6
        wt = sum(self.get_waiting_time_per_lane())/100.00
        norm = (wt - MIN)/ (MAX - MIN)
        return norm

    ### REWARDS ##             
    def passed_cars_reward(self):
        new_passed_cars =  self.passed_cars()
        last_cars = self.last_passed_cars
        change = new_passed_cars - last_cars
        #print(change,'c')
   
     
        if last_cars == 0:
            reward = -1
        else:
            if change >= 0: 
                increase = (change/last_cars)
                if increase <= 0.15:
                    reward = 1
                elif increase <= 0.25:
                    reward = 2
                elif increase <= 0.35:
                     reward = 3
                else:
                    reward = 4
            else:
                decrease = -(change/last_cars)
            
                if decrease <= 0.15:
                    reward = -1
                elif decrease <= 0.25:
                    reward = -2
                elif decrease <= 0.35:
                     reward = -3
                else:
                    reward = -4
     
        self.cars_rew.append(reward)
        self.last_passed_cars = self.passed_cars()
        return reward

    def noiseEmission_reward(self):  
        new_noise = self.noiseEmission()
        last_noise = self.last_noise
        change = last_noise - new_noise
 
        
        if last_noise == 0:
            reward = -1
        else:
            if change >= 0: 
                decrease = (change/last_noise)
                
                if decrease <= 0.15:
                    reward = 1
                elif decrease <= 0.25:
                    reward = 2
                elif decrease <= 0.35:
                     reward = 3
                else:
                    reward = 4
            else:
                increase = -(change/last_noise)
                if increase <= 0.15:
                    reward = -1
                    
                elif increase <= 0.25:
                    reward = -2
                elif increase <= 0.35:
                     reward = -3
                else:
                    reward = -4
  
        self.noise_rew.append(reward)
        self.last_noise = self.noiseEmission()
        return reward
    
    def fuel_reward(self):
         new_fuel = self.get_fuel_consumption()
         last_fuel = self.last_fuel
         change = last_fuel - new_fuel
 
         
         if last_fuel == 0:
             reward = -1
         else:
             if change >= 0: 
                 decrease = (change/last_fuel)
                 if decrease <= 0.15:
                     reward = 1
                 elif decrease <= 0.25:
                     reward = 2
                 elif decrease <= 0.35:
                      reward = 3
                 else:
                     reward = 4
             else:
                 increase = -(change/last_fuel)
                 if increase <= 0.15:
                     reward = -1 
                 elif increase <= 0.25:
                     reward = -2
                 elif increase <= 0.35:
                      reward = -3
                 else:
                     reward = -4

         self.fuel_rew.append(reward)
         self.last_fuel = self.get_fuel_consumption()
         return reward
        
    def waiting_time_reward(self):
        new_wait = self.waitingTime()
        last_wait = self.last_wt
        change = last_wait - new_wait 
    
  
        
        if last_wait == 0:
            reward = -1
        else:
            if change >= 0: 
                decrease = (change/last_wait)
       
                if decrease <= 0.15:
                    reward = 1
                elif decrease <= 0.25:
                    reward = 2
                elif decrease <= 0.35:
                     reward = 3
                else:
                    reward = 4
            else:
                increase = -(change/last_wait)
     
                if increase <= 0.15:
                    reward = -1
                elif increase <= 0.25:
                    reward = -2
                elif increase <= 0.35:
                     reward = -3
                else:
                    reward = -4
       
        self.time_rew.append(reward)
        self.last_wt = self.waitingTime()
        return reward
    
    # useful funcs
    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.out_lanes]

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]
        
    def get_lanes_queue(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepHaltingNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]
    
    def get_waiting_time_per_lane(self):
        
        wait_time_per_lane = []
        wait= 0.0
        for lane in self.lanes:
            wait = traci.lane.getWaitingTime(lane)
            wait_time_per_lane.append(wait)
        return wait_time_per_lane
            
        # wait_time_per_lane = []
        # for lane in self.lanes:
        #     veh_list = traci.lane.getLastStepVehicleIDs(lane)
        #     wait_time = 0.0
        #     for veh in veh_list:
        #         veh_lane = traci.vehicle.getLaneID(veh)
        #         acc = traci.vehicle.getAccumulatedWaitingTime(veh)
        #         if veh not in self.env.vehicles:
        #             self.env.vehicles[veh] = {veh_lane: acc}
        #         else:
        #             self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
        #         wait_time += self.env.vehicles[veh][veh_lane]
        #     wait_time_per_lane.append(wait_time)
        
        # return wait_time_per_lane # [4]
        
    def _rewards(self):
        return {
            'queue_average_reward': self.queue_average_reward(),
            'passed_cars_reward': self.passed_cars_reward(),
            'pressure_reward': self.pressure_reward(),
            'waiting_time_reward': self.waiting_time_reward(),
            'noiseEmission_reward': self.noiseEmission_reward(),
            'COEmission_reward': self.COEmission_reward(),
            'CO2Emission_reward': self.CO2Emission_reward(), 
            'fuel_reward': self.fuel_reward(),
            'occupancy_reward': self.occupancy_reward(),
            'reward': self.compute_reward()
              }
    
    def _features(self):
        return {
            'waitingTime': self.waitingTime(),
            'passedCars': self.passed_cars(),
            'noiseEmission': self.noiseEmission(),
            'fuelConsumption': self.get_fuel_consumption(),
              }
    
    def values(self):
        self.last_fuel = self.get_fuel_consumption()
        self.last_noise = self.noiseEmission()
        self.last_passed_cars = self.passed_cars()
        self.last_wt = self.waitingTime()

        

                


