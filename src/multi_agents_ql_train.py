### main script for training or testing the multi-agent systems
import argparse
import os
import sys
from datetime import datetime
import pandas as pd
import ujson , json
import dataframe_image as dfi
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
import statistics
 
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_env_sixth import SumoEnvironments
from QL_algorithm import  QLAgent,  EpsilonGreedy, Testing


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Double-Intersection""")
    prs.add_argument("-route", dest="route", type=str, default='../Network/sixth/routes0.4.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.036, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.2, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=1.0, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.0002, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=0.999, required=False, help="Epsilon decay.\n")
    
    prs.add_argument("-fixed", action="store_true", default =False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=5, help="Number of runs.\n")
    args = prs.parse_args()
    
    out_csv = '../f'
    env = SumoEnvironments(network='../Network/sixth/sixth.net.xml',
                          route_file=args.route,
                          output_file = out_csv,
                          sumo_gui=args.gui,
                          num_seconds=args.seconds,
                          max_depart_delay=0)
    
    # put qtables for testing
    #opentable = {ts : open('../QTables/sixth/{}.json'.format(ts), "r+") for ts in env.ts_ids}
    #qtables = {ts:  json.load(opentable[ts]) for ts in env.ts_ids }
 
    
    ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts]),
                                  state_space=env.observation_space,
                                  action_space=env.action_space,
                                  alpha=args.alpha,
                                  gamma=args.gamma,
                                  qtable =qtables[ts],
                                      exploration_strategy= EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay)) for ts in env.ts_ids}
    # or  Testing()) for ts in env.ts_ids} for testing
    
                             
    x_values = []
    waiting = []
    fuel = []
    cars = []
    noise = []
    
    wr = []
    fr = []
    cr = []
    nr = []
    
    eprewards=[]
    stops= []
    waitingTime =[]
    fuelConsumption = []
    noiseEmission = []
    waiting=[]
    for run in range(1, args.runs+1):
        x_values.append(run)
        initial_states = env.reset()
           
        
        env.reset()
        infos = []
        done = {'__all__': False}
        if args.fixed: 
            while not done['__all__']:
                _, _, done, _ = env.step({})
        while not done['__all__']:
    
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
            

            s, r, done, info = env.step(action=actions)
            
            
            for agent_id in s.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id]), reward=r[agent_id])
        


        data = pd.DataFrame(env.metrics)
        statistic = data.describe()
        df = pd.DataFrame(statistic)
        mean = df.total_wait_time[1:2]
        waiting.append(mean.item())
        print('Mean waiting time after {} run simulation of 100000 seconds : '.format(run) , mean.item())
        
        wr.append(np.mean(env.traffic_lights['gneJ10'].time_rew + env.traffic_lights['1'].time_rew + env.traffic_lights['gneJ0'].time_rew + env.traffic_lights['gneJ5'].time_rew ))# + env.traffic_lights['gneJ15'].time_rew +env.traffic_lights['gneJ13'].time_rew))
        fr.append(np.mean(env.traffic_lights['gneJ10'].fuel_rew + env.traffic_lights['1'].fuel_rew + env.traffic_lights['gneJ0'].fuel_rew + env.traffic_lights['gneJ5'].fuel_rew )) #+ env.traffic_lights['gneJ15'].fuel_rew + env.traffic_lights['gneJ13'].fuel_rew ))
        cr.append(np.mean(env.traffic_lights['gneJ10'].cars_rew + env.traffic_lights['1'].cars_rew + env.traffic_lights['gneJ0'].cars_rew + env.traffic_lights['gneJ5'].cars_rew )) # + env.traffic_lights['gneJ15'].cars_rew + env.traffic_lights['gneJ13'].cars_rew))
        nr.append(np.mean(env.traffic_lights['gneJ10'].noise_rew + env.traffic_lights['1'].noise_rew + env.traffic_lights['gneJ0'].noise_rew + env.traffic_lights['gneJ5'].noise_rew )) # + env.traffic_lights['gneJ15'].noise_rew + env.traffic_lights['gneJ13'].noise_rew ))
    
        eprewards.append(df.reward[1:2].item()) 
        stops.append(df.total_stopped[1:2].item())
        waitingTime.append(df.total_wait_time[1:2].item())
        fuelConsumption.append(df.fuel_consumption[1:2].item())
        noiseEmission.append(df.noise_emission[1:2].item())
        
        
    
                
    y_values = [eprewards, stops, waitingTime, fuelConsumption, noiseEmission]
    y_names = ['Rewards', 'Stop of Cars', 'Waiting Time', 'Fuel Consumption', 'Noise Emission']
    
    for y, names in zip(y_values, y_names):
        # Plot mean waiting time in every episode
        plt.plot(x_values, y,color= 'cyan')
         
        # naming the x axis
        plt.xlabel('Episodes')
        # naming the y axis
        plt.ylabel(names)
         
        # giving a title to my graph
        plt.title('Training Progress')
         
        # function to show the plot
        plt.show()
        
    y_val = [wr, fr, cr,  nr]
    ynames = ['Waiting time rewards', 'Fuel consumption rewards', 'Passed cars rewars', 'Noise emission rewards']
    
    for y, names in zip(y_val, ynames):
        # Plot mean waiting time in every episode
        plt.plot(x_values, y, color='red')
         
        # naming the x axis
        plt.xlabel('Episodes')
        # naming the y axis
        plt.ylabel(names)
         
        # giving a title to my graph
        plt.title('Training Progress')
         
        # function to show the plot
        plt.show()
        
    for ts in ql_agents.keys():
        with open('../QTables/fourth/{}.json'.format(ts), 'w') as convert_file:
          convert_file.write(ujson.dumps(ql_agents[ts].q_table))
    print('Mean : ', np.mean(waiting), statistics.mean(waiting))
    print('Standard deviation :' , statistics.stdev(waiting))            
    env.save_csv(out_csv)
    env.close()

