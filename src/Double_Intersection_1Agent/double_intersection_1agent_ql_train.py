### main script for training or testing signle agent of the double intersection network
import argparse
import os
import sys
from datetime import datetime
import pandas as pd
import ujson , json
import dataframe_image as dfi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ujson , json
import copy
from pathlib import Path 

 
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_env_double_1agent import SumoEnvironments
from QL_algorithm import  QLAgent,  EpsilonGreedy, Testing


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Double-Intersection Training""")
    prs.add_argument("-route", dest="route", type=str, default='../Network/double/doubleRoutes.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.036, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.2, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=1.0, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.0002, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=0.999, required=False, help="Epsilon decay.\n")
    
    prs.add_argument("-fixed", action="store_true", default = False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-gui", action="store_true", default= False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=40, help="Number of runs.\n")
    args = prs.parse_args()
    
    out_csv = '../Train/double/1Agent/metrics'
    env = SumoEnvironments(network='../Network/double/double.net.xml',
                          route_file=args.route,
                          output_file = out_csv,
                          sumo_gui=args.gui,
                          num_seconds=args.seconds,
                          max_depart_delay=0)

    

    ### put qtable for testing the model 
    # openTable = open('../QTables/double/final0.6!.json', "r+")
    # Qtable= json.load(openTable)
        
    initial_states = env.reset()
    initial = env.encode(initial_states)
    ql_agents = QLAgent(starting_state= initial,
                                  state_space=env.obs_space,
                                  action_space=env.act_space,
                                  alpha=args.alpha,
                                  gamma=args.gamma,
                                  qtable =  None,
                                  exploration_strategy= EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay))
    # or Testing() as exploration_strategy for testing )

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
    
    for run in range(1, args.runs+1):
        x_values.append(run)
        env.reset()

        infos = []
        done = {'__all__': False}
        if args.fixed: 
            while not done['__all__']:
                _, _, done, _ = env.step({})
        while not done['__all__']:
            actions = {ts: ql_agents.act(ts) for ts in env.ts_ids}
    
            
            s, r, done, info = env.step(actions)
            
            x = actions.get('1')
            y = actions.get('gneJ13')
            if x==0 and y==1:
                ql_agents.action = 0
            if x==1 and y==0:
                ql_agents.action = 1
            if x==0 and y==0:
                ql_agents.action = 2
            if x==1 and y==1:
                ql_agents.action = 3

         
            ql_agents.learn(next_state=env.encode(s), reward=r)

        
                
        data = pd.DataFrame(env.metrics)
        statistics = data.describe()
        df = pd.DataFrame(statistics)
        mean = df.total_wait_time[1:2]
        print('Mean waiting time after {} run simulation of 100000 seconds : '.format(run) , mean.item())
        
        wr.append(np.mean(env.traffic_lights['1'].time_rew + env.traffic_lights['gneJ13'].time_rew))
        fr.append(np.mean(env.traffic_lights['1'].fuel_rew + env.traffic_lights['gneJ13'].fuel_rew))
        cr.append(np.mean(env.traffic_lights['1'].cars_rew + env.traffic_lights['gneJ13'].cars_rew))
        nr.append(np.mean(env.traffic_lights['1'].noise_rew + env.traffic_lights['gneJ13'].noise_rew))
    
        eprewards.append(df.reward[1:2].item()) 
        stops.append(df.total_stopped[1:2].item())
        waitingTime.append(df.total_wait_time[1:2].item())
        fuelConsumption.append(df.fuel_consumption[1:2].item())
        noiseEmission.append(df.noise_emission[1:2].item())
    
    with open('../QTables/double/1Agent/final.json', 'w') as convert_file:
        convert_file.write(ujson.dumps(ql_agents.q_table))

            

    
    y_values = [eprewards, stops, waitingTime, fuelConsumption, noiseEmission]
    y_names = ['Rewards', 'Stop of Cars', 'Waiting Time', 'Fuel Consumption', 'Noise Emission']
     
    for y, names in zip(y_values, y_names):
         # Plot mean waiting time in every episode
         plt.plot(x_values, y, color = 'cyan')
          
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
  
    env.save_csv(out_csv)
    env.close()