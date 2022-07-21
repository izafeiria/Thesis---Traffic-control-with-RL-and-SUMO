### main script for training or testing the model
import argparse
import os
import sys
from datetime import datetime
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
import ujson , json
import copy
from pathlib import Path 
import statistics

 
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_env import SumoEnvironment
from tlofficial import Traffic_Light
from QL_algorithm import  QLAgent,  EpsilonGreedy, Testing


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Single-Intersection""")
    prs.add_argument("-route", dest="route", type=str, default='../Network/single/routes.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.069, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.553, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=1.0, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.0003, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=0.999, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=  True, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default = False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=50, help="Number of runs.\n")
    args = prs.parse_args()
    out_csv = '../Tr'
    env = SumoEnvironment(network='../Network/single/single_intersection.net.xml',
                          route_file=args.route,
                          output_file = out_csv,
                          sumo_gui=args.gui,
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          max_green=args.max_green,
                          max_depart_delay=0)

    
    # openTable = open('../QTables/single/final.json', "r+")
    # Qtable= json.load(openTable)
    
      
    # qtable eksw alla edw exw ena mono run
    initial_states = env.reset()
    ql_agent = QLAgent(starting_state=env.encode(initial_states),
                                 state_space=env.observation_space,
                                 action_space=env.action_space,
                                 alpha=args.alpha,
                                 gamma=args.gamma,
                                 qtable = Qtable, exploration_strategy = EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay))
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

    waiting=[]
    episodes= 1
    for episode in range(1,episodes+1):
        rewards = []


        x_values.append(episode)
        env.reset()
        
        done = False 
        infos = [] 
        if args.fixed: 
            while not done:
                _, _, done, _ = env.step({})
        
        
        else:
            while not done:
          
                action = ql_agent.act()
     
                s, r, done, _ = env.step(action)
                
                
                next_state=env.encode(s)
 
                ql_agent.learn(next_state, reward=r)
                
                waiting.append(env.Traffic_Light.waiting_time_reward())
                fuel.append(env.Traffic_Light.fuel_reward())
                cars.append(env.Traffic_Light.passed_cars_reward())
                noise.append(env.Traffic_Light.noiseEmission_reward())
                print(fuel,noise)
                
        data = pd.DataFrame(env.metrics)
        statistic = data.describe()
        df = pd.DataFrame(statistic)
        mean = df.total_wait_time[1:2]
        waiting.append(mean.item())
        print('Mean waiting time after {} episodes : '.format(episode) ,mean.item())

    
        wr.append(np.mean(env.Traffic_Light.time_rew))
        fr.append(np.mean(env.Traffic_Light.fuel_rew))
        cr.append(np.mean(env.Traffic_Light.cars_rew))
        nr.append(np.mean(env.Traffic_Light.noise_rew))

        eprewards.append(df.reward[1:2].item()) 
        stops.append(df.total_stopped[1:2].item())
        waitingTime.append(df.total_wait_time[1:2].item())
        fuelConsumption.append(df.fuel_consumption[1:2].item())
        noiseEmission.append(df.noise_emission[1:2].item())
                

                
    with open('../QTables/single/final.json', 'w') as convert_file:
        convert_file.write(ujson.dumps(ql_agent.q_table))

            
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
    
    
    print('Mean : ', np.mean(waiting), statistics.mean(waiting))
    print('Standard deviation :' , statistics.stdev(waiting))  
    env.save_csv(out_csv)
    env.close()
    
    
