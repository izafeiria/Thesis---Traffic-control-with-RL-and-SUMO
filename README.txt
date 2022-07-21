# Optimization of traffic lights timing using Reinforcement learning to minimize car queueing time


## Abstract

Traffic congestion is increasing worldwide, and the problem needs to be addressed.
In a dynamically changing and interconnected transport environment, current traffic
regulations are not adaptable. An intelligent transport system is needed to improve the
efficiency of the road network of smart cities.

The present Diploma Thesis proposes a system for calculating the timing of traffic
lights in order to minimize the waiting time of vehicles. Each traffic light at an intersection
is trained to learn to change its phase according to traffic. The proposed road system has
a flexible structure that is modified by adding more intersections to the original structure
of the simple intersection.

Q-learning is an RL algorithm used to select the next optimal signal action in a given
state. It works by sequentially improving the rewards for the state-action pairs, which
are stored in a Q-table as traffic light information. The tool SUMO was used to simulate the road networks. 
The models were trained and studied in the environments of road networks with N intersections, where N = 1,2,4,6, 
and the traffic lights of each intersection were trained to reduce traffic. The results of the training are compared with the responses of the current traffic management models. 
In addition, Q-tables of simple structures (N = 1,2) are applied to the most complex networks to assess the correspondence of systems with the experience of simple structures.

According to the results of the training of the models and the experiments, all models responded efficiently to a variety of traffic situations, 
although the training time increases with complexity. An optimal model requires more training time than a simply good model, 
so there is a trade-off between training time and optimal response that every researcher should consider.

Zafeiria Iatropoulou

Electrical and Computer Engineering School,

Aristotle University of Thessaloniki, Greece

June 2022


## Folders Explanation
### Networks
Includes Network files and Route files of all implementations of SUMO simulations. There are route files with different traffic for testing purposes of the created models.
The ranodomTrips.py file is responsive for the creation of route files. Read [SUMO](https://sumo.dlr.de/docs/) documentation for more.
### Parameters
Extraction of optimal parameters for the 2 main problems: single agent & multi-agent systems. The parameters of double simulation used for all N-agents systems. Brute force method used for optimization of parameters.
### RewardFunction
Display correlation matrixes of characteristics during simulation. Created in order to choose the uncorrelated characteristics which provide valuable information. These configured the reward function of the Q Learning algorithm.
### src 
* sumo_env_*.py: sumo environments implementations for each system. Main difference in the observation of agents.
* QL_algorithm.py: construct the QL algorithm and refresh items on the Q table.
* tlofficial.py: class which represents the agent which is the intersection of the networks.
* single_intersection_ql_train.py & multi_intersection_ql_train.py: python implementation for running single agent or multi-agent systems, accordingly. Depending on the providing of an existing Q Table or not, it offers both testing and training of a model. 
* Double_Intersection_1Agent file: includes sumo_env and traffic_light class for training a double intersection with 1 agent.

### TrainingResults
Contains the training results for each system. Diagrams, the standard deviation of waiting time, and statistics of the model are included for each training system.

### Qtables
Contains the Q tables of the training systems (N= 1,2,4,6) which are JSON files. Each system has one Q table for each intersection, which is the agent of the algorithm. Q table constitutes the experience and learning of the agent.

## Requirements
* Python 
* Simulation of Urban MObility (SUMO)

#### Install SUMO latest version:
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc 
```
Don't forget to set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

## Usage
Check single_intersection_ql_train.py & multi_intersection_ql_train.py on how to instantiate an environment and train your RL agent. Depending on the providing parameters, you can train your own environment. 
Train your single agent system or test the model, providing an existing Q-table:
```bash
python single_intersection_ql_train.py 
```
or train your multi-agent systems or test the model, providing an existing Q-table:
```bash
 python multi_intersection_ql_train.py
```