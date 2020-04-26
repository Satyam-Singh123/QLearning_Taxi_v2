
## Smart_Cab

### Importing Libraries


```python
import gym
# Importing libraries
import numpy as np
import random
import math
from collections import deque
import collections
import pickle
#for text processing
import spacy
import re
import pandas as pd
env = gym.make("Taxi-v2").env
env.render()
```

    +---------+
    |R: |[43m [0m: :[34;1mG[0m|
    | : : : : |
    | : : : : |
    | | : | : |
    |[35mY[0m| : |B: |
    +---------+
    



```python
env.reset() # reset environment to a new, random state
env.render()
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
```

    +---------+
    |[34;1mR[0m: | : :G|
    | : : : :[43m [0m|
    | : : : : |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
    
    Action Space Discrete(6)
    State Space Discrete(500)


#### There are 4 locations (labeled by different letters), and our job is to pick up the passenger at one location and drop him off at another. We receive +20 points for a successful drop-off and lose 1 point for every time-step it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions."

### Mapping City


```python
def create_loc_dict(city_df):
    loc_dict = {}
    ## Create dictionary example, loc_dict['dwarka sector 23] = 0
    for place , maps in zip(city_df.location, city_df.mapping):
        loc_dict[place] = maps
    return loc_dict
```

### Fetching Origing, Destination, and Time of Pickup from the sms data 


```python
def fetch_pickup_drop(text):
    
    s = text.lower()
    if ('from' in s):
        pick_pat = re.compile(r'from (airport|hauz khaas|dwarka sector 23|dwarka sector 21)')
        origin = re.findall(pick_pat , s)[0] #Pick-up location of Passenger
        time_pat = re.compile('at (\w+ (pm|am))')
        time_of_pickup = re.findall(time_pat,s)[0][0] #Time for Pick up
        dest_pat = re.compile(r'(airport|hauz khaas|dwarka sector 23|dwarka sector 21)')
        destination = [word for word in re.findall(dest_pat,s) if word!= origin][0]#Drop-off location of Passenger
        
    else:
        dest_pat = re.compile(r'(to|for) (airport|hauz khaas|dwarka sector 23|dwarka sector 21)')
        destination = re.findall(dest_pat , s)[0][-1]
        time_pat = re.compile('at (\w+ (pm|am))')
        time_of_pickup = re.findall(time_pat,s)[0][0]
        pick_pat = re.compile(r'(airport|hauz khaas|dwarka sector 23|dwarka sector 21)')
        origin = [word for word in re.findall(pick_pat,s) if word != destination][0]
        
    return [origin, destination, time_of_pickup.upper()]
```

### Checking If Fetched Locations Value Matches With Original Data.


```python
def check_pick_up_drop_correction(picks, drops, index, orig_df):
    original_origin = orig_df.iloc[index]['origin']
    original_destination = orig_df.iloc[index]['dest']
    if original_origin == picks and original_destination == drops:
        return True
    else:
        return False
```

## Summing up the Q-Learning Process
Breaking it down into steps, we get

Initialize the Q-table by all zeros.

Start exploring actions: 

For each state, select any one among all possible actions for the current state (S).

Travel to the next state (S') as a result of that action (a).

For all possible actions from the state (S') select the one with the highest Q-value.

Update Q-table values using the equation.

Set the next state as the current state.

If goal state is reached, then end and repeat the process.



```python
def decode(pick_up):
    if pick ==  0:
        taxi_row, taxi_column  = 0,0
    elif pick == 1:
        taxi_row, taxi_column  = 0,3
    elif pick == 2:
        taxi_row, taxi_column  = 3,0
    else: 
        taxi_row, taxi_column  = 3,3
        
    return taxi_row, taxi_column
```

### Generating Q table


```python
def generate_q_table(q_table):
    """Training the agent"""

    # Hyperparameters
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1

    total_penalties = 0
    total_epochs = 0
    ##Write your code here
    for i in range(1, 100001):
        state = env.reset()
        epochs, penalties, reward, = 0, 0, 0
    
        done = False
        while not done:
            
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Exploring action space
            else:
                action = np.argmax(q_table[state]) # Exploiting learned values
            
            next_state, reward, done, info = env.step(action) 
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma*next_max)
            q_table[state, action] = new_value
            
            if i == 10000:
                env.render()
            
            if done:
                break
            
            elif True:
                penalties += 1
   
            state = next_state
            epochs += 1
        total_epochs += epochs
        total_penalties += penalties
           

    print("q_table Created.\n")
    #print('Total Penalties: ', total_penalties)
    np.save("./q_table.npy", q_table)
```

###  The Q-table


```python
q_table = np.zeros([env.observation_space.n, env.action_space.n])
generate_q_table(q_table)
q_table
```

    +---------+
    |R: | : :[42mG[0m|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (Pickup)
    +---------+
    |R: | : :[34;1m[43mG[0m[0m|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (Dropoff)
    +---------+
    |R: | : :[42mG[0m|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (Pickup)
    +---------+
    |R: | : :G|
    | : : : :[42m_[0m|
    | : : : : |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (South)
    +---------+
    |R: | : :G|
    | : : :[42m_[0m: |
    | : : : : |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (West)
    +---------+
    |R: | :[42m_[0m:G|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (North)
    +---------+
    |R: | : :G|
    | : : :[42m_[0m: |
    | : : : : |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (South)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : :[42m_[0m: |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (South)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : : : |
    | | : |[42m_[0m: |
    |Y| : |[35mB[0m: |
    +---------+
      (South)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |[35m[42mB[0m[0m: |
    +---------+
      (South)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |[35m[34;1m[43mB[0m[0m[0m: |
    +---------+
      (Dropoff)
    q_table Created.
    





    array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ],
           [ 1.62261351,  2.91400926,  1.62261251,  2.91401597,  4.348907  ,
            -6.08598366],
           [ 4.34890603,  5.94322987,  4.34890671,  5.94322971,  7.7147    ,
            -3.05677   ],
           ...,
           [-1.12875179,  9.68299979, -0.25110817,  1.28622905, -1.73512079,
            -1.73215214],
           [-2.52095463, -1.10499268, -2.11902451,  2.91401439, -5.35436052,
            -5.28994038],
           [ 9.17381968,  7.1941436 ,  7.85681049, 17.        ,  1.55785106,
             0.197     ]])



### Implementation


```python
f = open("./sms.txt", "r")
num_of_lines = 1000

city = pd.read_csv("./city.csv")
loc_dict = create_loc_dict(city)
print(loc_dict)

org_df = pd.read_csv("./org_df.csv")
total_epochs, total_penalties, total_reward, wrong_predictions, right_predictions = 0, 0, 0, 0, 0
count = 0

line_num = 0
for line in f:
    #print(line)
    
    '''For Fetching Variables from Sms'''
    origin, destination, time_of_pickup = fetch_pickup_drop(line)
    #print('Origin: ',origin, 'destination: ',destination, 'time_of_pick_up: ', time_of_pickup)
    
    '''true_bool = True for Correct Prediction Else False'''
    true_bool = check_pick_up_drop_correction(origin, destination, line_num, org_df) 
    line_num += 1
    if not true_bool:
        wrong_predictions += 1
        reward = -10
        total_reward += reward 
        total_penalties += 1
    else:
        right_predictions += 1
        
    '''Setting Random State'''
    rand_state = env.reset()
    taxi_row, taxi_column, pick, drop = env.decode(rand_state)
    #print('Random State Generated:\n', taxi_row, taxi_column, pick, drop)
    
    '''Setting Env Parameter Based on Fetched PickUp and Drop and Return'''
    pick = loc_dict[origin]
    drop = loc_dict[destination]
    taxi_row, taxi_column = decode(pick)
    
    state = env.encode(taxi_row, taxi_column, int(pick), int(drop))
    env.s = state
    #taxi_row, taxi_column, pick, drop = env.decode(state)
    #print(' State Generated:\n', taxi_row, taxi_column, pick, drop)
    
    '''Loading trained q_table for evaluation'''
    q_table = np.load("./q_table.npy")
    
    """Evaluate agent's performance after Q-learning"""

    epochs, penalties, Reward = 0, 0, 0
    done = False
    
    while not done:
        
        action = np.argmax(q_table[state])
        new_state, reward, done, info = env.step(action)
        
        if done:
            pass
       
        else:
            penalties += 1
        
        total_reward += reward
        state = new_state
        epochs += 1
        
    
    total_penalties += penalties
    total_epochs += epochs
    
    
print(f"Results after {num_of_lines} episodes:")
print(f"Average timesteps per episode: {total_epochs / num_of_lines}")
print(f"Average penalties per episode: {total_penalties / num_of_lines}")
print(f"Total number of wrong predictions is: {wrong_predictions} and right predictions is :{right_predictions}", )
print("Total Reward is", total_reward)
```

    {'dwarka sector 23': 0, 'dwarka sector 21': 1, 'hauz khaas': 2, 'airport': 3}
    Results after 1000 episodes:
    Average timesteps per episode: 8.925
    Average penalties per episode: 7.925
    Total number of wrong predictions is: 0 and right predictions is :1000
    Total Reward is 12075

