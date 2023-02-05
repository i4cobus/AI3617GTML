import sys
from os import path
from olympics.ruleagent import RuleAgent
from agents.rl.submission import agent as rl_agent
father_path = path.dirname(__file__)
sys.path.append(str(father_path))
from olympics.generator import create_scenario
import argparse
from olympics.scenario.running import Running
import random
import numpy as np

actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}           #dicretise action space

RENDER = True  #show the game or not

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    agent1 = RuleAgent()   
    agent2 = RuleAgent() 
    map_index_seq = list(range(1, 12))
    for i in range(200000):
        ind = map_index_seq.pop(0)
        Gamemap = create_scenario("map"+str(ind))
        map_index_seq.append(ind)
        agent1.reset()
        agent2.reset()
        rnd_seed = random.randint(0, 1000)
        game = Running(Gamemap, seed = rnd_seed)
        game.map_num =ind
        obs = game.reset()
        if RENDER: game.render()
        done = False
        step = 0
        if RENDER:
            game.render('MAP {}'.format(ind))
        while not done:
            step += 1
            action1 = agent1.act(obs[0])
            #these 2 lines are for baseline agent
            raw_action2 = rl_agent.choose_action(obs[1].flatten())
            action2 = actions_map[raw_action2]
            #this line is for random agent
            #action2 = [random.randint(-100,200),random.randint(-30,30)]
            #this line is for not moving agent
            #action2 = [0,0]                                         
            #update env
            obs, reward, done, _ = game.step([action1, action2])
            if RENDER:
                game.render()

        #save the game log of not moving v.s. rule_baesd
        #with open(file = 'com_log.txt', mode = 'a') as f:                      #This is for 1 not moving agent and 1 rule-based agent
            #f.write(str(ind) + '\t' +str(step) + '\n' + str(reward) + '\n')

        #save the game log of random v.s. rule_based
        #with open(file = 'com_log_1.txt', mode = 'a') as f:                       #This is for 1 random agent and 1 rule-based agent
            #f.write(str(ind) + '\t' +str(step) + '\n' + str(reward) + '\n') 

        #save the game log of baseline v.s. rule_based
        #with open(file = 'com_log_2.txt', mode = 'a') as f:                      #This is for 1 not moving agent and 1 rule-based agent
            #f.write(str(ind) + '\t' +str(step) + '\n' + str(reward) + '\n')