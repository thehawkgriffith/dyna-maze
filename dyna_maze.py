import numpy as np
import os
import matplotlib.pyplot as plt
from time import sleep


StateMemory = []
ActionMemory = {}

class Environment:

    def __init__(self):
        self.states = [State(i) for i in range(54)]
        self.states[7].accessible = False
        self.states[11].accessible = False
        self.states[16].accessible = False
        self.states[20].accessible = False
        self.states[25].accessible = False
        self.states[29].accessible = False
        self.states[41].accessible = False
        self.states[8].reward = 1
        self.player_pos = 18
        self.done = False
        self.accessible_states = [state.id for state in self.states if state.accessible == True]

    def reset(self):
        self.player_pos = 18
        self.done = False
        return self.player_pos

    def step(self, action):
        if action == 1: # up
            if self.states[self.player_pos].id <= 8:
                return self.player_pos, self.states[self.player_pos].reward, self.done
            self.player_pos_change(self.player_pos, -9)
            self.check_done()
            return self.player_pos, self.states[self.player_pos].reward, self.done

        if action == 0: # left
            if self.states[self.player_pos].id in [0, 9, 18, 27, 36, 45]:
                return self.player_pos, self.states[self.player_pos].reward, self.done
            self.player_pos_change(self.player_pos, -1)
            self.check_done()
            return self.player_pos, self.states[self.player_pos].reward, self.done

        if action == 2: # right
            if self.states[self.player_pos].id in [8, 17, 26, 35, 44, 53]:
                return self.player_pos, self.states[self.player_pos].reward, self.done
            self.player_pos_change(self.player_pos, 1)
            self.check_done()
            return self.player_pos, self.states[self.player_pos].reward, self.done

        if action == 3: # down
            if self.states[self.player_pos].id >= 45:
                return self.player_pos, self.states[self.player_pos].reward, self.done
            self.player_pos_change(self.player_pos, 9)
            self.check_done()
            return self.player_pos, self.states[self.player_pos].reward, self.done

    def check_done(self):
        if self.player_pos == 8:
            self.done = True

    def player_pos_change(self, pos, value):
        if pos + value in self.accessible_states:
            self.player_pos += value

    def render(self):
        row1 = ['-', '-', '-', '-', '-', '-', '-', 'X', 'G']
        row2 = ['-', '-', 'X', '-', '-', '-', '-', 'X', '-']
        row3 = ['S', '-', 'X', '-', '-', '-', '-', 'X', '-']
        row4 = ['-', '-', 'X', '-', '-', '-', '-', '-', '-']
        row5 = ['-', '-', '-', '-', '-', 'X', '-', '-', '-']
        row6 = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
        rows = [row1, row2, row3, row4, row5, row6]
        loc = self.player_pos
        row_num = loc//9
        col_num = loc%9
        rows[row_num][col_num] = 'o'
        print(rows[0])
        print(rows[1])
        print(rows[2])
        print(rows[3])
        print(rows[4])
        print(rows[5])

    def clear(self):
        os.system("clear")


class State:

    def __init__(self, id):
        self.id = id
        self.reward = -0.01
        self.accessible = True


class Agent:

    def __init__(self, env):
        self.env = env
        self.Q = {}
        self.model = {}
        self.env.accessible_states.pop(8)
        for s in self.env.accessible_states:
            self.Q[s] = []
            self.model[s] = []
            for a in range(4):
                self.Q[s] += [np.random.random()]
                self.model[s] += [np.random.random()]

    def train(self, episode_nums, env, alpha, gamma, eval_epochs):
        total_reward = 0
        episode_num = 0
        running_average = []
        while episode_num < episode_nums:
            s = env.player_pos
            a = self.sample_action(s)
            p_s = s
            StateMemory.append(s)
            if s not in ActionMemory:
                ActionMemory[s] = []
            ActionMemory[s] += [a]
            s, r, done = env.step(a)
            env.clear()
            env.render()
            print("Cumulative Reward this episode: %.2f"%total_reward)
            total_reward += r
            self.Q[p_s][a] += alpha * (r + (gamma * np.max(self.Q[s])) - self.Q[p_s][a])
            self.model[p_s][a] = (r, s)
            if done:
                s = env.reset()
                env.clear()
                episode_num += 1
                # print("Attained total reward at {}th episode: {}".format(episode_num, total_reward))
                # sleep(1.5)
                running_average.append(total_reward)
                total_reward = 0
            for n in range(eval_epochs):
                s1 = np.random.choice(StateMemory)
                a1 = np.random.choice(ActionMemory[s1])
                r1, s_p1 = self.model[s1][a1]
                self.Q[s1][a1] += alpha * (r1 + (gamma * np.max(self.Q[s_p1])) - self.Q[s1][a1])
        return running_average

    def sample_action(self, s):
        if np.random.random() < 0.1:
            return np.random.choice([0, 1, 2, 3])
        return np.argmax(self.Q[s])

    def print_policy(self):
        best_actions = {}
        for s in self.env.accessible_states:
            a = np.argmax(self.Q[s])
            if a == 1:
                a = '^'
            if a == 0:
                a = '<'
            if a == 2:
                a = '>'
            if a == 3:
                a = 'v'
            best_actions[s] = a
        self.env.clear()
        print("----------------BEST POLICY----------------")
        row1 = ['-', '-', '-', '-', '-', '-', '-', 'X', 'G']
        row2 = ['-', '-', 'X', '-', '-', '-', '-', 'X', '-']
        row3 = ['S', '-', 'X', '-', '-', '-', '-', 'X', '-']
        row4 = ['-', '-', 'X', '-', '-', '-', '-', '-', '-']
        row5 = ['-', '-', '-', '-', '-', 'X', '-', '-', '-']
        row6 = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
        rows = [row1, row2, row3, row4, row5, row6]
        for s in self.env.accessible_states:
            row_num = s//9
            col_num = s%9
            rows[row_num][col_num] = best_actions[s]
        rows[0][8] = 'G'
        print(rows[0])
        print(rows[1])
        print(rows[2])
        print(rows[3])
        print(rows[4])
        print(rows[5])
        print("-------------------------------------------")



def play_human(env):
    s = env.reset()
    done = False
    total_reward = 0
    env.render()
    while not done:
        a = input("Enter action: ")
        if a == 'w':
            a = 1
        if a == 'a':
            a = 0
        if a == 's':
            a = 3
        if a == 'd':
            a = 2
        s, r, done = env.step(a)
        env.clear()
        env.render()
        total_reward += r
    print("Total reward attained is: ", total_reward)

env = Environment()
agent = Agent(env)
running_average = agent.train(20, env, 0.1, 0.95, 100)
agent.print_policy()
plt.plot(running_average)
plt.title("Running Average")
plt.show()
