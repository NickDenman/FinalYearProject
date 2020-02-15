import math
import random

import numpy as np
import environment.PCB_Board_2D_Static as pcb
import matplotlib.pyplot as plt
import json

num_agents = 1
env = pcb.PCBBoard(num_agents=num_agents)
Q = {}
epsilon_start = 0.5
epsilon_end = 0.05
alpha = 0.3
discount_factor = 0.9
games_won = []
k_played = []
interaction_count = 0
total_interaction_count = 75000
learning_interactions = 25
samples = 200


def setup_learning():
    reset_env()


def add_states_to_q(s):
    for a in range(1, 9):
        if (s, a) not in Q:
            Q[(s, a)] = 0


def update_q_table(s_a, s_prime, reward):
    Q[s_a] = Q[s_a] + alpha * (reward + discount_factor * get_max_reward(s_prime) - Q[s_a])


# Returns the best reward expected from the current game state.
def get_max_reward(s):
    best = -math.inf
    for a in range(1, 9):
        if Q[(s, a)] > best:
            best = Q[(s, a)]
    return best


def get_random_action(s):
    return random.randrange(1, 9)


def get_best_action(s):
    best = -math.inf

    # pick random move in accordance with epsilon to maintain exploration
    if random.random() < (epsilon_start - ((epsilon_start - epsilon_end) * (interaction_count / total_interaction_count))):
        return random.randrange(1, 9)

    choices = []
    for a in range(1, 9):
        qval = Q[(s, a)]
        if qval > best:
            choices = [a]
            best = qval
        elif qval == best:
            choices.append(a)

    return random.choice(choices)


def learn(k, n, m):
    global interaction_count

    count = 0
    while count < k:
        print("iteration: " + str(count) + " [" + str(len(Q)) + "]")
        reset_env()
        interactions = 0
        while interactions < n:
            move()
            interactions += 1

        num_games = 0
        total = 0
        while num_games < m:
            total += route_board()
            num_games += 1

        games_won.append(total / m)
        count += 1
        k_played.append(count * n)
        interaction_count += 1


def move():
    actions = {}
    observations = {}
    state_action = {}
    for ag in range(num_agents):
        s = get_observation(ag)
        if s is None:
            continue

        a = get_best_action(s)
        actions[ag] = a
        observations[ag] = s
        state_action[ag] = (s, a)

    reward = env.joint_act(actions)

    if env.is_game_over():
        for _, s_a in state_action.items():
            Q[s_a] = reward

        reset_env()

    else:
        for ag in range(num_agents):
            s_prime = get_observation(ag)
            add_states_to_q(s_prime)
            update_q_table(state_action.get(ag), s_prime, reward)


def get_observation(agent_id):
    grid, location, destination = env.observe(agent_id)
    if grid is None and location is None and destination is None:
        return None

    return np.array_str(grid) + str(location) + str(destination)


def route_board():
    reset_env()
    interactions = 0
    total_reward = 0

    while not env.is_game_over() and interactions < 50:
        actions = {}
        for ag in range(num_agents):
            s = get_observation(ag)
            if (s, 1) in Q:
                a = get_best_action(s)
            else:
                a = get_random_action(s)

            actions[ag] = a

        total_reward += env.joint_act(actions)
        interactions += 1

    return total_reward


def reset_env():
    env.reset_env()
    s = get_observation(0)
    add_states_to_q(s)


setup_learning()
learn(total_interaction_count, learning_interactions, samples)
plt.plot(k_played, games_won)
plt.show()

plt.plot(k_played, games_won)
plt.savefig("plots/plot.png")

json_q = json.dumps(str(Q))
f = open("q.json", "w")
f.write(json_q)
f.close()
