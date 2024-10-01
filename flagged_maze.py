
from os import stat
from tabnanny import check
import numpy as np
import copy
import math
import time
import pygame
import matplotlib.pyplot as plt
import networkx as nx
grid = np.array([
              ["A" , "B" , "W" , "W" , "W" , "W" , "W" , "W" , "W" , "W"],

              ["W" , "W" , "W" , "F" , "W" , "B" , "W" , "W" , "W" , "W"],

              ["F" , "W" , "W" , "W" , "W" , "B" , "F" , "W" , "W" , "W"],

              ["B" , "B" , "W" , "B" , "B" , "W" , "B" , "W" , "W" , "W"],

              ["W" , "F" , "B" , "W" , "B" , "F" , "B" , "B" , "B" , "W"],

              ["W" , "W" , "B" , "W" , "B" , "W" , "W" , "W" , "W" , "W"],

              ["W" , "W" , "W" , "W" , "W" , "W" , "W" , "W" , "W" , "W"],

              ["W" , "F" , "W" , "W" , "W" , "W" , "B" , "B" , "B" , "B"],
              
              ["W" , "B" , "B" , "B" , "B" , "B" , "W" , "W" , "W" , "W"],
              
              ["W" , "W" , "W" , "W" , "W" , "W" , "W" , "B" , "F" , "T"]


])
pygame.init()
display_size = (800, 600)
screen = pygame.display.set_mode(display_size)
cell_size = 50
pygame.display.set_caption('Maze')

states = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
actions = ['up', 'down', 'left', 'right']

# Define the Q-table
q_table = {state: {action: 0 for action in actions} for state in states}
previous_qtable = {}
# Define the learning parameters
alpha = 0.5
gamma = 0.5
epsilon = 0.5
#it choses a action for agent
def take_action(state, action):
    i, j = state
    if action == 'up':
        i -= 1
    elif action == 'down':
        i += 1
    elif action == 'left':
        j -= 1
    elif action == 'right':
        j += 1
    return (i, j)
#it checks if our q_table is converged or not
def check_convergence (q_table , previous_qtable , hyper_paramter):
    for state in q_table.keys():
         for movement in q_table[state].keys():
            if (abs(q_table[state][movement] - previous_qtable[state][movement]) > hyper_paramter):
               return False
    return True
# Train the agent

for episode in range(1):
    state = (0, 0)
    running = True
    flags = [(1,3) , (2 , 0) , (2 , 6) , (4 , 5), (4 , 1)  , (7 , 1) , (9 , 8)]
    visited_falgs = set()
    episode_path = []
    while running:
        while (True):
            screen.fill((255, 255, 255))
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    #this secction is for pygame
                    if grid[i, j] == "B":
                        pygame.draw.rect(screen, (0, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
                    if(grid[i , j] == "A"):
                        pygame.draw.rect(screen, (127, 127, 127), (j * cell_size, i * cell_size, cell_size, cell_size))
                    if(grid[i , j] == "T"):
                        pygame.draw.rect(screen, (255, 255, 0) , (j * cell_size, i * cell_size, cell_size, cell_size))
                    if(grid[i ,j] == "F"):
                        pygame.draw.rect(screen,(0, 255, 0)  , (j * cell_size, i * cell_size, cell_size, cell_size))
                    pygame.draw.rect(screen , (0, 0, 255) , (state[1] * cell_size, state[0] * cell_size, cell_size, cell_size))
            pygame.time.wait(1)    
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            if(state == (9,9)):
                running = False
                break
            else:
                if np.random.uniform(0, 1) < epsilon:
                    action = np.random.choice(actions)
                else:
                    action = max(q_table[state], key=q_table[state].get)
                next_state = take_action(state, action)
                if(next_state[0] < 0 or next_state[0] >= grid.shape[0] or next_state[1] < 0 or next_state[1] >= grid.shape[1]):
                    reward = -1
                    q_table[state][action] = reward
                else:    
                    #we gave our agent reward throw this section
                    if (grid[next_state[0]][next_state[1]] == "W" or grid[next_state[0]][next_state[1]] == "A"):
                        reward = -0.01
                    elif(grid [next_state[0]][next_state[1]] == "B"):
                        reward = -1
                    elif(grid[next_state[0]][next_state[1]] == "F" and next_state != state):
                        if(next_state in visited_falgs):
                            reward  = -0.01
                        else :
                            reward = 0.5
                            grid[next_state[0]][next_state[1]] == "W"
                        for items in flags:
                            if items == next_state:
                                visited_falgs.add(items)
                    if(next_state == (9,9)):
                        reward = 10 
                    previous_qtable = copy.deepcopy(q_table)
                    q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * max(q_table[next_state].values()))
                    if(grid [next_state[0]][next_state[1]] != "B"):
                        #before reaching T we check if our table is converged or not
                        if(grid [next_state[0]][next_state[1]] == "T"and check_convergence(q_table , previous_qtable , 0.01) == False and len(visited_falgs) == len(flags) ):
                            continue
                        else:#stop our agent from going throw blocks
                            episode_path.append(action)
                            state = next_state
                
def find_neighbors(neighbor , node):
    if(neighbor == 'up'):
        return (node[0] -1 , node[1])
    if(neighbor == 'down'):
        return (node[0] +1 , node[1])
    if(neighbor == 'left'):
        return (node[0]  , node[1]-1)
    if(neighbor == 'right'):
        return (node[0] , node[1] + 1)
# this section is for plotting the graph
def plot_weighted_graph(graph):
    pos = {(i, j) : ( j , (grid.shape[1]-i)) for i in range(grid.shape[0]) for j in range(grid.shape[1])}
    G = nx.DiGraph()
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            if(weight != -1 and weight != 0):
                G.add_edge( node,find_neighbors(neighbor , tuple(node)) ,  weight=round(weight , 5))
    edges = G.edges()
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw(G, pos ,  with_labels=True ,  connectionstyle='arc3, rad = 0.3')
    nx.draw_networkx_edge_labels(G, pos , edge_labels=labels , font_size= 4 , horizontalalignment='center' , verticalalignment= 'center' , label_pos= 0.6 , clip_on= True)
    plt.show()
# this section is for showing q table
def print_q_table(q_table):
    print("Q-table:")
    for state, actions in q_table.items():
        print(f"State: {state}")
        for action, value in actions.items():
            print(f"  Action: {action}, Value: {value}")
print("episode:")
print(episode_path)
print("")
print("path.length:" , len(episode_path))

print("-----------------------------------------------------------------------------")
print("")
print("")
print_q_table(q_table)
plot_weighted_graph(q_table)
print( "total time:" , time.process_time())

