
# coding: utf-8

# In[1]:


import numpy as np
import gym

env = gym.make("FrozenLake8x8-v0")
env.reset()
epsilon = 1
alpha = .005
gamma = .94

Q = np.zeros([env.observation_space.n, env.action_space.n]) # inicializa Q com zeros
for episode in range(1, 1000001):
    done = False
    obs_old = env.reset()  
    while not done: # done é true quando se morre ou quando se pega o frisbie
        if np.random.rand(1) < epsilon:
            action = env.action_space.sample() # Pegamos uma ação aleatória de acordo com a probabilidade epsilon
            epsilon = max(.1, epsilon - 2*1e-7) # Diminuir o epsilon com o tempo até o mínimo de 10%
        else:
            action = np.argmax(Q[obs_old]) # caso contrário pegamos a ação ideal já aprendida
        obs_new, reward, done, _ = env.step(action) # Usamos essa ação para gerar um novo estado
        Q[obs_old,action] += alpha * (reward + gamma * np.max(Q[obs_new]) - Q[obs_old,action]) # Bellman
        obs_old = obs_new
    
    if episode % 10000 == 0: # Mostrar o andamento do aprendizado
        rew_total = 0
        for i in range(100):
            obs = env.reset()
            done = False
            while not done: 
                action = np.argmax(Q[obs])
                obs, reward, done, info = env.step(action)
                rew_total += reward
        rew = rew_total/100
        print("Episode {} epsilon: {}".format(episode, epsilon))
        print('Episode {} reward: {}'.format(episode,rew))
        print()
        
        if rew >= 0.83:
            print("FIM!!")
            break


# In[6]:


rew_total = 0.
obs = env.reset()
done = False
while not done: 
    action = np.argmax(Q[obs])
    obs, rew, done, info = env.step(action)
    rew_total += rew
    env.render()

print("Reward:", rew_total)  

