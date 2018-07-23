import gym
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    def forward(self, x):
        return self.net(x)

def generate_batch(env,batch_size, t_max=1000):
    
    activation = nn.Softmax(dim=1)
    batch_actions,batch_states, batch_rewards = [],[],[]
    
    for b in range(batch_size):
        states,actions = [],[]
        total_reward = 0
        s = env.reset()
        for t in range(t_max):
            
            s_v = torch.FloatTensor([s])
            act_probs_v = activation(net(s_v))
            act_probs = act_probs_v.data.numpy()[0]
            a = np.random.choice(len(act_probs), p=act_probs)

            new_s, r, done, info = env.step(a)

            #record sessions like you did before
            states.append(s)
            actions.append(a)
            total_reward += r

            s = new_s
            if done:
                batch_actions.append(actions)
                batch_states.append(states)
                batch_rewards.append(total_reward)
                break
                
    return batch_states, batch_actions, batch_rewards

def filter_batch(states_batch,actions_batch,rewards_batch,percentile=50):
    
    reward_threshold = np.percentile(rewards_batch, percentile)
    
    elite_states = []
    elite_actions = []
    
    
    for i in range(len(rewards_batch)):
        if rewards_batch[i] > reward_threshold:
            for j in range(len(states_batch[i])):
                elite_states.append(states_batch[i][j])
                elite_actions.append(actions_batch[i][j])
    
    return elite_states,elite_actions
    

if __name__ == "__main__":
	batch_size = 100
	session_size = 100
	percentile = 80
	hidden_size = 200
	completion_score = 50

	env = gym.make("LunarLander-v2")
	n_states = env.observation_space.shape[0]
	n_actions = env.action_space.n

	#neural network
	net = Net(n_states, hidden_size, n_actions)
	#loss function
	objective = nn.CrossEntropyLoss()
	#optimisation function
	optimizer = optim.Adam(params=net.parameters(), lr=0.01)

	for i in range(session_size):
		#generate new sessions
		batch_states,batch_actions,batch_rewards = generate_batch(env, batch_size, t_max=5000)

		elite_states, elite_actions = filter_batch(batch_states,batch_actions,batch_rewards,percentile)

		optimizer.zero_grad()
		tensor_states = torch.FloatTensor(elite_states)
		tensor_actions = torch.LongTensor(elite_actions)
		action_scores_v = net(tensor_states)
		loss_v = objective(action_scores_v, tensor_actions)
		loss_v.backward()
		optimizer.step()

		#show results
		mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, percentile)
		print("%d: loss=%.3f, reward_mean=%.1f, reward_threshold=%.1f" % (
		        i, loss_v.item(), mean_reward, threshold))

		#check if 
		if np.mean(batch_rewards)> completion_score:
		    print("Environment has been successfullly completed!")

	#record sessions (use on local machine)
	# import gym.wrappers
	# env = gym.wrappers.Monitor(gym.make("LunarLander-v2"), directory="videos", force=True)
	# generate_batch(env, 1, t_max=5000)
	# env.close()

	#floyd hub rendering fix
	from support import save_frames_as_gif

	frames = []
	env = gym.make("LunarLander-v2")
	activation = nn.Softmax(dim=1)

	for t in range(t_max):
	        
	        s_v = torch.FloatTensor([s])
	        act_probs_v = activation(net(s_v))
	        act_probs = act_probs_v.data.numpy()[0]
	        a = np.random.choice(len(act_probs), p=act_probs)
	        
	        new_s, r, done, info = env.step(a)
	        frames.append(new_s)
	        
	        s = new_s
	        if done: break

	# Save the run
	save_frames_as_gif(frames, filename='lunarlander-cem.gif')
	