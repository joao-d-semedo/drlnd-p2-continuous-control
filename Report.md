[//]: # (Image References)

[image1]: Figure_1.png "Performance"
[image2]: Figure_2.png "Performance Final"

# Project 1: Navigation

## Learning Algorithm

The training procedure leverages the DDPG architecture. At a high level, the training loop involves:

1. Environment interaction: the agent interacts with the environment using the current actor network (with an additive exploration noise process) and stores the outcome of the interactions in memory (the experience replay buffer).

2. Network update: Every few interactions (controlled by the hyperparameter `UPDATE_EVERY` in `ddpg_agent.py`) the agent draws experiences from memory (uniformly at random) and trains the actor and critic networks using the corresponding target networks.

For full details, see [original publication](https://arxiv.org/abs/1509.02971).

### Architecture

The function approximator network employed in this solution is a simple multi-layer perceptron (MLP). The implementation provides options for the activation types, as well as the number and size of the hidden layers. The default parameters for the actor network are:
- `sizes = [state_space_size, 400, 300, action_space_size]`, which defines a MLP with two hidden layers with 400 and 300 units respectively (number of input and output units are matched to the environment, 33 and 4 in this case). The default parameters for the critic network are:
- `sizes = [action_space_size+state_space_size, 400, 300, 1]`, which defines a MLP with two hidden layers with 400 and 300 units respectively (number of input units is matched to the environment, 37 in this case).
- `activation = torch.nn.ReLU()`, which sets the activation functions between layers as rectified linear units (ReLU).
- `output_activation` was set to `nn.Tanh` for the actor network and left as identity (no output transformation) for the critic network.

### Agent

The agent object contains both actor and critic MLP networks (`agent.actor_local` and `agent.critic_local`), the target actor and critic MLP networks (`agent.actor_target` and `agent.critic_target`), and the memory (the experience replay buffer, `agent.memory`). The agent's hyperparameters control the two steps of the training procedure outlined above.
- `BUFFER_SIZE = 1e6` controls the size of the experience replay buffer
- `BATCH_SIZE = 64` controls the number of experiences drawn from the replay buffer whenever a DQN update is triggered
- `GAMMA = .99` is the discont factor used when computing returns (between 0 and 1)
- `TAU = 1e-3` controls the rate at which the target DQN is updated (between 0 and 1: 0 means the target DQN is never updated, 1 means the target network is always equal to the main network)
- `ACTOR_LR = 1e-4` is the learning rate used when updating the weights of the local actor network
- `CRITIC_LR = 1e-3` is the learning rate used when updating the weights of the local critic network
- `UPDATE_EVERY = 1` controls how frequently to trigger a network update step (1 means the network weights are updated after every interaction with the environment, 4 means the weights are updated every 4 interactions, etc)

### Training

- `n_episodes = 1000` controls the total number of episodes used for training
- `max_t = 1000` controls the maximum duration of an episode
- `std_start=0.2`, `std_end=0.01`, and `std_decay=.995` control the standard deviation of the exploration noise process. Specifically, the standard deviation is set to `std_start` at the beggining of training and decays by `std_decay` every episode until it reaches `std_end`, after which it stays constant.

The exploration noise process used here is an Ornstein-Uhlenbeck Process and the implementation was adapted from [this repository](https://github.com/ShangtongZhang/DeepRL).

## Learning Performance

Bellow is the result of one training run, showing the average total rewards over 100 episodes over the course of 1000 episodes. In this instance, the agent needed 284 episodes to solve the environment.

![Learning Performance][image1]

While the agent achieves great performance, it was clear that after 400 episodes training became unstable. In order to try to stability training and improve performance further, a learning rate scheduler was used that decreases the learning rate of both actor and critic networks by one order of magnitude every 400 episodes. This change led to more stable training.

![Learning Performance][image2]

The two agent weights were saved in `checkpoint_284.pth` and `checkpoint_final.pth`, respectively.

## Ideas for Future Work

While the approach employed here was fairly successful, it represents a fairly straightforward implementation that can be improved in a number of ways. While these improvements might have a limited impact in solving this simple environment, they might prove much more important when tackling more complex environments (for example, learning using pixel-level representation of states):
- Explore parallel processing environment with 20 agents acting simultaneously
- Prioritized Experience Replay
- Further hyper-parameter tuning

