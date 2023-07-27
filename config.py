import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Tuple, Dict

class DreamerConfig():
    '''default HPs that are known to work for MinAtar envs '''
    
    ########## DreamerV2 ##########
    
    #env desc
    env: str                                           
    obs_shape: Tuple                                            
    action_size: int  
    pixel: bool = True                                             # if true: use convolutional encoder and decoder; else: use DenseModel encoder and decoder
    
    #buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    #training desc
    train_steps: int = int(5e6)
    train_every: int = 50                                          # number of frames to skip while training; reduce this to potentially improve sample requirements
    collect_intervals: int = 5                                     # number of batches to be sampled from buffer, at every "train-every" iteration
    batch_size: int = 50 
    seq_len: int = 50                                              # length of trajectory sequence to be sampled from buffer
    eval_episode: int = 4       
    eval_render: bool = True
    save_every: int = int(1e5)
    seed_steps: int = 4000                                         # seed steps to collect data
    model_dir: int = 'results'                                     # mayde str
    gif_dir: int = 'results'                                       # maybe str
    
    #latent space desc
    rssm_type: str = 'discrete'                                    # categorical ('discrete') or gaussian ('continuous') random variables for stochastic states
    embedding_size: int = 200                                      # size of embedding vector that is output by observation encoder 
    rssm_node_size: int = 200                                      # size of hidden layers of temporal posteriors and priors
    rssm_info: Dict = field(default_factory=lambda:{
        'deter_size':200,                                          # deter_size: size of deterministic part of recurrent state.
        'stoch_size':20,                                           # stoch_size: size of stochastic part of recurrent state.
        'class_size':20,                                           # class_size: number of classes for each categorical random variable
        'category_size':20,                                        # category_size: number of categorical random variables.
        'min_std':0.1})
                     
    #objective desc
    grad_clip: float = 100.0
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 10                                              # horizon for imagination in future latent state space
    lr: Dict = field(default_factory=lambda:{
        'model':2e-4, 
        'actor':4e-5, 
        'critic':1e-4})
    loss_scale: Dict = field(default_factory=lambda:{
        'kl':0.1, 
        'reward':1.0, 
        'discount':5.0})
    kl: Dict = field(default_factory=lambda:{
        'use_kl_balance':True, 
        'kl_balance_scale':0.8,                                    # kl_balance_scale: scale for kl balancing (=alpha in _kl_loss)
        'use_free_nats':False,                                     
        'free_nats':0.0})                                          # necessary in order to clip values of terms of the kl loss
    use_slow_target: float = True                                  # delayed TargetValueModel update
    slow_target_update: int = 100                                  # update TargetValueModel every 'slow_target_update' steps
    slow_target_fraction: float = 1.04                             # determines the proportion of TargetValueModel parameters changes (trainer.py)
                                                                   # def update_target(self):
                                                                   #     mix = self.config.slow_target_fraction if self.config.use_slow_target else 1
                                                                   #     for param, target_param in zip(self.ValueModel.parameters(), self.TargetValueModel.parameters()):
                                                                   #     target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

    #actor critic
    actor: Dict = field(default_factory=lambda:{
        'layers':3, 
        'node_size':100, 
        'dist':'one_hot', 
        'min_std':1e-4, 
        'init_std':5, 
        'mean_scale':5, 
        'activation':nn.ELU})
    critic: Dict = field(default_factory=lambda:{
        'layers':3, 
        'node_size':100, 
        'dist': 'normal', 
        'activation':nn.ELU})
    expl: Dict = field(default_factory=lambda:{
        'train_noise':0.4, 
        'eval_noise':0.0, 
        'expl_min':0.05, 
        'expl_decay':7000.0, 
        'expl_type':'epsilon_greedy'})
    actor_grad: str ='reinforce'                                   # if self.config.actor_grad == 'reinforce':
                                                                   #     advantage = (lambda_returns-imag_value[:-1]).detach()
                                                                   # elif self.config.actor_grad == 'dynamics':
                                                                   #     objective = lambda_returns
    actor_entropy_scale: float = 1e-3                              # scale for policy entropy regularization in latent state space

    #learnt world-models desc
    obs_encoder: Dict = field(default_factory=lambda:{
        'layers':3, 
        'node_size':100,                                           # node_size - size of hidden layers in DenseModel
        'dist': None, 
        'activation':nn.ELU, 
        'kernel':3, 
        'depth':16})
    obs_decoder: Dict = field(default_factory=lambda:{
        'layers':3, 
        'node_size':100, 
        'dist':'normal', 
        'activation':nn.ELU, 
        'kernel':3, 
        'depth':16})
    reward: Dict = field(default_factory=lambda:{
        'layers':3, 
        'node_size':100, 
        'dist':'normal', 
        'activation':nn.ELU})
    discount: Dict = field(default_factory=lambda:{
        'layers':3, 
        'node_size':100, 
        'dist':'binary', 
        'activation':nn.ELU, 
        'use':True})
    
    # other parameters
    _id: str = '0'                                                 # experiment id
    seed: int = 123                                                # random seed
    device: str = 'cuda'                                           # CUDA or CPU


class MAXConfig():
    
    ########## MAX ##########
    
    # MAX agent policies
    max_exploration: bool = False                                
    random_exploration: bool = False
    
    # MAX env config
    env_name: str                                                  # environment out of the defined magellan environments with `Magellan` prefix
    n_eval_episodes: int = 3                                       # number of episodes evaluated for each task
    env_noise_stdev: int = 0                                       # standard deviation of noise added to state
    n_warm_up_steps: int = 256                                     # number of steps to populate the initial buffer, actions selected randomly
    n_exploration_steps: int = 20000                               # total number of steps (including warm up) of exploration
    eval_freq: int = 2000                                          # interval in steps for evaluating models on tasks in the environment
    data_buffer_size: int = n_exploration_steps + 1                # size of the data buffer (FIFO queue - "external" buffer)
    d_state: int = 10                                              # env.observation_space.shape[0] - dimensionality of state
    d_action: int = 3                                              # env.action_space.shape[0] - dimensionality of action (ours: env.action_space.n)
    
    # MAX infra config
    verbosity: int = 0                                             # level of logging/printing on screen
    render: bool = False                                           # render the environment visually (warning: could open too many windows)
    record: bool = False                                           # record videos of episodes (warning: could be slower and use up disk space)
    save_eval_agents: bool = False                                 # save evaluation agent (sac module objects)
    checkpoint_frequency: int = 2000                               # dump buffer with normalizer every checkpoint_frequency steps
    disable_cuda: bool = False                                     # if true: do not use cuda even though its available
    omp_num_threads: int = 1                                       # for high CPU count machines

    # MAX model arch config
    ensemble_size: int = 32                                        # number of models in the bootstrap ensemble
    n_hidden: int = 512                                            # number of hidden units in each hidden layer (hidden layer size)
    n_layers: int = 4                                              # number of hidden layers in the model (at least 2)
    non_linearity: str = 'swish'                                   # activation function: can be 'leaky_relu' or 'swish'
    
    # MAX model training config
    exploring_model_epochs: int = 50                               # number of training epochs in each training phase during exploration
    evaluation_model_epochs: int = 200                             # number of training epochs for evaluating the tasks
    batch_size: int = 256                                          # batch size for training models
    learning_rate: float = 1e-3                                    # learning rate for training models
    normalize_data: bool = True                                    # normalize states, actions, next states to zero mean and unit variance
    weight_decay: float = 0                                        # L2 weight decay on model parameters (good: 1e-5, default: 0)
    training_noise_stdev: float = 0                                # standard deviation of training noise applied on states, actions, next states
    grad_clip: float = 5                                           # gradient clipping to train model
    
    # policy config (common to both exploration and exploitation)
    policy_actors: int = 128                                       # number of parallel actors in imagination MDP
    policy_warm_up_episodes: int = 3                               # number of episodes with random actions before SAC on-policy data is collected (as a part of init)

    policy_replay_size: int = int(1e7)                             # SAC replay size
    policy_batch_size: int = 4096                                  # SAC training batch size
    policy_reactive_updates: int = 100                             # number of SAC off-policy updates of `batch_size`
    policy_active_updates: int = 1                                 # number of SAC on-policy updates per step in the imagination/environment

    policy_n_hidden: int = 256                                     # policy hidden size (2 layers)
    policy_lr: float = 1e-3                                        # SAC learning rate
    policy_gamma: float = 0.99                                     # discount factor for SAC
    policy_tau: float = 0.005                                      # soft target network update mixing factor

    buffer_reuse: bool = True                                      # transfer the main exploration buffer as off-policy samples to SAC
    use_best_policy: bool = False                                  # execute the best policy or the last one

    # exploration coefs
    policy_explore_horizon: int = 50                               # length of sampled trajectories (planning horizon)
    policy_explore_episodes: int = 50                              # number of iterations of SAC before each episode
    policy_explore_alpha: float = 0.02                             # entropy scaling factor in SAC for exploration (utility maximisation)

    # exploitation coefs
    policy_exploit_horizon: int = 100                              # length of sampled trajectories (planning horizon)
    policy_exploit_episodes: int = 250                             # number of iterations of SAC before each episode
    policy_exploit_alpha: float = 0.4                              # entropy scaling factor in SAC for exploitation (task return maximisation)
    
    #exploration config
    exploration_mode: str = 'active'                               # active or reactive
    model_train_freq: int = 25                                     # interval in steps for training models. if `np.inf`, models are trained after every episode
    utility_measure: str = 'renyi_div'                             # measure for calculating exploration utility of a particular (state, action). 'cp_stdev', 'renyi_div'
    renyi_decay: float = 0.1                                       # decay to be used in calculating Renyi entropy
    utility_action_norm_penalty: float = 0                         # regularize to actions even when exploring
    action_noise_stdev: float = 0                                  # noise added to actions
    
