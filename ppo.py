import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import cv2
import matplotlib.pyplot as plt
import os
from collections import deque
import time
import logging
import gc
import signal
import json
from datetime import datetime
import pickle
import argparse

# https://spinningup.openai.com/en/latest/algorithms/ppo.html - openai ppo docs

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PreprocessingWrapper:
    """Environment wrapper for Atari frame preprocessing"""
    def __init__(self, env, height=84, width=84, frame_stack=4):
        self.env = env
        self.height = height
        self.width = width
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)  # Store last 4 frames
        
    def reset(self):
        obs, info = self.env.reset()
        frame = self._preprocess_frame(obs)
        # Fill frame stack with initial frame
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return self._get_stacked_frames(), info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self._preprocess_frame(obs)
        self.frames.append(frame)  # Add new frame, automatically removes oldest
        return self._get_stacked_frames(), reward, terminated, truncated, info
        
    def _preprocess_frame(self,frame):
        # Convert to grayscale and resize to 84x84
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height))
        return resized.astype(np.float32) / 255.0  # Normalize to [0,1]
        
    def _get_stacked_frames(self):
        #Stack frames along first dimension: (4, 84, 84) tensor
        return np.stack(self.frames, axis=0)
        
    def __getattr__(self, name):
        # Forward all other attributes to the base environment
        return getattr(self.env, name)

class ActorCriticNetwork(nn.Module):
    """Neural network with shared CNN features and separate actor/critic heads"""
    def __init__(self, input_shape, num_actions):
        super(ActorCriticNetwork, self).__init__()
        
        # Convolutional layers for feature extraction from game frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # Extract large features
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Medium-scale features
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # Fine-grained features
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),  # Enhanced representation
            nn.ReLU()
        )
        

        # Calculate size after convolution
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # Shared fully connected layers
        self.shared = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head: outputs action probabilities
        self.actor= nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)  # One output per action
        )
        
        # Critic head: outputs state value estimate
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Single value output
        )
        
        # Initialize weights using orthogonal initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        # Initialize linear and conv layers with orthogonal weights
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
        
    def _get_conv_out_size(self, shape):
        # Calculate output size after convolution layers
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self.conv(dummy_input)
        return int(np.prod(dummy_output.size()))
        
    def forward(self, x):
        # Add batch dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        # Extract features through CNN
        conv_out =self.conv(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        shared_out = self.shared(flattened)
        
        # Get both action logits and value estimate
        action_logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_logits, value

class PPOAgent:
    """PPO agent with clipped surrogate objective and GAE"""
    def __init__(self,input_shape, num_actions, 
                 lr=3e-4, gamma=0.995, gae_lambda=0.98, clip_epsilon=0.15,
                 value_coef=0.25, entropy_coef=0.02, max_grad_norm=0.5):
        
        # Set device for GPU/CPU computation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Store hyperparameters
        self.gamma = gamma
        self.gae_lambda= gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef= value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.initial_lr = lr
        
        # Log key hyperparameters
        logger.info("PPO HYPERPARAMETERS:")
        logger.info(f"Learning Rate: {lr}")
        logger.info(f"Gamma: {gamma}")
        logger.info(f"GAE Lambda: {gae_lambda}")
        logger.info(f"Clip Epsilon: {clip_epsilon}")
        logger.info(f"Entropy Coefficient: {entropy_coef}")
        
        # Initialize network and optimizer
        self.network = ActorCriticNetwork(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        
        # Training configuration
        self.update_epochs = 8
        self.mini_batch_size = 256
        self.update_count = 0
        
    def get_action(self, state, deterministic=False):
        # Convert state to tensor and move to device
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action_logits, value = self.network(state_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            
            if deterministic:
                # Use most likely action for evaluation
                action = torch.argmax(action_probs, dim=-1)
                log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1))).squeeze(1)
            else:
                # Sample action from probability distribution
                dist = Categorical(action_probs)
                action =dist.sample()
                log_prob = dist.log_prob(action)
        
        # Convert to numpy and clean up GPU memory
        action_cpu = action.cpu().numpy()
        log_prob_cpu = log_prob.cpu().numpy()
        value_cpu = value.cpu().numpy()
        
        del state_tensor,action_logits, action_probs, action, log_prob, value
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return action_cpu, log_prob_cpu, value_cpu
    
    def compute_gae(self, rewards, values, dones):
        #Compute Generalized Advantage Estimation
        advantages = []
        gae = 0
        
        # Work backwards through the episode
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0  # No next value at episode end
            else:
                next_value =values[i + 1]
            
            # Compute TD error and GAE
            delta = rewards[i] +self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        return advantages
    
    def update(self, states, actions, old_log_probs, rewards, dones, values):
        self.update_count += 1
        
        # Convert lists to numpy arrays first
        states_np =np.array(states)
        actions_np = np.array(actions)
        old_log_probs_np = np.array(old_log_probs)
        rewards_np = np.array(rewards)
        dones_np = np.array(dones)
        values_np = np.array(values)

        
        # Convert to PyTorch tensors and move to device
        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor(actions_np).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs_np).to(self.device)
        rewards = torch.FloatTensor(rewards_np).to(self.device)
        dones = torch.FloatTensor(dones_np).to(self.device)
        values = torch.FloatTensor(values_np).to(self.device)
        
        # Compute advantages using GAE
        advantages_list= self.compute_gae(rewards_np, values_np, dones_np)
        advantages = torch.FloatTensor(np.array(advantages_list)).to(self.device)
        
        # Compute returns (targets for value function)
        returns = advantages + values
        # Normalize advantages for stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Set up mini-batch training
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        #Initialize tracking variables
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates=0
        kl_divergence = 0
        
        # Perform multiple epochs of learning
        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)  # Shuffle data each epoch
            
            # Process data in mini-batches
            for start in range(0, dataset_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # Extract mini-batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_values = values[batch_indices]
                
                # Forward pass through network
                action_logits, current_values = self.network(batch_states)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                
                #Compute policy loss components
                new_log_probs= dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()  # Encourage exploration!!!!!
                
                # Importance sampling ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages  # Unclipped objective
                # Clipped objective to prevent large policy updates
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()  # Take minimum (pessimistic)
                
                # Compute value loss with clipping
                current_values_flat = current_values.view(-1)
                batch_returns_flat = batch_returns.view(-1)
                batch_values_flat = batch_values.view(-1)
                
                # Clip value function updates
                value_pred_clipped = batch_values_flat + torch.clamp(
                    current_values_flat - batch_values_flat, 
                    -self.clip_epsilon, self.clip_epsilon
                )
                
                # Compute both clipped and unclipped value losses
                value_loss1 = F.mse_loss(current_values_flat, batch_returns_flat)
                value_loss2 = F.mse_loss(value_pred_clipped, batch_returns_flat)
                value_loss = torch.max(value_loss1, value_loss2)
                
                # Combine all losses
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Perform gradient update
                self.optimizer.zero_grad()
                total_loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics for monitoring
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                kl_div += torch.mean(batch_old_log_probs - new_log_probs).item()
                num_updates += 1
        
        # Update learning rate periodically
        if self.update_count % 50 == 0:
            self.scheduler.step()
        
        # Clean up memory
        del states, actions, old_log_probs, rewards, dones, values, advantages, returns
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return averaged metrics
        if num_updates > 0:
            return (total_policy_loss / num_updates, 
                    total_value_loss / num_updates, 
                    total_entropy / num_updates,
                    kl_div / num_updates)
        else:
            return 0.0, 0.0, 0.0, 0.0

class Tracker:
    """Track training metrics and automatically save data"""
    def __init__(self, save_frequency=100):
        self.episode_rewards = []
        self.episode_lengths = []
        self.update_metrics = []
        self.save_frequency = save_frequency
        self.session_start = datetime.now()
        self.total_episodes = 0
        
    def add_episode(self, reward, length):
        # Record episode data
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.total_episodes += 1
        
        # Auto-save periodically
        if self.total_episodes % self.save_frequency == 0:
            self.save_data()
    
    def add_update_metrics(self, policy_loss, value_loss, entropy, kl_div):
        # Record training metrics from each update
        self.update_metrics.append({
            'episode': self.total_episodes,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'kl_divergence': kl_div,
            'timestamp': datetime.now()
        })
    
    def save_data(self):
        # Save all training data to pickle file
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'update_metrics': self.update_metrics,
            'total_episodes': self.total_episodes,
            'session_start': self.session_start,
            'last_update': datetime.now()
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"training_data_{timestamp}.pkl"
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Training data saved: {filename}")
        except Exception as e:
            logger.warning(f"Could not save training data: {e}")
    
    def get_stats(self, recent_window=100):
        # Calculate performance statistics
        if not self.episode_rewards:
            return None
        
        recent_rewards = self.episode_rewards[-recent_window:]
        recent_lengths = self.episode_lengths[-recent_window:]
        
        return {
            'total_episodes': self.total_episodes,
            'max_reward': max(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards),
            'recent_average_reward': np.mean(recent_rewards),
            'recent_average_length': np.mean(recent_lengths),
            'reward_std': np.std(self.episode_rewards),
            'recent_reward_std': np.std(recent_rewards)
        }

class Trainer:
    """Main training for PPO agent"""
    def __init__(self, env_name="ALE/MsPacman-v5", max_episodes=15000,
                 max_steps_per_episode=15000, update_frequency=2048):
        self.env_name = env_name
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_frequency = update_frequency
        
        # Set up environment with preprocessing
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.env = PreprocessingWrapper(self.env)
        
        # Get environment dimensions
        obs, _ = self.env.reset()
        input_shape = obs.shape
        num_actions = self.env.action_space.n
        
        # Log training configuration
        logger.info("PPO TRAINING SETUP")
        logger.info(f"Environment: {env_name}")
        logger.info(f"Max episodes: {max_episodes}")
        logger.info(f"Update frequency: {update_frequency}")
        logger.info("Target: 1500+ average reward")
        
        # Initialize agent and tracking
        self.agent = PPOAgent(input_shape, num_actions)
        self.tracker = Tracker()
        self.best_avg_reward = -np.inf
        self.performance_threshold = 1000
        
    def evaluate_agent(self, num_episodes=5):
        # Run evaluation episodes without training
        eval_rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while episode_length < self.max_steps_per_episode:
                # Use deterministic actions for evaluation
                action, _, _ = self.agent.get_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action[0])
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards), np.std(eval_rewards)
    
    def plot_training_curves(self):
        # Generate comprehensive training plots
        stats = self.tracker.get_stats()
        if not stats:
            logger.info("No episode data to plot")
            return
            
        rewards = self.tracker.episode_rewards
        lengths = self.tracker.episode_lengths
        
        # Create detailed plot with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training progress with moving average
        window = min(100, max(10, len(rewards)//10))
        if window > 1 and len(rewards) >= window:
            running_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0,0].plot(rewards, alpha=0.3, color='lightcoral', label='Raw')
            axes[0,0].plot(range(window-1, len(rewards)), running_avg, 
                          color='darkred', linewidth=2, label=f'{window}-ep avg')
        else:
            axes[0,0].plot(rewards, color='red', label='Rewards')
        
        axes[0,0].set_title(f'PPO Training - {len(rewards)} Episodes')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Reward')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Episode lengths over time
        axes[0,1].plot(lengths, alpha=0.6, color='green')
        axes[0,1].set_title('Episode Lengths')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Steps')
        axes[0,1].grid(True, alpha=0.3)
        
        # Reward distribution histogram
        recent_rewards = rewards[-min(500, len(rewards)):]
        axes[0,2].hist(recent_rewards, bins=min(30, len(recent_rewards)//5 + 1), 
                       alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,2].set_title('Recent Reward Distribution')
        axes[0,2].set_xlabel('Reward')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].grid(True, alpha=0.3)
        
        # Performance summary bar chart
        milestones = ['Max', 'Avg', 'Recent']
        values = [stats['max_reward'], stats['average_reward'], stats['recent_average_reward']]
        colors = ['gold', 'silver', 'orange']
        
        bars = axes[1,0].bar(milestones, values, color=colors, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('Performance Summary')
        axes[1,0].set_ylabel('Reward')
        for bar, value in zip(bars, values):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                          f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Smoothed learning curve
        if len(rewards) > 50:
            chunk_size = max(25, len(rewards) // 20)
            smoothed_rewards = []
            smoothed_positions = []
            for i in range(chunk_size, len(rewards), chunk_size//2):
                chunk = rewards[max(0, i-chunk_size):i]
                if len(chunk) > 5:
                    smoothed_rewards.append(np.mean(chunk))
                    smoothed_positions.append(i)
            
            if smoothed_rewards:
                axes[1,1].plot(smoothed_positions, smoothed_rewards, 'o-', 
                              color='purple', linewidth=3, markersize=4)
                axes[1,1].set_title('Learning Curve (Smoothed)')
                axes[1,1].set_xlabel('Episode')
                axes[1,1].set_ylabel('Average Reward')
                axes[1,1].grid(True, alpha=0.3)
        
        # Variance analysis for stability
        if len(rewards) > 100:
            variance_window = 50
            moving_var = []
            for i in range(variance_window, len(rewards)):
                chunk = rewards[i-variance_window:i]
                moving_var.append(np.var(chunk))
            
            axes[1,2].plot(range(variance_window, len(rewards)), moving_var, 
                          color='red', alpha=0.7, linewidth=2)
            axes[1,2].set_title('Learning Stability (Variance)')
            axes[1,2].set_xlabel('Episode')
            axes[1,2].set_ylabel('Reward Variance')
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save detailed plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'ppo_training_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create simple summary plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(rewards, alpha=0.3, color='lightcoral')
        if window > 1 and len(rewards) >= window:
            running_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), running_avg, 
                     color='darkred', linewidth=2)
        plt.title(f'PPO - {len(rewards)} Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(lengths, alpha=0.5, color='green')
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.hist(recent_rewards, bins=min(20, len(recent_rewards)//2 + 1), 
                 edgecolor='black', alpha=0.7, color='skyblue')
        plt.title('Recent Rewards')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        bars = plt.bar(milestones, values, color=colors, alpha=0.7)
        plt.title('Performance Summary')
        plt.ylabel('Reward')
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                    f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('ppo_latest.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PPO plots saved: {filename} and ppo_latest.png")
        
        # Log performance summary
        logger.info("PERFORMANCE SUMMARY:")
        logger.info(f"   Total Episodes: {stats['total_episodes']}")
        logger.info(f"   Max Reward: {stats['max_reward']:.0f}")
        logger.info(f"   Overall Average: {stats['average_reward']:.1f}")
        logger.info(f"   Recent Average: {stats['recent_average_reward']:.1f}")
        logger.info(f"   Recent Std: {stats['recent_reward_std']:.1f}")
        
        # Performance assessment
        if stats['max_reward'] >= 1500:
            logger.info("EXCELLENT: Outstanding performance!")
        elif stats['max_reward'] >= 1000:
            logger.info("GREAT: Strong performance!")
        elif stats['max_reward'] >= 600:
            logger.info("GOOD: Solid performance!")
        else:
            logger.info("DEVELOPING: Still learning!")
    
    def collect_experience(self):
        # Collect experience for one update cycle
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        
        state, _ = self.env.reset()
        total_steps = 0
        episode_reward = 0
        episode_length = 0
        episodes_this_update = 0
        
        # Collect until we have enough steps for update
        while total_steps < self.update_frequency:
            # Get action from current policy
            action, log_prob, value = self.agent.get_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action[0])
            
            # Apply reward shaping for better learning
            shaped_reward = reward
            if not (terminated or truncated):
                shaped_reward += 0.1  # Small bonus for staying alive
            if (terminated or truncated) and episode_length < 100:
                shaped_reward -= 1.0  # Penalty for dying too quickly
            
            # Store experience
            states.append(state.copy())
            actions.append(action[0])
            log_probs.append(log_prob[0])
            rewards.append(shaped_reward)
            dones.append(terminated or truncated)
            values.append(value[0])
            
            # Move to next state
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Handle episode termination
            if terminated or truncated or episode_length >= self.max_steps_per_episode:
                self.tracker.add_episode(episode_reward, episode_length)
                episodes_this_update += 1
                
                # Log first few episodes in each update
                if episodes_this_update <= 3:
                    logger.info(f"Episode: Reward={episode_reward:.1f}, Length={episode_length}, Total={self.tracker.total_episodes}")
                
                # Reset for new episode
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
        
        return states, actions, log_probs, rewards, dones, values
    
    def save_model(self, filename):
        # Save model with comprehensive metadata
        stats = self.tracker.get_stats()
        
        save_data = {
            'model_state_dict': self.agent.network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'scheduler_state_dict': self.agent.scheduler.state_dict(),
            'hyperparameters': {
                'lr': self.agent.initial_lr,
                'gamma': self.agent.gamma,
                'gae_lambda': self.agent.gae_lambda,
                'clip_epsilon': self.agent.clip_epsilon,
                'value_coef': self.agent.value_coef,
                'entropy_coef': self.agent.entropy_coef,
                'update_epochs': self.agent.update_epochs,
                'mini_batch_size': self.agent.mini_batch_size
            },
            'training_stats': stats,
            'update_count': self.agent.update_count,
            'save_timestamp': datetime.now().isoformat()
        }
        
        torch.save(save_data, filename)
        logger.info(f"Model saved: {filename}")
        
        # Create text summary file
        summary_file = filename.replace('.pth', '_summary.txt')
        try:
            with open(summary_file, 'w') as f:
                f.write("PPO Model Summary\n")
                f.write("=================\n\n")
                f.write(f"Save Date: {datetime.now()}\n")
                if stats:
                    f.write(f"Total Episodes: {stats['total_episodes']}\n")
                    f.write(f"Max Reward: {stats['max_reward']:.1f}\n")
                    f.write(f"Average Reward: {stats['average_reward']:.2f}\n")
                    f.write(f"Recent Average: {stats['recent_average_reward']:.2f}\n")
                else:
                    f.write("Total Episodes: N/A\n")
                    f.write("Max Reward: N/A\n")
                    f.write("Average Reward: N/A\n")
                    f.write("Recent Average: N/A\n")
                f.write(f"Update Count: {self.agent.update_count}\n")
                f.write("\nHyperparameters:\n")
                for key, value in save_data['hyperparameters'].items():
                    f.write(f"  {key}: {value}\n")
        except Exception as e:
            logger.warning(f"Could not save summary: {e}")
    
    def train(self):
        # Main training loop
        logger.info("Starting PPO training...")
        start_time = time.time()
        
        update_count = 0
        last_save_time = time.time()
        
        def signal_handler(signum, frame):
            # Handle Ctrl+C gracefully
            logger.info("Training interrupted - saving data...")
            self.tracker.save_data()
            self.plot_training_curves()
            self.save_model("ppo_interrupted.pth")
            logger.info("Graceful shutdown complete!")
            raise KeyboardInterrupt()
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Main training loop
            while self.tracker.total_episodes < self.max_episodes:
                # Collect experience from environment
                states, actions, log_probs, rewards, dones, values = self.collect_experience()
                
                # Update policy using collected experience
                policy_loss, value_loss, entropy, kl_div = self.agent.update(
                    states, actions, log_probs, rewards, dones, values
                )
                
                # Track training metrics
                self.tracker.add_update_metrics(policy_loss, value_loss, entropy, kl_div)
                
                # Clean up memory to prevent leaks
                del states, actions, log_probs, rewards, dones, values
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                update_count += 1
                
                # Get current performance statistics
                stats = self.tracker.get_stats()
                
                if stats:
                    # Check for performance improvement
                    if stats['recent_average_reward'] > self.best_avg_reward + 25:
                        improvement = stats['recent_average_reward'] - self.best_avg_reward
                        self.best_avg_reward = stats['recent_average_reward']
                        logger.info(f"NEW BEST! +{improvement:.1f} -> {self.best_avg_reward:.1f}")
                        self.save_model("ppo_best.pth")
                    
                    # Periodic evaluation with deterministic policy
                    if update_count % 25 == 0:
                        try:
                            eval_mean, eval_std = self.evaluate_agent()
                            logger.info(f"Evaluation: {eval_mean:.1f} +/- {eval_std:.1f}")
                        except Exception as e:
                            logger.warning(f"Evaluation failed: {e}")
                    
                    # Regular logging every 10 updates
                    if update_count % 10 == 0:
                        elapsed_time = time.time() - start_time
                        current_lr = self.agent.optimizer.param_groups[0]['lr']
                        
                        logger.info(f"Update {update_count} | Episodes: {stats['total_episodes']}")
                        logger.info(f"Recent Avg: {stats['recent_average_reward']:.1f} | Max: {stats['max_reward']:.0f}")
                        logger.info(f"Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}")
                        logger.info(f"Entropy: {entropy:.4f} | KL Div: {kl_div:.4f}")
                        logger.info(f"Learning Rate: {current_lr:.6f}")
                        logger.info(f"Time: {elapsed_time/3600:.2f}h | Best Avg: {self.best_avg_reward:.1f}")
                        logger.info("=" * 80)
                
                # Auto-save checkpoint every hour
                current_time = time.time()
                if current_time - last_save_time > 3600:
                    self.tracker.save_data()
                    if stats:
                        self.save_model(f"checkpoint_{stats['total_episodes']}.pth")
                        logger.info(f"Auto-saved checkpoint at episode {stats['total_episodes']}")
                    last_save_time = current_time
                
                # Generate plots every 100 episodes
                if stats and stats['total_episodes'] % 1000 == 0 and stats['total_episodes'] > 0:
                    try:
                        self.plot_training_curves()
                    except Exception as e:
                        logger.warning(f"Could not generate plot: {e}")
                
                # Early stopping check for good performance
                if (stats and stats['total_episodes'] > 2000 and 
                    stats['recent_average_reward'] > self.performance_threshold and
                    stats['recent_reward_std'] < 200):
                    
                    logger.info(f"TARGET ACHIEVED! Avg: {stats['recent_average_reward']:.1f}, Stable: {stats['recent_reward_std']:.1f}")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Final cleanup and saving
            logger.info("Generating final plots...")
            try:
                self.tracker.save_data()
                self.plot_training_curves()
                self.save_model("ppo_final.pth")
            except Exception as e:
                logger.warning(f"Final save failed: {e}")
            
            self.env.close()
            
            # Final summary
            total_time = time.time() - start_time
            stats = self.tracker.get_stats()
            
            logger.info("PPO TRAINING COMPLETE!")
            logger.info(f"Total Time: {total_time/3600:.2f} hours")
            if stats:
                logger.info(f"Total Episodes: {stats['total_episodes']}")
                logger.info(f"Best Performance: {self.best_avg_reward:.1f}")
                

def main():
    
    parser = argparse.ArgumentParser(description='PPO Training for Ms Pacman')
    parser.add_argument('--env', default='ALE/MsPacman-v5', help='Environment name')
    parser.add_argument('--episodes', type=int, default=15000, help='Max episodes')
    
    args = parser.parse_args()
    
    # Try multiple environment names in case of compatibility issues
    env_names = [
        args.env,
        "ALE/MsPacman-v5", 
        "MsPacmanNoFrameskip-v4",
        "MsPacman-v4"
    ]
    
    trainer = None
    for env_name in env_names:
        try:
            trainer = Trainer(
                env_name=env_name,
                max_episodes=args.episodes,
                max_steps_per_episode=15000,
                update_frequency=2048
            )
            logger.info(f"Successfully created environment: {env_name}")
            break
        except Exception as e:
            logger.warning(f"Failed to create {env_name}: {e}")
            continue
    
    if trainer is None:
        logger.error("Could not create any environment.")
        return
    
    trainer.train()

if __name__ == "__main__":
    main()