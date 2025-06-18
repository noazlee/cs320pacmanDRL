import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import os
from collections import deque
import time
import logging
import random
import gc
from datetime import datetime

# https://www.youtube.com/watch?v=EUrWGTCGzlA&t=840s - Very simple dql implementation on frozen lake

# set up logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PreprocessingWrapper:
    """Environment wrapper for Atari frame preprocessing"""
    def __init__(self, env, height=84, width=84, frame_stack=4):
        self.env = env
        self.height = height
        self.width= width
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)  # Store last 4 frames
        
    def reset(self):
        obs,info = self.env.reset()
        frame= self._preprocess_frame(obs)
        # fill frame stack with initial frame
        for i in range(self.frame_stack):
            self.frames.append(frame)
        return self._get_stacked_frames(), info
        
    def step(self, action):
        obs, reward,terminated, truncated, info = self.env.step(action)
        frame =self._preprocess_frame(obs)
        self.frames.append(frame) # Add new frame,automatically removes oldest
        return self._get_stacked_frames(), reward, terminated, truncated, info
        
    def _preprocess_frame(self, frame):
        # Convert to grayscale and resize to 84x84
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height))
        return resized.astype(np.float32) / 255.0  # Normalize to [0,1]
        
    def _get_stacked_frames(self):
        # Stack frames along first dimension: (4, 84, 84)
        return np.stack(self.frames, axis=0)
        
    def __getattr__(self, name):
        # Forward all other attributes to the base environment
        return getattr(self.env, name)

class DQNetwork(nn.Module):
    """Deep Q-Network for learning action values"""
    def __init__(self, input_shape, num_actions):
        super(DQNetwork, self).__init__()
        
        # Convolutional layers for feature extraction from game frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),     # Extract large features
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),                 # Medium-scale features
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),                 # Fine-grained features
            nn.ReLU()
        )
        
        # Calculate size after convolution
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # Fully connected layers to map features to Q-values
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)  # One Q-value per action
        )
        
    def _get_conv_out_size(self, shape):
        # Calculate output size after convolution layers
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self.conv(dummy_input)
        return int(np.prod(dummy_output.size()))
        
    def forward(self, x):
        # Add batch dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        # Extract features and compute Q-values
        conv_out = self.conv(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        q_values = self.fc(flattened)
        return q_values

class ReplayBuffer:
    """Experience replay buffer for storing and sampling training experiences"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)  # Circular buffer
    
    def push(self, state, action, reward, next_state, done):
        # Store experiences in memory-efficient uint8 format
        state_uint8 = (state * 255).astype(np.uint8)
        next_state_uint8 = (next_state * 255).astype(np.uint8)
        self.buffer.append((state_uint8, action, reward, next_state_uint8, done))
    
    def sample(self, batch_size):
        # Sample random batch of experiences
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert back to float32 for training
        states = np.array(states, dtype=np.float32) / 255.0
        next_states = np.array(next_states, dtype=np.float32) / 255.0
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQLAgent:
    """Deep Q-Learning agent with Double DQN and target networks"""
    def __init__(self, input_shape, num_actions, lr=2.5e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.02,epsilon_decay_steps=500000,
                 target_update=5000, batch_size=32, warmup_steps=10000):
        
        # Set device for GPU/CPU compuation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # store hyperparamaters
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end= epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update = target_update
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        
        # initialize networks - policy network and target network
        self.q_network = DQNetwork(input_shape, num_actions).to(self.device)
        self.target_network = DQNetwork(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, eps=1e-4)
        
        # Copy weights to target network for initial sync
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize experience replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training counters
        self.update_counter = 0
        self.total_steps = 0
        
    def get_epsilon(self):
        # Calculate current exploration rate
        if self.total_steps < self.warmup_steps:
            return self.epsilon
        
        # linear decay from epsilon_start to epsilon_end
        progress = min(1.0, (self.total_steps - self.warmup_steps) / self.epsilon_decay_steps)
        current_eps = self.epsilon_end + (self.epsilon - self.epsilon_end) * (1 - progress)
        
        return current_eps
        
    def get_action(self, state, training=True):
        # Get current exploration rate
        current_epsilon = self.get_epsilon() if training else 0.01
        
        # Epsilon-greedy action selection
        if training and random.random() < current_epsilon:
            action = random.randrange(self.num_actions)  # Random exploration
            q_values = None
        else:
            # use neural network to select best action
            state_tensor =torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()  # Greedy action
                q_values = q_values.cpu().numpy()
            
            # Clean up GPU memory
            del state_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Increment step counter during training
        if training:
            self.total_steps += 1
        return action, q_values
    
    def store_experience(self, state,action, reward, next_state, done):
        # add experience to replay buffer
        self.replay_buffer.push(state,action, reward, next_state, done)
    
    def update(self):
        # check if we have enough experiences to train
        if len(self.replay_buffer)<max(self.batch_size, self.warmup_steps):
            return 0.0, self.get_epsilon()
        
        # Sample batch of experiences from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to PyTorch tensors and move to device
        states =torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute current Q-values for taken actions
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use policy network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)  # Policy network selects
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()  # Target evaluates
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)  # Bellman equation
        
        # Compute Huber loss (less sensitive to outliers than MSE)
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Perform gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)  # Clip gradients
        self.optimizer.step()
        
        # Clean up GPU memory
        loss_value = loss.item()
        del states, actions, rewards, next_states, dones, current_q_values, target_q_values, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.info(f"Target network updated at step {self.total_steps:,} (eps: {self.get_epsilon():.3f})")
        
        return loss_value, self.get_epsilon()

class EarlyStopper:
    """Early stopping when performance goals are reached"""
    def __init__(self, min_episodes=2000, patience=400, target_performance=4000):
        self.min_episodes = min_episodes
        self.patience = patience
        self.target_performance = target_performance
        self.best_score = -np.inf
        self.counter = 0
        self.activated = False
        
    def __call__(self, episode_count, score):
        # Don't activate until minimum episodes reached
        if episode_count < self.min_episodes or score < self.target_performance:
            return False
        
        # Activate early stopping when target is first reached
        if not self.activated:
            self.activated = True
            self.best_score = score
            logger.info(f"TARGET ACHIEVED! Episode {episode_count:,}, Score {score:.1f}")
            return False
        
        # Track continued improvement or start patience counter
        if score > self.best_score + 100:  # Significant improvement
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

class Trainer:
    """Main training for DQL agent"""
    def __init__(self, env_name="ALE/MsPacman-v5", max_episodes=20000,
                 max_steps_per_episode=18000, update_frequency=4):
        self.env_name = env_name
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_frequency = update_frequency
        
        # set up environment with preprocessing
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.env = PreprocessingWrapper(self.env)
        
        # Get environment dimensions
        obs, _= self.env.reset()
        input_shape = obs.shape
        num_actions = self.env.action_space.n
        
        # Log training configuration
        logger.info(f"DQL TRAINING STARTED")
        logger.info(f"Environment: {env_name}")
        logger.info(f"Max episodes: {max_episodes:,}")
        logger.info(f"Max steps per episode: {max_steps_per_episode:,}")
        logger.info(f"TARGET: 4000+ average reward (Expert Level)")
        logger.info(f"Expected training time: 1-4 hours")
        
        # Initialize agent and tracking
        self.agent = DQLAgent(input_shape, num_actions)
        
        # Training metrics with limited memory
        self.episode_rewards = deque(maxlen=2000)
        self.episode_lengths = deque(maxlen=2000)
        
        # Initialize early stopping
        self.early_stopper = EarlyStopper()
        
        # Counter to prevent excessive saving
        self.save_counter = 0

    def plot_training_curves(self):
        # Generate comprehensive training plots
        if not self.episode_rewards:
            logger.info("No episode data to plot")
            return
            
        rewards = list(self.episode_rewards)
        lengths = list(self.episode_lengths)
        
        # Calculate performance statistics
        max_reward = max(rewards)
        average_reward = np.mean(rewards)
        recent_rewards = rewards[-min(100, len(rewards)):]
        recent_average_reward = np.mean(recent_rewards)
        recent_reward_std = np.std(recent_rewards)
        
        # Create detailed plot with multiple subplots
        fig,axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training progress with moving average
        window = min(100, max(10, len(rewards)//10))
        if window > 1 and len(rewards) >= window:
            running_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0,0].plot(rewards, alpha=0.3, color='lightblue', label='Raw')
            axes[0,0].plot(range(window-1, len(rewards)), running_avg, 
                          color='darkblue', linewidth=2, label=f'{window}-ep avg')
        else:
            axes[0,0].plot(rewards, color='blue', label='Rewards')
        
        axes[0,0].set_title(f'DQL Training - {len(rewards)} Episodes')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Reward')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # episode lengths over time
        axes[0,1].plot(lengths, alpha=0.6, color='green')
        axes[0,1].set_title('Episode Lengths')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Steps')
        axes[0,1].grid(True, alpha=0.3)
        
        # reward distribution histogram
        recent_plot_rewards = rewards[-min(500, len(rewards)):]
        axes[0,2].hist(recent_plot_rewards, bins=min(30, len(recent_plot_rewards)//5 + 1), 
                       alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,2].set_title('Recent Reward Distribution')
        axes[0,2].set_xlabel('Reward')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].grid(True, alpha=0.3)
        
        # Performance summary bar chart
        milestones = ['Max', 'Avg', 'Recent']
        values = [max_reward, average_reward, recent_average_reward]
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
        filename = f'dql_training_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create simple summary plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(rewards, alpha=0.3, color='lightblue')
        if window> 1 and len(rewards) >= window:
            running_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), running_avg, 
                     color='darkblue', linewidth=2)
        plt.title(f'DQL Training - {len(rewards)} Episodes')
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
        plt.hist(recent_plot_rewards, bins=min(20, len(recent_plot_rewards)//2 + 1), 
                 edgecolor='black', alpha=0.7, color='lightgreen')
        plt.title('Recent Rewards')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        bars =plt.bar(milestones, values, color=colors, alpha=0.7)
        plt.title('Performance Summary')
        plt.ylabel('Reward')
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                    f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('dql_latest.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"DQL plots saved: {filename} and dql_latest.png")
        
        # Log performance summary
        logger.info("DQL PERFORMANCE SUMMARY:")
        logger.info(f"Total Episodes: {len(rewards)}")
        logger.info(f"Max Reward: {max_reward:.0f}")
        logger.info(f"Overall Average: {average_reward:.1f}")
        logger.info(f"Recent Average: {recent_average_reward:.1f}")
        logger.info(f"Recent Std: {recent_reward_std:.1f}")
        
        # Performance assessment
        if max_reward >= 1500:
            logger.info(">1500")
        elif max_reward >= 1000:
            logger.info(">1000")
        elif max_reward >= 600:
            logger.info(">600")
        else:
            logger.info("<600 :(")
        
    def train(self):
        # Main training loop
        logger.info("Starting DQL training...")
        start_time = time.time()
        
        episode_count = 0
        best_avg_reward = -np.inf
        
        try:
            while episode_count < self.max_episodes:
                # Run single episode
                episode_reward, episode_length, avg_loss = self.run_episode()
                
                # Store episode results
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_count += 1
                
                # Clean up memory periodically
                if episode_count % 20 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Log recent performance every 100 episodes
                if episode_count % 100 == 0:
                    logger.info(f"Episode {episode_count:,}: Recent reward={episode_reward:.0f}")
                
                # Check for performance improvements and save sparingly
                if len(self.episode_rewards) >= 100:
                    recent_avg_reward = np.mean(list(self.episode_rewards)[-100:])
                    
                    # Only save on major improvements (100+ points)
                    if recent_avg_reward > best_avg_reward + 100:
                        old_best = best_avg_reward
                        best_avg_reward = recent_avg_reward
                        self.save_counter += 1
                        logger.info(f"IMPROVEMENT {old_best:.1f} -> {recent_avg_reward:.1f}")
                        
                        # Save only every 3rd major improvement to reduce file spam
                        if self.save_counter % 3 == 0:
                            self.save_model(f"dql_best.pth")
                            logger.info(f"Model saved (improvement #{self.save_counter})")
                    
                    # Check early stopping condition
                    if self.early_stopper(episode_count, recent_avg_reward):
                        logger.info(f"Training complete! Score: {recent_avg_reward:.2f}")
                        break
                else:
                    recent_avg_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0
                
                # Generate plots every 500 episodes
                if episode_count % 500 ==0 and episode_count>0:
                    try:
                        self.plot_training_curves()
                    except Exception as e:
                        logger.warning(f"Could not generate plot: {e}")
                
                # Detailed logging every 500 episodes
                if episode_count % 500 == 0:
                    elapsed_time = time.time() - start_time
                    current_epsilon = self.agent.get_epsilon()
                    
                    logger.info(f"UPDATE {episode_count:,}")
                    logger.info(f"Avg Reward (last 100): {recent_avg_reward:.2f}")
                    logger.info(f"Best Avg Reward: {best_avg_reward:.2f}")
                    logger.info(f"Total Steps: {self.agent.total_steps:,}")
                    logger.info(f"Epsilon: {current_epsilon:.4f}")
                    logger.info(f"Replay Buffer: {len(self.agent.replay_buffer):,}")
                    logger.info(f"Elapsed Time: {elapsed_time/3600:.1f} hours")
                    logger.info("-" * 80)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Final cleanup and saving
            try:
                self.plot_training_curves()
            except Exception as e:
                logger.warning(f"Final plot generation failed: {e}")
                
            self.save_model("dql_final.pth")
            self.env.close()
            
            # Final summary
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time/3600:.1f} hours")
            logger.info(f"Total episodes: {episode_count:,}")
    
    def run_episode(self):
        # Execute single training episode
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        losses = []
        
        while episode_length < self.max_steps_per_episode:
            # Get action from agent (epsilon-greedy)
            action, q_values = self.agent.get_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store experience in replay buffer
            self.agent.store_experience(state, action, reward, next_state, done)
            
            # Update network periodically
            if episode_length % self.update_frequency == 0:
                loss, epsilon = self.agent.update()
                if loss > 0:
                    losses.append(loss)
            
            # Move to next state
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # End episode if done
            if done:
                break
        
        return episode_reward, episode_length, np.mean(losses) if losses else 0.0

    def save_model(self, filename):
        # Save model state and training data
        torch.save({
            'q_network_state_dict': self.agent.q_network.state_dict(),
            'target_network_state_dict': self.agent.target_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'total_steps': self.agent.total_steps,
        }, filename)
        logger.info(f"Model saved to {filename}")

if __name__ == "__main__":
    # Try multiple environment names for compatibility
    env_names = [
        "ALE/MsPacman-v5",
        "MsPacmanNoFrameskip-v4", 
        "MsPacman-v4"
    ]
    
    trainer = None
    for env_name in env_names:
        try:
            trainer = Trainer(
                env_name=env_name,
                max_episodes=20000,
                max_steps_per_episode=18000,
                update_frequency=4
            )
            logger.info(f"Successfully created environment: {env_name}")
            break
        except Exception as e:
            logger.warning(f"Failed to create {env_name}: {e}")
            continue
    
    if trainer is None:
        logger.error("Could not create any environment")
        exit(1)
    
    # Start training
    trainer.train()