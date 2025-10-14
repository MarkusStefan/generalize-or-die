import numpy as np
import matplotlib.pyplot as plt
from env import create_environment
from rl import DDPG, RandomAgent
from collections import defaultdict


class GeneralizationBenchmark:
    def __init__(self, env_name, seed=42):
        self.env_name = env_name
        self.seed = seed
        self.results = defaultdict(list)
        self.train_variation = 'default'
        self.test_variations = ['default', 'high_gravity', 'low_friction']
        
    def create_env(self, variation='default'):
        return create_environment(self.env_name, self.seed, variation)
    
    def evaluate_agent(self, agent, variation='default', num_episodes=10):
        env = self.create_env(variation)
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state, add_noise=False)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)
    
    def train_agent(self, agent_class, num_episodes=500, **agent_kwargs):
        train_env = self.create_env(self.train_variation)
        action_bound = train_env.action_bound
        
        if agent_class == DDPG:
            agent = agent_class(
                state_dim=train_env.obs_dim,
                action_dim=train_env.action_dim,
                action_bound=action_bound,
                **agent_kwargs
            )
        else:
            agent = agent_class(
                action_dim=train_env.action_dim,
                action_bound=action_bound,
                **agent_kwargs
            )
        
        # Training loop
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = train_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = train_env.step(action)
                
                # Store transition for learning agents
                if hasattr(agent, 'replay_buffer'):
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                # Train agent
                agent.train()
            
            episode_rewards.append(episode_reward)
            
            if episode % 50 == 0:
                print(f"Episode {episode}: Average reward = {np.mean(episode_rewards[-50:]):.2f}")
        
        return agent, episode_rewards
    
    def run_benchmark(self, agent_classes, num_train_episodes=500, num_eval_episodes=20):
        """Run full generalization benchmark"""
        results = {}
        
        for agent_name, agent_class in agent_classes.items():
            print(f"\n=== Training {agent_name} ===")
            
            # Train agent
            agent, train_rewards = self.train_agent(agent_class, num_train_episodes)
            
            # Evaluate on all variations
            agent_results = {'train_rewards': train_rewards}
            
            for variation in self.test_variations:
                print(f"Evaluating on {variation} variation...")
                mean_reward, std_reward = self.evaluate_agent(agent, variation, num_eval_episodes)
                agent_results[variation] = {'mean': mean_reward, 'std': std_reward}
                print(f"{variation}: {mean_reward:.2f} ± {std_reward:.2f}")
            
            results[agent_name] = agent_results
        
        return results
    
    def plot_results(self, results):
        """Plot benchmark results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot training curves
        for agent_name, agent_results in results.items():
            train_rewards = agent_results['train_rewards']
            # Smooth the curve
            window_size = 50
            smoothed_rewards = [np.mean(train_rewards[i:i+window_size]) 
                              for i in range(0, len(train_rewards)-window_size+1, window_size)]
            ax1.plot(range(0, len(train_rewards)-window_size+1, window_size), 
                    smoothed_rewards, label=agent_name)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Training Performance')
        ax1.legend()
        ax1.grid(True)
        
        # Plot generalization results
        variations = self.test_variations
        agent_names = list(results.keys())
        x = np.arange(len(variations))
        width = 0.8 / len(agent_names)
        
        for i, agent_name in enumerate(agent_names):
            means = [results[agent_name][var]['mean'] for var in variations]
            stds = [results[agent_name][var]['std'] for var in variations]
            ax2.bar(x + i * width, means, width, yerr=stds, 
                   label=agent_name, alpha=0.7, capsize=5)
        
        ax2.set_xlabel('Environment Variation')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Generalization Performance')
        ax2.set_xticks(x + width * (len(agent_names) - 1) / 2)
        ax2.set_xticklabels(variations)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


if __name__ == "__main__":
    benchmark = GeneralizationBenchmark('Pendulum-v1')
    
    agent_classes = {
        'DDPG': DDPG,
        'Random': RandomAgent
    }
    
    results = benchmark.run_benchmark(
        agent_classes, 
        num_train_episodes=100,
        num_eval_episodes=5
    )
    
    benchmark.plot_results(results)
    
    print("\n=== Generalization Summary ===")
    for agent_name, agent_results in results.items():
        default_score = agent_results['default']['mean']
        generalization_scores = [agent_results[var]['mean'] for var in ['high_gravity', 'low_friction']]
        avg_generalization = np.mean(generalization_scores)
        generalization_gap = default_score - avg_generalization
        
        print(f"{agent_name}:")
        print(f"  Default environment: {default_score:.2f}")
        print(f"  Average on variations: {avg_generalization:.2f}")
        print(f"  Generalization gap: {generalization_gap:.2f}")