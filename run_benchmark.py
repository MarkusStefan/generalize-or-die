"""
Simple runner script to test the RL benchmark with fallback environments
"""

import os
import sys

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from bench import GeneralizationBenchmark
    from rl import DDPG, RandomAgent
    import torch
    print("✓ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def main():
    print("🚀 Starting RL Generalization Benchmark")
    print("=" * 50)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Available environments (using Gymnasium for Windows compatibility)
    available_envs = [
        ('Pendulum-v1', 'Continuous control - swing up pendulum'),
        ('CartPole-v1', 'Discrete control - balance cartpole'),
        ('Acrobot-v1', 'Continuous control - swing up acrobot'),
        ('MountainCarContinuous-v0', 'Continuous control - drive up mountain'),
    ]
    
    print("\\nAvailable environments:")
    for i, (env_name, description) in enumerate(available_envs):
        print(f"{i}: {env_name} - {description}")
    
    # Auto-select Pendulum for demo
    env_idx = 0
    env_name, description = available_envs[env_idx]
    print(f"\\nSelected: {env_name} - {description}")
    
    # Create benchmark
    benchmark = GeneralizationBenchmark(
        domain_name=env_name,
        task_name=None,
        env_type='gymnasium'
    )
    
    # Define agents
    agent_classes = {
        'DDPG': DDPG,
        'Random': RandomAgent
    }
    
    print(f"\\n🏋️ Training agents for generalization testing...")
    print("This may take a few minutes...")
    
    # Run benchmark with reasonable parameters for demo
    try:
        results = benchmark.run_benchmark(
            agent_classes, 
            num_train_episodes=50,   # Quick demo
            num_eval_episodes=3      # Quick evaluation
        )
        
        print("\\n📊 Plotting results...")
        benchmark.plot_results(results)
        
        # Print summary
        print("\\n📈 Generalization Summary:")
        print("=" * 30)
        for agent_name, agent_results in results.items():
            default_score = agent_results['default']['mean']
            
            # Calculate generalization scores
            other_variations = [v for v in benchmark.test_variations if v != 'default']
            if other_variations:
                generalization_scores = [agent_results[var]['mean'] for var in other_variations if var in agent_results]
                if generalization_scores:
                    avg_generalization = sum(generalization_scores) / len(generalization_scores)
                    generalization_gap = default_score - avg_generalization
                    
                    print(f"\\n{agent_name}:")
                    print(f"  Default environment: {default_score:.2f}")
                    print(f"  Average on variations: {avg_generalization:.2f}")
                    print(f"  Generalization gap: {generalization_gap:.2f}")
                    
                    if generalization_gap < 0:
                        print(f"  ✓ Agent performs better on variations!")
                    elif generalization_gap < 50:
                        print(f"  ✓ Good generalization")
                    else:
                        print(f"  ⚠️ Poor generalization")
            
        print("\\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()