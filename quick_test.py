import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bench import GeneralizationBenchmark
from rl import RandomAgent

def quick_test():
    print("🧪 Quick Benchmark Test")
    print("=" * 30)
    
    benchmark = GeneralizationBenchmark('Pendulum-v1')
    
    print("Testing environment variations...")
    for variation in ['default', 'high_gravity', 'low_friction']:
        env = benchmark.create_env(variation)
        print(f"  ✓ {variation}: obs_dim={env.obs_dim}, action_dim={env.action_dim}")
    
    # Test agents
    print("\\nTesting agents...")
    agent_classes = {
        'Random': RandomAgent,
    }
    
    # Quick run with minimal episodes
    print("Running quick benchmark (this will take ~1 minute)...")
    results = benchmark.run_benchmark(
        agent_classes,
        num_train_episodes=10,  # Very quick for testing
        num_eval_episodes=2
    )
    
    # Print results
    print("\\n📊 Results:")
    for agent_name, agent_results in results.items():
        print(f"\\n{agent_name}:")
        for variation in benchmark.test_variations:
            if variation in agent_results:
                mean_score = agent_results[variation]['mean']
                std_score = agent_results[variation]['std']
                print(f"  {variation}: {mean_score:.2f} ± {std_score:.2f}")
    
    print("\\n✅ Quick test completed successfully!")
    return results

if __name__ == "__main__":
    try:
        results = quick_test()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()