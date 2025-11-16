import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from environment import SQCLinkEnv # Import môi trường của chúng ta

# --- 1. CẤU HÌNH ---
MODEL_DIR = "./models/"
FIGURES_DIR = "../figures/" # Lưu hình ảnh ở thư mục figures ngoài src
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.zip")

NUM_EVAL_EPISODES = 50 # Chạy 50 episode để lấy kết quả trung bình, đáng tin cậy hơn

def evaluate_policy(model, env, num_episodes=10):
    """
    Hàm để đánh giá một policy (model RL hoặc baseline) trên môi trường.
    Trả về một list các dictionary, mỗi dictionary chứa kết quả của một episode.
    """
    results = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False

        # Biến để lưu trữ dữ liệu của episode này
        episode_rewards = []
        episode_actions = []
        episode_transmittances = []
        episode_qbers = []

        while not done:
            action_value = None # Biến tạm để chứa hành động cuối cùng
            if isinstance(model, PPO):
                # Nếu là model RL, dùng predict()
                action, _ = model.predict(obs, deterministic=True)
                
                # [THAY ĐỔI Ở ĐÂY]
                # Chuyển đổi numpy array thành số nguyên.
                # .item() là cách an toàn để lấy giá trị đơn lẻ từ một array.
                action_value = action.item() 
            else:
                # Nếu là baseline (một số nguyên hoặc một hàm)
                action_value = model(obs) if callable(model) else model

            # Sử dụng action_value đã được chuyển đổi
            obs, reward, done, _, info = env.step(action_value)

            # Lưu lại dữ liệu
            episode_rewards.append(reward)
            # Lưu lại action_value đã là số nguyên
            episode_actions.append(action_value)
            episode_transmittances.append(info.get('transmittance', 0))
            episode_qbers.append(info.get('qber', 0))

        results.append({
            "total_reward": sum(episode_rewards),
            "rewards": episode_rewards,
            "actions": episode_actions,
            "transmittances": episode_transmittances,
            "qbers": episode_qbers
        })
    return results

def plot_results(results_dict):
    """Hàm để vẽ và lưu các biểu đồ so sánh."""

    # --- Biểu đồ 1: So sánh tổng Reward trung bình ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    policy_names = list(results_dict.keys())
    mean_rewards = [np.mean([ep['total_reward'] for ep in results_dict[p]]) for p in policy_names]
    std_rewards = [np.std([ep['total_reward'] for ep in results_dict[p]]) for p in policy_names]

    ax1.bar(policy_names, mean_rewards, yerr=std_rewards, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel("Average Total SKR (Mbps-episode)")
    ax1.set_title("Performance Comparison of Different Policies")
    ax1.set_xticklabels(policy_names, rotation=45, ha="right")

    plt.tight_layout()
    fig1.savefig(os.path.join(FIGURES_DIR, "policy_comparison_bar.png"), dpi=600)
    print(f"Đã lưu biểu đồ so sánh tổng reward tại: {FIGURES_DIR}policy_comparison_bar.png")

    # --- Biểu đồ 2: Phân tích hành vi của Agent RL trong một episode điển hình ---
    rl_results_one_ep = results_dict['PPO Agent'][0] # Lấy kết quả của episode đầu tiên

    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    timesteps = np.arange(len(rl_results_one_ep['rewards']))

    # Biểu đồ SKR và Transmittance
    ax2.plot(timesteps, rl_results_one_ep['rewards'], label='Secret Key Rate (Mbps)', color='blue')
    ax2.set_ylabel('SKR (Mbps)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(timesteps, rl_results_one_ep['transmittances'], label='Link Transmittance', color='green', linestyle='--')
    ax2_twin.set_ylabel('Transmittance', color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    ax2.set_title("PPO Agent Behavior in a Typical Episode")
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    # Biểu đồ hành động đã chọn
    action_labels = {0: 'Low Power', 1: 'Medium Power', 2: 'High Power'}
    ax3.step(timesteps, rl_results_one_ep['actions'], where='post', label='Action Chosen', color='red')
    ax3.set_yticks(list(action_labels.keys()))
    ax3.set_yticklabels(list(action_labels.values()))
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Action')
    ax3.grid(True, axis='y')

    plt.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, "ppo_agent_behavior.png"), dpi=600)
    print(f"Đã lưu biểu đồ phân tích hành vi tại: {FIGURES_DIR}ppo_agent_behavior.png")


if __name__ == "__main__":
    # --- Tải môi trường và model ---
    eval_env = SQCLinkEnv(pass_duration=600)

    try:
        model = PPO.load(MODEL_PATH, env=eval_env)
        print("Đã tải model PPO thành công.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy model tại {MODEL_PATH}. Hãy chắc chắn rằng bạn đã huấn luyện model trước.")
        exit()

    # --- Định nghĩa các baseline policies ---
    # Baseline 1: Luôn chọn hành động 0 (Công suất thấp)
    low_power_policy = lambda obs: 0
    # Baseline 2: Luôn chọn hành động 2 (Công suất cao)
    high_power_policy = lambda obs: 2
    # Baseline 3: Chọn hành động ngẫu nhiên
    random_policy = lambda obs: eval_env.action_space.sample()

    # --- Chạy đánh giá cho tất cả các policies ---
    all_results = {}
    print(f"\nBắt đầu đánh giá PPO Agent trên {NUM_EVAL_EPISODES} episodes...")
    all_results['PPO Agent'] = evaluate_policy(model, eval_env, num_episodes=NUM_EVAL_EPISODES)

    print(f"\nBắt đầu đánh giá Low Power Policy trên {NUM_EVAL_EPISODES} episodes...")
    all_results['Low Power'] = evaluate_policy(low_power_policy, eval_env, num_episodes=NUM_EVAL_EPISODES)

    print(f"\nBắt đầu đánh giá High Power Policy trên {NUM_EVAL_EPISODES} episodes...")
    all_results['High Power'] = evaluate_policy(high_power_policy, eval_env, num_episodes=NUM_EVAL_EPISODES)

    print(f"\nBắt đầu đánh giá Random Policy trên {NUM_EVAL_EPISODES} episodes...")
    all_results['Random'] = evaluate_policy(random_policy, eval_env, num_episodes=NUM_EVAL_EPISODES)

    print("\n--- Đánh giá hoàn tất ---")

    # --- In kết quả ra console ---
    print("\n--- Kết quả tóm tắt ---")
    for policy_name, results in all_results.items():
        total_rewards = [ep['total_reward'] for ep in results]
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"Policy: {policy_name:<15} | Mean Reward: {mean_reward:,.2f} +/- {std_reward:,.2f}")

    # --- Vẽ và lưu biểu đồ ---
    plot_results(all_results)
