import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from environment import SQCLinkEnv # Import class môi trường của chúng ta
from gymnasium.wrappers import RecordEpisodeStatistics 


# --- 1. CẤU HÌNH CÁC THAM SỐ VÀ ĐƯỜNG DẪN ---
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Đặt tên file cho model sẽ lưu
MODEL_NAME = "ppo_sqc_link_model"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Cấu hình tổng số bước huấn luyện
# 200,000 timesteps là một con số hợp lý để bắt đầu
TOTAL_TIMESTEPS = 200_000

if __name__ == "__main__":

    # --- 2. TẠO MÔI TRƯỜNG ---
    # make_vec_env giúp tạo nhiều môi trường chạy song song để tăng tốc độ huấn luyện.
    # Tuy nhiên, để bắt đầu, chúng ta chỉ dùng 1 môi trường (n_envs=1).
    env = SQCLinkEnv(pass_duration=600)
    env = RecordEpisodeStatistics(env) 

    # --- 3. CẤU HÌNH THUẬT TOÁN PPO ---
    model = PPO(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device='cuda' 
    )

    # --- 4. THIẾT LẬP CALLBACKS ---
    eval_env = SQCLinkEnv(pass_duration=600)
    
    # [THAY ĐỔI Ở ĐÂY]
    # Môi trường eval cũng cần được bọc bởi wrapper mới
    eval_env = RecordEpisodeStatistics(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # --- 5. BẮT ĐẦU HUẤN LUYỆN ---
    print("--- Bắt đầu quá trình huấn luyện ---")

    # Phương thức learn() sẽ bắt đầu quá trình huấn luyện.
    # tb_log_name đặt tên cho thư mục log trên TensorBoard.
    # callback=eval_callback để sử dụng callback đã thiết lập.
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=MODEL_NAME,
        callback=eval_callback
    )

    print("--- Huấn luyện hoàn tất ---")

    # --- 6. LƯU LẠI MODEL CUỐI CÙNG ---
    # eval_callback đã lưu lại model tốt nhất (best_model.zip),
    # nhưng chúng ta cũng có thể lưu model cuối cùng để so sánh.
    model.save(MODEL_PATH)
    print(f"Model cuối cùng đã được lưu tại: {MODEL_PATH}.zip")
    print(f"Model tốt nhất đã được lưu tại: {MODEL_DIR}/best_model.zip")
    print("Để xem kết quả, chạy lệnh: tensorboard --logdir ./logs/")
