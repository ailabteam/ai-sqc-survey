import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class SQCLinkEnv(gym.Env):
    """
    Custom Environment for simulating a Satellite Quantum Communication Link.
    Follows the Gymnasium API.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, pass_duration=600, turbulence_model='random_walk'):
        super(SQCLinkEnv, self).__init__()

        # --- ACTION SPACE DEFINITION ---
        # 3 discrete actions: 0 (Low), 1 (Medium), 2 (High Power/Rate config)
        self.action_space = spaces.Discrete(3)
        self.action_configs = {
            0: {'power_watt': 0.5, 'rep_rate_hz': 1e9},  # Low power, High rate
            1: {'power_watt': 1.0, 'rep_rate_hz': 5e8},  # Medium power, Medium rate
            2: {'power_watt': 2.0, 'rep_rate_hz': 1e8}   # High power, Low rate
        }

        # --- OBSERVATION SPACE DEFINITION ---
        # Observation is a 3-element continuous vector:
        # 1. QBER (Quantum Bit Error Rate): [0, 0.5]
        # 2. Link Transmittance (Signal Strength): [0, 1.0]
        # 3. Time Remaining Ratio: [0, 1.0]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([0.5, 1.0, 1.0]),
            shape=(3,),
            dtype=np.float32
        )

        # --- ENVIRONMENT PARAMETERS ---
        self.pass_duration = pass_duration  # Total timesteps in an episode
        self.turbulence_model = turbulence_model
        self.current_step = 0

    def _get_obs(self):
        """Private method to get the current observation."""
        # Normalize time for the observation space
        time_ratio = (self.pass_duration - self.current_step) / self.pass_duration

        # Clip values to ensure they are within the defined space
        qber = np.clip(self.current_qber, 0.0, 0.5)
        transmittance = np.clip(self.current_transmittance, 0.0, 1.0)

        return np.array([qber, transmittance, time_ratio], dtype=np.float32)

    def _get_info(self):
        """Private method to get auxiliary information for logging."""
        return {
            "turbulence_cn2": self.current_cn2,
            "transmittance": self.current_transmittance,
            "qber": self.current_qber,
            "secret_key_rate": self.last_skr
        }

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        self.current_step = 0

        # --- ATMOSPHERIC TURBULENCE MODELING ---
        if self.turbulence_model == 'random_walk':
            self.cn2_series = self._generate_turbulence_random_walk()
        else: # Add more models later if needed
            self.cn2_series = self._generate_turbulence_random_walk()

        self.current_cn2 = self.cn2_series[self.current_step]
        self.current_transmittance = self._calculate_transmittance(self.current_cn2)

        # Assume an initial action to calculate initial QBER
        initial_config = self.action_configs[1] # Medium config
        self.current_qber = self._calculate_qber(self.current_transmittance, initial_config['power_watt'])
        self.last_skr = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """Executes one time step within the environment."""
        self.current_step += 1

        config = self.action_configs[action]

        # Update turbulence and channel state
        self.current_cn2 = self.cn2_series[self.current_step]
        self.current_transmittance = self._calculate_transmittance(self.current_cn2)
        self.current_qber = self._calculate_qber(self.current_transmittance, config['power_watt'])

        # --- REWARD CALCULATION ---
        # Reward is the Secret Key Rate (SKR)
        skr = self._calculate_skr_bb84(self.current_qber, config['rep_rate_hz'], self.current_transmittance)
        reward = skr
        self.last_skr = skr

        # --- DONE CONDITION ---
        # Episode ends when the satellite pass is over
        terminated = self.current_step >= self.pass_duration - 1

        observation = self._get_obs()
        info = self._get_info()

        # Gymnasium API returns 5 values
        return observation, reward, terminated, False, info

    def _generate_turbulence_random_walk(self):
        """
        Generates a time series simulating the refractive index structure parameter C_n^2.
        This is a simplified model of changing atmospheric turbulence.
        """
        # C_n^2 values for weak to strong turbulence typically range from 1e-17 to 1e-13
        log_series = np.zeros(self.pass_duration)
        log_series[0] = self.np_random.uniform(-16, -14) # Start in a log-scale range

        for i in range(1, self.pass_duration):
            step = self.np_random.uniform(-0.1, 0.1) # Small steps in log space
            log_series[i] = np.clip(log_series[i-1] + step, -17, -13)

        return 10**log_series


    def _calculate_transmittance(self, cn2):
        """
        [SỬA LỖI] Hiệu chỉnh mô hình transmittance.
        Thay vì một mô hình vật lý phức tạp, chúng ta sẽ tạo một mối liên hệ trực tiếp
        và dễ kiểm soát hơn giữa Cn2 và transmittance.
        """
        # Ánh xạ log của Cn2 (từ -17 đến -13) vào một thang transmittance (từ 0.8 đến 0.01)
        # Đây là một phép ánh xạ tuyến tính trong không gian log-linear.
        log_cn2 = np.log10(cn2)

        # Thang log Cn2: min=-17 (trời trong), max=-13 (nhiễu động mạnh)
        log_cn2_min = -17.0
        log_cn2_max = -13.0

        # Thang transmittance tương ứng:
        trans_max = 0.8  # Transmittance cao nhất khi trời trong
        trans_min = 0.01 # Transmittance thấp nhất khi nhiễu động mạnh

        # Công thức ánh xạ tuyến tính:
        transmittance = trans_max - ((log_cn2 - log_cn2_min) / (log_cn2_max - log_cn2_min)) * (trans_max - trans_min)

        return np.clip(transmittance, trans_min, trans_max)

    def _calculate_qber(self, transmittance, power, qber_dark=1e-7, qber_optic=0.01):
        """
        [SỬA LỖI] Hiệu chỉnh mô hình QBER.
        Giảm tác động của 'signal_dependent_error' để QBER không quá cao.
        """
        # Giảm hệ số của lỗi phụ thuộc tín hiệu từ 0.1 xuống 0.04
        signal_dependent_error = 0.04 * (1 - transmittance)

        # Giảm tác động của power một chút
        power_effect = 0.005 / power

        qber = qber_optic + qber_dark + signal_dependent_error + power_effect
        return np.clip(qber, 0, 0.5)

    def _calculate_skr_bb84(self, qber, R_rep, transmittance, mu=0.5, f_ec=1.16, eta_d=0.7):
        """
        [SỬA LỖI] Hiệu chỉnh mô hình SKR.
        Mô hình cũ đã khá ổn, nhưng chúng ta sẽ đơn giản hóa nó để dễ debug hơn
        và đảm bảo nó cho ra giá trị dương trong điều kiện QBER hợp lý.
        Công thức này dựa trên một phiên bản đơn giản hơn nhưng vẫn giữ được bản chất.
        """
        if qber >= 0.11: # Ngưỡng an toàn lý thuyết
            return 0

        def h2(p):
            if p <= 0 or p >= 1:
                return 0
            return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        # Công thức SKR đơn giản hóa: R = R_rep * T * [1 - h(e_d) - f*h(e_p)]
        # Trong đó e_d là QBER, e_p là lỗi pha (coi như bằng qber cho đơn giản)
        # Yếu tố quan trọng nhất là R_rep * transmittance * (1 - 2*h2(qber))

        gain = transmittance * eta_d # Tỷ lệ photon đến được detector

        # Yếu tố [1 - f_ec*h2(qber) - h2(qber)] = [1 - (f_ec+1)*h2(qber)]
        # Yếu tố này thường quá khắc nghiệt. Hãy dùng công thức gần đúng phổ biến hơn.
        # R ≈ 0.5 * R_rep * gain * (1 - 2*h2(qber))

        skr_factor = (1 - 2 * h2(qber))

        raw_skr = 0.5 * R_rep * gain * skr_factor

        # Chuyển đổi sang kbps và đảm bảo không âm
        return max(0, raw_skr) / 1e6



    def close(self):
        pass
