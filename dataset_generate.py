import os
import copy
import cv2
import random
import numpy as np
import scipy.ndimage
from scipy.io import savemat
from scipy.signal import filtfilt, butter

# ================= 配置区域 (Configuration) =================
CONFIG = {
    # 输出路径
    'paths': {
        'rc':    './dataset/test/rc_random',
        'synth': './dataset/test/synth_random',
        'pure':  './dataset/test/pure_random'
    },
    'sim_params': {
        'num_simulations': 50,       # 模拟样本数量
        'width': 500,                # 初始生成宽度
        'height': 500,               # 初始生成高度
        
        # --- 关键修改：区分输入尺寸和标签尺寸 ---
        'input_size': (128, 128),    # 网络输入尺寸 (Low-Res)
        # 标签尺寸由 crop_rect 决定: 306 - 50 = 256
        'crop_rect': (50, 306, 50, 306), # (y_s, y_e, x_s, x_e) -> 256x256
        
        'snr_db': 20,                # 目标信噪比 (dB)
        'ricker_freq': 35            # 雷克子波频率
    }
}

# ================= 核心工具函数 =================

def ricker(f, length=0.03, dt=0.001):
    t = np.arange(-length / 2, (length - dt) / 2, dt)
    y = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t ** 2))
    return t, y

def add_gaussian_noise(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    if signal_power == 0: return signal
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, signal.shape)
    return signal + noise

def butter_lowpass_filter_2d(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered[i, :] = filtfilt(b, a, data[i, :])
    for j in range(data.shape[1]):
        filtered[:, j] = filtfilt(b, a, filtered[:, j])
    return filtered

# ================= 地质建模函数 =================

def generate_initial_curve(width):
    x = np.linspace(0, width, width)
    params = sorted([random.uniform(1, 4) for _ in range(3)], reverse=True)
    freqs = sorted([random.uniform(-0.05, 0.05) for _ in range(3)])
    phases = [random.uniform(0, 2 * np.pi) for _ in range(3)]
    
    y1 = params[0] * np.sin(x / 2 * freqs[0] * np.pi + phases[0]) + \
         params[1] * (np.sin(x / 2 * freqs[1] * np.pi + phases[1]) ** 2)
    power = random.choice([1, 2, 3])
    y2 = params[2] * (np.sin(x / 2 * freqs[2] * np.pi + phases[2]) ** power)
    y3 = random.uniform(-10, 10) * x + random.uniform(-10, 10)
    
    curve = 600 * y1 + 500 * y2 + random.randint(8, 20) * y3
    if curve.max() - curve.min() != 0:
        curve = (curve - curve.min()) / (curve.max() - curve.min()) * 10 + 5
    return curve

def generate_velocity_model(height, width, num_layers, interfaces):
    velocity_model = np.zeros((height, width))
    layer_velocities = []
    current_impedance = 100
    for _ in range(num_layers + 1):
        coeff = np.random.uniform(0.05, 0.3) * np.random.choice([-1, 1])
        current_impedance = current_impedance * (1 + coeff) / (1 - coeff)
        layer_velocities.append(current_impedance)
        
    for i in range(len(interfaces) - 1):
        vel = layer_velocities[i]
        start_curve = interfaces[i]
        end_curve = interfaces[i+1]
        for x in range(width):
            start_y = int(np.clip(start_curve[x], 0, height))
            end_y = int(np.clip(end_curve[x], 0, height))
            velocity_model[start_y:end_y, x] = vel
    return velocity_model

def apply_data_augmentation(rc_data, width, height):
    aug_rc = rc_data.copy()
    fault_pos = 40
    aug_rc = np.hstack([aug_rc[:, :fault_pos], aug_rc[:, fault_pos:]]) 
    
    if random.random() < 0.8:
        left_data = np.zeros_like(aug_rc)
        right_data = np.zeros_like(aug_rc)
        a = random.uniform(-5, 5)
        if abs(a) < 1e-3: a = 0.01
        b = random.uniform(-200, 200)
        shear_factor = random.uniform(2, 5)
        
        grid_x = np.arange(width)
        split_indices = (grid_x / a + b).astype(int)
        split_indices = np.clip(split_indices, 0, width)
        
        for i in range(height):
            idx = split_indices[i] if i < len(split_indices) else 0
            left_data[i, :idx] = aug_rc[i, :idx]
            right_data[i, idx:] = aug_rc[i, idx:]
            
        mat_shift = np.float32([[1, 0, shear_factor], [0, 1, a * shear_factor]])
        right_shifted = cv2.warpAffine(right_data, mat_shift, (width, height), flags=cv2.INTER_NEAREST)
        aug_rc = right_shifted + left_data
        
    return aug_rc

# ================= 主程序 =================

if __name__ == "__main__":
    for p in CONFIG['paths'].values():
        if not os.path.exists(p):
            os.makedirs(p)
            
    # 参数提取
    W, H = CONFIG['sim_params']['width'], CONFIG['sim_params']['height']
    target_input_size = CONFIG['sim_params']['input_size'] # (128, 128)
    y_s, y_e, x_s, x_e = CONFIG['sim_params']['crop_rect']
    
    print(f"Start generating {CONFIG['sim_params']['num_simulations']} simulations...")
    print(f"Input Size (Synth): {target_input_size}")
    print(f"Label Size (RC/Pure): ({y_e-y_s}, {x_e-x_s})")
    
    for sim_num in range(1, CONFIG['sim_params']['num_simulations'] + 1):
        
        # 1. 生成几何与模型
        num_layers = np.random.randint(30, 40)
        interfaces = [np.zeros(W), generate_initial_curve(W)]
        for i in range(1, num_layers):
            adjusted = interfaces[-1] + generate_initial_curve(W) + random.randint(-3, 3)
            interfaces.append(adjusted)
        interfaces.append(np.ones(W) * H)
        
        vel_model = generate_velocity_model(H, W, num_layers, interfaces)
        rc_raw = (vel_model - np.roll(vel_model, 1, axis=0)) / (vel_model + np.roll(vel_model, 1, axis=0) + 1e-6)
        max_val = np.max(np.abs(rc_raw))
        rc_raw = rc_raw / max_val if max_val > 0 else rc_raw
        
        # 2. 增强与裁剪 (High Res)
        rc_aug = apply_data_augmentation(rc_raw, W, H)
        rc_label = rc_aug[y_s:y_e, x_s:x_e] # 这里尺寸是 256x256 (High Res)
        
        # 3. 合成记录
        _, wavelet = ricker(CONFIG['sim_params']['ricker_freq'])
        synth_data = np.apply_along_axis(lambda t: np.convolve(t, wavelet, mode='same'), axis=0, arr=rc_label)
        pure_data = synth_data.copy()
        
        # 4. 后处理
        # 滤波
        synth_filtered = butter_lowpass_filter_2d(synth_data.T, 10.0, 100.0, 6).T
        pure_filtered = scipy.ndimage.gaussian_filter(pure_data, sigma=0)
        pure_filtered = butter_lowpass_filter_2d(pure_filtered.T, 15.0, 100.0, 6).T
        
        # --- 关键修正区域 ---
        # RC (Label): 保持裁剪后的尺寸 (256x256)，不进行 Resize
        rc_final = rc_label 
        
        # Pure (Label): 保持裁剪后的尺寸 (256x256)，不进行 Resize
        pure_final = pure_filtered
        
        # Synth (Input): 进行 Resize 到低分辨率 (128x128)
        # 使用 INTER_CUBIC 模拟下采样
        synth_input = cv2.resize(synth_filtered, target_input_size, interpolation=cv2.INTER_CUBIC)
        
        # 5. 添加噪声 (仅对低分 Input 添加)
        synth_final = add_gaussian_noise(synth_input, CONFIG['sim_params']['snr_db'])
        
        # 6. 保存
        savemat(os.path.join(CONFIG['paths']['rc'], f'rc{sim_num}.mat'), {'rc': rc_final})
        savemat(os.path.join(CONFIG['paths']['synth'], f'synth{sim_num}.mat'), {'synth': synth_final})
        savemat(os.path.join(CONFIG['paths']['pure'], f'pure{sim_num}.mat'), {'pure': pure_final})
        
        if sim_num % 10 == 0:
            print(f"Progress: {sim_num} done.")

    print("Processing complete.")
