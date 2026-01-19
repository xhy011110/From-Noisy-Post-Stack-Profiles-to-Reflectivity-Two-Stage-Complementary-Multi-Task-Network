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
    # 输出路径 (已修改为通用相对路径，自动在当前脚本目录下创建)
    'paths': {
        'rc': './dataset/test/rc_random',
        'synth': './dataset/test/synth_random',
        'pure': './dataset/test/pure_random'
    },
    'sim_params': {
        'num_simulations': 50,  # 模拟样本数量
        'width': 500,  # 初始生成宽度
        'height': 500,  # 初始生成高度
        'target_size': (128, 128),  # 最终输出尺寸 (W, H)
        'snr_db': 20,  # 目标信噪比 (dB)
        'ricker_freq': 35,  # 雷克子波频率
        'crop_rect': (50, 306, 50, 306)  # 裁剪区域 (y_start, y_end, x_start, x_end)
    }
}


# ================= 核心工具函数 (Core Functions) =================

def ricker(f, length=0.03, dt=0.001):
    """生成雷克子波"""
    t = np.arange(-length / 2, (length - dt) / 2, dt)
    y = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t ** 2))
    return t, y


def add_gaussian_noise(signal, snr_db):
    """向信号添加指定SNR的高斯噪声"""
    signal_power = np.mean(signal ** 2)
    if signal_power == 0: return signal

    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, signal.shape)
    return signal + noise


def butter_lowpass_filter_2d(data, cutoff, fs, order=5):
    """二维巴特沃斯低通滤波"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    filtered = np.zeros_like(data)
    # 沿行滤波
    for i in range(data.shape[0]):
        filtered[i, :] = filtfilt(b, a, data[i, :])
    # 沿列滤波
    for j in range(data.shape[1]):
        filtered[:, j] = filtfilt(b, a, filtered[:, j])
    return filtered


# ================= 地质建模函数 (Geology Modeling) =================

def generate_initial_curve(width):
    """生成初始地层曲线"""
    x = np.linspace(0, width, width)
    # 随机参数生成混合正弦波
    params = sorted([random.uniform(1, 4) for _ in range(3)], reverse=True)
    freqs = sorted([random.uniform(-0.05, 0.05) for _ in range(3)])
    phases = [random.uniform(0, 2 * np.pi) for _ in range(3)]

    y1 = params[0] * np.sin(x / 2 * freqs[0] * np.pi + phases[0]) + \
         params[1] * (np.sin(x / 2 * freqs[1] * np.pi + phases[1]) ** 2)

    power = random.choice([1, 2, 3])
    y2 = params[2] * (np.sin(x / 2 * freqs[2] * np.pi + phases[2]) ** power)

    # 线性趋势
    y3 = random.uniform(-10, 10) * x + random.uniform(-10, 10)

    # 组合
    curve = 600 * y1 + 500 * y2 + random.randint(8, 20) * y3

    # 归一化
    if curve.max() - curve.min() != 0:
        curve = (curve - curve.min()) / (curve.max() - curve.min()) * 10 + 5
    return curve


def generate_velocity_model(height, width, num_layers, interfaces):
    """基于层位接口生成速度模型"""
    velocity_model = np.zeros((height, width))

    # 预计算每一层的声阻抗/速度
    layer_velocities = []
    current_impedance = 100
    for _ in range(num_layers + 1):
        # 简化反射系数生成逻辑
        coeff = np.random.uniform(0.05, 0.3) * np.random.choice([-1, 1])
        current_impedance = current_impedance * (1 + coeff) / (1 - coeff)
        layer_velocities.append(current_impedance)

    for i in range(len(interfaces) - 1):
        vel = layer_velocities[i]
        start_curve = interfaces[i]
        end_curve = interfaces[i + 1]

        for x in range(width):
            start_y = int(np.clip(start_curve[x], 0, height))
            end_y = int(np.clip(end_curve[x], 0, height))
            velocity_model[start_y:end_y, x] = vel

    return velocity_model


def apply_data_augmentation(rc_data, width, height):
    """应用断层错动和剪切变换"""
    aug_rc = rc_data.copy()

    # 1. 简单的水平错动模拟断层
    fault_pos = 40
    aug_rc = np.hstack([aug_rc[:, :fault_pos], aug_rc[:, fault_pos:]])

    # 2. 随机剪切/错断 (Random Shearing/Faulting)
    if random.random() < 0.8:
        left_data = np.zeros_like(aug_rc)
        right_data = np.zeros_like(aug_rc)

        a = random.uniform(-5, 5)
        if abs(a) < 1e-3: a = 0.01
        b = random.uniform(-200, 200)
        shear_factor = random.uniform(2, 5)

        # 创建分割掩码
        grid_x = np.arange(width)
        split_indices = (grid_x / a + b).astype(int)
        split_indices = np.clip(split_indices, 0, width)

        for i in range(height):
            idx = split_indices[i] if i < len(split_indices) else 0
            left_data[i, :idx] = aug_rc[i, :idx]
            right_data[i, idx:] = aug_rc[i, idx:]

        # 仿射变换 (剪切右侧)
        mat_shift = np.float32([[1, 0, shear_factor], [0, 1, a * shear_factor]])
        right_shifted = cv2.warpAffine(right_data, mat_shift, (width, height), flags=cv2.INTER_NEAREST)
        aug_rc = right_shifted + left_data

    return aug_rc


# ================= 主程序 (Main Execution) =================

if __name__ == "__main__":
    # 1. 初始化路径 (确保目录存在)
    for p in CONFIG['paths'].values():
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"Created directory: {p}")

    # 获取参数
    W, H = CONFIG['sim_params']['width'], CONFIG['sim_params']['height']
    target_W, target_H = CONFIG['sim_params']['target_size']
    y_s, y_e, x_s, x_e = CONFIG['sim_params']['crop_rect']

    print(f"Start generating {CONFIG['sim_params']['num_simulations']} simulations...")

    for sim_num in range(1, CONFIG['sim_params']['num_simulations'] + 1):

        # --- A. 生成几何结构 ---
        num_layers = np.random.randint(30, 40)
        interfaces = [np.zeros(W), generate_initial_curve(W)]

        for i in range(1, num_layers):
            prev = interfaces[-1]
            # 下一层生成逻辑
            new_curve = generate_initial_curve(W)
            separation = random.randint(-3, 3)
            adjusted = prev + new_curve + separation
            interfaces.append(adjusted)

        interfaces.append(np.ones(W) * H)  # 底部边界

        # --- B. 生成速度模型与反射系数 ---
        vel_model = generate_velocity_model(H, W, num_layers, interfaces)

        # 计算反射系数 (纵向差分)
        # rc = (v2 - v1) / (v2 + v1) -> 近似为 shift 后的差分
        rc_raw = (vel_model - np.roll(vel_model, 1, axis=0)) / (vel_model + np.roll(vel_model, 1, axis=0) + 1e-6)
        # 简单的归一化
        max_val = np.max(np.abs(rc_raw))
        rc_raw = rc_raw / max_val if max_val > 0 else rc_raw

        # --- C. 数据增强 (断层/剪切) ---
        rc_aug = apply_data_augmentation(rc_raw, W, H)

        # --- D. 裁剪 (Cropping) ---
        rc_crop = rc_aug[y_s:y_e, x_s:x_e]

        # --- E. 合成地震记录 (Convolution) ---
        _, wavelet = ricker(CONFIG['sim_params']['ricker_freq'])

        # 对每一道(Trace)进行卷积
        synth_data = np.apply_along_axis(lambda t: np.convolve(t, wavelet, mode='same'), axis=0, arr=rc_crop)
        pure_data = synth_data.copy()  # 纯净数据副本

        # --- F. 后处理 (滤波 & Resize) ---
        # 1. 滤波
        synth_filtered = butter_lowpass_filter_2d(synth_data.T, 10.0, 100.0, 6).T
        pure_filtered = scipy.ndimage.gaussian_filter(pure_data, sigma=0)
        pure_filtered = butter_lowpass_filter_2d(pure_filtered.T, 15.0, 100.0, 6).T

        # 2. 统一 Resize (所有数据都必须 Resize 到相同尺寸)
        rc_final = cv2.resize(rc_crop, (target_W, target_H), interpolation=cv2.INTER_NEAREST)
        synth_resized = cv2.resize(synth_filtered, (target_W, target_H), interpolation=cv2.INTER_CUBIC)
        pure_final = cv2.resize(pure_filtered, (target_W, target_H), interpolation=cv2.INTER_CUBIC)

        # --- G. 添加噪声 (Add Noise) ---
        synth_final = add_gaussian_noise(synth_resized, CONFIG['sim_params']['snr_db'])

        # --- H. 保存文件 ---
        savemat(os.path.join(CONFIG['paths']['rc'], f'rc{sim_num}.mat'), {'rc': rc_final})
        savemat(os.path.join(CONFIG['paths']['synth'], f'synth{sim_num}.mat'), {'synth': synth_final})
        savemat(os.path.join(CONFIG['paths']['pure'], f'pure{sim_num}.mat'), {'pure': pure_final})

        if sim_num % 10 == 0:
            print(
                f"Progress: {sim_num}/{CONFIG['sim_params']['num_simulations']} | SNR: {CONFIG['sim_params']['snr_db']}dB")

    print("Processing complete.")