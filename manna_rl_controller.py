"""
Manna Nanorobot Reinforcement Learning Controller
Model: manna_core.xml
Framework: Gymnasium & Stable-Baselines3 (PPO)
Physics: Low Reynolds Number (Re ~ 10^-5) Stokesian Propulsion with RFT Cross-Coupling
"""

import os
import tempfile
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# ==============================================================================
# 1. MJCF 物理模型定义 (内嵌完整规范)
# ==============================================================================
MANNA_XML = """
<mujoco model="manna_nanorobot">
    <compiler coordinate="local" angle="radian" inertiafromgeom="auto"/>
    
    <option gravity="0 0 0" integrator="RK4" timestep="0.0002">
        <flag energy="enable"/>
    </option>

    <default>
        <joint damping="1.5e-2"/>
        <geom density="1050" friction="0.1 0.005 0.001"/>
    </default>

    <worldbody>
        <light pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        
        <!-- Nanorobot 主体 (特征长度 L ≈ 5 um, 坐标缩放 1 unit = 1 um) -->
        <body name="chassis" pos="0 0 0">
            <freejoint name="root"/>
            <geom name="capsule_body" type="ellipsoid" size="2.5 1.2 1.2" rgba="0.2 0.7 0.9 0.8"/>
            <site name="mass_center" pos="0 0 0" size="0.1"/>

            <!-- 推进马达转子：刚性微螺旋尾纤接口 -->
            <body name="flagellar_motor" pos="-2.5 0 0">
                <joint name="propeller_joint" type="hinge" axis="1 0 0" damping="8.0e-3"/>
                <geom name="helix_seg_1" type="capsule" fromto="0 0 0 -0.6 0.5 0" size="0.08" rgba="0.8 0.8 0.2 1"/>
                <geom name="helix_seg_2" type="capsule" fromto="-0.6 0.5 0 -1.2 0 0.5" size="0.08" rgba="0.8 0.8 0.2 1"/>
                <geom name="helix_seg_3" type="capsule" fromto="-1.2 0 0.5 -1.8 -0.5 0" size="0.08" rgba="0.8 0.8 0.2 1"/>
                <geom name="helix_seg_4" type="capsule" fromto="-1.8 -0.5 0 -2.4 0 -0.5" size="0.08" rgba="0.8 0.8 0.2 1"/>
                <geom name="helix_seg_5" type="capsule" fromto="-2.4 0 -0.5 -3.0 0.5 0" size="0.08" rgba="0.8 0.8 0.2 1"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor joint="propeller_joint" name="magnetic_torque_drive" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
    </actuator>
</mujoco>
"""

# ==============================================================================
# 2. Gymnasium 环境封装 (带阻力理论 RFT 与微尺度物理动力学)
# ==============================================================================
class MannaNanoEnv(gym.Env):
    """
    针对低雷诺数微纳机器人 '吗哪 (Manna)' 的强化学习交互环境。
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # 编译 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_string(MANNA_XML)
        self.data = mujoco.MjData(self.model)

        # 尺度缩放系数与物理常数
        self.SCALE_L = 1e6     # 1 unit (sim) = 1e-6 m (real)
        self.SCALE_M = 1e9     # 1 unit (sim) = 1e-9 kg (real)
        self.ETA_WATER = 1e-3  # 水在常温下的动力粘度: 1.0 mPa·s
        self.TAU_SHEAR_MAX = 0.05  # 剥离临界剪切应力: 0.05 N/m^2 (Pa)

        # 阻力理论 (Resistive Force Theory, RFT) 手性推进耦合系数
        # 在 Stokes 流下: Fx = K_thrust * omega_helix - Gamma_trans * vx
        self.K_THRUST = 4.2e-2      # 螺旋各向异性手性推进耦合系数 (Sim Units)
        self.GAMMA_TRANS = 8.5e-2   # 前进方向平移平流粘滞阻力系数
        self.HELIX_RADIUS = 0.5e-6  # 螺旋有效半径: 0.5 um
        self.BL_THICKNESS = 0.1e-6  # 酶催化层流体特征边界层厚度: 0.1 um

        # 索引与句柄定位
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "magnetic_torque_drive")
        self.motor_qvel_idx = 6  # root freejoint 占用 0~5, 螺旋铰链为索引 6

        # 动作空间: 电磁驱动扭矩 ctrl ∈ [-10, 10]
        ctrl_limit = self.model.actuator_ctrlrange[self.actuator_id]
        self.action_space = spaces.Box(
            low=float(ctrl_limit[0]),
            high=float(ctrl_limit[1]),
            shape=(1,),
            dtype=np.float32
        )

        # 状态空间 (8维):
        # [vx, vy, vz, wx, wy, wz, E_k, omega_propeller]
        # 所有量均为当前模拟尺度下的物理观测值
        obs_high = np.array([np.finfo(np.float32).max] * 8, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.max_episode_steps = 1500
        self.current_step = 0

    def _compute_shear_stress(self, omega_prop):
        """
        计算表面酶催化层所承受的流体剪切应力 (SI 真实单位: N/m^2)。
        剪切应力 tau = eta * (v_tip / delta) = eta * (|omega| * R / delta)
        """
        # omega_prop 为 rad/s (因时间比例尺为 1:1)
        v_tip = np.abs(omega_prop) * self.HELIX_RADIUS
        tau_shear = self.ETA_WATER * (v_tip / self.BL_THICKNESS)
        return float(tau_shear)

    def _get_obs(self):
        # 质心线速度 (qvel[0:3]) 与 角速度 (qvel[3:6])
        lin_vel = self.data.qvel[0:3].copy()  # vx, vy, vz (sim units/s = um/s)
        ang_vel = self.data.qvel[3:6].copy()  # wx, wy, wz (rad/s)
        
        # 动力学能量 (Sim 单位: 10^-21 J)
        mujoco.mj_energyPos(self.model, self.data)
        mujoco.mj_energyVel(self.model, self.data)
        kinetic_energy = float(self.data.energy[1])
        
        omega_prop = float(self.data.qvel[self.motor_qvel_idx])

        obs = np.array([
            lin_vel[0], lin_vel[1], lin_vel[2],
            ang_vel[0], ang_vel[1], ang_vel[2],
            kinetic_energy,
            omega_prop
        ], dtype=np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        ctrl_torque = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        self.data.ctrl[self.actuator_id] = ctrl_torque

        # --- 核心 Stokes 流动力学耦合 (RFT 注入) ---
        omega_helix = self.data.qvel[self.motor_qvel_idx]
        vx_sim = self.data.qvel[0]

        # 计算并向主体施加净手性推进力与平移流体阻力
        hydrodynamic_thrust = self.K_THRUST * omega_helix - self.GAMMA_TRANS * vx_sim
        self.data.xfrc_applied[self.body_id, 0] = hydrodynamic_thrust

        # 执行物理积分步
        mujoco.mj_step(self.model, self.data)

        # 观测值解析
        obs = self._get_obs()
        vx, vy, vz = obs[0], obs[1], obs[2]
        wx, wy, wz = obs[3], obs[4], obs[5]
        omega_prop = obs[7]

        # --- 核心奖励函数 (Reward Function) ---
        # 1. 主轴推进正奖励 (um/s)
        r_forward = 3.5 * vx

        # 2. 侧向平移漂移与轴向侧摆惩罚 (保持纯直线稳定推进)
        r_lateral = -1.2 * (vy**2 + vz**2) - 0.8 * (wy**2 + wz**2)

        # 3. 电磁驱动能量消耗惩罚: 真实电磁做功功率 P = |tau * omega|
        sim_power = np.abs(ctrl_torque * omega_prop)
        r_ctrl = -0.05 * sim_power

        # 4. 界面雷诺/剪切应力超限惩罚 (避免外壳固载酶脱落, tau > 0.05 N/m^2)
        tau_shear = self._compute_shear_stress(omega_prop)
        r_shear_penalty = 0.0
        if tau_shear > self.TAU_SHEAR_MAX:
            # 引入二次壁面惩罚
            excess = (tau_shear - self.TAU_SHEAR_MAX) / self.TAU_SHEAR_MAX
            r_shear_penalty = -25.0 * (excess ** 2)

        total_reward = float(r_forward + r_lateral + r_ctrl + r_shear_penalty)

        # 终止条件
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        # 真实 SI 物理量与指标统计
        real_power_watts = sim_power * 1e-21  # Sim 做功功率转 SI 瓦特
        info = {
            "vx_um_per_s": float(vx),
            "sim_power": float(sim_power),
            "real_power_W": float(real_power_watts),
            "shear_stress_Pa": float(tau_shear),
            "shear_violation": bool(tau_shear > self.TAU_SHEAR_MAX)
        }

        return obs, total_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # 施加微弱的热布朗运动扰动初态
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.normal(0, 1e-3, size=self.model.nv)
        self.data.qvel[:] = noise
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

# ==============================================================================
# 3. 强化学习训练与全流程基准评估
# ==============================================================================
def main():
    print("=" * 80)
    print("初始化 '吗哪 (Manna)' 纳米机器人 RL 控制工程环境...")
    print("特征尺度: L ≈ 5 μm | Stokes 流机制 | 临界剪切应力阈值: 0.05 N/m²")
    print("=" * 80)

    # 1. 实例化与环境自检
    env = MannaNanoEnv()
    check_env(env, warn=True)
    print("[✓] Gymnasium 环境规范检查通过。\n")

    # 2. PPO 超参数调优配置
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="auto"
    )

    # 3. 执行策略训练 (快速微型基准与测试验证: 1000 steps)
    TEST_TIMESTEPS = 1000
    print(f"开始执行微型强化学习训练验证，训练步数: {TEST_TIMESTEPS} steps...")
    model.learn(total_timesteps=TEST_TIMESTEPS)
    print(f"\n[✓] 训练完成！成功完成 {TEST_TIMESTEPS} 步 PPO 训练验证。\n")

    # 4. 评估闭环控制策略性能与综合能效
    eval_episodes = 3
    total_vx = []
    total_powers = []
    total_shears = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=42 + ep)
        ep_vx, ep_power, ep_shear = [], [], []
        
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            
            ep_vx.append(info["vx_um_per_s"])
            ep_power.append(info["real_power_W"])
            ep_shear.append(info["shear_stress_Pa"])
            
            if term or trunc:
                break
                
        total_vx.append(np.mean(ep_vx))
        total_powers.append(np.mean(ep_power))
        total_shears.append(np.mean(ep_shear))

    # 指标统计与打印
    avg_vx = np.mean(total_vx)                  # μm/s
    avg_power_W = np.mean(total_powers)         # Watts (真实物理值)
    avg_shear = np.mean(total_shears)           # N/m^2 (Pa)
    
    # 微纳机器人能效比: (μm/s) / (Femtowatt: 1e-15 W)
    avg_power_fW = avg_power_W * 1e15
    propulsion_efficiency = avg_vx / (avg_power_W + 1e-30)

    print("=" * 80)
    print("               【吗哪 (Manna) 纳米机器人 - 动力学基准评估结果】")
    print("=" * 80)
    print(f"平均轴向推进速度 (Average Forward Velocity)  : {avg_vx:12.4f} μm/s")
    print(f"平均机械做功功耗 (Average Power Consumption) : {avg_power_W:12.4e} W ({avg_power_fW:.2f} fW)")
    print(f"平均壁面剪切应力 (Average Wall Shear Stress) : {avg_shear:12.5f} N/m²")
    print(f"剪切应力安全冗余 (Safety Boundary Status)    : {'[安全]' if avg_shear <= 0.05 else '[超标危险]'}")
    print("-" * 80)
    print(f"综合推进效率 (Metric: Velocity / Real Power)  : {propulsion_efficiency:12.4e} (μm/s) / W")
    print(f"工程标准化效率 (Velocity per Femtowatt)       : {(avg_vx / (avg_power_fW + 1e-12)):12.4f} (μm/s) / fW")
    print("=" * 80)

    # 5. 按照需求显式补充的实例化与微型训练独立测试代码段
    print("\n" + "=" * 80)
    print("【补充验证模块】独立实例化 MannaNanoEnv 与 PPO 1000 步训练测试")
    print("=" * 80)
    test_env = MannaNanoEnv()
    test_model = PPO("MlpPolicy", test_env, verbose=0)
    test_model.learn(total_timesteps=1000)
    print("[SUCCESS] MannaNanoEnv 1000 步微型强化学习训练成功！模型与环境运行正常。")
    print("=" * 80)

if __name__ == "__main__":
    main()
