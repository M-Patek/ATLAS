"""
Predictive Attention System - Visualization
预测性注意力系统的可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
import sys
sys.path.append('.')

from predictive_attention_system import (
    PredictiveAgent, SimpleEnvironment, Action, PredictiveField
)


class Visualizer:
    """预测性注意力系统的可视化器"""

    def __init__(self, agent: PredictiveAgent, env: SimpleEnvironment):
        self.agent = agent
        self.env = env

        # 创建图形
        self.fig = plt.figure(figsize=(16, 10))

        # 布局
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 子图1: 世界 + 智能体 + 注意力
        self.ax_world = self.fig.add_subplot(gs[0, 0])
        self.ax_world.set_title('World + Agent + Attention', fontsize=12, fontweight='bold')

        # 子图2: 预测不确定性场
        self.ax_uncertainty = self.fig.add_subplot(gs[0, 1])
        self.ax_uncertainty.set_title('Prediction Uncertainty', fontsize=12, fontweight='bold')

        # 子图3: 访问热力图
        self.ax_visits = self.fig.add_subplot(gs[0, 2])
        self.ax_visits.set_title('Visit Heatmap', fontsize=12, fontweight='bold')

        # 子图4: 预测误差历史
        self.ax_errors = self.fig.add_subplot(gs[1, 0])
        self.ax_errors.set_title('Prediction Error History', fontsize=12, fontweight='bold')

        # 子图5: 注意力分布细节
        self.ax_attention = self.fig.add_subplot(gs[1, 1])
        self.ax_attention.set_title('Current Attention Map', fontsize=12, fontweight='bold')

        # 子图6: 认知指标
        self.ax_metrics = self.fig.add_subplot(gs[1, 2])
        self.ax_metrics.set_title('Cognitive Metrics', fontsize=12, fontweight='bold')

        self.metrics_history = {
            'uncertainty': [],
            'entropy': [],
            'reward': []
        }

    def update(self, frame: int):
        """更新可视化"""
        # 获取当前状态
        state = self.agent.get_cognitive_state()

        # 清空所有子图
        for ax in [self.ax_world, self.ax_uncertainty, self.ax_visits,
                   self.ax_attention]:
            ax.clear()

        self.ax_errors.clear()
        self.ax_metrics.clear()

        # 重新设置标题
        self.ax_world.set_title(f'World + Agent (Step {frame})', fontsize=12, fontweight='bold')

        # 1. 世界视图：显示环境、智能体、注意力中心
        # 背景：世界特征（奖励密度）
        world_bg = self.env.world_features[:, :, 3]  # 奖励维度
        im1 = self.ax_world.imshow(world_bg, cmap='YlGn', origin='lower',
                                    extent=[0, self.env.width, 0, self.env.height])

        # 智能体位置
        agent_x, agent_y = state['position']
        self.ax_world.plot(agent_x, agent_y, 'ro', markersize=15, label='Agent')

        # 轨迹
        if len(state['position_history']) > 1:
            hist = np.array(state['position_history'])
            self.ax_world.plot(hist[:, 0], hist[:, 1], 'r-', alpha=0.3, linewidth=1)

        # 注意力中心
        for i, (cx, cy) in enumerate(state['attention_centers']):
            weight = state['attention_weights'][i] if i < len(state['attention_weights']) else 0.5
            circle = mpatches.Circle((cx, cy), 1.5 + weight * 2,
                                     fill=False, color='blue',
                                     alpha=0.3 + weight * 0.5, linewidth=2)
            self.ax_world.add_patch(circle)

        self.ax_world.set_xlim(0, self.env.width)
        self.ax_world.set_ylim(0, self.env.height)
        self.ax_world.legend(loc='upper left')

        # 2. 预测不确定性场
        uncertainty = np.mean(self.agent.predictive_field.uncertainty_field, axis=2)
        im2 = self.ax_uncertainty.imshow(uncertainty, cmap='hot', origin='lower',
                                          vmin=0, vmax=1)
        self.ax_uncertainty.set_title('Prediction Uncertainty')
        plt.colorbar(im2, ax=self.ax_uncertainty, fraction=0.046)

        # 3. 访问热力图
        visits = self.agent.predictive_field.visit_count
        visit_max = visits.max() if visits.max() > 0 else 1
        im3 = self.ax_visits.imshow(visits, cmap='Blues', origin='lower',
                                     vmin=0, vmax=visit_max)
        self.ax_visits.set_title(f'Visit Count (Total: {visits.sum():.0f})')
        plt.colorbar(im3, ax=self.ax_visits, fraction=0.046)

        # 4. 预测误差历史
        errors = state['prediction_errors']
        if len(errors) > 0:
            self.ax_errors.plot(errors, 'g-', alpha=0.7)
            if len(errors) > 10:
                # 移动平均
                window = min(20, len(errors) // 5)
                ma = np.convolve(errors, np.ones(window)/window, mode='valid')
                self.ax_errors.plot(range(window-1, len(errors)), ma, 'r-', linewidth=2, label='Moving Avg')
            self.ax_errors.set_xlabel('Step')
            self.ax_errors.set_ylabel('Prediction Error')
            self.ax_errors.set_ylim(0, max(errors) * 1.2 if max(errors) > 0 else 1)
            self.ax_errors.legend()
        else:
            self.ax_errors.text(0.5, 0.5, 'No errors yet', ha='center', va='center',
                               transform=self.ax_errors.transAxes)

        # 5. 注意力分布细节
        attention_map = self.agent.attention.get_attention_mask(
            self.env.height, self.env.width
        )
        im5 = self.ax_attention.imshow(attention_map, cmap='viridis', origin='lower')
        self.ax_attention.set_title('Attention Distribution')
        plt.colorbar(im5, ax=self.ax_attention, fraction=0.046)

        # 6. 认知指标
        self.metrics_history['uncertainty'].append(state['field_uncertainty'])
        self.metrics_history['entropy'].append(state['visit_entropy'])

        if len(self.metrics_history['uncertainty']) > 0:
            x = range(len(self.metrics_history['uncertainty']))
            self.ax_metrics.plot(x, self.metrics_history['uncertainty'],
                               'r-', label='Avg Uncertainty', linewidth=2)
            self.ax_metrics.plot(x, self.metrics_history['entropy'],
                               'b-', label='Exploration Entropy', linewidth=2)

            # 归一化到相同尺度
            max_unc = max(self.metrics_history['uncertainty']) if self.metrics_history['uncertainty'] else 1
            max_ent = max(self.metrics_history['entropy']) if self.metrics_history['entropy'] else 1

            self.ax_metrics.set_ylim(0, max(max_unc, max_ent) * 1.1)
            self.ax_metrics.set_xlabel('Step')
            self.ax_metrics.legend()
            self.ax_metrics.grid(True, alpha=0.3)

        return [self.ax_world, self.ax_uncertainty, self.ax_visits,
                self.ax_errors, self.ax_attention, self.ax_metrics]


def run_visualized_simulation(steps: int = 300, save_animation: bool = False):
    """运行带有可视化的模拟"""
    print("=" * 60)
    print("预测性认知智能体 - 可视化模拟")
    print("=" * 60)

    # 创建环境
    env = SimpleEnvironment(width=20, height=20)

    # 创建智能体
    agent = PredictiveAgent(env)

    # 创建可视化器
    viz = Visualizer(agent, env)

    # 模拟步骤数据
    step_data = []
    total_reward = 0

    for step in range(steps):
        pos_before = tuple(env.agent_pos)

        # 智能体循环
        agent.perceive()
        action = agent.think()
        percept, reward, done = agent.act(action)
        agent.learn_from_experience(pos_before, action, percept, reward)

        total_reward += reward

        # 存储数据用于可视化
        step_data.append({
            'step': step,
            'agent': agent,
            'env': env,
            'reward': reward,
            'total_reward': total_reward
        })

        if step % 50 == 0:
            state = agent.get_cognitive_state()
            print(f"\n[Step {step}] 总奖励: {total_reward:.2f}, "
                  f"不确定性: {state['field_uncertainty']:.3f}, "
                  f"探索熵: {state['visit_entropy']:.3f}")

    print(f"\n模拟完成。总计奖励: {total_reward:.3f}")

    # 创建动画
    def update_frame(frame_idx):
        # 恢复到该步骤的状态
        data = step_data[frame_idx]
        viz.agent = data['agent']
        viz.env = data['env']
        return viz.update(frame_idx)

    print("\n正在生成可视化...")

    # 采样帧（每5帧显示一帧）
    frame_indices = range(0, steps, max(1, steps // 100))

    # 保存静态最终状态
    viz.agent = agent
    viz.env = env
    viz.update(steps)
    plt.savefig('predictive_system_final_state.png', dpi=150, bbox_inches='tight')
    print("已保存最终状态图: predictive_system_final_state.png")

    # 显示动画
    anim = FuncAnimation(viz.fig, update_frame, frames=frame_indices,
                        interval=100, blit=False, repeat=True)

    plt.tight_layout()
    plt.show()

    if save_animation:
        print("正在保存动画...（这可能需要几分钟）")
        anim.save('predictive_system_animation.mp4', writer='ffmpeg', fps=10)
        print("已保存动画: predictive_system_animation.mp4")

    return agent, viz


def run_interactive_demo():
    """运行交互式演示"""
    print("=" * 60)
    print("预测性注意力系统 - 交互式演示")
    print("=" * 60)

    env = SimpleEnvironment(width=15, height=15)
    agent = PredictiveAgent(env)

    print("\n系统组件:")
    print(f"  - 预测场: {env.width}x{env.height} 网格")
    print(f"  - 注意力点: {agent.attention.max_spots}")
    print(f"  - 价值函数: 预测后果链评估")

    print("\n运行中（按Ctrl+C停止）...\n")

    total_reward = 0
    step = 0

    try:
        while True:
            pos_before = tuple(env.agent_pos)

            # 智能体循环
            agent.perceive()
            action = agent.think()
            percept, reward, done = agent.act(action)
            agent.learn_from_experience(pos_before, action, percept, reward)

            total_reward += reward
            step += 1

            # 简洁输出
            if step % 30 == 0:
                state = agent.get_cognitive_state()
                print(f"Step {step:4d} | Pos {state['position']} | "
                      f"Action {action.name:10s} | "
                      f"Unc {state['field_uncertainty']:.3f} | "
                      f"Entropy {state['visit_entropy']:.3f} | "
                      f"Reward {total_reward:7.2f}")

    except KeyboardInterrupt:
        print(f"\n\n演示结束。总步数: {step}, 总奖励: {total_reward:.2f}")
        return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predictive Attention System Demo')
    parser.add_argument('--mode', choices=['visual', 'interactive', 'simple'],
                       default='interactive',
                       help='运行模式: visual(可视化动画), interactive(交互式), simple(简单运行)')
    parser.add_argument('--steps', type=int, default=300,
                       help='模拟步数')

    args = parser.parse_args()

    if args.mode == 'visual':
        run_visualized_simulation(steps=args.steps)
    elif args.mode == 'interactive':
        run_interactive_demo()
    else:
        # 简单模式
        from predictive_attention_system import run_simulation
        run_simulation(steps=args.steps)
