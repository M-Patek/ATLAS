"""
Tool Integration Module - 工具整合模块

实现"上手"状态（ready-to-hand）：
- 工具使用前：工具是对象，需要意识操作（高维空间）
- 工具使用中：工具成为身体延伸，动作透明（空间降维）

关键概念：
- 工具 = 空间的局部共形变换
- 内化程度（embodiment_level）：0=对象, 1=身体一部分
- 可达集（reachable_set）：工具能到达的状态
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Set, Tuple, Dict, Optional, Callable
from enum import Enum
import copy


@dataclass
class ToolCapability:
    """工具能力描述"""
    # 工具能执行的动作类型
    action_types: Set[str]
    # 作用范围（相对于持有位置）
    reach_radius: float
    # 精度（0-1，1=精确）
    precision: float
    # 能量消耗
    energy_cost: float


class ToolState(Enum):
    """工具的认知状态"""
    OBJECT = "object"           # 对象状态：需要注意操作
    TRANSITION = "transition"   # 转换状态：学习中
    READY_TO_HAND = "embodied"  # 上手状态：透明使用


class Tool:
    """
    工具类

    工具不是独立物体，而是改变空间形状的算子。
    """
    def __init__(self,
                 tool_id: str,
                 name: str,
                 capability: ToolCapability,
                 initial_embodiment: float = 0.0):
        self.id = tool_id
        self.name = name
        self.capability = capability

        # 内化程度：0=完全陌生, 1=完全内化（身体一部分）
        self.embodiment_level = initial_embodiment

        # 使用统计
        self.usage_count = 0
        self.success_count = 0

        # 认知状态
        self.state = ToolState.OBJECT if initial_embodiment < 0.3 else \
                    ToolState.READY_TO_HAND if initial_embodiment > 0.8 else \
                    ToolState.TRANSITION

        # 工具坐标系到身体坐标系的映射（学习得来）
        self.coord_mapping: Optional[Callable] = None

    def update_embodiment(self, success: bool, prediction_error: float):
        """
        更新内化程度

        类比：练习使用工具，逐渐变得"自然"
        """
        self.usage_count += 1

        if success:
            self.success_count += 1
            # 成功使用增加内化
            increment = 0.1 * (1.0 - prediction_error)
            self.embodiment_level = min(1.0, self.embodiment_level + increment)
        else:
            # 失败降低内化
            self.embodiment_level = max(0.0, self.embodiment_level - 0.2)

        # 更新状态
        if self.embodiment_level < 0.3:
            self.state = ToolState.OBJECT
        elif self.embodiment_level > 0.8:
            self.state = ToolState.READY_TO_HAND
        else:
            self.state = ToolState.TRANSITION

    def get_effective_metric_factor(self) -> float:
        """
        获取有效度量因子

        内化程度高 -> 距离感觉更短（空间压缩）
        """
        # 内化度0.8时，距离感觉缩短40%
        return 1.0 - self.embodiment_level * 0.5

    def get_attention_demand(self) -> float:
        """
        获取注意力需求

        内化程度低 -> 需要更多注意力（意识参与）
        """
        return 1.0 - self.embodiment_level

    def __repr__(self):
        return f"Tool({self.name}, embodiment={self.embodiment_level:.2f}, state={self.state.value})"


class ToolEquippedEnvironment:
    """
    支持工具的环境

    环境空间会根据当前工具动态变形
    """
    def __init__(self, base_env):
        self.base_env = base_env

        # 代理基础环境的属性
        self.width = base_env.width
        self.height = base_env.height
        self.perception_dim = base_env.perception_dim
        self.world_features = base_env.world_features
        self.reward_regions = base_env.reward_regions

        # 可用工具
        self.available_tools: Dict[str, Tool] = {}

        # 当前持有的工具
        self.held_tool: Optional[Tool] = None

        # 原始空间（无工具）
        self.original_space = self._capture_space_state()

        # 变形后的空间（有工具时）
        self.transformed_space = None

        # 工具放置位置（工具作为环境中的对象）
        self.tool_locations: Dict[str, Tuple[float, float]] = {}

    @property
    def agent_pos(self):
        """代理agent_pos属性"""
        return self.base_env.agent_pos

    @agent_pos.setter
    def agent_pos(self, value):
        """设置agent_pos"""
        self.base_env.agent_pos = value

    def _capture_space_state(self):
        """捕获当前空间状态"""
        return {
            'agent_pos': self.base_env.agent_pos.copy(),
            'world_features': self.base_env.world_features.copy()
        }

    def register_tool(self, tool: Tool, location: Tuple[float, float]):
        """注册工具及其位置"""
        self.available_tools[tool.id] = tool
        self.tool_locations[tool.id] = location

    def pickup_tool(self, tool_id: str) -> bool:
        """
        拾取工具

        拾取后，空间开始变形
        """
        if tool_id not in self.available_tools:
            return False

        # 检查是否在工具附近
        agent_pos = tuple(self.base_env.agent_pos)
        tool_pos = self.tool_locations[tool_id]
        distance = np.sqrt((agent_pos[0] - tool_pos[0])**2 +
                          (agent_pos[1] - tool_pos[1])**2)

        if distance > 2.0:  # 太远
            return False

        self.held_tool = self.available_tools[tool_id]
        self._apply_spatial_transformation()
        return True

    def drop_tool(self):
        """放下工具，空间恢复原状"""
        self.held_tool = None
        self.transformed_space = None
        self._restore_original_space()

    def _apply_spatial_transformation(self):
        """
        应用空间变换

        这是核心：工具内化程度高时，空间局部压缩
        """
        if self.held_tool is None:
            return

        # 获取工具的内化程度
        emb = self.held_tool.embodiment_level
        metric_factor = self.held_tool.get_effective_metric_factor()

        # 创建变形后的世界特征
        # 策略：在工具可达范围内，奖励信号被放大（感觉更近）
        transformed_features = self.base_env.world_features.copy()

        agent_pos = self.base_env.agent_pos
        reach = self.held_tool.capability.reach_radius

        # 局部空间变形：在工具范围内，特征被"拉近"
        for y in range(self.base_env.height):
            for x in range(self.base_env.width):
                dist = np.sqrt((x - agent_pos[0])**2 + (y - agent_pos[1])**2)

                if dist < reach * 2:  # 工具影响范围
                    # 距离感知被压缩
                    effective_dist = dist * (1.0 - emb * 0.3)

                    # 在变形空间中，远处的目标感觉更近
                    # 这通过调整特征强度实现
                    if effective_dist < reach:
                        # 工具可达范围内：奖励增强（更容易到达的感觉）
                        boost = 1.0 + emb * 0.5
                        transformed_features[y, x, 3] *= boost  # 奖励维度

        self.transformed_space = transformed_features

    def _restore_original_space(self):
        """恢复原始空间"""
        self.transformed_space = None

    def get_percept(self, x=None, y=None) -> 'Percept':
        """
        获取感知（考虑工具变形）
        """
        import sys
        sys.path.append('.')
        from multi_scale_predictive_system import Percept

        if x is None:
            x, y = self.base_env.agent_pos

        x_int, y_int = int(x), int(y)

        if not (0 <= x_int < self.base_env.width and 0 <= y_int < self.base_env.height):
            # 返回空感知
            return Percept(
                local_features=np.zeros(self.base_env.perception_dim),
                proprioception=np.array([x, y]),
                prediction_error=1.0
            )

        # 选择使用原始空间还是变形空间
        features = self.transformed_space if self.transformed_space is not None \
                   else self.base_env.world_features

        local = features[y_int, x_int].copy()
        local += np.random.randn(self.base_env.perception_dim) * 0.05
        local = np.clip(local, 0, 1)

        # 如果有工具，添加工具感知
        if self.held_tool is not None:
            # 工具内化程度影响感知
            # 内化度高时，工具"消失"在感知中（上手状态）
            tool_salience = 1.0 - self.held_tool.embodiment_level
            local = np.concatenate([local, [tool_salience]])

        return Percept(
            local_features=local,
            proprioception=np.array([x, y]),
            prediction_error=0.05
        )

    def step_with_tool(self, action: 'Action', agent) -> Tuple['Percept', float, bool]:
        """
        执行动作（考虑工具效果）

        有工具时，工具能力被激活
        """
        import sys
        sys.path.append('.')
        from multi_scale_predictive_system import Action

        base_action = action
        reward_multiplier = 1.0

        if self.held_tool is not None:
            tool = self.held_tool

            # 工具扩展动作空间
            # 例如，INTERACT动作在有工具时效果增强
            if action == Action.INTERACT:
                # 工具内化程度影响效果
                reward_multiplier = 1.0 + tool.embodiment_level

                # 在工具可达范围内搜索奖励
                agent_pos = self.base_env.agent_pos
                reach = tool.capability.reach_radius

                # 标记这次工具使用（用于学习）
                agent.record_tool_usage(tool.id, success=False)  # 先标记，成功后更新

            # 工具提高效率（减少移动成本感）
            elif action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
                # 内化程度高时，移动感觉更轻松
                pass

        # 执行基础动作
        percept, base_reward, done = self.base_env.step(base_action, agent)

        # 应用工具效果
        reward = base_reward * reward_multiplier

        return percept, reward, done

    def get_tool_status(self) -> Dict:
        """获取工具状态信息"""
        return {
            'held_tool': self.held_tool.name if self.held_tool else None,
            'embodiment_level': self.held_tool.embodiment_level if self.held_tool else 0.0,
            'tool_state': self.held_tool.state.value if self.held_tool else None,
            'space_transformed': self.transformed_space is not None,
            'available_tools': list(self.available_tools.keys())
        }

    def step(self, action, agent):
        """代理step方法到底层环境"""
        return self.base_env.step(action, agent)


class ToolAwareAgent:
    """
    工具感知智能体

    扩展多尺度智能体，增加工具使用能力
    """
    def __init__(self, multi_scale_agent):
        self.agent = multi_scale_agent
        self.tool_usage_history: Dict[str, list] = {}
        self.current_tool_target: Optional[str] = None

    def record_tool_usage(self, tool_id: str, success: bool):
        """记录工具使用"""
        if tool_id not in self.tool_usage_history:
            self.tool_usage_history[tool_id] = []
        self.tool_usage_history[tool_id].append(success)

    def decide_tool_action(self, env: ToolEquippedEnvironment) -> Optional[str]:
        """
        决定是否获取/使用工具

        策略：
        1. 如果远处有高奖励目标，找工具
        2. 如果工具内化程度高，优先使用
        3. 如果预测误差高，尝试工具
        """
        current_pos = tuple(env.base_env.agent_pos)

        # 检查附近是否有工具
        nearest_tool = None
        nearest_dist = float('inf')

        for tool_id, location in env.tool_locations.items():
            dist = np.sqrt((current_pos[0] - location[0])**2 +
                          (current_pos[1] - location[1])**2)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_tool = tool_id

        # 决策逻辑
        if env.held_tool is not None:
            # 已有工具，检查是否内化
            if env.held_tool.embodiment_level > 0.7:
                # 内化程度高，继续使用
                return None  # None表示使用当前工具
            elif env.held_tool.usage_count > 10 and env.held_tool.embodiment_level < 0.3:
                # 用了多次还没内化，放弃
                return 'DROP'
        else:
            # 没有工具，考虑获取
            if nearest_tool and nearest_dist < 5.0:
                # 检查是否需要工具（远处有目标）
                if self._distant_reward_exists(env):
                    return f'PICKUP_{nearest_tool}'

        return None

    def _distant_reward_exists(self, env: ToolEquippedEnvironment) -> bool:
        """检查远处是否有高奖励"""
        agent_pos = env.base_env.agent_pos

        # 搜索范围内的高奖励
        for y in range(env.base_env.height):
            for x in range(env.base_env.width):
                reward = env.base_env.world_features[y, x, 3]
                if reward > 0.5:
                    dist = np.sqrt((x - agent_pos[0])**2 + (y - agent_pos[1])**2)
                    if dist > 3.0:  # 远处
                        return True
        return False


def demo_tool_integration():
    """
    演示工具整合
    """
    import sys
    sys.path.append('.')
    from multi_scale_predictive_system import SimpleEnvironment, MultiScalePredictiveAgent

    print("=" * 70)
    print("工具整合演示")
    print("=" * 70)
    print()
    print("场景：智能体需要到达远处的高奖励区域")
    print("工具：'extender' - 扩展可达范围，内化后远处的目标感觉更近")
    print()

    # 创建基础环境
    base_env = SimpleEnvironment(width=20, height=20)

    # 在远处放置高奖励
    base_env.world_features[5, 15, 3] = 2.0  # 位置(15,5)的高奖励
    base_env.reward_regions.append((15, 5, 1.5, np.array([1.0, 0.0, 0.0])))

    # 创建工具环境
    tool_env = ToolEquippedEnvironment(base_env)

    # 创建工具
    extender = Tool(
        tool_id='extender',
        name='Reach Extender',
        capability=ToolCapability(
            action_types={'REACH', 'GRAB'},
            reach_radius=5.0,
            precision=0.8,
            energy_cost=0.1
        ),
        initial_embodiment=0.0  # 初始完全陌生
    )

    # 注册工具（放在起点附近）
    tool_env.register_tool(extender, (10, 10))

    # 创建智能体
    base_agent = MultiScalePredictiveAgent(base_env)
    agent = ToolAwareAgent(base_agent)

    print(f"工具位置: {tool_env.tool_locations['extender']}")
    print(f"高奖励目标位置: (15, 5)")
    print(f"初始距离: {np.sqrt((15-10)**2 + (5-10)**2):.2f}")
    print()

    # 阶段1：无工具，尝试到达远处
    print("【阶段1】无工具尝试（困难）")
    print("-" * 50)

    for step in range(30):
        agent.agent.perceive()
        action, meta = agent.agent.think_and_act()
        percept, reward, done = tool_env.step_with_tool(action, agent)
        agent.agent.execute_and_learn(action, meta)

        if step % 10 == 0:
            pos = tool_env.base_env.agent_pos
            print(f"Step {step}: 位置({pos[0]:.0f}, {pos[1]:.0f}), "
                  f"奖励{reward:.2f}, 层{meta['layer']}")

    print()
    print("【阶段2】获取工具")
    print("-" * 50)

    # 移动到工具位置并拾取
    success = tool_env.pickup_tool('extender')
    print(f"拾取工具: {success}")
    print(f"工具状态: {tool_env.held_tool}")
    print()

    # 阶段3：使用工具，学习内化
    print("【阶段3】使用工具学习（内化过程）")
    print("-" * 50)

    # 更新智能体的环境引用
    agent.agent.env = tool_env

    for step in range(30, 100):
        # 感知（现在通过工具环境）
        percept = tool_env.get_percept()
        agent.agent.current_percept = percept

        # 思考
        action, meta = agent.agent.think_and_act()

        # 执行
        percept, reward, done = tool_env.step_with_tool(action, agent)

        # 学习
        agent.agent.execute_and_learn(action, meta)

        # 更新工具内化
        if tool_env.held_tool and reward > 0.5:
            tool_env.held_tool.update_embodiment(success=True, prediction_error=0.1)
            agent.record_tool_usage('extender', success=True)

        if step % 15 == 0:
            pos = tool_env.base_env.agent_pos
            tool_status = tool_env.get_tool_status()
            print(f"Step {step}: 位置({pos[0]:.0f}, {pos[1]:.0f}), "
                  f"内化度{tool_status['embodiment_level']:.2f}, "
                  f"状态{tool_status['tool_state']}")

    print()
    print("【阶段4】工具内化后（透明使用）")
    print("-" * 50)

    for step in range(100, 150):
        percept = tool_env.get_percept()
        agent.agent.current_percept = percept
        action, meta = agent.agent.think_and_act()
        percept, reward, done = tool_env.step_with_tool(action, agent)
        agent.agent.execute_and_learn(action, meta)

        if step % 20 == 0:
            pos = tool_env.base_env.agent_pos
            awareness = agent.agent.get_cognitive_state()['awareness_level']
            print(f"Step {step}: 位置({pos[0]:.0f}, {pos[1]:.0f}), "
                  f"意识水平{awareness:.2f}, 奖励{reward:.2f}")

    print()
    print("=" * 70)
    print("演示完成")
    print(f"最终工具内化度: {tool_env.held_tool.embodiment_level:.2f}")
    print(f"总工具使用次数: {tool_env.held_tool.usage_count}")
    print("=" * 70)


if __name__ == "__main__":
    demo_tool_integration()
