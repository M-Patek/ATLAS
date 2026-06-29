"""
Continuous SSFR Kitchen Integration - 连续SSFR与物理厨房的集成

完全移除离散网格，所有操作在连续坐标上进行�?"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
import time

from src.kitchen import (
    Kitchen, Robot, PhysicsConfig, ObjectType, RobotAction,
    SensorSystem, KitchenSSFRInterface, TASK_LIBRARY, Task, TaskStep,
    create_demo_kitchen,
)
from src.spaces.continuous import (
    ContinuousCognitiveSpace,
    ContinuousRicciSpace,
    ContinuousFisherSpace,
    ContinuousWassersteinSpace,
)
from src.spaces.continuous_ssfr import ContinuousSSFR


# ============================================================================
# 1. 连续空间适配�?# ============================================================================

class ContinuousKitchenAdapter:
    """连续厨房适配�?- 无需离散网格"""

    def __init__(self, kitchen: Kitchen):
        self.kitchen = kitchen

    def encode_observation(self, robot_id: str, task: Optional[Task] = None) -> Dict[str, Any]:
        """编码物理观测（连续坐标）"""
        if robot_id not in self.kitchen.robots:
            return {}

        robot = self.kitchen.robots[robot_id]
        sensor = SensorSystem(robot, self.kitchen.space)

        # 获取附近物体
        all_entities = list(self.kitchen.entities.values())
        nearby = sensor.get_nearby_objects(all_entities, max_distance=3.0)

        # 障碍物（连续坐标�?        obstacles = []
        for obj, dist in nearby:
            if dist < 1.5 and obj.properties.obj_type != ObjectType.ROBOT:
                obstacles.append(obj.position)

        # 目标位置
        goal_position = None
        if task and task.get_current_step():
            target_name = task.get_current_step().target
            if target_name:
                for obj in self.kitchen.objects.values():
                    if target_name.lower() in obj.properties.name.lower():
                        goal_position = obj.position
                        break

                if goal_position is None:
                    # 预定义位�?                    predefined = {
                        'coffee_machine': (1.5, 6.5),
                        'microwave': (3.0, 6.5),
                        'sink': (8.0, 6.5),
                        'stove': (5.0, 6.5),
                        'table': (5.0, 3.0),
                    }
                    goal_position = predefined.get(target_name.lower())

        # 计算不确定�?        velocity = robot.body.velocity
        speed = math.sqrt(velocity.x**2 + velocity.y**2)
        speed_uncertainty = min(speed / PhysicsConfig.ROBOT_MAX_SPEED, 1.0)

        density_uncertainty = min(len(nearby) / 10.0, 1.0)

        goal_uncertainty = 0.0
        if goal_position:
            dist_to_goal = math.sqrt(
                (robot.position[0] - goal_position[0])**2 +
                (robot.position[1] - goal_position[1])**2
            )
            goal_uncertainty = min(dist_to_goal / 10.0, 1.0)

        uncertainty = (speed_uncertainty + density_uncertainty + goal_uncertainty) / 3.0

        return {
            'position': robot.position,
            'goal_position': goal_position,
            'obstacles': obstacles,
            'uncertainty': uncertainty,
            'velocity': (velocity.x, velocity.y),
            'speed': speed,
            'nearby_objects_count': len(nearby),
            'robot_angle': robot.angle,
            'task_step': task.get_current_step().action if task and task.get_current_step() else None,
            'task_target': task.get_current_step().target if task and task.get_current_step() else None,
            'step_size': 0.5,
        }


# ============================================================================
# 2. 连续物理SSFR
# ============================================================================

class ContinuousPhysicalSSFR:
    """连续物理SSFR - 无需网格"""

    def __init__(self, kitchen: Kitchen,
                 space_names: List[str] = None):
        self.kitchen = kitchen
        self.adapter = ContinuousKitchenAdapter(kitchen)

        # 初始化连续SSFR
        space_names = space_names or ['ricci', 'fisher', 'wasserstein']
        self.ssfr = ContinuousSSFR(
            space_names=space_names,
            max_structures=50,
            evolution_interval=20,
        )

        # 机器人状�?        self.robot_states: Dict[str, Dict[str, Any]] = {}

        # 统计
        self.stats = {
            'perception_count': 0,
            'competition_count': 0,
            'total_time': 0.0,
        }

    def perceive(self, robot_id: str, task: Optional[Task] = None,
                 active_space: Optional[str] = None) -> List[Any]:
        """感知"""
        start = time.time()

        observation = self.adapter.encode_observation(robot_id, task)
        if not observation:
            return []

        position = observation['position']

        hypotheses = self.ssfr.perceive(
            position=position,
            observation=observation,
            active_space_name=active_space
        )

        self.robot_states[robot_id] = {
            'observation': observation,
            'hypotheses': [h.id for h in hypotheses],
            'timestamp': self.kitchen.time,
        }

        self.stats['perception_count'] += 1
        self.stats['total_time'] += time.time() - start

        return hypotheses

    def compete(self, robot_id: str, actual_position: Tuple[float, float]) -> Optional[Any]:
        """竞争"""
        if robot_id not in self.robot_states:
            return None

        observation = self.robot_states[robot_id]['observation']
        actual = {'position': actual_position}

        winner = self.ssfr.compete(observation, actual)
        self.stats['competition_count'] += 1

        return winner

    def get_space_validity(self, robot_id: str) -> Dict[str, float]:
        """获取空间有效�?""
        validities = {}
        for name, space in self.ssfr.spaces.items():
            if robot_id in self.robot_states:
                obs = self.robot_states[robot_id]['observation']
                pos = obs['position']
                validities[name] = space.compute_validity(pos, obs)
            else:
                validities[name] = 0.5
        return validities

    def get_statistics(self) -> Dict[str, Any]:
        """统计"""
        ssfr_stats = self.ssfr.get_statistics()
        return {
            **ssfr_stats,
            'physical_perceptions': self.stats['perception_count'],
            'physical_competitions': self.stats['competition_count'],
            'avg_time_per_step': self.stats['total_time'] / max(1, self.stats['perception_count']),
        }


# ============================================================================
# 3. 连续任务规划�?# ============================================================================

class ContinuousSSFRTaskPlanner:
    """连续SSFR任务规划�?""

    def __init__(self, physical_ssfr: ContinuousPhysicalSSFR):
        self.physical_ssfr = physical_ssfr
        self.kitchen = physical_ssfr.kitchen

        self.active_tasks: Dict[str, Task] = {}
        self.task_progress: Dict[str, Dict[str, Any]] = {}

    def assign_task(self, robot_id: str, task_name: str) -> bool:
        """分配任务"""
        if task_name not in TASK_LIBRARY:
            return False

        if robot_id not in self.kitchen.robots:
            return False

        import copy
        task = copy.deepcopy(TASK_LIBRARY[task_name])
        self.active_tasks[robot_id] = task
        self.task_progress[robot_id] = {
            'start_time': self.kitchen.time,
            'steps_completed': 0,
            'current_step_start': self.kitchen.time,
        }

        return True

    def step(self, robot_id: str) -> Dict[str, Any]:
        """执行一步规�?""
        if robot_id not in self.active_tasks:
            return {'status': 'no_task'}

        if robot_id not in self.kitchen.robots:
            return {'status': 'robot_not_found'}

        robot = self.kitchen.robots[robot_id]
        task = self.active_tasks[robot_id]

        if task.completed:
            return {'status': 'completed', 'task': task.name, 'progress': 1.0}

        if task.failed:
            return {'status': 'failed', 'task': task.name}

        elapsed = self.kitchen.time - self.task_progress[robot_id]['start_time']
        if elapsed > task.time_limit:
            task.failed = True
            return {'status': 'timeout'}

        step = task.get_current_step()
        if step is None:
            task.completed = True
            return {'status': 'completed'}

        # SSFR感知（连续）
        hypotheses = self.physical_ssfr.perceive(robot_id, task)

        # 获取最佳空�?        validity = self.physical_ssfr.get_space_validity(robot_id)
        best_space = max(validity, key=validity.get) if validity else 'ricci'

        # 执行步骤
        result = self._execute_step(robot_id, step, robot, task, best_space)

        # SSFR竞争
        actual_pos = robot.position
        winner = self.physical_ssfr.compete(robot_id, actual_pos)

        return {
            'status': 'running',
            'task': task.name,
            'step': step.action,
            'target': step.target,
            'progress': task.get_progress(),
            'best_space': best_space,
            'space_validity': validity,
            'num_hypotheses': len(hypotheses),
            'winner_id': winner.id if winner else None,
            'step_result': result,
        }

    def _execute_step(self, robot_id: str, step: TaskStep,
                     robot: Robot, task: Task, best_space: str) -> str:
        """执行单个步骤"""
        action = step.action
        target = step.target

        if action == 'move_to':
            return self._execute_move_to(robot_id, robot, target, best_space)
        elif action == 'grab':
            return self._execute_grab(robot_id, robot, target)
        elif action == 'place':
            return self._execute_place(robot_id, robot, target)
        elif action == 'press_button':
            return self._execute_press_button(robot_id, robot, target)
        elif action == 'wait':
            return self._execute_wait(robot_id, robot, task)

        return 'unknown_action'

    def _execute_move_to(self, robot_id: str, robot: Robot,
                        target_name: str, best_space: str) -> str:
        """执行移动"""
        target_pos = None
        for obj in self.kitchen.objects.values():
            if target_name.lower() in obj.properties.name.lower():
                target_pos = obj.position
                break

        if target_pos is None:
            predefined = {
                'coffee_machine': (1.5, 6.5),
                'microwave': (3.0, 6.5),
                'sink': (8.0, 6.5),
                'stove': (5.0, 6.5),
                'table': (5.0, 3.0),
            }
            target_pos = predefined.get(target_name.lower())

        if target_pos is None:
            return 'target_not_found'

        robot_pos = robot.position
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        distance = math.sqrt(dx**2 + dy**2)

        arrival_threshold = 0.5

        if distance < arrival_threshold:
            robot.stop()
            self.active_tasks[robot_id].complete_step()
            return 'arrived'

        # 使用连续空间的启发式进行导航
        space = self.physical_ssfr.ssfr.spaces.get(best_space)
        if space:
            # 使用空间的启发式来指�?            heuristic_cost = space.get_heuristic(robot_pos, target_pos)
            # 启发式越小越好，所以直接导�?
        # 转向控制
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - robot.angle

        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        if abs(angle_diff) > 0.3:
            if angle_diff > 0:
                robot.turn_left(torque=2.0)
            else:
                robot.turn_right(torque=2.0)
            robot.move_forward(force=1.0)
        else:
            robot.move_forward(force=5.0)

        return 'moving'

    def _execute_grab(self, robot_id: str, robot: Robot, target_name: str) -> str:
        if robot.arm.is_gripping:
            self.active_tasks[robot_id].complete_step()
            return 'already_gripping'

        # 简化的抓取逻辑
        all_entities = list(self.kitchen.entities.values())
        sensor = SensorSystem(robot, self.kitchen.space)
        nearby = sensor.get_nearby_objects(all_entities, max_distance=0.5)

        for obj, dist in nearby:
            if obj.properties.grabbable and obj.body != robot.body:
                import pymunk
                joint = pymunk.PinJoint(robot.body, obj.body, (0, 0), (0, 0))
                joint.max_force = PhysicsConfig.GRIP_STRENGTH
                self.kitchen.space.add(joint)
                robot.arm.is_gripping = True
                robot.arm.gripped_object = obj
                robot.arm.grip_joint = joint
                self.active_tasks[robot_id].complete_step()
                return 'grabbed'

        return 'approaching_for_grab'

    def _execute_place(self, robot_id: str, robot: Robot, target_name: str) -> str:
        if not robot.arm.is_gripping:
            self.active_tasks[robot_id].complete_step()
            return 'nothing_to_place'

        success = robot.arm.release()
        if success:
            self.active_tasks[robot_id].complete_step()
            return 'placed'
        return 'releasing'

    def _execute_press_button(self, robot_id: str, robot: Robot, target_name: str) -> str:
        robot_pos = robot.position
        target_pos = None

        for obj in self.kitchen.objects.values():
            if target_name.lower() in obj.properties.name.lower():
                target_pos = obj.position
                break

        if target_pos:
            dist = math.sqrt(
                (robot_pos[0] - target_pos[0])**2 +
                (robot_pos[1] - target_pos[1])**2
            )
            if dist < 0.8:
                robot.stop()
                self.active_tasks[robot_id].complete_step()
                return 'button_pressed'

        return 'approaching_button'

    def _execute_wait(self, robot_id: str, robot: Robot, task: Task) -> str:
        robot.stop()
        step_time = self.kitchen.time - self.task_progress[robot_id]['current_step_start']
        if step_time > 2.0:
            self.active_tasks[robot_id].complete_step()
            return 'wait_complete'
        return 'waiting'

    def get_task_status(self, robot_id: str) -> Dict[str, Any]:
        if robot_id not in self.active_tasks:
            return {'status': 'no_task'}

        task = self.active_tasks[robot_id]
        return {
            'task': task.name,
            'status': 'completed' if task.completed else 'failed' if task.failed else 'running',
            'progress': task.get_progress(),
            'current_step': task.current_step,
            'total_steps': len(task.steps),
            'elapsed_time': self.kitchen.time - self.task_progress[robot_id]['start_time'],
        }


# ============================================================================
# 4. 测试
# ============================================================================

def test_continuous_field():
    """测试连续�?""
    print("=" * 70)
    print("TEST: Continuous Field")
    print("=" * 70)

    from src.spaces.continuous import ContinuousField

    field = ContinuousField(default_value=0.0)

    # 添加采样�?    field.add_sample((0.0, 0.0), 1.0)
    field.add_sample((1.0, 0.0), 2.0)
    field.add_sample((0.0, 1.0), 3.0)
    field.add_sample((1.0, 1.0), 4.0)

    # 查询
    print(f"Samples: {field.num_samples()}")
    print(f"Query (0.5, 0.5): {field.query((0.5, 0.5)):.3f}")
    print(f"Query (0.0, 0.0): {field.query((0.0, 0.0)):.3f}")
    print(f"Query (1.0, 1.0): {field.query((1.0, 1.0)):.3f}")

    assert field.num_samples() == 4
    print("[PASS] Continuous field works")
    return True


def test_continuous_space():
    """测试连续空间"""
    print("\n" + "=" * 70)
    print("TEST: Continuous Space")
    print("=" * 70)

    space = ContinuousRicciSpace()

    # 测试距离计算
    dist = space.compute_distance((0.0, 0.0), (1.0, 0.0))
    print(f"Distance (0,0) -> (1,0): {dist:.3f}")
    assert dist > 0

    # 测试更新
    space.update_from_observation((0.5, 0.5), {
        'obstacles': [(0.3, 0.3)],
        'goal_position': (2.0, 2.0),
    })

    print(f"Uncertainty samples: {space.uncertainty_field.num_samples()}")
    print(f"Curvature samples: {space.curvature_field.num_samples()}")
    assert space.uncertainty_field.num_samples() > 0

    # 测试预测
    prediction = space.predict_next_state((0.5, 0.5), {
        'goal_position': (2.0, 2.0),
        'obstacles': [],
    })
    print(f"Predicted position: {prediction['predicted_position']}")
    print(f"Predicted cost: {prediction['predicted_cost']:.3f}")

    print("[PASS] Continuous space works")
    return True


def test_continuous_ssfr():
    """测试连续SSFR"""
    print("\n" + "=" * 70)
    print("TEST: Continuous SSFR")
    print("=" * 70)

    ssfr = ContinuousSSFR(space_names=['ricci', 'fisher', 'wasserstein'])

    # 测试感知
    observation = {
        'position': (1.0, 2.0),
        'goal_position': (5.0, 5.0),
        'obstacles': [(2.0, 2.0)],
        'uncertainty': 0.3,
    }

    hypotheses = ssfr.perceive((1.0, 2.0), observation)
    print(f"Hypotheses: {len(hypotheses)}")
    assert len(hypotheses) > 0

    for h in hypotheses:
        print(f"  {h.name}: fitness={h.fitness:.3f}")

    # 测试竞争
    actual = {'position': (1.5, 2.5)}
    winner = ssfr.compete(observation, actual)
    if winner:
        print(f"Winner: {winner.name} (fitness={winner.fitness:.3f})")

    # 测试统计
    stats = ssfr.get_statistics()
    print(f"Pool size: {stats['pool_stats']['num_structures']}")
    assert stats['pool_stats']['num_structures'] > 0

    print("[PASS] Continuous SSFR works")
    return True


def test_continuous_kitchen_integration():
    """测试连续厨房集成"""
    print("\n" + "=" * 70)
    print("TEST: Continuous Kitchen Integration")
    print("=" * 70)

    kitchen = create_demo_kitchen()
    robot_id = list(kitchen.robots.keys())[0]

    # 创建连续SSFR
    physical_ssfr = ContinuousPhysicalSSFR(kitchen)
    planner = ContinuousSSFRTaskPlanner(physical_ssfr)

    # 分配任务
    success = planner.assign_task(robot_id, 'make_coffee')
    print(f"Task assigned: {success}")
    assert success

    # 执行几步
    for i in range(5):
        kitchen.step()
        result = planner.step(robot_id)
        print(f"Step {i+1}: {result['step']} -> {result['step_result']}")

    # 验证SSFR统计
    stats = physical_ssfr.get_statistics()
    print(f"\nPerceptions: {stats['physical_perceptions']}")
    print(f"Pool size: {stats['pool_stats']['num_structures']}")
    assert stats['physical_perceptions'] > 0

    print("[PASS] Continuous kitchen integration works")
    return True


def run_all_tests():
    """运行所有测�?""
    print("=" * 70)
    print("Continuous SSFR Tests")
    print("=" * 70)

    tests = [
        test_continuous_field,
        test_continuous_space,
        test_continuous_ssfr,
        test_continuous_kitchen_integration,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()

    print("\n" + "=" * 70)
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 70)
