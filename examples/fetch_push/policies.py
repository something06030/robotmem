"""策略模块 — HeuristicPolicy + MemoryPolicy + PhaseAwareMemoryPolicy + SlidePolicy

来源：圆桌 Round 4 Abbeel（heuristic）+ Levine（MemoryPolicy v1）
修正：四阶段策略（升高→绕后→下降→推），避免绕后时撞飞物体。
阶段感知：圆桌 Round 1-3 Levine 提出，只在推送阶段施加 bias。
SlidePolicy：Issue #21，FetchSlide 专用策略（惯性滑动）。
"""

import numpy as np


class HeuristicPolicy:
    """四阶段 heuristic：升高 → 绕后 → 下降 → 推

    FetchPush-v4 动作空间 4 维 [-1, 1]：[dx, dy, dz, grip]
    每步位移约 0.02m（action=1.0 时）

    策略：
    1. behind = 物体后方 0.04m（推方向的反方向）
    2. 先升到物体上方 0.08m（避免触碰物体）
    3. xy 移动到 behind 位置上方
    4. 下降到物体高度
    5. 直推向目标
    """
    ABOVE_OFFSET = 0.08  # 悬停高度偏移
    BEHIND_DIST = 0.04   # 后方距离
    XY_THRESH = 0.015    # xy 到位阈值
    Z_THRESH = 0.01      # z 到位阈值

    # 阶段常量
    PHASE_RISE = 1
    PHASE_MOVE_BEHIND = 2
    PHASE_DESCEND = 3
    PHASE_PUSH = 4

    def get_phase(self, obs):
        """返回当前阶段: 1=升高, 2=绕后, 3=下降, 4=推"""
        grip = obs["observation"][0:3]
        obj = obs["observation"][3:6]
        target = obs["desired_goal"]

        push_vec = target[:2] - obj[:2]
        push_norm = np.linalg.norm(push_vec)
        if push_norm < 0.005:
            return self.PHASE_PUSH

        push_dir = push_vec / push_norm
        behind_xy = obj[:2] - push_dir * self.BEHIND_DIST

        grip_xy = grip[:2]
        grip_z = grip[2]
        dist_to_behind_xy = np.linalg.norm(grip_xy - behind_xy)
        dist_to_obj_xy = np.linalg.norm(grip_xy - obj[:2])
        is_above = grip_z > obj[2] + self.ABOVE_OFFSET * 0.5
        is_behind = dist_to_behind_xy < self.XY_THRESH
        is_at_obj_height = abs(grip_z - obj[2]) < self.Z_THRESH

        if is_behind and is_at_obj_height:
            return self.PHASE_PUSH
        if is_behind and not is_at_obj_height:
            return self.PHASE_DESCEND
        if not is_above and dist_to_obj_xy < 0.06:
            return self.PHASE_RISE
        return self.PHASE_MOVE_BEHIND

    def act(self, obs):
        grip = obs["observation"][0:3]
        obj = obs["observation"][3:6]
        target = obs["desired_goal"]

        # 推方向（xy 平面）
        push_vec = target[:2] - obj[:2]
        push_norm = np.linalg.norm(push_vec)
        if push_norm < 0.005:
            # 物体已在目标上，不动
            return [0.0, 0.0, 0.0, 0.0]
        push_dir = push_vec / push_norm

        # 后方位置
        behind_xy = obj[:2] - push_dir * self.BEHIND_DIST
        above_z = obj[2] + self.ABOVE_OFFSET

        # 当前状态
        grip_xy = grip[:2]
        grip_z = grip[2]
        dist_to_behind_xy = np.linalg.norm(grip_xy - behind_xy)
        dist_to_obj_xy = np.linalg.norm(grip_xy - obj[:2])
        is_above = grip_z > obj[2] + self.ABOVE_OFFSET * 0.5
        is_behind = dist_to_behind_xy < self.XY_THRESH
        is_at_obj_height = abs(grip_z - obj[2]) < self.Z_THRESH

        # 判断是否已在推送位置（后方 + 物体高度）
        if is_behind and is_at_obj_height:
            # 阶段 4：直推向目标
            action_xy = push_dir * 1.0
            dz = (obj[2] - grip_z) * 3.0
            return np.clip([action_xy[0], action_xy[1], dz, 0.0], -1, 1).tolist()

        if is_behind and not is_at_obj_height:
            # 阶段 3：下降到物体高度
            dz = (obj[2] - grip_z) / (abs(obj[2] - grip_z) + 1e-6)
            return [0.0, 0.0, float(np.clip(dz, -1, 1)), 0.0]

        # 还没到后方位置
        if not is_above and dist_to_obj_xy < 0.06:
            # 阶段 1：先升高（避免碰物体）
            return [0.0, 0.0, 1.0, 0.0]

        # 阶段 2：xy 移动到后方位置（保持高度）
        move_xy = behind_xy - grip_xy
        move_norm = np.linalg.norm(move_xy)
        if move_norm > 1e-6:
            move_dir = move_xy / move_norm
        else:
            move_dir = np.zeros(2)
        # z 保持在 above_z
        dz = (above_z - grip_z) * 3.0
        return np.clip([move_dir[0], move_dir[1], dz, 0.0], -1, 1).tolist()


class MemoryPolicy:
    """记忆驱动策略 — base + recall 偏置

    v1: 固定权重 0.7 * base + 0.3 * memory_bias
    v2（待实现）: 距离自适应权重（Levine Round 4）
    """

    def __init__(self, base_policy, recalled_memories, memory_weight=0.3):
        self.base = base_policy
        self.weight = memory_weight
        self.bias = self._extract_bias(recalled_memories)

    def _extract_bias(self, memories):
        """从成功经验中提取策略偏置"""
        if not memories:
            return np.zeros(4)

        successful = []
        for m in memories:
            params = m.get("params", {})
            task = m.get("task", {})
            if task.get("success") and "approach_velocity" in params:
                successful.append(params)

        if not successful:
            return np.zeros(4)

        velocities = [s["approach_velocity"]["value"] for s in successful]
        avg_vel = np.mean(velocities, axis=0)
        avg_grip = np.mean([s.get("grip_force", {}).get("value", 0.0) for s in successful])

        return np.array([avg_vel[0], avg_vel[1], avg_vel[2], avg_grip])

    def act(self, obs):
        base_action = np.array(self.base.act(obs))

        if np.allclose(self.bias, 0):
            return base_action

        return np.clip((1 - self.weight) * base_action + self.weight * self.bias, -1.0, 1.0)


class PhaseAwareMemoryPolicy(MemoryPolicy):
    """阶段感知记忆策略 — 只在推送阶段（phase 4）施加 bias

    来源：圆桌 Round 1-3 Levine 提出，Round 3 全体共识"属于用户侧优化"。
    原理：bias 是成功经验的平均 approach velocity，只在推送阶段有意义。
    升高/绕后/下降阶段施加 bias 会干扰精确定位，反而降低成功率。
    """

    def act(self, obs):
        base_action = np.array(self.base.act(obs))

        if np.allclose(self.bias, 0):
            return base_action

        # 只在推送阶段施加 bias
        phase = self.base.get_phase(obs)
        if phase != HeuristicPolicy.PHASE_PUSH:
            return base_action

        return np.clip((1 - self.weight) * base_action + self.weight * self.bias, -1.0, 1.0)


class SlidePolicy:
    """FetchSlide 专用策略 — 惯性滑动

    FetchSlide vs FetchPush:
    - 目标通常更远（超出机械臂直接推达范围）
    - 需要给物体足够初速度使其靠惯性滑到目标
    - 摩擦力会减速，需要更大推力
    - 不需要精确跟随推送，需要正确的初始推力方向和大小

    策略：
    1. 计算推力方向（物体→目标）
    2. 在物体后方更远处定位（留出加速空间）
    3. 下降到物体高度
    4. 全力推向目标方向
    """
    ABOVE_OFFSET = 0.08
    BEHIND_DIST = 0.06   # 比 Push 更远，留出加速空间
    XY_THRESH = 0.02     # 略宽松，快速到位优先
    Z_THRESH = 0.015

    # 阶段常量（与 HeuristicPolicy 一致）
    PHASE_RISE = 1
    PHASE_MOVE_BEHIND = 2
    PHASE_DESCEND = 3
    PHASE_PUSH = 4

    def get_phase(self, obs):
        """返回当前阶段: 1=升高, 2=绕后, 3=下降, 4=推"""
        grip = obs["observation"][0:3]
        obj = obs["observation"][3:6]
        target = obs["desired_goal"]

        slide_vec = target[:2] - obj[:2]
        slide_norm = np.linalg.norm(slide_vec)
        if slide_norm < 0.005:
            return self.PHASE_PUSH

        slide_dir = slide_vec / slide_norm
        behind_xy = obj[:2] - slide_dir * self.BEHIND_DIST

        grip_xy = grip[:2]
        grip_z = grip[2]
        dist_to_behind = np.linalg.norm(grip_xy - behind_xy)
        dist_to_obj = np.linalg.norm(grip_xy - obj[:2])
        is_above = grip_z > obj[2] + self.ABOVE_OFFSET * 0.5
        is_behind = dist_to_behind < self.XY_THRESH
        is_at_height = abs(grip_z - obj[2]) < self.Z_THRESH

        if is_behind and is_at_height:
            return self.PHASE_PUSH
        if is_behind and not is_at_height:
            return self.PHASE_DESCEND
        if not is_above and dist_to_obj < 0.08:
            return self.PHASE_RISE
        return self.PHASE_MOVE_BEHIND

    def act(self, obs):
        grip = obs["observation"][0:3]
        obj = obs["observation"][3:6]
        target = obs["desired_goal"]

        # 滑动方向（xy 平面）
        slide_vec = target[:2] - obj[:2]
        slide_norm = np.linalg.norm(slide_vec)
        if slide_norm < 0.005:
            return [0.0, 0.0, 0.0, 0.0]
        slide_dir = slide_vec / slide_norm

        # 后方位置（加速跑道）
        behind_xy = obj[:2] - slide_dir * self.BEHIND_DIST
        above_z = obj[2] + self.ABOVE_OFFSET

        grip_xy = grip[:2]
        grip_z = grip[2]
        dist_to_behind = np.linalg.norm(grip_xy - behind_xy)
        dist_to_obj = np.linalg.norm(grip_xy - obj[:2])
        is_above = grip_z > obj[2] + self.ABOVE_OFFSET * 0.5
        is_behind = dist_to_behind < self.XY_THRESH
        is_at_height = abs(grip_z - obj[2]) < self.Z_THRESH

        # 阶段 4：全力推
        if is_behind and is_at_height:
            return np.clip([slide_dir[0], slide_dir[1], 0.0, 0.0], -1, 1).tolist()

        # 阶段 3：下降到物体高度
        if is_behind and not is_at_height:
            dz = (obj[2] - grip_z) / (abs(obj[2] - grip_z) + 1e-6)
            return [0.0, 0.0, float(np.clip(dz, -1, 1)), 0.0]

        # 阶段 1：升高避碰
        if not is_above and dist_to_obj < 0.08:
            return [0.0, 0.0, 1.0, 0.0]

        # 阶段 2：移到后方位置
        move_xy = behind_xy - grip_xy
        move_norm = np.linalg.norm(move_xy)
        move_dir = move_xy / move_norm if move_norm > 1e-6 else np.zeros(2)
        dz = (above_z - grip_z) * 3.0
        return np.clip([move_dir[0], move_dir[1], dz, 0.0], -1, 1).tolist()
