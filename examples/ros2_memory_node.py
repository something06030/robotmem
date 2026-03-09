"""RobotMem + ROS2 — 给你的 ROS2 节点加上长期记忆

需要:
    pip install robotmem
    # ROS2 Humble/Iron/Jazzy 任一

运行:
    ros2 run your_package memory_node
    # 或直接: python3 examples/ros2_memory_node.py

这个示例展示了一个 ROS2 节点如何在执行 pick-and-place 任务时
保存经验，并在下次启动时回忆过去的经验来改进策略。
"""

try:
    import rclpy
    from rclpy.node import Node
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    print("⚠ ROS2 not installed — running in standalone demo mode")

from robotmem import save_perception, recall, start_session, end_session


def simulate_grasp_task():
    """模拟一次抓取任务（替换为你的真实机器人控制代码）"""
    import random
    force = round(random.uniform(8.0, 20.0), 1)
    success = force < 18.0
    return {
        "object": random.choice(["red cup", "blue mug", "green bottle"]),
        "force": force,
        "steps": random.randint(15, 40),
        "success": success,
    }


def run_with_memory():
    """核心逻辑：记忆驱动的机器人任务"""

    # 1. 开始新 session
    session = start_session(context='{"task": "pick_and_place", "robot": "franka"}')
    print(f"📋 Session started: {session['session_id'][:8]}...")
    print(f"   Existing memories: {session['active_memories_count']}")

    # 2. 先回忆过去的经验
    past = recall("grasping strategy tips", n=3, session_id=session["session_id"])
    if past["total"] > 0:
        print(f"\n🧠 Recalled {past['total']} past experiences:")
        for m in past["memories"]:
            print(f"   → {m['content'][:70]}")
    else:
        print("\n🧠 No past experiences found — first run!")

    # 3. 执行任务并记录经验
    for i in range(3):
        result = simulate_grasp_task()
        status = "success" if result["success"] else "FAILED"
        desc = f"Grasped {result['object']}: force={result['force']}N, {result['steps']} steps, {status}"

        save_perception(
            description=desc,
            perception_type="procedural",
            data=str(result),
            session_id=session["session_id"],
        )
        print(f"  ✓ Task {i+1}: {desc}")

    # 4. 结束 session
    outcome = end_session(session["session_id"], outcome_score=0.8)
    print(f"\n📊 Session ended — {outcome['summary']}")


if HAS_ROS2:
    class MemoryNode(Node):
        def __init__(self):
            super().__init__("robotmem_node")
            self.get_logger().info("RobotMem memory node started")
            # 运行记忆驱动的任务
            self.timer = self.create_timer(0.1, self._run_once)
            self._done = False

        def _run_once(self):
            if self._done:
                return
            self._done = True
            run_with_memory()
            self.get_logger().info("Task complete — memories saved for next run")

    def main():
        rclpy.init()
        node = MemoryNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
else:
    def main():
        print("=== RobotMem + ROS2 Demo (standalone mode) ===\n")
        run_with_memory()
        print("\n=== Done! Restart to see memory recall in action ===")


if __name__ == "__main__":
    main()
