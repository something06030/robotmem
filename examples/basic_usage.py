"""RobotMem Basic Usage — 3 分钟入门

pip install robotmem
python examples/basic_usage.py
"""

from robotmem import save_perception, recall

# 1. 保存一次抓取经验
result = save_perception(
    description="Grasped red cup from table: force=12.5N, 30 steps, success",
    perception_type="procedural",
    data='{"actions": [[0.1, -0.3, 0.05]], "force_peak": 12.5, "steps": 30}',
)
print(f"✓ Saved memory #{result['memory_id']}  (embedding: {'yes' if result['has_embedding'] else 'no'})")

# 2. 保存更多经验
save_perception(
    description="Approached blue mug from left side, grip_force=15.3N, stable grasp",
    perception_type="procedural",
    data='{"approach_angle": "left", "grip_force": 15.3}',
)
save_perception(
    description="Failed to grasp bottle: slipped at step 12, force exceeded 20N",
    perception_type="procedural",
    data='{"failure": true, "slip_step": 12, "max_force": 20.1}',
)
print("✓ Saved 2 more memories")

# 3. 搜索相关经验
memories = recall("how to grasp a cup", n=3)
print(f"\n▸ Recall: {memories['total']} results ({memories['mode']} mode, {memories['query_ms']}ms)")
for m in memories["memories"]:
    print(f"  {m['content'][:60]:<60}  score={m.get('_rrf_score', 'N/A')}")
