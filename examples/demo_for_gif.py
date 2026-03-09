"""RobotMem 30-Second Demo — 用于录屏制作 GIF

运行方式:
    PYTHONPATH=src python3 examples/demo_for_gif.py

录屏建议:
    - 终端背景深色、字体 16pt+
    - 窗口宽度 80 列
    - 用 asciinema 或 macOS 截屏录制
"""

import time
import sys
import random

def slow_print(text, delay=0.02):
    """逐字打印，增加视觉效果"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def section(title):
    print()
    slow_print(f"{'─' * 50}")
    slow_print(f"  {title}")
    slow_print(f"{'─' * 50}")
    print()
    time.sleep(0.5)

# 每次运行用不同的随机数据，避免去重拦截
_run_id = random.randint(1000, 9999)

# ══════════════════════════════════════════════
section("Step 1: Save robot experiences")

slow_print(">>> from robotmem import save_perception, recall")
time.sleep(0.3)

from robotmem import save_perception, recall

force1 = round(random.uniform(10.0, 15.0), 1)
steps1 = random.randint(25, 40)
desc1 = f"Grasped red cup: force={force1}N, {steps1} steps, run#{_run_id}"

slow_print(f'>>> save_perception("{desc1[:50]}...")')
time.sleep(0.2)
r1 = save_perception(
    description=desc1,
    perception_type="procedural",
    data=f'{{"force_peak": {force1}, "steps": {steps1}}}',
)
if "memory_id" in r1:
    slow_print(f"✓ memory #{r1['memory_id']} saved  (embedding: {'384d' if r1['has_embedding'] else 'none'})")
else:
    slow_print(f"✓ saved  ({r1})")
time.sleep(0.3)

force2 = round(random.uniform(13.0, 18.0), 1)
desc2 = f"Picked blue mug from shelf, grip={force2}N, run#{_run_id}"

slow_print(f'>>> save_perception("{desc2[:50]}...")')
time.sleep(0.2)
r2 = save_perception(
    description=desc2,
    perception_type="procedural",
    data=f'{{"grip_force": {force2}, "approach": "top-down"}}',
)
if "memory_id" in r2:
    slow_print(f"✓ memory #{r2['memory_id']} saved")
else:
    slow_print(f"✓ saved  ({r2})")
time.sleep(0.5)


# ══════════════════════════════════════════════
section("Step 2: Robot restarts... memories persist!")

slow_print("  (simulating power cycle...)")
time.sleep(1.0)
slow_print("  🔄 Robot rebooted.")
time.sleep(0.5)


# ══════════════════════════════════════════════
section("Step 3: Recall past experiences")

slow_print('>>> memories = recall("how to grasp a cup")')
time.sleep(0.3)

memories = recall("how to grasp a cup", n=5)

slow_print(f"\n▸ Found {memories['total']} memories ({memories['mode']} search, {memories['query_ms']}ms)")
print()

for m in memories["memories"]:
    score = m.get("_rrf_score", "?")
    if isinstance(score, float):
        score = f"{score:.2f}"
    content = m["content"][:55]
    slow_print(f"  {content:<55}  score={score}")
    time.sleep(0.2)

print()
slow_print("✨ Robot remembers everything. Even after restart.")
print()
