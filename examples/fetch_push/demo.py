"""robotmem FetchPush 快速 Demo — 5 分钟验证记忆驱动决策

运行方式:
  cd examples/fetch_push
  pip install gymnasium-robotics  # 需要 MuJoCo
  PYTHONPATH=../../src python demo.py

三阶段（各 30 episodes）:
  Phase A: 基线（heuristic，无记忆）
  Phase B: 记忆写入（learn + save_perception）
  Phase C: 记忆利用（recall → MemoryPolicy）

预期结果: Phase C 成功率比 Phase A 高 10-20%
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

# 数据隔离 — 必须在 import robotmem 之前设置
_DEMO_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-demo")
os.environ["ROBOTMEM_HOME"] = _DEMO_HOME
os.makedirs(_DEMO_HOME, exist_ok=True)

try:
    import gymnasium_robotics  # noqa: F401
    import gymnasium
    import numpy as np
except ImportError:
    print("需要安装: pip install gymnasium-robotics")
    sys.exit(1)

from robotmem.config import load_config
from robotmem.db_cog import CogDatabase
from robotmem.embed import create_embedder
from robotmem.search import recall as do_recall
from robotmem.db import floats_to_blob
from robotmem.ops.memories import insert_memory
from robotmem.ops.sessions import get_or_create_session, mark_session_ended
from robotmem.auto_classify import classify_category, estimate_confidence

from policies import HeuristicPolicy, MemoryPolicy

COLLECTION = "demo"
EPISODES = 30  # 每阶段 30 episodes（快速验证）
MEMORY_WEIGHT = 0.3
RECALL_N = 5


def build_context(obs, actions, success, steps, total_reward):
    """构建 context JSON"""
    recent = actions[-10:] if len(actions) >= 10 else actions
    avg_action = np.mean(recent, axis=0) if recent else np.zeros(4)
    return {
        "params": {
            "approach_velocity": {"value": avg_action[0:3].tolist(), "type": "vector"},
            "grip_force": {"value": float(avg_action[3]), "type": "scalar"},
            "final_distance": {
                "value": float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])),
                "unit": "m",
            },
        },
        "spatial": {
            "grip_position": obs["observation"][0:3].tolist(),
            "object_position": obs["observation"][3:6].tolist(),
            "target_position": obs["desired_goal"].tolist(),
        },
        "task": {
            "name": "push_to_target",
            "success": bool(success),
            "steps": steps,
            "total_reward": float(total_reward),
        },
    }


async def run_episode(env, policy, phase, ep, db, embedder):
    """执行单个 episode"""
    ext_id = f"demo_{phase}_{ep:03d}"
    get_or_create_session(db.conn, ext_id, COLLECTION)

    # Phase C: recall 成功经验
    recalled = []
    if phase == "C":
        result = await do_recall(
            "push cube to target position",
            db, embedder, collection=COLLECTION, top_k=RECALL_N,
            context_filter={"task.success": True},
        )
        recalled = result.memories

    active_policy = MemoryPolicy(policy, recalled, MEMORY_WEIGHT) if recalled else policy

    # 跑 episode
    obs, _ = env.reset()
    actions = []
    total_reward = 0.0
    for _ in range(50):
        action = active_policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action.copy())
        total_reward += reward
        if terminated or truncated:
            break

    success = info.get("is_success", False)

    # Phase B/C: 写记忆
    if phase in ("B", "C"):
        ctx = build_context(obs, actions, success, len(actions), total_reward)
        ctx_str = json.dumps(ctx, ensure_ascii=False)
        dist = ctx["params"]["final_distance"]["value"]

        content = f"FetchPush: {'成功' if success else '失败'}, 距离 {dist:.3f}m, {len(actions)} 步"
        emb = await embedder.embed_one(content)
        insert_memory(db.conn, {
            "content": content, "context": ctx_str, "collection": COLLECTION,
            "session_id": ext_id, "type": "fact",
            "category": classify_category(content),
            "confidence": estimate_confidence(content),
            "embedding": floats_to_blob(emb),
        }, vec_loaded=db.vec_loaded)

    mark_session_ended(db.conn, ext_id)
    return success


async def run_phase(env, policy, phase, db, embedder):
    """执行一个 Phase"""
    successes = 0
    for ep in range(EPISODES):
        ok = await run_episode(env, policy, phase, ep, db, embedder)
        successes += int(ok)
        if (ep + 1) % 10 == 0:
            print(f"  Phase {phase} [{ep+1}/{EPISODES}] 成功率: {successes/(ep+1):.0%}")
    return successes / EPISODES


async def main():
    print("=" * 50)
    print("robotmem FetchPush Demo")
    print(f"每阶段 {EPISODES} episodes，约 2-3 分钟")
    print("=" * 50)

    config = load_config()
    db = CogDatabase(config)
    embedder = create_embedder(config)
    if not await embedder.check_availability():
        print(f"Embedder 不可用: {embedder.unavailable_reason}")
        return

    env = gymnasium.make("FetchPush-v4")
    policy = HeuristicPolicy()

    # Phase A
    print(f"\n--- Phase A: 基线（无记忆）---")
    t0 = time.time()
    rate_a = await run_phase(env, policy, "A", db, embedder)

    # Phase B
    print(f"\n--- Phase B: 写入记忆 ---")
    rate_b = await run_phase(env, policy, "B", db, embedder)

    # Phase C
    print(f"\n--- Phase C: 利用记忆 ---")
    rate_c = await run_phase(env, policy, "C", db, embedder)
    elapsed = time.time() - t0

    # 结果
    delta = rate_c - rate_a
    verdict = "失败" if delta <= 0 else "微弱" if delta <= 0.05 else "有效" if delta <= 0.15 else "显著"

    print(f"\n{'=' * 50}")
    print(f"结果")
    print(f"{'=' * 50}")
    print(f"  Phase A: {rate_a:.0%}")
    print(f"  Phase B: {rate_b:.0%}")
    print(f"  Phase C: {rate_c:.0%}")
    print(f"  提升:    {delta:+.0%} ({verdict})")
    print(f"  耗时:    {elapsed:.0f}s")
    print(f"\n数据存储于: {_DEMO_HOME}")

    env.close()
    await embedder.close()
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
