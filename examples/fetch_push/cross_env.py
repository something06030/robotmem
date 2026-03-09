"""跨环境泛化实验 — FetchPush 经验是否帮助 FetchSlide

验证 L3 核心命题："学一次，换个环境还能用"
Issue: #21

FetchSlide vs FetchPush:
- 相同：7-DOF Fetch 机器人，4 维动作，桌面场景
- 不同：Slide 需要把物体滑到远处目标（物体可能滑出桌面）
  Slide 的目标通常更远，需要更大力度和惯性

实验设计（三阶段）:
  Phase A: FetchSlide + SlidePolicy 基线（无记忆）— 50 episodes
  Phase B: FetchSlide + SlidePolicy + FetchPush 记忆（跨环境）— 50 episodes
  Phase C: FetchSlide + SlidePolicy + FetchSlide 记忆（同环境）— 50 episodes

Phase C 需要先写入 FetchSlide 专属记忆（50 episodes），然后利用。

运行:
  cd examples/fetch_push
  PYTHONPATH=../../src ROBOTMEM_HOME=.robotmem .venv/bin/python3 cross_env.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time

_EXP_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem")
os.environ.setdefault("ROBOTMEM_HOME", _EXP_HOME)

import gymnasium_robotics  # noqa: F401
import gymnasium
import numpy as np

from robotmem.config import load_config
from robotmem.db_cog import CogDatabase
from robotmem.embed import create_embedder
from robotmem.search import recall as do_recall
from robotmem.db import floats_to_blob
from robotmem.ops.memories import insert_memory
from robotmem.ops.sessions import get_or_create_session, mark_session_ended
from robotmem.auto_classify import classify_category, estimate_confidence

from policies import SlidePolicy, PhaseAwareMemoryPolicy

PUSH_COLLECTION = "exp_run_001"     # FetchPush 经验
SLIDE_COLLECTION = "exp_slide_001"  # FetchSlide 专属经验
EPISODES = 50
MEMORY_WEIGHT = 0.3
RECALL_N = 5


def build_slide_context(obs, actions, success, steps, total_reward):
    """构建 FetchSlide context JSON"""
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
            "frame": "world",
            "grip_position": obs["observation"][0:3].tolist(),
            "object_position": obs["observation"][3:6].tolist(),
            "target_position": obs["desired_goal"].tolist(),
            "scene_tag": "tabletop_slide",
        },
        "task": {
            "name": "slide_to_target",
            "object": "puck",
            "success": bool(success),
            "steps": steps,
            "total_reward": float(total_reward),
        },
    }


async def run_slide_episode(env, policy, use_memory, db, embedder,
                            collection=None, write_memory=False, ep_num=0):
    """跑单个 FetchSlide episode"""
    ext_id = f"slide_{ep_num:03d}"
    if write_memory:
        session = get_or_create_session(db.conn, ext_id, collection)
        if not session:
            print(f"  警告: session 创建失败 ({ext_id})")
            return False

    obs, _ = env.reset()

    # recall
    recalled = []
    if use_memory and collection:
        obj_pos = obs["observation"][3:6].tolist()
        target_pos = obs["desired_goal"].tolist()
        query = f"slide from [{obj_pos[0]:.2f}, {obj_pos[1]:.2f}] to [{target_pos[0]:.2f}, {target_pos[1]:.2f}]"
        result = await do_recall(
            query, db, embedder,
            collection=collection, top_k=RECALL_N,
            context_filter={"task.success": True},
            spatial_sort={"field": "spatial.object_position", "target": obj_pos},
        )
        recalled = result.memories

    active = PhaseAwareMemoryPolicy(policy, recalled, MEMORY_WEIGHT) if recalled else policy

    actions = []
    total_reward = 0.0
    for _ in range(50):
        action = active.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action.copy())
        total_reward += reward
        if terminated or truncated:
            break

    success = info.get("is_success", False)

    # 写记忆
    if write_memory and collection:
        ctx = build_slide_context(obs, actions, success, len(actions), total_reward)
        ctx_str = json.dumps(ctx, ensure_ascii=False)
        content = f"FetchSlide: {'成功' if success else '失败'}, 距离 {ctx['params']['final_distance']['value']:.3f}m, {len(actions)} 步"
        emb = await embedder.embed_one(content)
        insert_memory(db.conn, {
            "content": content, "context": ctx_str, "collection": collection,
            "session_id": ext_id, "type": "fact",
            "category": classify_category(content),
            "confidence": estimate_confidence(content),
            "embedding": floats_to_blob(emb),
        }, vec_loaded=db.vec_loaded)
        mark_session_ended(db.conn, ext_id)

    return success


async def run_phase(env, policy, use_memory, db, embedder,
                    collection=None, write_memory=False, label=""):
    """执行一个 Phase"""
    successes = 0
    for ep in range(EPISODES):
        ok = await run_slide_episode(
            env, policy, use_memory, db, embedder,
            collection=collection, write_memory=write_memory, ep_num=ep,
        )
        successes += int(ok)
        if (ep + 1) % 10 == 0:
            print(f"  {label} [{ep+1}/{EPISODES}] 成功率: {successes/(ep+1):.0%}")
    return successes / EPISODES


async def main():
    print("=" * 60)
    print("跨环境泛化实验: FetchSlide + SlidePolicy")
    print("=" * 60)

    if not os.path.exists(os.path.join(_EXP_HOME, "memory.db")):
        print("错误: 需要先运行 experiment.py 生成 FetchPush 经验")
        return

    config = load_config()
    db = CogDatabase(config)
    embedder = create_embedder(config)
    if not await embedder.check_availability():
        print(f"Embedder 不可用: {embedder.unavailable_reason}")
        return

    # 统计已有记忆
    push_count = db.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE collection=? AND status='active'",
        (PUSH_COLLECTION,),
    ).fetchone()[0]
    print(f"已有 FetchPush 记忆: {push_count} 条")

    env = gymnasium.make("FetchSlide-v4")
    policy = SlidePolicy()
    t0 = time.time()

    # Phase A: FetchSlide 基线（SlidePolicy，无记忆）
    print(f"\n--- Phase A: FetchSlide 基线（SlidePolicy，无记忆）---")
    rate_a = await run_phase(env, policy, False, db, embedder, label="Phase A")

    # Phase B: FetchSlide + FetchPush 记忆（跨环境）
    print(f"\n--- Phase B: FetchSlide + FetchPush 记忆（跨环境）---")
    rate_b = await run_phase(
        env, policy, True, db, embedder,
        collection=PUSH_COLLECTION, label="Phase B",
    )

    # Phase B-write: 写入 FetchSlide 专属记忆（50 episodes）
    print(f"\n--- 写入 FetchSlide 记忆（{EPISODES} episodes）---")
    rate_bw = await run_phase(
        env, policy, False, db, embedder,
        collection=SLIDE_COLLECTION, write_memory=True, label="Write",
    )

    slide_count = db.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE collection=? AND status='active'",
        (SLIDE_COLLECTION,),
    ).fetchone()[0]
    print(f"  写入 FetchSlide 记忆: {slide_count} 条")

    # Phase C: FetchSlide + FetchSlide 记忆（同环境）
    print(f"\n--- Phase C: FetchSlide + Slide 记忆（同环境）---")
    rate_c = await run_phase(
        env, policy, True, db, embedder,
        collection=SLIDE_COLLECTION, label="Phase C",
    )

    elapsed = time.time() - t0

    # 结果
    delta_cross = rate_b - rate_a
    delta_same = rate_c - rate_a
    verdict_cross = "失败" if delta_cross <= 0 else "微弱" if delta_cross <= 0.05 else "有效" if delta_cross <= 0.15 else "显著"
    verdict_same = "失败" if delta_same <= 0 else "微弱" if delta_same <= 0.05 else "有效" if delta_same <= 0.15 else "显著"

    print(f"\n{'=' * 60}")
    print(f"跨环境泛化结果")
    print(f"{'=' * 60}")
    print(f"  Phase A (基线):        {rate_a:.0%}")
    print(f"  Phase B (Push 记忆):   {rate_b:.0%}  Δ={delta_cross:+.0%} ({verdict_cross})")
    print(f"  Phase C (Slide 记忆):  {rate_c:.0%}  Δ={delta_same:+.0%} ({verdict_same})")
    print(f"  耗时: {elapsed:.0f}s")

    if delta_cross > 0:
        print(f"\n  跨环境: FetchPush 经验对 FetchSlide 有效！")
    else:
        print(f"\n  跨环境: FetchPush 经验对 FetchSlide 无效。")

    if delta_same > 0:
        print(f"  同环境: FetchSlide 记忆对 FetchSlide 有效！")
    else:
        print(f"  同环境: FetchSlide 记忆暂无显著效果。")

    env.close()
    await embedder.close()
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
