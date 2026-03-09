"""FetchPush-v4 仿真实验 — 验证 robotmem 记忆驱动决策

来源：圆桌讨论（机器人专家1，2026-03-09）
Issue: #16
结论文档: project-docs/roundtable/robotmem-core-competition-20260309/conclusion.md

三阶段实验:
  Phase A: 100 episodes 基线（heuristic，无记忆）
  Phase B: 100 episodes 记忆写入（learn + save_perception）
  Phase C: 100 episodes 记忆利用（recall → MemoryPolicy）

集成方式: 直接 import robotmem ops 层（Round 2 Brooks 共识）
数据隔离: ROBOTMEM_HOME 指向实验专用目录（Round 4 Brooks/Fox）
实验结果: 存入 robotmem DB，Web UI 查看（不存独立文件）

运行方式:
  cd examples/fetch_push
  source .venv/bin/activate
  PYTHONPATH=../../src ROBOTMEM_HOME=.robotmem python experiment.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time

# 数据隔离 — 必须在 import robotmem 之前设置（config.py 在 import 时计算 ROBOTMEM_HOME）
_EXP_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem")
os.environ.setdefault("ROBOTMEM_HOME", _EXP_HOME)
os.makedirs(_EXP_HOME, exist_ok=True)

import gymnasium_robotics  # noqa: F401 — 注册 Fetch 环境
import gymnasium
import numpy as np

# robotmem 直接 import（绕过 MCP/Pydantic，Round 4 Levine 共识）
from robotmem.config import load_config
from robotmem.db_cog import CogDatabase
from robotmem.embed import create_embedder
from robotmem.search import recall as do_recall
from robotmem.db import floats_to_blob
from robotmem.ops.memories import insert_memory, consolidate_session
from robotmem.ops.sessions import get_or_create_session, mark_session_ended
from robotmem.auto_classify import classify_category, estimate_confidence

from policies import HeuristicPolicy, PhaseAwareMemoryPolicy

# ── 配置 ──

COLLECTION = "exp_run_001"
PHASE_A_EPISODES = 100
PHASE_B_EPISODES = 100
PHASE_C_EPISODES = 100
B_CHECK_AT = 50  # Phase B 中间抽检点（Round 3 Rus）
MEMORY_WEIGHT = 0.3
RECALL_N = 5


def build_context_json(obs, actions_history, success, steps, total_reward):
    """构建 context JSON 四分区（Round 4 Abbeel 终版）"""
    # 最后 10 步平均动作作为 approach_velocity
    recent_actions = actions_history[-10:] if len(actions_history) >= 10 else actions_history
    avg_action = np.mean(recent_actions, axis=0) if recent_actions else np.zeros(4)

    final_obs = obs["observation"]
    return {
        "params": {
            "approach_velocity": {
                "value": avg_action[0:3].tolist(),
                "type": "vector",
            },
            "grip_force": {
                "value": float(avg_action[3]),
                "type": "scalar",
                "range": [-1, 1],
            },
            "final_distance": {
                "value": float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])),
                "unit": "m",
                "type": "scalar",
            },
        },
        "spatial": {
            "frame": "world",
            "grip_position": final_obs[0:3].tolist(),
            "object_position": final_obs[3:6].tolist(),
            "target_position": obs["desired_goal"].tolist(),
            "scene_tag": "tabletop_push",
        },
        "robot": {
            "id": "fetch-001",
            "type": "Fetch",
            "end_effector": "parallel_gripper",
            "dof": 7,
        },
        "task": {
            "name": "push_to_target",
            "object": "cube",
            "success": bool(success),
            "steps": steps,
            "total_reward": float(total_reward),
        },
    }


async def run_episode(env, policy, phase, ep_num, db, embedder):
    """执行单个 episode，返回 (success, final_distance, recall_count)"""
    ext_id = f"ep_{phase}_{ep_num:03d}"
    session = get_or_create_session(db.conn, ext_id, COLLECTION)
    if not session:
        print(f"  警告: session 创建失败 ({ext_id})")
        return False, 1.0, 0

    # 跑 episode — 先 reset 获取空间信息，再 recall
    obs, info = env.reset()

    # Phase C: recall（含空间信息的动态查询 + spatial_sort）
    recalled = []
    if phase == "C":
        obj_pos = obs["observation"][3:6].tolist()
        target_pos = obs["desired_goal"].tolist()
        query = f"push from [{obj_pos[0]:.2f}, {obj_pos[1]:.2f}] to [{target_pos[0]:.2f}, {target_pos[1]:.2f}]"
        result = await do_recall(
            query,
            db, embedder,
            collection=COLLECTION, top_k=RECALL_N,
            context_filter={"task.success": True},
            spatial_sort={"field": "spatial.object_position", "target": obj_pos},
        )
        recalled = result.memories

    # 构建策略（PhaseAwareMemoryPolicy: 只在推送阶段施加 bias）
    if recalled:
        active_policy = PhaseAwareMemoryPolicy(policy, recalled, memory_weight=MEMORY_WEIGHT)
    else:
        active_policy = policy
    actions_history = []
    total_reward = 0.0
    steps = 0

    for step in range(50):
        action = active_policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        actions_history.append(action.copy())
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    success = info.get("is_success", False)
    final_distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))

    # Phase B/C: 写入记忆
    if phase in ("B", "C"):
        context_json = build_context_json(obs, actions_history, success, steps, total_reward)
        context_str = json.dumps(context_json, ensure_ascii=False)

        # learn: 经验摘要
        content = f"FetchPush episode: {'成功' if success else '失败'}, 最终距离 {final_distance:.3f}m, {steps} 步"
        embedding = await embedder.embed_one(content)
        insert_memory(db.conn, {
            "content": content,
            "context": context_str,
            "collection": COLLECTION,
            "session_id": ext_id,
            "type": "fact",
            "category": classify_category(content),
            "confidence": estimate_confidence(content),
            "embedding": floats_to_blob(embedding),
        }, vec_loaded=db.vec_loaded)

        # save_perception: 轨迹摘要（每 10 步采样）
        sampled_actions = [list(actions_history[i]) for i in range(0, len(actions_history), 10)]
        perception_content = (
            f"push 轨迹: {steps} 步, "
            f"物体从 {obs['observation'][3:6].tolist()} 朝目标 {obs['desired_goal'].tolist()}"
        )
        p_embedding = await embedder.embed_one(perception_content)
        insert_memory(db.conn, {
            "content": perception_content,
            "context": context_str,
            "collection": COLLECTION,
            "session_id": ext_id,
            "type": "perception",
            "perception_type": "procedural",
            "perception_data": json.dumps({"sampled_actions": sampled_actions}),
            "embedding": floats_to_blob(p_embedding),
        }, vec_loaded=db.vec_loaded)

    # 结束 session
    mark_session_ended(db.conn, ext_id)
    if phase in ("B", "C"):
        try:
            consolidate_session(db.conn, ext_id, COLLECTION)
        except Exception as e:
            print(f"  警告: consolidate_session 失败 ({ext_id}): {e}")

    return success, final_distance, len(recalled)


async def check_recall(db, embedder, n_queries=5):
    """Phase B 中间抽检（Round 3 Rus 快速失败机制）"""
    queries = [
        "push cube to target position",
        "成功 push 经验",
        "FetchPush 最终距离小",
        "推物体到目标",
        "push trajectory tabletop",
    ]
    hits = 0
    for q in queries[:n_queries]:
        result = await do_recall(q, db, embedder, collection=COLLECTION, top_k=3)
        if result.memories:
            hits += 1
    return hits, n_queries


async def run_phase(env, policy, phase, episodes, db, embedder):
    """执行一个 Phase，返回统计"""
    successes = 0
    distances = []
    recall_counts = []

    for ep in range(episodes):
        success, dist, n_recalled = await run_episode(env, policy, phase, ep, db, embedder)
        successes += int(success)
        distances.append(dist)
        recall_counts.append(n_recalled)

        if (ep + 1) % 20 == 0:
            rate = successes / (ep + 1)
            print(f"  Phase {phase} [{ep+1}/{episodes}] 成功率: {rate:.1%} 平均距离: {np.mean(distances):.3f}")

        # Phase B 中间抽检
        if phase == "B" and ep + 1 == B_CHECK_AT:
            hits, total = await check_recall(db, embedder)
            print(f"\n  >>> Phase B 抽检 ({B_CHECK_AT} ep): recall 命中 {hits}/{total}")
            if hits == 0:
                print("  >>> 命中率为零，停止实验，转 #17 搜索层分析")
                return {
                    "phase": phase,
                    "episodes": ep + 1,
                    "success_rate": successes / (ep + 1),
                    "avg_distance": float(np.mean(distances)),
                    "aborted": True,
                    "abort_reason": "recall 命中率为零",
                }
            print(f"  >>> 命中率 {hits}/{total}，继续\n")

    return {
        "phase": phase,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "avg_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "avg_recall_count": float(np.mean(recall_counts)) if recall_counts else 0,
        "aborted": False,
    }


async def learn_summary(db, embedder, phase_results):
    """将 Phase 汇总存入 DB（替代 metrics.json）"""
    r = phase_results
    content = (
        f"实验汇总 Phase {r['phase']}: "
        f"成功率 {r['success_rate']:.1%}, "
        f"{r['episodes']} episodes, "
        f"平均距离 {r['avg_distance']:.3f}m"
    )
    context = json.dumps({"task": {
        "phase": r["phase"],
        "success_rate": r["success_rate"],
        "episodes": r["episodes"],
        "avg_distance": r["avg_distance"],
    }}, ensure_ascii=False)

    embedding = await embedder.embed_one(content)
    insert_memory(db.conn, {
        "content": content,
        "context": context,
        "collection": COLLECTION,
        "category": "observation",
        "confidence": 0.95,
        "embedding": floats_to_blob(embedding),
    }, vec_loaded=db.vec_loaded)


async def main():
    print("=" * 60)
    print("robotmem FetchPush-v4 仿真实验")
    print(f"Collection: {COLLECTION}")
    print(f"ROBOTMEM_HOME: {os.environ['ROBOTMEM_HOME']}")
    print("=" * 60)

    # 初始化（CogDatabase 通过 conn property 延迟连接）
    config = load_config()
    db = CogDatabase(config)
    embedder = create_embedder(config)

    # 检查 embedder 可用性
    available = await embedder.check_availability()
    if not available:
        print(f"错误: Embedder 不可用 — {embedder.unavailable_reason}")
        return
    print(f"Embedder: {embedder.model} ({embedder.dim}d)")

    env = gymnasium.make("FetchPush-v4")
    base_policy = HeuristicPolicy()

    # Phase A: 基线
    print(f"\n{'─'*40}")
    print(f"Phase A: 基线（{PHASE_A_EPISODES} episodes，无记忆）")
    print(f"{'─'*40}")
    t0 = time.time()
    result_a = await run_phase(env, base_policy, "A", PHASE_A_EPISODES, db, embedder)
    t_a = time.time() - t0
    print(f"\n  Phase A 完成: 成功率 {result_a['success_rate']:.1%}, 耗时 {t_a:.1f}s")
    await learn_summary(db, embedder, result_a)

    # Phase B: 记忆写入
    print(f"\n{'─'*40}")
    print(f"Phase B: 记忆写入（{PHASE_B_EPISODES} episodes）")
    print(f"{'─'*40}")
    t0 = time.time()
    result_b = await run_phase(env, base_policy, "B", PHASE_B_EPISODES, db, embedder)
    t_b = time.time() - t0
    print(f"\n  Phase B 完成: 成功率 {result_b['success_rate']:.1%}, 耗时 {t_b:.1f}s")
    await learn_summary(db, embedder, result_b)

    if result_b.get("aborted"):
        print(f"\n实验中止: {result_b['abort_reason']}")
        env.close()
        db.close()
        return

    # Phase C: 记忆利用
    print(f"\n{'─'*40}")
    print(f"Phase C: 记忆利用（{PHASE_C_EPISODES} episodes，recall → MemoryPolicy）")
    print(f"{'─'*40}")
    t0 = time.time()
    result_c = await run_phase(env, base_policy, "C", PHASE_C_EPISODES, db, embedder)
    t_c = time.time() - t0
    print(f"\n  Phase C 完成: 成功率 {result_c['success_rate']:.1%}, 耗时 {t_c:.1f}s")
    await learn_summary(db, embedder, result_c)

    # 最终汇总
    delta = result_c["success_rate"] - result_a["success_rate"]
    if delta <= 0:
        verdict = "失败"
    elif delta <= 0.05:
        verdict = "微弱"
    elif delta <= 0.15:
        verdict = "有效"
    else:
        verdict = "显著"

    print(f"\n{'='*60}")
    print("实验结果")
    print(f"{'='*60}")
    print(f"  Phase A 成功率: {result_a['success_rate']:.1%}")
    print(f"  Phase B 成功率: {result_b['success_rate']:.1%}")
    print(f"  Phase C 成功率: {result_c['success_rate']:.1%}")
    print(f"  提升: {delta:+.1%}")
    print(f"  判定: {verdict}（四级标准: 失败/微弱/有效/显著）")
    print(f"  Phase C 平均 recall 数: {result_c['avg_recall_count']:.1f}")
    print(f"  总耗时: {t_a + t_b + t_c:.1f}s")

    # 写入最终汇总
    final_content = (
        f"实验最终结论: Phase A {result_a['success_rate']:.1%} → Phase C {result_c['success_rate']:.1%}, "
        f"提升 {delta:+.1%}, 判定: {verdict}"
    )
    final_embedding = await embedder.embed_one(final_content)
    insert_memory(db.conn, {
        "content": final_content,
        "context": json.dumps({
            "task": {
                "experiment": "fetch_push_v4",
                "phase_a_rate": result_a["success_rate"],
                "phase_c_rate": result_c["success_rate"],
                "delta": delta,
                "verdict": verdict,
                "total_episodes": PHASE_A_EPISODES + PHASE_B_EPISODES + PHASE_C_EPISODES,
            }
        }, ensure_ascii=False),
        "collection": COLLECTION,
        "category": "observation",
        "confidence": 0.99,
        "embedding": floats_to_blob(final_embedding),
    }, vec_loaded=db.vec_loaded)

    env.close()
    await embedder.close()
    db.close()
    print(f"\n结果已存入 DB，用 Web UI 查看: ROBOTMEM_HOME={_EXP_HOME} python -m robotmem web --port 7888")


if __name__ == "__main__":
    asyncio.run(main())
