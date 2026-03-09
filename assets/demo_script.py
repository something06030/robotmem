"""Terminal demo for README GIF recording"""
import sqlite3
import sys
import time
import os

# 隔离 demo DB
os.environ["ROBOTMEM_HOME"] = "/tmp/robotmem-demo"
os.makedirs("/tmp/robotmem-demo", exist_ok=True)

# 清理旧 demo 数据
db_path = "/tmp/robotmem-demo/memory.db"
if os.path.exists(db_path):
    os.remove(db_path)

def slow_print(text, delay=0.02):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def section(title):
    print()
    slow_print(f"\033[1;32m>>> {title}\033[0m", 0.015)
    time.sleep(0.3)

# ── 导入 ──
section("Import robotmem")
slow_print("from robotmem.db_cog import CogDatabase")
slow_print("from robotmem.ops.memories import insert_memory")
slow_print("from robotmem.ops.sessions import get_or_create_session, mark_session_ended")
slow_print("from robotmem.ops.search import fts_search_memories")
time.sleep(0.3)

from robotmem.config import Config
from robotmem.db_cog import CogDatabase
from robotmem.ops.memories import insert_memory
from robotmem.ops.sessions import get_or_create_session, mark_session_ended
from robotmem.ops.search import fts_search_memories

cfg = Config()
db = CogDatabase(cfg)

# ── Start Session ──
section("1. Start episode")
session = get_or_create_session(db.conn, external_id=None, collection="fetch-push")
sid = str(session["id"])
slow_print(f"  Session: \033[1;36m{sid}\033[0m  (collection: fetch-push)")
time.sleep(0.3)

# ── Learn ──
section("2. Learn from experience")
experiences = [
    ("grip_force=12.5N gives best success rate for cube pushing",
     '{"params": {"grip_force": {"value": 12.5, "unit": "N"}}, "task": {"success": true}}'),
    ("approach_speed=0.03 m/s prevents object displacement before grasp",
     '{"params": {"approach_speed": {"value": 0.03, "unit": "m/s"}}, "task": {"success": true}}'),
    ("target overshoot when push_force > 8N on smooth surface",
     '{"params": {"push_force": {"value": 8.0, "unit": "N"}}, "task": {"success": false}}'),
]

for insight, ctx in experiences:
    mem = {
        "content": insight, "context": ctx,
        "session_id": int(sid), "memory_type": "fact",
        "source": "experiment", "collection": "fetch-push",
    }
    mid = insert_memory(db.conn, mem)
    slow_print(f"  \033[1;33m+\033[0m {insight[:55]}...")
    time.sleep(0.2)

# ── Recall ──
section("3. Recall relevant experiences")
slow_print('  query: "force parameters for pushing"')
time.sleep(0.3)

results = fts_search_memories(db.conn, "force parameters pushing", collection="fetch-push", limit=5)
slow_print(f"  Found \033[1;36m{len(results)}\033[0m memories:")
for i, r in enumerate(results):
    content = r.get("content", r.get("assertion", ""))
    slow_print(f"    {i+1}. {content[:60]}...")
    time.sleep(0.15)

# ── End Session ──
section("4. End episode")
mark_session_ended(db.conn, int(sid))
slow_print(f"  Session \033[1;36m{sid[:8]}...\033[0m closed")
slow_print(f"  Memories persisted to \033[1;33m~/.robotmem/memory.db\033[0m")
time.sleep(0.3)

# ── Summary ──
count = db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
print()
slow_print(f"\033[1;32m✅ {count} memories stored. Ready for next episode.\033[0m")
print()

db.close()
