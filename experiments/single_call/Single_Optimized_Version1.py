import sys
from pathlib import Path
import json
import re
from typing import TypedDict, List, Dict, Optional
from openai import OpenAI   # 使用 background.docx 里给的初始化方式

# 2026 migration: use the shared local Qwen settings from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from npc_runtime_config import WORKSPACE_ROOT, load_qwen_config

qwen_config = load_qwen_config()
client = OpenAI(
    base_url=qwen_config["base_url"],
    api_key=qwen_config["api_key"],
    timeout=50.0
)

# === State 类型定义 ===
class Notes(TypedDict, total=False):
    dialogue_history: List[str]
    player: str
    self_state: str
    emotion_pool: List[str]
    emotion: Optional[str]
    trigger_status: Dict[str, str]                 # {"0":"yes","1":"no"}
    trigger_goals: Dict[str, str]                  # {"0":"...", "1":"..."}
    trigger_conditions: Dict[str, Optional[str]]   # {"0": None, "1": "keyword:any: leave,exit,..."/regex/.../自然语言句子}
    trigger_cond_present: Dict[str, bool]          # {"0": True, "1": True}
    trigger_queue: List[str]                       # 当轮新达成
    constraints: List[str]

class State(TypedDict):
    npc_profile: str
    scene_context: str
    player_description: str
    world_view: str
    notes: Notes
    final_intention: Optional[str]
    final_behavior: Optional[str]
    valid_input: Optional[bool]

# === 工具函数 ===
def _patch_emotion_pool(text: str, pool: List[str]) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        if line.strip().lower().startswith("- emotion change:"):
            out.append(f"- Emotion Change: [Pool: {', '.join(pool)}]")
        else:
            out.append(line)
    return "\n".join(out)

def _history_indexed_lines(history: List[str]) -> Dict[int, str]:
    return {i+1: line for i, line in enumerate(history or [])}

def _safe_get(d: dict, key: str, default):
    try:
        return d.get(key, default)
    except Exception:
        return default

def _extract_json_block(text: str) -> str:
    if not text:
        return text
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL|re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```+\s*(\{.*?\})\s*```+", text, flags=re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return text.strip()

def normalize_condition_value(val):
    """把 'null'/'None'/''/None 统一为 None，其余保持原样。"""
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in ("null", "none", ""):
        return None
    return val

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _auto_compile_nl_condition_to_rule(nl: str) -> Optional[str]:
    """
    自然语言条件 → 规则：leave/exit/go/out，hostile/angry/threat/attack/shout/yell/fight/hit
    """
    text = _normalize(nl)
    leave_kw = ["leave", "exit", "go", "get out", "leave directly", "walk out", "out"]
    hostile_kw = ["hostility", "hostile", "angry", "threat", "attack", "shout", "yell", "fight", "hit"]
    rules = []
    if any(k in text for k in leave_kw):
        rules.append("keyword:any: leave, exit, go, out")
    if any(k in text for k in hostile_kw):
        rules.append("keyword:any: hostile, hostility, angry, threat, attack, shout, yell, fight, hit")
    if not rules:
        return None
    if len(rules) == 1:
        return rules[0]
    return "keyword:any: leave, exit, go, out, hostile, hostility, angry, threat, attack, shout, yell, fight, hit"

# === 本地触发判断（确定性先行） ===
def evaluate_triggers_local(state: State, latest_player_text: str) -> List[str]:
    new_yes: List[str] = []
    latest = _normalize(latest_player_text)

    trig_status = state['notes'].get('trigger_status', {})
    conds = state['notes'].get('trigger_conditions', {})
    present = state['notes'].get('trigger_cond_present', {})

    for tid, _goal in state['notes']['trigger_goals'].items():
        prev = trig_status.get(tid, "no")
        if prev == "yes":
            continue

        if not present.get(tid, False):
            continue  # <absent> 不在本地升级

        cond_val = conds.get(tid, None)

        # 显式 null：本地直接 yes（初始化也会设）
        if cond_val is None:
            trig_status[tid] = "yes"
            new_yes.append(tid)
            continue

        rule = normalize_condition_value(cond_val)
        if rule is None:
            trig_status[tid] = "yes"; new_yes.append(tid); continue

        rule_s = _normalize(str(rule))

        # 若为自然语言，尝试转规则
        if not (rule_s.startswith("keyword:any:") or rule_s.startswith("keyword:all:") or rule_s.startswith("regex:")):
            auto = _auto_compile_nl_condition_to_rule(rule_s)
            if auto:
                rule_s = auto

        if rule_s.startswith("keyword:any:"):
            words = [w.strip() for w in rule_s.split("keyword:any:",1)[1].split(",") if w.strip()]
            if any(w in latest for w in words):
                trig_status[tid] = "yes"; new_yes.append(tid); continue

        if rule_s.startswith("keyword:all:"):
            words = [w.strip() for w in rule_s.split("keyword:all:",1)[1].split(",") if w.strip()]
            if all(w in latest for w in words):
                trig_status[tid] = "yes"; new_yes.append(tid); continue

        if rule_s.startswith("regex:"):
            pat = rule_s.split("regex:",1)[1].strip()
            try:
                if re.search(pat, latest, flags=re.IGNORECASE):
                    trig_status[tid] = "yes"; new_yes.append(tid); continue
            except re.error:
                pass

    state['notes']['trigger_status'] = trig_status
    return new_yes

# === 主流程（单次调用；触发 goal 作为覆盖项传入，不早停） ===
def intention_and_behavior(state: State) -> State:
    history_list = state['notes'].get('dialogue_history', [])
    hist_indexed = _history_indexed_lines(history_list)
    hist_index_block = "\n".join([f"h-{i}: {txt}" for i, txt in hist_indexed.items()])

    # 触发定义（展示给 LLM 的）
    trig_defs = []
    for tid, goal in state['notes']['trigger_goals'].items():
        cond_present = state['notes']['trigger_cond_present'].get(tid, False)
        cond_val = state['notes']['trigger_conditions'].get(tid, None) if cond_present else None
        cond_text = "null" if (cond_present and cond_val is None) else (cond_val if cond_present else "<absent>")
        trig_defs.append(f"{tid}\tgoal={goal}\tcondition={cond_text}")
    trigger_defs = "\n".join(trig_defs)

    # —— 先做本地触发判断，收集“覆盖目标” —— 
    latest_player_text = ""
    for h in reversed(history_list):
        if h.startswith("【player→NPC】"):
            latest_player_text = h.split("】",1)[1]
            break
    newly_met = evaluate_triggers_local(state, latest_player_text)
    if newly_met:
        state['notes'].setdefault('trigger_queue', [])
        for tid in newly_met:
            if tid not in state['notes']['trigger_queue']:
                state['notes']['trigger_queue'].append(tid)

    # 组装“覆盖目标”：把当轮新达成的 trigger 的 goal 合并，给到 LLM 强制使用
    forced_goal = ""
    if state['notes'].get('trigger_queue'):
        goals = state['notes']['trigger_goals']
        status = state['notes']['trigger_status']
        granted = [goals[tid] for tid in state['notes']['trigger_queue'] if status.get(tid) == "yes"]
        forced_goal = " and ".join(granted) if granted else ""

    # —— 单次 LLM 调用（要求若有 Forced Goal，则必须用它覆盖 final_intention.goal）——
    prompt = f"""
You are an immersive NPC in a role-playing game. Perform a full cycle.
Return strict JSON with keys: analysis, triggers, emotions, candidates, chosen, final_intention, final_behavior.

[NPC Profile]
{state['npc_profile']}

[Scene Context]
{state['scene_context']}

[World View]
{state['world_view']}

[Player Description]
{state['player_description']}

[Dialogue History (indexed)]
{hist_index_block}

[Trigger Definitions]
# Format: <id>\\tgoal=<...>\\tcondition=<... | null | <absent>>
{trigger_defs}

[Valid Input Status]
{state['valid_input']}

[Local Trigger Status — DO NOT DOWNGRADE or contradict]
{json.dumps(state['notes'].get('trigger_status', {}), ensure_ascii=False)}

[Forced Goal Override]
{forced_goal}

---

### Step 1–4. Analysis (compact)
- Constraint lines ONLY, TOTAL <= 8 across [ENV]/[PLAYER]/[SELF]/[DIALOGUE].
- Provide one 'thread_goal' line.

### Step 5. Trigger Fulfillment Evaluation (strict)
Return: {{"<id>": {{"met": true/false, "evidence": ["h-<n>", ...], "reason": "<one line>"}}}}
Rules:
- If condition == null (explicitly null), met=true even without evidence.
- If condition == <absent>, treat unknown and set met=false unless there is clear evidence.
- For any non-null condition, set met=true ONLY IF evidence references existing h-<n> lines (or explicit player description).
- Never contradict Local Trigger Status; never downgrade yes->no; never claim "Condition is null" unless the definition shows null.

### Step 6. Emotion Selection
Return 'emotions' (Top 3).

### Step 7. Candidate Intentions (EXACTLY 3)
Each candidate: {{"type":"conversation","style":"<tone>","goal":"<motivation>","target":"player|object","emotion_change":"","detail":"<intended message or action>"}}

### Step 8. Choose Best
Return 'chosen' as 1|2|3.

### Step 9. Final Intention (with Forced Goal if provided)
- If [Forced Goal Override] is non-empty, you MUST set final_intention.goal = that value.
- Keep type="conversation", fill emotion_change with the first item in 'emotions'.

### Step 10. Behavior Realization
Rules:
- If includes physical states/gestures, wrap them in parentheses BEFORE the speech.
- Natural speech; no meta-comment; ensure continuity; avoid restating long context.
- Tone reflects emotion but do not explicitly name it.
- Use referential continuity when appropriate ("as I said…", "you mentioned…").
- Prefer concrete sensory detail and succinct pacing.
"""
    resp = client.chat.completions.create(
        model=qwen_config["model"],
        extra_body=qwen_config["extra_body"],
        messages=[{"role": "user", "content": prompt}]
    )
    raw = resp.choices[0].message.content.strip()
    cleaned = _extract_json_block(raw)
    print("\n=== Raw LLM Output ===\n", raw, "\n")

    # 解析
    try:
        data = json.loads(cleaned)
    except Exception as e:
        print(f"⚠️ JSON 解析失败：{e}. 回退为原文行为。")
        state['final_behavior'] = raw
        return state

    # 合并触发：只允许 no->yes；显式 null 永远 yes；<absent>/非空须证据
    old_status = state['notes'].get('trigger_status', {}).copy() or {}
    report = _safe_get(data, "triggers", {})
    merged = old_status.copy()
    queue = state['notes'].setdefault('trigger_queue', [])

    # 索引存在性表
    history_indexed = {i for i in _history_indexed_lines(state['notes'].get('dialogue_history', [])).keys()}

    for tid in state['notes']['trigger_goals'].keys():
        cond_present = state['notes']['trigger_cond_present'].get(tid, False)
        cond_val = state['notes']['trigger_conditions'].get(tid, None) if cond_present else None
        prev = old_status.get(tid, "no")

        if cond_present and cond_val is None:
            merged[tid] = "yes"
            if prev != "yes" and tid not in queue:
                queue.append(tid)
            continue

        r = report.get(tid, {})
        met = bool(r.get("met", False))
        evidence = r.get("evidence", [])
        reason = (r.get("reason") or "").lower()

        valid_ev = any(
            isinstance(ev, str) and ev.startswith("h-") and ev[2:].isdigit() and int(ev[2:]) in history_indexed
            for ev in (evidence or [])
        )
        if "null" in reason and not (cond_present and cond_val is None):
            met = False; valid_ev = False

        if prev != "yes":
            if met and valid_ev:
                merged[tid] = "yes"
                if tid not in queue:
                    queue.append(tid)
            else:
                merged[tid] = "no"
        else:
            merged[tid] = "yes"

    state['notes']['trigger_status'] = merged

    # —— 结果落盘：强制 goal 已经由 LLM 执行在 final_intention 内（Step 9）
    # 为稳妥：如果我们本地传了 forced_goal，但模型没改，就再本地覆盖一次 final_intention.goal
    forced_goal_safe = forced_goal.strip()
    emotions = _safe_get(data, "emotions", []) or ["neutral"]
    fi = _safe_get(data, "final_intention", None)
    if isinstance(fi, dict):
        if forced_goal_safe:
            fi["goal"] = forced_goal_safe  # 二次兜底覆盖
        if "emotion_change" not in fi or not fi["emotion_change"]:
            fi["emotion_change"] = emotions[0]
        state['final_intention'] = json.dumps(fi, ensure_ascii=False, indent=2)
    else:
        # 如果不是 dict，就尽力构造一个
        state['final_intention'] = json.dumps({
            "type": "conversation",
            "style": "accommodating",
            "goal": forced_goal_safe if forced_goal_safe else "N/A",
            "target": "player",
            "emotion_change": emotions[0],
            "detail": "..."
        }, ensure_ascii=False, indent=2)

    state['final_behavior'] = _safe_get(data, "final_behavior", "...")
    # 本轮触发队列已用过，这里清空
    state['notes']['trigger_queue'] = []
    return state

# === 对话循环 ===
def dialogue_loop(state: State):
    state['notes']['dialogue_history'] = state['notes'].get('dialogue_history', [])
    while True:
        player_input = input("\n🎮 Player: ")
        if player_input.lower().strip() == "exit":
            print("🛑 Exit."); break
        validTF = input("Is this input valid? (yes/no): ").strip().lower()
        state['valid_input'] = (validTF == "yes")
        state['notes'].setdefault('dialogue_history', []).append(f"【player→NPC】{player_input}")
        state = intention_and_behavior(state)
        print(f"\n🗣️ NPC: {state['final_behavior']}")
        state['notes']['dialogue_history'].append(f"【NPC→player】{state['final_behavior']}")

# === 示例入口 ===
def main():
    # 2026 migration: load the copied scenario materials within this workspace.
    data_dir = WORKSPACE_ROOT / "data" / "story"
    npc_json = data_dir / "NPC_HuangQiye.json"
    scene_json = data_dir / "Scene_Beginning.json"
    world_json = data_dir / "WorldView.json"

    with open(npc_json, "r", encoding="utf-8") as f:
        npc_data = json.load(f)
    npc_profile = f"- Name: {npc_data.get('name','')}\n- Personality: {npc_data['basic_information'].get('personality','')}"

    with open(scene_json, "r", encoding="utf-8") as f:
        scene = json.load(f)
    scene_context = scene.get("environment", "")

    # 读取 emotion_pool[*].trigger_condition
    trigger_goals: Dict[str, str] = {}
    trigger_conditions: Dict[str, Optional[str]] = {}
    trigger_cond_present: Dict[str, bool] = {}

    inter = scene.get("interactive_npc", [])
    if inter:
        npc_info = inter[0]
        for emo in npc_info.get("emotion_pool", []):
            tid = str(emo.get("id"))
            trigger_goals[tid] = emo.get("goal")
            # 注意：字段名是 trigger_condition
            if "trigger_condition" in emo:
                raw = emo.get("trigger_condition")  # 可能是 None / "null" / 自然语言 / 规则串
                trigger_conditions[tid] = normalize_condition_value(raw)
                trigger_cond_present[tid] = True
            else:
                trigger_conditions[tid] = None
                trigger_cond_present[tid] = False

    with open(world_json, "r", encoding="utf-8") as f:
        world = json.load(f)

    # 初始化触发状态：显式 null → yes 入队；其余 no
    trigger_status: Dict[str, str] = {}
    trigger_queue: List[str] = []
    for tid in trigger_goals.keys():
        cond_present = trigger_cond_present.get(tid, False)
        cond_val = trigger_conditions.get(tid, None) if cond_present else None
        if cond_present and cond_val is None:
            trigger_status[tid] = "yes"
            trigger_queue.append(tid)
        else:
            trigger_status[tid] = "no"

    state: State = {
        "npc_profile": npc_profile,
        "scene_context": scene_context,
        "world_view": json.dumps(world, ensure_ascii=False),
        "player_description": "",
        "notes": {
            "trigger_goals": trigger_goals,
            "trigger_conditions": trigger_conditions,
            "trigger_cond_present": trigger_cond_present,
            "trigger_status": trigger_status,
            "trigger_queue": trigger_queue,
            "dialogue_history": []
        },
        "final_intention": None,
        "final_behavior": None,
        "valid_input": None
    }
    dialogue_loop(state)

if __name__ == "__main__":
    main()


