import sys
from pathlib import Path
import json
import re
from typing import TypedDict, List, Dict, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 纠错版本

# 2026 migration: use the shared local Qwen settings from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from npc_runtime_config import WORKSPACE_ROOT, load_qwen_config

qwen_config = load_qwen_config()
llm = ChatOpenAI(**qwen_config)

# === OCEAN 五大性格框架说明（整体文段） ===
OCEAN_EXPLANATION = """
# OCEAN Big Five Personality Framework (scores 0–5)

**Openness**  
Definition: Interest in new experiences, ideas, creativity, and imagination.  
- Low (0): Conventional, practical, prefers routine and tradition.  
- High (5): Curious, imaginative, enjoys novelty and abstract thinking.

**Conscientiousness**  
Definition: Level of self-discipline, organization, reliability, and goal-directed behavior.  
- Low (0): Spontaneous, disorganized, easily distracted, lacks follow-through.  
- High (5): Thorough, dependable, plans ahead, sees tasks through to completion.

**Extraversion**  
Definition: Sociability, energy, assertiveness, and preference for stimulation.  
- Low (0): Reserved, quiet, prefers solitude or small groups.  
- High (5): Outgoing, talkative, seeks social engagement and excitement.

**Agreeableness**  
Definition: Compassion, cooperativeness, and trust in others.  
- Low (0): Competitive, skeptical, does not easily trust or cooperate.  
- High (5): Friendly, empathetic, helpful, values harmony.

**Neuroticism**  
Definition: Tendency toward emotional instability, anxiety, and moodiness.  
- Low (0): Calm, even-tempered, resilient under stress.  
- High (5): Prone to worry, easily upset, experiences mood swings.
"""


# === 接口函数：从 JSON 文件加载新设定 ===
def load_npc_profile(json_path: str) -> str:
    """
    接口：加载 NPC 个人信息并格式化为 prompt 所需的字符串。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    basic = data['basic_information']
    lines: List[str] = []
    # 基本属性
    lines.append(f"- Name: {data.get('name', '')}")
    if data.get('nickname'):
        lines.append(f"- Nickname: {data['nickname']}")
    lines.append(f"- Background: {basic.get('background', '')}")
    lines.append(f"- Personality: {basic.get('personality', '')}")
    lines.append(f"- Obsession: {basic.get('obsession', '')}")
    lines.append(f"- Initial Goal: {basic.get('initial_goal', '')}")
    lines.append(f"- Attitude toward Yaoguai: {basic.get('attitude_toward_yaoguai', '')}")
    # OCEAN 分数转换
    scores = basic.get('big_five_scores', {})
    oceans: List[str] = []
    for dim, val in scores.items():
        score = round(val / 2 * 2) / 2  # 四舍五入到最近的 0.5
        oceans.append(f"{dim}:{score}")
    lines.append("# OCEAN Profile: " + ", ".join(oceans))
    # 整体插入 OCEAN 说明文段
    lines.append("")
    lines.append(OCEAN_EXPLANATION.strip())
    return "\n".join(lines)


def load_scene_context(json_path: str) -> str:
    """
    接口：加载场景信息并格式化为 prompt 所需的字符串。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    env_text = data.get('environment', '')
    return "\n".join([f"- {line.strip()}" for line in env_text.splitlines() if line.strip()])


def load_scene_triggers_and_general(json_path: str) -> tuple[Dict[str, str], str]:
    """
    接口：从 scene JSON 中提取 emotion_pool 里的 trigger_conditions 映射 和 scene_general_goal。
    返回 trigger_goals: {id: goal}, scene_general_goal: str
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    interactive = data.get('interactive_npc', [])
    trigger_goals: Dict[str, str] = {}
    scene_general_goal = ''
    if interactive:
        npc_info = interactive[0]
        scene_general_goal = npc_info.get('goal', '')
        for emo in npc_info.get('emotion_pool', []):
            tid = emo.get('id')
            goal = emo.get('goal')
            if tid is not None and goal is not None:
                trigger_goals[str(tid)] = goal
    return trigger_goals, scene_general_goal

def load_world_config(json_path: str) -> Dict:
    """
    接口：加载全局世界观设定，返回原始 JSON 数据以供扩展使用。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# === State 类型定义（保持不变） ===
class Notes(TypedDict, total=False):
    environment: str
    player: str
    self_state: str
    dialogue: str
    constraints: List[str]
    emotion: Optional[str]
    emotion_pool: List[str]
    dialogue_history: List[str]
    trigger_status: Dict[str, str]
    trigger_goals: Dict[str, Optional[str]]
    tradeoff_summary: str

class State(TypedDict):
    npc_profile: str
    scene_context: str
    player_description: str
    notes: Notes
    decision: Optional[str]
    initiative: Optional[bool]
    candidates: List[str]
    final_intention: Optional[str]
    final_behavior: Optional[str]
    use_tradeoff_evaluation: Optional[bool]

# === 可选的 OCEAN 性格配置（保留供参考，但不再用于对话中选择） ===
personality_profiles: Dict[str, Dict[str, float]] = {
    "Default": {"Openness":2.0, "Conscientiousness":4.5, "Extraversion":2.0, "Agreeableness":3.5, "Neuroticism":1.5},
    "Irritable": {"Openness":2.0, "Conscientiousness":3.0, "Extraversion":2.0, "Agreeableness":1.5, "Neuroticism":4.0},
    "Timid": {"Openness":2.0, "Conscientiousness":3.0, "Extraversion":1.0, "Agreeableness":3.0, "Neuroticism":4.5},
    "Arrogant": {"Openness":3.5, "Conscientiousness":2.5, "Extraversion":4.0, "Agreeableness":1.0, "Neuroticism":2.0},
    "Friendly": {"Openness":3.0, "Conscientiousness":3.5, "Extraversion":3.5, "Agreeableness":5.0, "Neuroticism":1.5},
}


def _patch_emotion_pool(text: str, pool: List[str]) -> str:
    lines = text.splitlines()
    new_lines = []

    for line in lines:
        if line.strip().lower().startswith("- emotion change:"):
            # 删除原有行，替换成新的
            new_line = f"- Emotion Change: [Pool: {', '.join(pool)}]"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)

def clean_emotion_text(text: str) -> List[str]:
    # 移除 markdown ** 加粗符号
    cleaned = re.sub(r"\*\*", "", text)
    return [e.strip() for e in cleaned.split(',') if e.strip()]

# === 模块函数定义 ===
def combined_analysis_node(state: State) -> State:
    dialogue_summary = "\n".join(state['notes'].get('dialogue_history', []))
    old_status = state['notes'].get('trigger_status', {}).copy()
    trigger_defs = "\n".join(f"Trigger {tid}: {goal}" for tid, goal in state['notes']['trigger_goals'].items())
    prompt = f"""
You are analyzing the NPC's situation across the following six structured steps. Each step has a defined task. Generate the analysis of different parts separately in the form below.

[NPC Profile]
{state['npc_profile']}

[Scene Context]
{state['scene_context']}

[Player Description]
{state['player_description']}

[Dialogue History]
{dialogue_summary}

[Trigger Definitions]
{trigger_defs}

[Contextual Continuity Reminder]
The dialogue history is not static. You are responding in an ongoing conversation. Your current response must reflect the emotional tone, topic, or inferred intent from the last player input. Each response is a reply to what the current player has said and a continuation of what you said last time. Don't treat each sentence as an independent reply. Avoid restarting or repeating information unless it is necessary.

---
[Environment Analysis]
Task: Evaluate the current physical environment and its impact on NPC awareness, decision-making, and safety.
Summary: <...>
Constraints:
- [ENV] ...
- [ENV] ...
- [ENV] ...
---
[Player Analysis]
Task: Analyze the player's appearance, behavior, words and any known information. Mainly focus on evaluating the purpose player's intentnion and their possible underlying meanings. Evaluate the rationality of the player's statements (including whether the player should know this information and whether it is nonsense).
Summary: <...>
Constraints:
- [PLAYER] ...
- [PLAYER] ...
- [PLAYER] ...
---
[Self Analysis]
Task: Reflect on the NPC's personality traits, past experiences, goals, psychological state, and any information about the NPC. Identify how these internal states and external information influence current priorities and decisions.
Summary: <...>
Constraints:
- [SELF] ...
- [SELF] ...
- [SELF] ...
---
[Dialogue Thread Analysis]
Task: Identify what ongoing topic, conversational thread, or goal the NPC is currently pursuing. Consider whether the NPC is trying to confirm identity, de-escalate tension, extract information, etc. Suggest whether to maintain, shift, or escalate the topic based on the history.
Summary: <...>
Current Thread Goal: <e.g., confirm identity / calm threat / delay / request info>
Constraints:
- [DIALOGUE] ...
- [DIALOGUE] ...
- [DIALOGUE] ...
---
[Trigger Fulfillment Evaluation]
Task: …
{trigger_defs.replace(':', ' Met: <yes/no>:')}
---
[Emotion Selection]
Task: Based strictly on the self-analysis above and NPC's personality, identify five reasonable emotions that reflect the NPC's current internal state. From those, choose three that best represent the dominant emotional tone. These will serve as the only valid options for Emotion Change in subsequent intentions.
All Relevant Emotions: <five emotion words>
Top 3 Representative Emotions: <three emotion words>
"""
    response = llm.invoke(prompt)
    content = response.content.strip()
    print(f"\n📝 [combined_analysis] node output:\n{content}\n")

    # 解析并存储各触发条件状态
    triggers: Dict[str, str] = {}
    if 'Trigger Fulfillment Evaluation' in content:
        for line in content.splitlines():
            m = re.match(r'Trigger\s*(\d+)\s*Met:\s*(yes|no)', line, re.IGNORECASE)
            if m:
                triggers[m.group(1)] = m.group(2).lower()
    state['notes']['trigger_status'] = triggers

    # 更新 trigger_queue: 收集状态变化的 trigger id
    queue = state['notes'].setdefault('trigger_queue', [])
    for tid, new_st in triggers.items():
        old_st = old_status.get(tid)
        if new_st == 'yes' and old_st != 'yes':
            queue.append(tid)

    # 解析 constraints 来源
    try:
        parts = content.split('---')
        for part in parts:
            tag = None
            if 'Environment Analysis' in part:
                state['notes']['environment'] = part.strip(); tag='[ENV]'
            elif 'Player Analysis' in part:
                state['notes']['player'] = part.strip(); tag='[PLAYER]'
            elif 'Self Analysis' in part:
                state['notes']['self_state'] = part.strip(); tag='[SELF]'
            elif 'Dialogue Thread Analysis' in part:
                state['notes']['dialogue'] = part.strip(); tag='[DIALOGUE]'
            elif 'Top 3 Representative Emotions' in part:
                lines = part.splitlines()
                pool_line = next((l for l in lines if 'Top 3 Representative Emotions' in l), None)
                if pool_line:
                    pool = pool_line.split(':')[-1].strip()
                    emotions = clean_emotion_text(pool)
                    if emotions:
                        state['notes']['emotion_pool'] = emotions
                        state['notes']['emotion'] = emotions[0]
                        print("🎯 Emotion Pool Set:", emotions)
                    else:
                        print("⚠️ Emotion line found but no emotions extracted.")
                else:
                    print("⚠️ No emotion line found in the analysis.")
            else:
                print(f"⚠️ Unrecognized part in analysis: {part.strip()[:30]}...")


            if tag:
                constraints = part.split('Constraints:')[-1].strip().splitlines()
                for c in constraints:
                    if c.strip().startswith(tag):
                        state['notes'].setdefault('constraints', []).append(c.strip())
    except Exception as e:
        pass
    return state


def evaluate_tradeoffs(state: State) -> State:
    summary = "\n".join(f"{k}: {v}" for k, v in state['notes'].items())
    prompt = f"""
You are performing a tradeoff evaluation for the NPC's next possible strategies.
Based on the following context, propose 2 or 3 possible strategies.
For each strategy, analyze the pros and cons from the following 3 perspectives:
- Interpersonal (relationships and their influence on achieving the goal)
- Personal Gain (benefits or losses to the NPC)
- Moral Duty (alignment with the NPC's values or responsibilities)
Finally, recommend one strategy that best supports the NPC's long-term goal.

[NPC Profile]
{state['npc_profile']}

[Scene Context]
{state['scene_context']}

[Player Description]
{state['player_description']}

[Combined Analysis and Dialogue History]
{summary}

Output Format:
Strategy 1: <description>
Pros:
Interpersonal: ...
Personal Gain: ...
Moral Duty: ...
Cons:
Interpersonal: ...
Personal Gain: ...
Moral Duty: ...

Strategy 2: ...
...

Suggested Best Strategy: <your choice with reasoning>
"""
    resp = llm.invoke(prompt)
    result = resp.content.strip()
    print("\n📊 [Tradeoff Evaluation Output]:\n" + result + "\n")
    state['notes']['tradeoff_summary'] = result
    return state


def generate_candidates(state: State) -> State:
    summary = '\n'.join(f"{k}: {v}" for k, v in state['notes'].items() if k != 'constraints')
    constraints = '\n'.join(state['notes'].get('constraints', []))
    pool = state['notes'].get('emotion_pool', ['neutral'])
    trigger_status = state['notes'].get('trigger_status', {})
    # 构造 Trigger Status 文本块
    trigger_lines = "\n".join(f"Trigger {tid} Met: {status}" for tid, status in trigger_status.items())


    prompt = f"""
Generate 3 candidate intentions for the NPC.
Leave the Emotion Change field blank; it will be filled later.
DO NOT OMIT "- Type:" line. 
Each candidate must start with: 
- Type: conversation
The behavior type has already been decided and must be: {state['decision']}.
Only generate intentions of this type. Do not mix or switch behavior types.
Valid Emotion Change from: {', '.join(pool)}

[Constraint Source Tags]
[ENV]: Environmental conditions
[PLAYER]: Player behavior/appearance
[SELF]: NPC's internal limitations
[DIALOGUE]: Strategic dialogue limitations

[NPC Profile]
{state['npc_profile']}

[Scene Context]
{state['scene_context']}

[Player Description]
{state['player_description']}

[Summarized Analysis]
{summary}

[Trigger Status]
{trigger_lines}

[Constraints from Analysis]
{constraints}

[Tradeoff Summary Guidance]
{state['notes'].get('tradeoff_summary', 'None')}

Format for each candidate:
- Type: {state['decision']}
- Style: <tone>
- Goal: <motivation>
- Target: <player | object>
- Emotion Change:
- Detail: <intended message or action>
"""
    raw = None
    while True:
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
        parts = raw.split('- Type: ')
        cands = [('- Type: ' + blk.strip()) if i>0 else blk.strip() for i, blk in enumerate(parts) if blk.strip()]
        if len(cands) >= 3:
            state['candidates'] = cands[:3]
            for idx, cand in enumerate(state['candidates'], 1):
                print(f"Candidate {idx}: {cand}")
            break
        else:
            print("Error generating candidates, retrying...")
    return state



def select_best(state: State) -> State:
        # Guard: ensure we have candidates
    if not state.get('candidates'):
        print("⚠️ No candidates available for selection, skipping select_best.")
        return state
    
    queue = state['notes'].get('trigger_queue', [])
    if queue:
        # 把所有新变为 yes 的 trigger 对应的 goal 聚合
        goals = state['notes']['trigger_goals']
        granted = [goals[tid] for tid in queue if state['notes']['trigger_status'].get(tid)=='yes']
        # 生成聚合目标字符串
        agg_goal = ' and '.join(granted)
        # 构造 final_intention，合并 emotion_pool
        pool = state['notes'].get('emotion_pool', [])
        lines = [
            "- Type: conversation",
            "- Style: accommodating",
            f"- Goal: {agg_goal}",
            "- Target: player",
            "- Emotion Change:",
            f"- Detail: \"Your credentials satisfy: {', '.join(queue)}. You may enter.\""
        ]
        sel = "\n".join(lines)
        if pool:
            sel = _patch_emotion_pool(sel, pool)
        state['final_intention'] = sel
        # 清空队列
        state['notes']['trigger_queue'] = []
        print(f"🌟 Final Intention after triggers: {sel}")
        return state
    prompt = f"""
From the three candidates below, choose the best by outputting only the number 1, 2, or 3, based on goal, constraints, and current NPC state (Emotion, Personality, etc.).
Candidates:
1. {state['candidates'][0]}
2. {state['candidates'][1]}
3. {state['candidates'][2]}

[Constraint Source Tags]
[ENV]: Environmental conditions
[PLAYER]: Player behavior/appearance
[SELF]: NPC's internal limitations
[DIALOGUE]: Strategic dialogue limitations

[NPC Profile]
{state['npc_profile']}

[Context]
{state['scene_context']}

[Constraints]
{state['notes'].get('constraints', [])}

[Candidates]
{state['candidates'][0]}

Return only the chosen number
"""
    resp = llm.invoke(prompt)
    choice = resp.content.strip()
    try:
        idx = int(choice)
        sel = state['candidates'][idx - 1]
    except:
        sel = state['candidates'][0]

    # ✅ 插入 emotion pool 处理
    pool = state['notes'].get('emotion_pool', [])
    if pool:
        print("Entered emotion pool patching!")
        print(pool)
        sel = _patch_emotion_pool(sel, pool)
    elif pool==[]:
        print("No emotion pool available, skipping patching.")

    state['final_intention'] = sel
    print(f"Selected Candidate {choice}")
    print(f"🌟 Final Intention:\n{sel}")
    return state


def generate_behavior(state: State) -> State:
    prompt = f"""
You are the behavior realization module for the NPC.
Generate a behavior based on the intention and constraints.
If the behavior is of type 'conversation', and includes physical states or gestures, wrap them in parentheses before the speech.
Use natural, human-like speech. Most analysis shouldn't remain in the generated final behavior.
The emotion that the NPC has will affect the tone and word choice, but should not be explicitly stated.

[Dialogue Continuity Instruction]
This is a multi-turn conversation. Treat this response as part of an ongoing exchange. Response what the opposite side just said and what you said last time.
Unless the player has such a need or is in a situation that requires it, do not repeat what you have already said.
The reply should follow naturally from what the player just said and what the NPC just replied previously.
Avoid restarting the topic, repeating context, or acting as if this is the first message.
Use referential continuity where appropriate (e.g., “as I said…”, “you mentioned…”, “but earlier…”).

[NPC Profile]
{state['npc_profile']}

[Scene Context]
{state['scene_context']}

[Player Description]
{state['player_description']}

[Final Intention]
{state['final_intention']}

[Dialogue History]
{state['notes']['dialogue_history']}

[World View]
{state['world_view']}

Output only the final behavior.
"""
    if state['valid_input'] is False:
        prompt += "\n\n[Invalid Input Handling]\nThe player input was invalid. Generate a behavior that acknowledges this and redirects the conversation."
    elif state['valid_input'] is None:
        prompt += "\n\n[Default Handling]\nThe player input is valid. Generate a behavior that continues the conversation naturally."
        print("⚠️ Valid input status is None, assuming valid input.")
    resp = llm.invoke(prompt)
    state['final_behavior'] = resp.content.strip()
    return state


def dialogue_loop(state: State):
    state['notes']['dialogue_history'] = []
    while True:
        try:
            player_input = input("\n🎮 Player: ")
            if player_input.lower().strip() == "exit":
                print("\n🛑 退出对话。"); break
            # 询问是否使用 tradeoff evaluation
            use_eval = input("Use tradeoff evaluation this turn? (yes/no): ").strip().lower()
            state['use_tradeoff_evaluation'] = use_eval == "yes"
            validTF = input("Is this input valid? (yes/no): ").strip().lower()
            if validTF == "yes":
                state['valid_input'] = True
            else:
                state['valid_input'] = False

        except KeyboardInterrupt:
            print("\n🛑 输入中断，自动退出。"); break
        

        state['notes']['dialogue_history'].append(f"【player→NPC】{player_input}")
        state['player_description'] = "- [Player Description loaded externally or set during character creation]"
        state = combined_analysis_node(state)
        ### 此处删除了decide behavior type
        if state['use_tradeoff_evaluation']:
            state = evaluate_tradeoffs(state)
        state = generate_candidates(state)
        state = select_best(state)
        state = generate_behavior(state)

        response = state['final_behavior']
        print(f"\n🤖 Player: {player_input}")
        print(f"\n🗣️ NPC: {response}")
        state['notes']['dialogue_history'].append(f"【NPC→player】{response}")
        
        state['use_tradeoff_evaluation'] = False # 重置 tradeoff 使用状态
        validTF = None


def main():
    # 2026 migration: load the copied scenario materials within this workspace.
    data_dir = WORKSPACE_ROOT / "data" / "story"
    npc_json = data_dir / "NPC_HuangQiye.json"
    scene_json = data_dir / "Scene_Beginning.json"
    world_json = data_dir / "WorldView.json"

    # 加载设定
    npc_profile = load_npc_profile(npc_json)
    scene_context = load_scene_context(scene_json)
    trigger_goals, scene_general_goal = load_scene_triggers_and_general(scene_json)

    # 将 scene_general_goal 追加到 npc_profile
    full_npc_profile = npc_profile + "\n- Scene General Goal: " + scene_general_goal

    state: State = {
        'npc_profile': full_npc_profile,
        'scene_context': scene_context,
        'player_description': '',
        'world_view': '',
        'notes': {},
        'decision': None,
        'initiative': None,
        'candidates': [],
        'final_intention': None,
        'final_behavior': None,
        'valid_input': None
    }
    # 设置触发条件目标
    state['notes']['trigger_goals'] = trigger_goals

    world = load_world_config(world_json)
    worldCheck = input("Add the world view? (yes/no): ").strip().lower()
    if worldCheck == "yes":
        state['world_view'] = json.dumps(world, ensure_ascii=False, indent=2)
        print("\n🌍 世界观设定已加载。")
    dialogue_loop(state)

if __name__ == '__main__':
    main()

