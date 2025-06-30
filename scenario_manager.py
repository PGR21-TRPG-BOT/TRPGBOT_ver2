# -*- coding: utf-8 -*-
import os
import json
import logging
from datetime import datetime
from enum import Enum

# NPC 매니저 임포트 추가
try:
    from npc_manager import npc_manager
except ImportError:
    logger.warning("⚠️ NPC 매니저를 임포트할 수 없습니다. NPC 기능이 제한됩니다.")
    npc_manager = None

logger = logging.getLogger(__name__)

class ScenarioStage(Enum):
    """시나리오 생성 단계"""
    OVERVIEW = "개요"
    EPISODES = "에피소드"
    NPCS = "NPC"
    HINTS = "힌트"
    DUNGEONS = "던전"
    COMPLETED = "완료"

class ScenarioProgress(Enum):
    """시나리오 진행 상태"""
    NOT_STARTED = "시작_전"
    IN_PROGRESS = "진행_중"
    COMPLETED = "완료"
    PAUSED = "일시정지"

class ScenarioManager:
    """
    TRPG 시나리오 생성 및 진척도 관리 클래스
    
    ✨ 점진적 LLM 생성 시스템 포함:
    
    **핵심 개념:**
    - 기존: 한 번에 모든 필드를 LLM에게 요청하여 생성
    - 개선: 가장 중요한 테마를 중심으로 빈칸을 하나씩 찾아 점진적으로 생성
    
    **주요 특징:**
    1. 우선순위 기반 처리: 테마(1순위) → 제목(2순위) → 배경(3순위) 등
    2. 단계별 생성: 하나의 빈칸을 채우고 → 다음 빈칸 탐지 → 반복
    3. 컨텍스트 기반 생성: 이미 채워진 정보를 바탕으로 다음 정보 생성
    4. 실시간 진행도 추적
    
    **사용 예시:**
    ```python
    # 1. 점진적 생성 시작
    scenario_manager.start_progressive_generation(user_id, "중세 판타지 시나리오")
    
    # 2. 다음 빈칸 확인
    next_gap = scenario_manager.get_next_gap_for_user(user_id)
    
    # 3. 자동 생성 또는 수동 입력
    result = scenario_manager.process_next_gap_automatically(user_id)
    # 또는
    result = scenario_manager.process_user_input_for_gap(user_id, "사용자 입력")
    
    # 4. 진행도 확인
    progress = scenario_manager.get_generation_progress(user_id)
    ```
    """
    
    def __init__(self):
        """ScenarioManager 초기화"""
        self.ensure_directories()
        
    def ensure_directories(self):
        """필요한 디렉토리 생성"""
        os.makedirs('scenarios', exist_ok=True)
        
    def get_scenario_file_path(self, user_id):
        """시나리오 파일 경로 반환"""
        return f'scenarios/scenario_{user_id}.json'
        
    def init_scenario_creation(self, user_id):
        """시나리오 생성 초기화"""
        scenario_data = {
            "user_id": user_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_stage": ScenarioStage.OVERVIEW.value,
            "progress": ScenarioProgress.NOT_STARTED.value,
            "scenario": {
                "title": "",
                "overview": {
                    "title": "",
                    "theme": "",
                    "setting": "",
                    "main_conflict": "",
                    "objective": "",
                    "rewards": ""
                },
                "episodes": [],
                "npcs": [],
                "hints": [],
                "dungeons": [],
                "sessions": []
            }
        }
        
        self.save_scenario(user_id, scenario_data)
        logger.info(f"사용자 {user_id}의 시나리오 생성 초기화")
        return scenario_data
        
    def load_scenario(self, user_id):
        """시나리오 데이터 로드"""
        file_path = self.get_scenario_file_path(user_id)
        
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"시나리오 파일 로드 오류: {e}")
            return None
            
    def save_scenario(self, user_id, scenario_data):
        """시나리오 데이터 저장"""
        file_path = self.get_scenario_file_path(user_id)
        
        try:
            scenario_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(scenario_data, f, ensure_ascii=False, indent=2)
            logger.info(f"시나리오 데이터 저장 완료: {file_path}")
            return True
        except Exception as e:
            logger.error(f"시나리오 데이터 저장 오류: {e}")
            return False
            
    def get_current_stage(self, user_id):
        """현재 시나리오 생성 단계 반환"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return ScenarioStage.OVERVIEW.value
        return scenario_data.get("current_stage", ScenarioStage.OVERVIEW.value)
        
    def set_current_stage(self, user_id, stage):
        """현재 시나리오 생성 단계 설정"""
        scenario_data = self.load_scenario(user_id)
        if scenario_data:
            scenario_data["current_stage"] = stage
            self.save_scenario(user_id, scenario_data)
            
    def get_next_stage(self, current_stage):
        """다음 시나리오 생성 단계 반환"""
        stage_flow = {
            ScenarioStage.OVERVIEW.value: ScenarioStage.EPISODES.value,
            ScenarioStage.EPISODES.value: ScenarioStage.NPCS.value,
            ScenarioStage.NPCS.value: ScenarioStage.HINTS.value,
            ScenarioStage.HINTS.value: ScenarioStage.DUNGEONS.value,
            ScenarioStage.DUNGEONS.value: ScenarioStage.COMPLETED.value
        }
        return stage_flow.get(current_stage, ScenarioStage.COMPLETED.value)
        
    def is_stage_complete(self, user_id, stage):
        """특정 단계가 완료되었는지 확인"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return False
            
        scenario = scenario_data.get("scenario", {})
        
        if stage == ScenarioStage.OVERVIEW.value:
            overview = scenario.get("overview", {})
            return all([
                overview.get("theme"),
                overview.get("setting"),
                overview.get("main_conflict"),
                overview.get("objective")
            ])
        elif stage == ScenarioStage.EPISODES.value:
            return len(scenario.get("episodes", [])) >= 3
        elif stage == ScenarioStage.NPCS.value:
            if npc_manager:
                return self.is_npc_stage_complete(user_id)
            else:
                return len(scenario.get("npcs", [])) >= 3
        elif stage == ScenarioStage.HINTS.value:
            return len(scenario.get("hints", [])) >= 3
        elif stage == ScenarioStage.DUNGEONS.value:
            return len(scenario.get("dungeons", [])) >= 1
            
        return False
        
    def update_scenario_overview(self, user_id, overview_data):
        """시나리오 개요 업데이트"""
        scenario_data = self.load_scenario(user_id) or self.init_scenario_creation(user_id)
        scenario_data["scenario"]["overview"].update(overview_data)
        scenario_data["progress"] = ScenarioProgress.IN_PROGRESS.value
        self.save_scenario(user_id, scenario_data)
        
    def add_episode(self, user_id, episode_data):
        """에피소드 추가"""
        scenario_data = self.load_scenario(user_id)
        if scenario_data:
            episode_data["id"] = len(scenario_data["scenario"]["episodes"]) + 1
            scenario_data["scenario"]["episodes"].append(episode_data)
            self.save_scenario(user_id, scenario_data)
            
    def add_npc(self, user_id, npc_data):
        """NPC 추가"""
        scenario_data = self.load_scenario(user_id)
        if scenario_data:
            npc_data["id"] = len(scenario_data["scenario"]["npcs"]) + 1
            scenario_data["scenario"]["npcs"].append(npc_data)
            self.save_scenario(user_id, scenario_data)
            
    def add_hint(self, user_id, hint_data):
        """힌트 추가"""
        scenario_data = self.load_scenario(user_id)
        if scenario_data:
            hint_data["id"] = len(scenario_data["scenario"]["hints"]) + 1
            scenario_data["scenario"]["hints"].append(hint_data)
            self.save_scenario(user_id, scenario_data)
            
    def add_dungeon(self, user_id, dungeon_data):
        """던전 추가"""
        scenario_data = self.load_scenario(user_id)
        if scenario_data:
            dungeon_data["id"] = len(scenario_data["scenario"]["dungeons"]) + 1
            scenario_data["scenario"]["dungeons"].append(dungeon_data)
            self.save_scenario(user_id, scenario_data)
            
    def get_stage_prompt(self, stage):
        """단계별 프롬프트 반환"""
        prompts = {
            ScenarioStage.OVERVIEW.value: """
🎭 **시나리오 개요 생성 단계**

다음 요소들을 포함한 시나리오 개요를 만들어보겠습니다:

1. **테마**: 어떤 종류의 모험인가요? (미스터리, 탐험, 구출, 조사 등)
2. **배경**: 언제, 어디서 일어나는 이야기인가요?
3. **주요 갈등**: 해결해야 할 핵심 문제는 무엇인가요?
4. **목표**: 플레이어들이 달성해야 할 것은 무엇인가요?
5. **보상**: 성공 시 얻을 수 있는 것은 무엇인가요?

원하시는 시나리오의 테마나 아이디어를 알려주세요!
""",
            ScenarioStage.EPISODES.value: """
📖 **에피소드 구성 단계**

시나리오를 3-5개의 주요 에피소드로 나누어 구성하겠습니다:

각 에피소드마다 다음을 포함합니다:
- 에피소드 제목과 목표
- 주요 사건들
- 플레이어 행동 옵션
- 성공/실패 결과

어떤 흐름으로 이야기를 전개하고 싶으신가요?
""",
            ScenarioStage.NPCS.value: """
👥 **NPC 설정 단계**

시나리오에 등장할 주요 NPC들을 만들어보겠습니다:

각 NPC마다 다음을 설정합니다:
- 이름과 외모
- 성격과 동기
- 플레이어와의 관계 (적, 동료, 중립)
- 가진 정보나 능력
- 대화 스타일

어떤 NPC들이 필요할까요?
""",
            ScenarioStage.HINTS.value: """
🔍 **힌트 시스템 설정**

플레이어들이 발견할 수 있는 단서와 힌트들을 설정하겠습니다:

각 힌트마다 다음을 포함합니다:
- 힌트 내용
- 발견 방법 (조사, 대화, 관찰 등)
- 연결되는 정보
- 난이도

어떤 종류의 힌트들이 필요할까요?
""",
            ScenarioStage.DUNGEONS.value: """
🏰 **던전/탐험지 설정**

플레이어들이 탐험할 장소들을 설계하겠습니다:

각 장소마다 다음을 포함합니다:
- 장소 설명과 분위기
- 주요 방/구역들
- 함정이나 퍼즐
- 몬스터나 수호자
- 숨겨진 보물이나 정보

어떤 장소를 탐험하게 하고 싶으신가요?
"""
        }
        return prompts.get(stage, "알 수 없는 단계입니다.")
        
    def get_scenario_context_for_mastering(self, user_id, current_session_type):
        """마스터링용 시나리오 컨텍스트 생성"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return ""
            
        # 현재 세션에 맞는 진척도 업데이트
        self.update_session_progress(user_id, current_session_type)
        
        scenario = scenario_data.get("scenario", {})
        context_parts = []
        
        # 사용자 선호도 정보 (시나리오 생성에 활용)
        user_preferences = scenario_data.get("user_preferences")
        if user_preferences and user_preferences.get("preferences_detected"):
            context_parts.append(f"""
🎯 **사용자 선호도**
사용자 요청: "{user_preferences.get('user_input', '')}"
이 요청을 바탕으로 시나리오 생성 및 마스터링을 진행해주세요.
""")
        
        # 시나리오 개요
        overview = scenario.get("overview", {})
        if overview.get("theme"):
            context_parts.append(f"""
🎭 **현재 진행중인 시나리오**
- 제목: {overview.get('title', '제목 미정')}
- 테마: {overview.get('theme', '')}
- 배경: {overview.get('setting', '')}
- 주요 갈등: {overview.get('main_conflict', '')}
- 목표: {overview.get('objective', '')}
""")
        
        # 에피소드 정보
        episodes = scenario.get("episodes", [])
        if episodes:
            context_parts.append("📖 **에피소드 구성**")
            for i, episode in enumerate(episodes, 1):
                status = self.get_episode_status(user_id, episode.get("id"))
                context_parts.append(f"{i}. {episode.get('title', f'에피소드 {i}')} [{status}]")
        
        # NPC 정보
        npcs = scenario.get("npcs", [])
        if npcs:
            context_parts.append("\n👥 **주요 NPC들**")
            for npc in npcs:
                relationship = npc.get('relationship', npc.get('role', '역할미정'))
                context_parts.append(f"- {npc.get('name', '이름없음')}: {relationship}")
                if npc.get('personality'):
                    context_parts.append(f"  └ 성격: {npc.get('personality')}")
                if npc.get('information'):
                    context_parts.append(f"  └ 정보: {npc.get('information')}")
        
        # 현재 세션 관련 힌트
        hints = scenario.get("hints", [])
        relevant_hints = [h for h in hints if current_session_type in h.get("relevant_sessions", [])]
        if relevant_hints:
            context_parts.append(f"\n🔍 **{current_session_type} 관련 힌트들**")
            for hint in relevant_hints:
                context_parts.append(f"- {hint.get('content', '')}")
                if hint.get('discovery_method'):
                    context_parts.append(f"  └ 발견방법: {hint.get('discovery_method')}")
        
        # 던전 정보 (해당 세션에서 필요한 경우)
        dungeons = scenario.get("dungeons", [])
        if dungeons and current_session_type in ["던전_탐험", "모험_진행"]:
            context_parts.append("\n🏰 **탐험 가능한 장소들**")
            for dungeon in dungeons:
                context_parts.append(f"- {dungeon.get('name', '이름없음')}: {dungeon.get('type', '유형미정')}")
                if dungeon.get('description'):
                    context_parts.append(f"  └ {dungeon.get('description')}")
        
        return "\n".join(context_parts)
        
    def update_session_progress(self, user_id, session_type):
        """세션 진행도 업데이트"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return
            
        sessions = scenario_data["scenario"].get("sessions", [])
        session_found = False
        
        for session in sessions:
            if session.get("type") == session_type:
                session["last_played"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                session["play_count"] = session.get("play_count", 0) + 1
                session_found = True
                break
                
        if not session_found:
            sessions.append({
                "type": session_type,
                "first_played": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_played": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "play_count": 1,
                "status": "진행중"
            })
            
        self.save_scenario(user_id, scenario_data)
        
    def get_episode_status(self, user_id, episode_id):
        """에피소드 진행 상태 반환"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return "미시작"
            
        # 에피소드별 진행 상태 추적
        episode_progress = scenario_data.get("episode_progress", {})
        episode_key = f"episode_{episode_id}"
        
        if episode_key in episode_progress:
            return episode_progress[episode_key].get("status", "미시작")
        
        # 세션 기록을 바탕으로 진행도 판단
        sessions = scenario_data["scenario"].get("sessions", [])
        if not sessions:
            return "미시작"
            
        # 간단한 진행도 판단 로직 (실제로는 더 복잡할 수 있음)
        adventure_sessions = [s for s in sessions if s.get("type") in ["모험_진행", "던전_탐험"]]
        if adventure_sessions:
            return "진행중"
        else:
            return "준비중"
    
    def update_episode_progress(self, user_id, episode_id, status, location=None):
        """에피소드 진행 상태 업데이트"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return False
            
        if "episode_progress" not in scenario_data:
            scenario_data["episode_progress"] = {}
        
        episode_key = f"episode_{episode_id}"
        scenario_data["episode_progress"][episode_key] = {
            "status": status,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": location
        }
        
        self.save_scenario(user_id, scenario_data)
        logger.info(f"에피소드 {episode_id} 진행 상태 업데이트: {status}")
        return True
    
    def get_current_episode(self, user_id):
        """현재 진행중인 에피소드 반환"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return None
            
        episodes = scenario_data.get("scenario", {}).get("episodes", [])
        if not episodes:
            return None
        
        episode_progress = scenario_data.get("episode_progress", {})
        
        # 진행중인 에피소드 찾기
        for i, episode in enumerate(episodes):
            episode_key = f"episode_{i+1}"
            status = episode_progress.get(episode_key, {}).get("status", "미시작")
            
            if status == "진행중":
                return {"index": i, "episode": episode, "id": i+1}
        
        # 진행중인 에피소드가 없으면 첫 번째 미시작 에피소드 반환
        for i, episode in enumerate(episodes):
            episode_key = f"episode_{i+1}"
            status = episode_progress.get(episode_key, {}).get("status", "미시작")
            
            if status == "미시작":
                return {"index": i, "episode": episode, "id": i+1}
        
        # 모든 에피소드가 완료되었으면 마지막 에피소드 반환
        return {"index": len(episodes)-1, "episode": episodes[-1], "id": len(episodes)}
    
    def advance_to_next_episode(self, user_id):
        """다음 에피소드로 진행"""
        current_episode = self.get_current_episode(user_id)
        if not current_episode:
            return False
            
        # 현재 에피소드를 완료로 표시
        self.update_episode_progress(user_id, current_episode["id"], "완료")
        
        # 다음 에피소드를 진행중으로 표시
        next_episode_id = current_episode["id"] + 1
        scenario_data = self.load_scenario(user_id)
        episodes = scenario_data.get("scenario", {}).get("episodes", [])
        
        if next_episode_id <= len(episodes):
            self.update_episode_progress(user_id, next_episode_id, "진행중")
            logger.info(f"에피소드 {next_episode_id}로 진행")
            return True
        else:
            logger.info("모든 에피소드가 완료되었습니다")
            return False
    
    def find_empty_fields(self, user_id):
        """시나리오에서 빈 필드들을 찾아서 반환"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return {}
            
        scenario = scenario_data.get("scenario", {})
        empty_fields = {}
        
        # 개요 빈 필드 검사
        overview = scenario.get("overview", {})
        empty_overview = {}
        for field in ["title", "theme", "setting", "main_conflict", "objective", "rewards"]:
            if not overview.get(field) or overview.get(field).strip() == "":
                empty_overview[field] = field
        if empty_overview:
            empty_fields["overview"] = empty_overview
        
        # 에피소드 빈 필드 검사
        episodes = scenario.get("episodes", [])
        incomplete_episodes = []
        for i, episode in enumerate(episodes):
            empty_episode = {}
            for field in ["title", "objective", "events", "player_options", "success_result", "failure_result"]:
                if not episode.get(field) or (isinstance(episode.get(field), list) and len(episode.get(field)) == 0):
                    empty_episode[field] = field
            if empty_episode:
                incomplete_episodes.append({"index": i, "empty_fields": empty_episode})
        if incomplete_episodes:
            empty_fields["episodes"] = incomplete_episodes
        
        # NPC 빈 필드 검사
        npcs = scenario.get("npcs", [])
        incomplete_npcs = []
        for i, npc in enumerate(npcs):
            empty_npc = {}
            for field in ["name", "appearance", "personality", "motivation", "relationship", "information", "abilities"]:
                if not npc.get(field) or npc.get(field).strip() == "":
                    empty_npc[field] = field
            if empty_npc:
                incomplete_npcs.append({"index": i, "name": npc.get("name", f"NPC {i+1}"), "empty_fields": empty_npc})
        if incomplete_npcs:
            empty_fields["npcs"] = incomplete_npcs
        
        # 힌트 빈 필드 검사
        hints = scenario.get("hints", [])
        incomplete_hints = []
        for i, hint in enumerate(hints):
            empty_hint = {}
            for field in ["content", "discovery_method", "connected_info", "difficulty", "relevant_sessions"]:
                if not hint.get(field) or (isinstance(hint.get(field), list) and len(hint.get(field)) == 0):
                    empty_hint[field] = field
            if empty_hint:
                incomplete_hints.append({"index": i, "empty_fields": empty_hint})
        if incomplete_hints:
            empty_fields["hints"] = incomplete_hints
        
        # 던전 빈 필드 검사
        dungeons = scenario.get("dungeons", [])
        incomplete_dungeons = []
        for i, dungeon in enumerate(dungeons):
            empty_dungeon = {}
            for field in ["name", "type", "description", "atmosphere", "rooms", "traps", "puzzles", "monsters", "treasures"]:
                if not dungeon.get(field) or (isinstance(dungeon.get(field), list) and len(dungeon.get(field)) == 0):
                    empty_dungeon[field] = field
            if empty_dungeon:
                incomplete_dungeons.append({"index": i, "name": dungeon.get("name", f"던전 {i+1}"), "empty_fields": empty_dungeon})
        if incomplete_dungeons:
            empty_fields["dungeons"] = incomplete_dungeons
        
        return empty_fields
    
    def get_field_priority(self, category, field_name):
        """필드별 우선순위 반환 (낮을수록 높은 우선순위)"""
        priority_map = {
            "overview": {
                "theme": 1,        # 가장 중요한 테마
                "title": 2,        # 제목
                "setting": 3,      # 배경
                "main_conflict": 4, # 주요 갈등
                "objective": 5,    # 목표
                "rewards": 6       # 보상
            },
            "episodes": {
                "title": 1,
                "objective": 2,
                "events": 3,
                "player_options": 4,
                "success_result": 5,
                "failure_result": 6
            },
            "npcs": {
                "name": 1,
                "relationship": 2,
                "personality": 3,
                "motivation": 4,
                "appearance": 5,
                "information": 6,
                "abilities": 7
            },
            "hints": {
                "content": 1,
                "relevant_sessions": 2,
                "discovery_method": 3,
                "connected_info": 4,
                "difficulty": 5
            },
            "dungeons": {
                "name": 1,
                "type": 2,
                "description": 3,
                "atmosphere": 4,
                "rooms": 5,
                "traps": 6,
                "puzzles": 7,
                "monsters": 8,
                "treasures": 9
            }
        }
        return priority_map.get(category, {}).get(field_name, 99)
    
    def find_next_most_important_gap(self, user_id):
        """가장 중요한 다음 빈칸 하나를 찾아서 반환"""
        empty_fields = self.find_empty_fields(user_id)
        if not empty_fields:
            return None
        
        # 모든 빈 필드를 우선순위와 함께 수집
        all_gaps = []
        
        # 개요 필드들
        if "overview" in empty_fields:
            for field in empty_fields["overview"]:
                priority = self.get_field_priority("overview", field)
                all_gaps.append({
                    "category": "overview",
                    "field": field,
                    "priority": priority,
                    "index": None
                })
        
        # 에피소드 필드들
        if "episodes" in empty_fields:
            for episode_info in empty_fields["episodes"]:
                for field in episode_info["empty_fields"]:
                    priority = self.get_field_priority("episodes", field)
                    all_gaps.append({
                        "category": "episodes",
                        "field": field,
                        "priority": priority,
                        "index": episode_info["index"]
                    })
        
        # NPC 필드들
        if "npcs" in empty_fields:
            for npc_info in empty_fields["npcs"]:
                for field in npc_info["empty_fields"]:
                    priority = self.get_field_priority("npcs", field)
                    all_gaps.append({
                        "category": "npcs",
                        "field": field,
                        "priority": priority,
                        "index": npc_info["index"],
                        "name": npc_info["name"]
                    })
        
        # 힌트 필드들
        if "hints" in empty_fields:
            for hint_info in empty_fields["hints"]:
                for field in hint_info["empty_fields"]:
                    priority = self.get_field_priority("hints", field)
                    all_gaps.append({
                        "category": "hints",
                        "field": field,
                        "priority": priority,
                        "index": hint_info["index"]
                    })
        
        # 던전 필드들
        if "dungeons" in empty_fields:
            for dungeon_info in empty_fields["dungeons"]:
                for field in dungeon_info["empty_fields"]:
                    priority = self.get_field_priority("dungeons", field)
                    all_gaps.append({
                        "category": "dungeons",
                        "field": field,
                        "priority": priority,
                        "index": dungeon_info["index"],
                        "name": dungeon_info["name"]
                    })
        
        # 우선순위로 정렬하여 가장 중요한 것 반환
        if all_gaps:
            all_gaps.sort(key=lambda x: (x["priority"], x["category"]))
            return all_gaps[0]
        
        return None
    
    def generate_single_gap_prompt(self, user_id, gap_info):
        """단일 빈칸을 채우기 위한 프롬프트 생성"""
        if not gap_info:
            return None
            
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return None
            
        scenario = scenario_data.get("scenario", {})
        prompt_parts = []
        
        # 현재 시나리오 컨텍스트 제공
        overview = scenario.get("overview", {})
        prompt_parts.append("🎭 **현재 시나리오 정보:**")
        if overview.get("title"):
            prompt_parts.append(f"제목: {overview.get('title')}")
        if overview.get("theme"):
            prompt_parts.append(f"테마: {overview.get('theme')}")
        if overview.get("setting"):
            prompt_parts.append(f"배경: {overview.get('setting')}")
        
        # 필드명 한국어 매핑
        field_name_kr = {
            # 개요
            "title": "시나리오 제목", "theme": "테마", "setting": "배경설정", 
            "main_conflict": "주요 갈등", "objective": "목표", "rewards": "보상",
            # 에피소드
            "title": "에피소드 제목", "objective": "에피소드 목표", "events": "주요 사건들",
            "player_options": "플레이어 선택지", "success_result": "성공 결과", "failure_result": "실패 결과",
            # NPC
            "name": "NPC 이름", "appearance": "외모", "personality": "성격",
            "motivation": "동기", "relationship": "플레이어와의 관계", "information": "보유 정보", "abilities": "특수 능력",
            # 힌트
            "content": "힌트 내용", "discovery_method": "발견 방법", "connected_info": "연결된 정보",
            "difficulty": "난이도", "relevant_sessions": "관련 세션",
            # 던전
            "name": "던전 이름", "type": "던전 유형", "description": "던전 설명", "atmosphere": "분위기",
            "rooms": "방/구역", "traps": "함정", "puzzles": "퍼즐", "monsters": "몬스터", "treasures": "보물"
        }
        
        category = gap_info["category"]
        field = gap_info["field"]
        korean_field = field_name_kr.get(field, field)
        
        prompt_parts.append(f"\n🎯 **현재 채울 필드:** {korean_field}")
        
        # 카테고리별 구체적인 요청
        if category == "overview":
            prompt_parts.append(f"\n📝 **요청사항:** 시나리오의 {korean_field}를 생성해주세요.")
            if field == "theme":
                prompt_parts.append("• 어떤 종류의 모험인지 (미스터리, 탐험, 구출, 조사 등)")
            elif field == "setting":
                prompt_parts.append("• 언제, 어디서 일어나는 이야기인지 구체적으로")
            elif field == "main_conflict":
                prompt_parts.append("• 해결해야 할 핵심 문제나 갈등")
            elif field == "objective":
                prompt_parts.append("• 플레이어들이 달성해야 할 명확한 목표")
            elif field == "rewards":
                prompt_parts.append("• 성공 시 얻을 수 있는 보상")
                
        elif category == "episodes":
            episode_index = gap_info.get("index", 0)
            episodes = scenario.get("episodes", [])
            if episode_index < len(episodes):
                episode = episodes[episode_index]
                prompt_parts.append(f"\n📖 **에피소드 {episode_index + 1}:** {episode.get('title', '제목미정')}")
                prompt_parts.append(f"**요청사항:** 이 에피소드의 {korean_field}를 생성해주세요.")
                
        elif category == "npcs":
            npc_index = gap_info.get("index", 0)
            npc_name = gap_info.get("name", f"NPC {npc_index + 1}")
            prompt_parts.append(f"\n👤 **NPC:** {npc_name}")
            prompt_parts.append(f"**요청사항:** 이 NPC의 {korean_field}를 생성해주세요.")
            
        elif category == "hints":
            hint_index = gap_info.get("index", 0)
            prompt_parts.append(f"\n🔍 **힌트 {hint_index + 1}**")
            prompt_parts.append(f"**요청사항:** 이 힌트의 {korean_field}를 생성해주세요.")
            
        elif category == "dungeons":
            dungeon_index = gap_info.get("index", 0)
            dungeon_name = gap_info.get("name", f"던전 {dungeon_index + 1}")
            prompt_parts.append(f"\n🏰 **던전:** {dungeon_name}")
            prompt_parts.append(f"**요청사항:** 이 던전의 {korean_field}를 생성해주세요.")
        
        prompt_parts.append(f"\n⚠️ **중요:** {korean_field}만 간결하고 명확하게 생성하여 답변해주세요. 다른 내용은 포함하지 마세요.")
        
        return "\n".join(prompt_parts)
    
    def update_single_gap(self, user_id, gap_info, value):
        """단일 빈칸을 업데이트"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return False
            
        scenario = scenario_data.get("scenario", {})
        category = gap_info["category"]
        field = gap_info["field"]
        
        try:
            if category == "overview":
                if "overview" not in scenario:
                    scenario["overview"] = {}
                scenario["overview"][field] = value
                
            elif category == "episodes":
                index = gap_info.get("index", 0)
                episodes = scenario.get("episodes", [])
                if index < len(episodes):
                    episodes[index][field] = value
                    
            elif category == "npcs":
                index = gap_info.get("index", 0)
                npcs = scenario.get("npcs", [])
                if index < len(npcs):
                    npcs[index][field] = value
                    
            elif category == "hints":
                index = gap_info.get("index", 0)
                hints = scenario.get("hints", [])
                if index < len(hints):
                    hints[index][field] = value
                    
            elif category == "dungeons":
                index = gap_info.get("index", 0)
                dungeons = scenario.get("dungeons", [])
                if index < len(dungeons):
                    dungeons[index][field] = value
            
            # 시나리오 저장
            self.save_scenario(user_id, scenario_data)
            logger.info(f"사용자 {user_id}의 {category}.{field} 업데이트 완료: {value[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"단일 필드 업데이트 오류: {e}")
            return False
    
    def progressive_scenario_generation(self, user_id, max_iterations=50):
        """점진적 시나리오 생성 - 가장 중요한 빈칸부터 하나씩 처리"""
        iteration_count = 0
        completed_fields = []
        
        print(f"\n🎯 사용자 {user_id}의 점진적 시나리오 생성 시작")
        print("=" * 80)
        
        while iteration_count < max_iterations:
            iteration_count += 1
            
            # 다음 가장 중요한 빈칸 찾기
            next_gap = self.find_next_most_important_gap(user_id)
            
            if not next_gap:
                print(f"\n✅ 모든 중요한 필드가 완료되었습니다!")
                print(f"📊 총 {len(completed_fields)}개 필드 생성 완료")
                break
            
            # 필드명 한국어 변환
            field_name_kr = self._get_korean_field_name(next_gap['field'])
            category_kr = self._get_korean_category_name(next_gap['category'])
            
            print(f"\n🔍 [{iteration_count}/{max_iterations}] {category_kr} > {field_name_kr} 생성 중...")
            
            # 단일 빈칸용 프롬프트 생성
            prompt = self.generate_single_gap_prompt(user_id, next_gap)
            if not prompt:
                print(f"❌ 프롬프트 생성 실패: {next_gap}")
                break
            
            # 요청 내용 출력
            print(f"📝 LLM 요청:")
            print("-" * 60)
            print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
            print("-" * 60)
            
            # LLM에게 요청
            generated_value = self._call_llm_for_gap(prompt, next_gap)
            
            if generated_value:
                # 생성된 값으로 빈칸 채우기
                success = self.update_single_gap(user_id, next_gap, generated_value)
                
                if success:
                    field_name = f"{next_gap['category']}.{next_gap['field']}"
                    completed_fields.append(field_name)
                    
                    # 성공 결과 출력
                    print(f"✅ LLM 응답:")
                    print("-" * 60)
                    print(f"🎯 필드: {field_name_kr}")
                    print(f"📄 내용: {generated_value}")
                    print("-" * 60)
                    print(f"💾 파일 저장 완료!")
                    
                else:
                    print(f"❌ {field_name} 업데이트 실패")
                    break
            else:
                print(f"❌ LLM 응답 생성 실패: {next_gap}")
                break
            
            # 진행률 표시
            progress = self.get_generation_progress(user_id)
            print(f"📊 진행률: {progress['completed']}/{progress['total']} ({progress['progress']:.1f}%)")
        
        if iteration_count >= max_iterations:
            print(f"⚠️ 최대 반복 횟수({max_iterations})에 도달하여 생성을 중단합니다.")
        
        print("=" * 80)
        print(f"🎉 점진적 생성 완료! 총 {len(completed_fields)}개 필드 생성")
        
        return {
            "completed": iteration_count < max_iterations,
            "iterations": iteration_count,
            "completed_fields": completed_fields
        }
    
    def _call_llm_for_gap(self, prompt, gap_info):
        """LLM 호출하여 빈칸 채우기 - trpgbot_ragmd_sentencetr.py 함수 사용"""
        try:
            # trpgbot_ragmd_sentencetr.py의 generate_answer_without_rag 함수 임포트 및 사용
            try:
                from trpgbot_ragmd_sentencetr import generate_answer_without_rag
                print("🤖 trpgbot_ragmd_sentencetr.py의 LLM 함수로 요청 중...")
                
                # 시나리오 생성 세션으로 요청
                generated_text = generate_answer_without_rag(
                    query=prompt,
                    session_type="시나리오_생성",
                    character_context=""
                )
                
                if generated_text and generated_text.strip():
                    # 응답 길이 체크 및 정리
                    cleaned_text = generated_text.strip()
                    if len(cleaned_text) > 300:
                        cleaned_text = cleaned_text[:300] + "..."
                    
                    print(f"✅ LLM 응답 성공 (trpgbot_ragmd_sentencetr 사용)")
                    print(f"📄 응답: {cleaned_text[:100]}...")
                    return cleaned_text
                else:
                    print("⚠️ LLM 응답이 비어있습니다.")
                    return self._get_dummy_response(gap_info)
                    
            except ImportError as e:
                print(f"⚠️ trpgbot_ragmd_sentencetr.py 임포트 실패: {e}")
                print("💡 더미 응답을 사용합니다.")
                return self._get_dummy_response(gap_info)
            except Exception as e:
                print(f"⚠️ trpgbot_ragmd_sentencetr LLM 호출 실패: {e}")
                print("💡 더미 응답을 사용합니다.")
                return self._get_dummy_response(gap_info)
                
        except Exception as e:
            print(f"❌ LLM 시스템 오류: {e}")
            print("💡 더미 응답을 사용합니다.")
            return self._get_dummy_response(gap_info)
    
    def _get_dummy_response(self, gap_info):
        """더미 응답 생성 (LLM 호출 실패 시 사용)"""
        category = gap_info["category"]
        field = gap_info["field"]
        index = gap_info.get("index")
        if index is None:
            index = 0
        
        print(f"💡 더미 응답 생성: {category}.{field} (index: {index})")
        
        # 더미 응답 생성
        dummy_responses = {
            "overview": {
                "theme": "미스터리 조사 모험",
                "title": "사라진 마법사의 비밀",
                "setting": "중세 판타지 왕국 '알렌시아'의 변경 마을 '미스트하운드'",
                "main_conflict": "마을의 유명한 마법사 '엘드린'이 갑자기 사라지면서 마을에 이상한 현상들이 발생하기 시작했다",
                "objective": "마법사 엘드린의 행방을 찾고 마을에 일어나는 이상 현상의 원인을 규명하여 마을을 구하기",
                "rewards": "엘드린의 마법서, 고급 마법 아이템, 마을 사람들의 감사와 금전적 보상"
            },
            "episodes": {
                "title": [
                    "첫 번째 단서",
                    "숨겨진 진실",
                    "최후의 대결"
                ][index % 3],
                "objective": [
                    "마법사의 실종 원인을 조사하고 첫 번째 단서를 찾기",
                    "발견한 단서를 바탕으로 숨겨진 음모를 파헤치기",
                    "진짜 배후를 찾아 마을을 구하기"
                ][index % 3],
                "events": [
                    "마을 사람들의 증언 수집", "마법사의 연구실 조사", "이상한 현상 목격"
                ],
                "player_options": [
                    "마을 사람들과 대화하기", "연구실을 조사하기", "숲 속을 탐색하기"
                ],
                "success_result": "중요한 단서를 발견하고 다음 목표가 명확해진다",
                "failure_result": "시간을 낭비하지만 다른 방법을 시도할 기회가 있다"
            },
            "npcs": {
                "name": [
                    "마을 이장 브람",
                    "여관 주인 에밀리",
                    "의문의 방랑자 카이엔"
                ][index % 3],
                "appearance": [
                    "흰 수염을 기른 노인, 걱정스러운 표정",
                    "중년 여성, 친절하지만 예민해 보임",
                    "검은 망토를 입은 젊은 남성, 날카로운 눈빛"
                ][index % 3],
                "personality": [
                    "신중하고 책임감이 강하며 마을을 걱정하는 성격",
                    "친절하고 수다스럽지만 관찰력이 뛰어난 성격",
                    "신비롭고 말수가 적으며 뭔가 숨기는 듯한 성격"
                ][index % 3],
                "motivation": [
                    "마을의 평화를 되찾고 싶어함",
                    "손님들의 안전을 걱정하고 정보를 수집하고 싶어함",
                    "자신만의 목적이 있어 보이며 정보를 은밀히 찾고 있음"
                ][index % 3],
                "relationship": [
                    "정보 제공자이자 의뢰인",
                    "정보 제공자이자 휴식처 제공자",
                    "잠재적 동료 또는 라이벌"
                ][index % 3],
                "information": [
                    "마을의 역사와 마법사에 대한 정보",
                    "마을 사람들의 소문과 목격담",
                    "마법사의 과거와 관련된 비밀 정보"
                ][index % 3],
                "abilities": [
                    "마을 행정과 사람들을 설득하는 능력",
                    "정보 수집과 사람들의 마음을 읽는 능력",
                    "전투 기술과 마법에 대한 지식"
                ][index % 3]
            },
            "hints": {
                "content": [
                    "마법사의 연구실에서 발견된 이상한 마법진 그림",
                    "마을 근처 숲에서 들려오는 기이한 소리",
                    "마법사가 마지막으로 언급한 '금지된 실험'"
                ][index % 3],
                "discovery_method": [
                    "연구실을 세밀히 조사할 때 발견",
                    "마을 사람들과의 대화를 통해 수집",
                    "마법사의 일기나 메모에서 발견"
                ][index % 3],
                "connected_info": [
                    "고대 마법진과 소환술에 연관됨",
                    "숲 속의 비밀스러운 장소와 연관됨",
                    "위험한 마법 실험과 그 결과에 연관됨"
                ][index % 3],
                "difficulty": ["쉬움", "보통", "어려움"][index % 3],
                "relevant_sessions": [["조사", "탐험"], ["탐험", "전투"], ["추리", "마법"]][index % 3]
            },
            "dungeons": {
                "name": [
                    "버려진 마법탑",
                    "지하 실험실",
                    "고대 유적지"
                ][index % 3],
                "type": [
                    "마법 연구소",
                    "비밀 실험실",
                    "고대 신전"
                ][index % 3],
                "description": [
                    "높이 솟은 석탑으로 이상한 마법 기운이 감돌고 있다",
                    "지하 깊숙한 곳에 숨겨진 실험실로 위험한 냄새가 난다",
                    "오래된 돌기둥과 조각상들이 서 있는 신비로운 장소"
                ][index % 3],
                "atmosphere": [
                    "어둡고 신비로우며 마법적 위험이 느껴짐",
                    "밀폐되고 답답하며 불안한 기운이 감돎",
                    "고요하고 장엄하지만 고대의 힘이 잠들어 있음"
                ][index % 3],
                "rooms": [
                    ["입구 홀", "연구실", "서고", "마법진 방", "탑 꼭대기"],
                    ["비밀 통로", "실험실", "표본실", "창고", "중앙 실험실"],
                    ["입구", "기도실", "제단", "보물고", "성역"]
                ][index % 3],
                "traps": [
                    ["마법 함정", "환상 미로", "에너지 장벽"],
                    ["독가스 함정", "산성 웅덩이", "폭발 함정"],
                    ["수호자 석상", "마법 봉인", "저주 트랩"]
                ][index % 3],
                "puzzles": [
                    ["마법진 퍼즐", "고대 문자 해독", "원소 조합"],
                    ["비밀번호 찾기", "실험 재현하기", "장치 수리"],
                    ["고대 의식 재현", "성스러운 시 암송", "보석 배치"]
                ][index % 3],
                "monsters": [
                    ["마법 골렘", "떠도는 영혼", "소환된 정령"],
                    ["변이된 생물", "독성 슬라임", "실험체"],
                    ["고대 수호자", "언데드 사제", "신성한 야수"]
                ][index % 3],
                "treasures": [
                    ["마법서", "마법 지팡이", "마나 포션", "마법 반지"],
                    ["실험 자료", "강화 무기", "치료 약품", "연금술 재료"],
                    ["고대 유물", "축복받은 무기", "성스러운 보석", "신의 가호"]
                ][index % 3]
            }
        }
        
        if category in dummy_responses and field in dummy_responses[category]:
            response = dummy_responses[category][field]
            # 리스트인 경우 문자열로 변환
            if isinstance(response, list):
                return ", ".join(response)
            return response
        
        return f"[더미] {field}에 대한 생성된 내용"
    
    def _get_korean_field_name(self, field):
        """필드명을 한국어로 변환"""
        field_name_kr = {
            # 개요
            "title": "시나리오 제목", "theme": "테마", "setting": "배경설정", 
            "main_conflict": "주요 갈등", "objective": "목표", "rewards": "보상",
            # 에피소드
            "objective": "에피소드 목표", "events": "주요 사건들",
            "player_options": "플레이어 선택지", "success_result": "성공 결과", "failure_result": "실패 결과",
            # NPC
            "name": "NPC 이름", "appearance": "외모", "personality": "성격",
            "motivation": "동기", "relationship": "플레이어와의 관계", "information": "보유 정보", "abilities": "특수 능력",
            # 힌트
            "content": "힌트 내용", "discovery_method": "발견 방법", "connected_info": "연결된 정보",
            "difficulty": "난이도", "relevant_sessions": "관련 세션",
            # 던전
            "name": "던전 이름", "type": "던전 유형", "description": "던전 설명", "atmosphere": "분위기",
            "rooms": "방/구역", "traps": "함정", "puzzles": "퍼즐", "monsters": "몬스터", "treasures": "보물"
        }
        return field_name_kr.get(field, field)
    
    def _get_korean_category_name(self, category):
        """카테고리명을 한국어로 변환"""
        category_name_kr = {
            "overview": "시나리오 개요",
            "episodes": "에피소드",
            "npcs": "NPC",
            "hints": "힌트",
            "dungeons": "던전"
        }
        return category_name_kr.get(category, category)
    
    def start_progressive_generation(self, user_id, user_request=None):
        """점진적 시나리오 생성 시작"""
        # 시나리오 초기화 (존재하지 않는 경우)
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            scenario_data = self.init_scenario_creation(user_id)
        
        # 사용자 요청이 있으면 저장
        if user_request:
            scenario_data["user_preferences"] = {
                "user_input": user_request,
                "preferences_detected": True,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.save_scenario(user_id, scenario_data)
        
        logger.info(f"🎯 사용자 {user_id}의 점진적 시나리오 생성 시작")
        return True
    
    def get_next_gap_for_user(self, user_id):
        """사용자에게 다음으로 채워야 할 빈칸 정보 반환"""
        next_gap = self.find_next_most_important_gap(user_id)
        
        if not next_gap:
            return None
        
        # 사용자 친화적인 정보로 변환
        field_name_kr = {
            # 개요
            "title": "시나리오 제목", "theme": "테마", "setting": "배경설정", 
            "main_conflict": "주요 갈등", "objective": "목표", "rewards": "보상",
            # 에피소드
            "title": "에피소드 제목", "objective": "에피소드 목표", "events": "주요 사건들",
            "player_options": "플레이어 선택지", "success_result": "성공 결과", "failure_result": "실패 결과",
            # NPC
            "name": "NPC 이름", "appearance": "외모", "personality": "성격",
            "motivation": "동기", "relationship": "플레이어와의 관계", "information": "보유 정보", "abilities": "특수 능력",
            # 힌트
            "content": "힌트 내용", "discovery_method": "발견 방법", "connected_info": "연결된 정보",
            "difficulty": "난이도", "relevant_sessions": "관련 세션",
            # 던전
            "name": "던전 이름", "type": "던전 유형", "description": "던전 설명", "atmosphere": "분위기",
            "rooms": "방/구역", "traps": "함정", "puzzles": "퍼즐", "monsters": "몬스터", "treasures": "보물"
        }
        
        korean_field = field_name_kr.get(next_gap["field"], next_gap["field"])
        
        return {
            "category": next_gap["category"],
            "field": next_gap["field"],
            "korean_field": korean_field,
            "priority": next_gap["priority"],
            "context_info": self._get_context_for_gap(next_gap)
        }
    
    def _get_context_for_gap(self, gap_info):
        """빈칸에 대한 컨텍스트 정보 생성"""
        category = gap_info["category"]
        field = gap_info["field"]
        
        if category == "overview":
            return f"시나리오의 핵심이 되는 {field}를 설정하는 단계입니다."
        elif category == "episodes":
            return f"에피소드 {gap_info.get('index', 0) + 1}의 {field}를 설정하는 단계입니다."
        elif category == "npcs":
            npc_name = gap_info.get('name', f"NPC {gap_info.get('index', 0) + 1}")
            return f"{npc_name}의 {field}를 설정하는 단계입니다."
        elif category == "hints":
            return f"힌트 {gap_info.get('index', 0) + 1}의 {field}를 설정하는 단계입니다."
        elif category == "dungeons":
            dungeon_name = gap_info.get('name', f"던전 {gap_info.get('index', 0) + 1}")
            return f"{dungeon_name}의 {field}를 설정하는 단계입니다."
        
        return "현재 단계의 정보를 설정합니다."
    
    def process_next_gap_automatically(self, user_id):
        """다음 빈칸을 자동으로 LLM을 통해 채우기"""
        next_gap = self.find_next_most_important_gap(user_id)
        
        if not next_gap:
            return {"completed": True, "message": "모든 필드가 완료되었습니다!"}
        
        # 프롬프트 생성
        prompt = self.generate_single_gap_prompt(user_id, next_gap)
        if not prompt:
            return {"completed": False, "error": "프롬프트 생성 실패"}
        
        # LLM 호출
        generated_value = self._call_llm_for_gap(prompt, next_gap)
        
        if not generated_value:
            return {"completed": False, "error": "LLM 응답 생성 실패"}
        
        # 값 업데이트
        success = self.update_single_gap(user_id, next_gap, generated_value)
        
        if success:
            field_name_kr = {
                "title": "시나리오 제목", "theme": "테마", "setting": "배경설정", 
                "main_conflict": "주요 갈등", "objective": "목표", "rewards": "보상"
            }.get(next_gap["field"], next_gap["field"])
            
            return {
                "completed": False,
                "success": True,
                "field": field_name_kr,
                "value": generated_value,
                "message": f"✅ {field_name_kr}이(가) 생성되었습니다!"
            }
        else:
            return {"completed": False, "error": "필드 업데이트 실패"}
    
    def process_user_input_for_gap(self, user_id, user_input):
        """사용자 입력으로 다음 빈칸 채우기"""
        next_gap = self.find_next_most_important_gap(user_id)
        
        if not next_gap:
            return {"completed": True, "message": "모든 필드가 완료되었습니다!"}
        
        # 사용자 입력으로 값 업데이트
        success = self.update_single_gap(user_id, next_gap, user_input)
        
        if success:
            field_name_kr = {
                "title": "시나리오 제목", "theme": "테마", "setting": "배경설정", 
                "main_conflict": "주요 갈등", "objective": "목표", "rewards": "보상"
            }.get(next_gap["field"], next_gap["field"])
            
            return {
                "completed": False,
                "success": True,
                "field": field_name_kr,
                "value": user_input,
                "message": f"✅ {field_name_kr}이(가) 설정되었습니다!"
            }
        else:
            return {"completed": False, "error": "필드 업데이트 실패"}
    
    def get_generation_progress(self, user_id):
        """현재 생성 진행도 반환"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return {"progress": 0, "total": 0, "completed": 0}
        
        # 전체 필요한 필드 수 계산 (기본적인 필드들만)
        total_basic_fields = 6  # overview의 6개 필드 (title, theme, setting, main_conflict, objective, rewards)
        
        # 완료된 필드 수 계산
        overview = scenario_data.get("scenario", {}).get("overview", {})
        completed_fields = sum(1 for field in ["title", "theme", "setting", "main_conflict", "objective", "rewards"] 
                              if overview.get(field) and overview.get(field).strip())
        
        progress_percentage = (completed_fields / total_basic_fields) * 100 if total_basic_fields > 0 else 0
        
        return {
            "progress": round(progress_percentage, 1),
            "total": total_basic_fields,
            "completed": completed_fields,
            "remaining": total_basic_fields - completed_fields
        }
    
    def ensure_scenario_npcs(self, user_id):
        """시나리오에 필요한 NPC들이 생성되어 있는지 확인하고 없으면 생성"""
        if not npc_manager:
            logger.warning("⚠️ NPC 매니저를 사용할 수 없습니다.")
            return False
            
        try:
            # 현재 시나리오 데이터 로드
            scenario_data = self.load_scenario(user_id)
            if not scenario_data:
                logger.warning("⚠️ 시나리오 데이터가 없어 NPC를 생성할 수 없습니다.")
                return False
            
            # NPC 생성 또는 확인
            logger.info(f"🎭 사용자 {user_id}의 시나리오 NPC 확인/생성 중...")
            npc_success = npc_manager.ensure_npcs_exist(user_id, scenario_data)
            
            if npc_success:
                logger.info(f"✅ 사용자 {user_id}의 시나리오 NPC 준비 완료")
                
                # 시나리오에 NPC 정보 추가 (참조만 저장)
                scenario_data["npc_generated"] = True
                scenario_data["npc_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.save_scenario(user_id, scenario_data)
                
                return True
            else:
                logger.error(f"❌ 사용자 {user_id}의 시나리오 NPC 생성 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ 시나리오 NPC 확인/생성 중 오류: {e}")
            return False
    
    def generate_npcs_for_current_scenario(self, user_id, force_regenerate=False):
        """현재 시나리오에 맞는 NPC 강제 생성"""
        if not npc_manager:
            logger.warning("⚠️ NPC 매니저를 사용할 수 없습니다.")
            return False
            
        try:
            # 현재 시나리오 데이터 로드
            scenario_data = self.load_scenario(user_id)
            if not scenario_data:
                logger.warning("⚠️ 시나리오 데이터가 없어 NPC를 생성할 수 없습니다.")
                return False
            
            # 기존 NPC가 있고 강제 재생성이 아니면 스킵
            if not force_regenerate:
                existing_npcs = npc_manager.load_npcs(user_id)
                if existing_npcs and len(existing_npcs) >= 3:
                    logger.info(f"✅ 기존 NPC가 충분히 있습니다: {len(existing_npcs)}명")
                    return True
            
            logger.info(f"🎭 사용자 {user_id}의 시나리오 기반 NPC 생성 시작...")
            
            # NPC 생성
            npc_success = npc_manager.create_npcs_for_scenario(user_id, scenario_data, npc_count=5)
            
            if npc_success:
                # 시나리오에 NPC 생성 기록
                scenario_data["npc_generated"] = True
                scenario_data["npc_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                scenario_data["npc_force_regenerated"] = force_regenerate
                self.save_scenario(user_id, scenario_data)
                
                logger.info(f"✅ 사용자 {user_id}의 NPC 생성 완료")
                return True
            else:
                logger.error(f"❌ 사용자 {user_id}의 NPC 생성 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ NPC 생성 중 오류: {e}")
            return False
    
    def get_npc_summary_for_scenario(self, user_id):
        """시나리오용 NPC 요약 정보 반환"""
        if not npc_manager:
            return "NPC 매니저를 사용할 수 없습니다."
            
        try:
            return npc_manager.get_npc_summary(user_id)
        except Exception as e:
            logger.error(f"❌ NPC 요약 정보 조회 오류: {e}")
            return "NPC 정보를 조회할 수 없습니다."
    
    def is_npc_stage_complete(self, user_id):
        """NPC 단계가 완료되었는지 확인"""
        if not npc_manager:
            return False
            
        try:
            # NPC 매니저에서 NPC 존재 여부 확인
            existing_npcs = npc_manager.load_npcs(user_id)
            
            # 최소 3명의 NPC가 있어야 완료로 간주
            if existing_npcs and len(existing_npcs) >= 3:
                logger.info(f"✅ NPC 단계 완료 확인: {len(existing_npcs)}명")
                return True
            else:
                logger.info(f"⚠️ NPC 단계 미완료: {len(existing_npcs) if existing_npcs else 0}명")
                return False
                
        except Exception as e:
            logger.error(f"❌ NPC 단계 완료 확인 중 오류: {e}")
            return False

    def get_current_episode(self, user_id):
        """현재 진행중인 에피소드 정보 반환"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return None
            
        episode_progress = scenario_data.get("episode_progress", {})
        
        # 진행중인 에피소드 찾기
        for episode_key, progress in episode_progress.items():
            if progress.get("status") == "진행중":
                # episode_key에서 ID 추출 (episode_1 -> 1)
                episode_id = episode_key.replace("episode_", "")
                episodes = scenario_data.get("scenario", {}).get("episodes", [])
                for episode in episodes:
                    if str(episode.get("id", "")) == episode_id:
                        return episode
        
        # 진행중인 에피소드가 없으면 첫 번째 에피소드 반환
        episodes = scenario_data.get("scenario", {}).get("episodes", [])
        return episodes[0] if episodes else None
    
    def get_next_episode_info(self, user_id):
        """다음 에피소드 정보 반환"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return None
            
        episodes = scenario_data.get("scenario", {}).get("episodes", [])
        if not episodes:
            return None
        
        episode_progress = scenario_data.get("episode_progress", {})
        
        # 현재 진행중인 에피소드 다음 에피소드 찾기
        current_episode_index = -1
        for i, episode in enumerate(episodes):
            episode_key = f"episode_{episode.get('id', i + 1)}"
            if episode_progress.get(episode_key, {}).get("status") == "진행중":
                current_episode_index = i
                break
        
        # 다음 에피소드가 있으면 반환
        if current_episode_index >= 0 and current_episode_index + 1 < len(episodes):
            return episodes[current_episode_index + 1]
        
        return None
    
    def advance_to_next_episode(self, user_id):
        """다음 에피소드로 진행"""
        try:
            scenario_data = self.load_scenario(user_id)
            if not scenario_data:
                return False
            
            episodes = scenario_data.get("scenario", {}).get("episodes", [])
            if not episodes:
                return False
            
            if "episode_progress" not in scenario_data:
                scenario_data["episode_progress"] = {}
            
            episode_progress = scenario_data["episode_progress"]
            
            # 현재 에피소드를 완료로 변경
            current_episode_index = -1
            for i, episode in enumerate(episodes):
                episode_key = f"episode_{episode.get('id', i + 1)}"
                if episode_progress.get(episode_key, {}).get("status") == "진행중":
                    episode_progress[episode_key]["status"] = "완료"
                    current_episode_index = i
                    break
            
            # 다음 에피소드를 진행중으로 설정
            if current_episode_index >= 0 and current_episode_index + 1 < len(episodes):
                next_episode = episodes[current_episode_index + 1]
                next_episode_key = f"episode_{next_episode.get('id', current_episode_index + 2)}"
                episode_progress[next_episode_key] = {
                    "status": "진행중",
                    "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "round_count": 0
                }
                
                # 진행 상태 저장
                self.save_scenario(user_id, scenario_data)
                
                logger.info(f"에피소드 진행: 사용자 {user_id}, {current_episode_index + 1}번째 → {current_episode_index + 2}번째 에피소드")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"에피소드 진행 오류: {e}")
            return False

    def generate_complete_scenario(self, user_id, user_request=None, max_iterations=50):
        """완전한 시나리오를 점진적으로 생성"""
        print(f"\n🎬 완전한 시나리오 생성 시작!")
        print(f"👤 사용자 ID: {user_id}")
        if user_request:
            print(f"📝 요청사항: {user_request}")
        
        # 1. 점진적 생성 시작
        self.start_progressive_generation(user_id, user_request)
        
        # 2. 기본 시나리오 구조 설정 (빈 에피소드, NPC, 힌트, 던전 추가)
        self._ensure_basic_scenario_structure(user_id)
        
        # 3. 점진적 생성 실행
        result = self.progressive_scenario_generation(user_id, max_iterations)
        
        # 4. 최종 결과 출력
        self._print_final_scenario_summary(user_id)
        
        return result
    
    def _ensure_basic_scenario_structure(self, user_id):
        """기본 시나리오 구조 보장 (빈 에피소드, NPC 등 추가)"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return
        
        scenario = scenario_data.get("scenario", {})
        
        # 에피소드가 없으면 3개 추가
        if not scenario.get("episodes"):
            scenario["episodes"] = [
                {"id": 1, "title": "", "objective": "", "events": [], "player_options": [], "success_result": "", "failure_result": ""},
                {"id": 2, "title": "", "objective": "", "events": [], "player_options": [], "success_result": "", "failure_result": ""},
                {"id": 3, "title": "", "objective": "", "events": [], "player_options": [], "success_result": "", "failure_result": ""}
            ]
        
        # NPC가 없으면 3개 추가
        if not scenario.get("npcs"):
            scenario["npcs"] = [
                {"id": 1, "name": "", "appearance": "", "personality": "", "motivation": "", "relationship": "", "information": "", "abilities": ""},
                {"id": 2, "name": "", "appearance": "", "personality": "", "motivation": "", "relationship": "", "information": "", "abilities": ""},
                {"id": 3, "name": "", "appearance": "", "personality": "", "motivation": "", "relationship": "", "information": "", "abilities": ""}
            ]
        
        # 힌트가 없으면 3개 추가
        if not scenario.get("hints"):
            scenario["hints"] = [
                {"id": 1, "content": "", "discovery_method": "", "connected_info": "", "difficulty": "", "relevant_sessions": []},
                {"id": 2, "content": "", "discovery_method": "", "connected_info": "", "difficulty": "", "relevant_sessions": []},
                {"id": 3, "content": "", "discovery_method": "", "connected_info": "", "difficulty": "", "relevant_sessions": []}
            ]
        
        # 던전이 없으면 1개 추가
        if not scenario.get("dungeons"):
            scenario["dungeons"] = [
                {"id": 1, "name": "", "type": "", "description": "", "atmosphere": "", "rooms": [], "traps": [], "puzzles": [], "monsters": [], "treasures": []}
            ]
        
        self.save_scenario(user_id, scenario_data)
        print("📋 기본 시나리오 구조 준비 완료")
    
    def _print_final_scenario_summary(self, user_id):
        """최종 시나리오 요약 출력"""
        scenario_data = self.load_scenario(user_id)
        if not scenario_data:
            return
        
        print("\n" + "="*80)
        print("🎭 생성된 시나리오 요약")
        print("="*80)
        
        scenario = scenario_data.get("scenario", {})
        
        # 개요
        overview = scenario.get("overview", {})
        if overview.get("title"):
            print(f"📖 제목: {overview['title']}")
        if overview.get("theme"):
            print(f"🎯 테마: {overview['theme']}")
        if overview.get("setting"):
            print(f"🌍 배경: {overview['setting']}")
        if overview.get("main_conflict"):
            print(f"⚔️ 주요 갈등: {overview['main_conflict']}")
        if overview.get("objective"):
            print(f"🎯 목표: {overview['objective']}")
        if overview.get("rewards"):
            print(f"🏆 보상: {overview['rewards']}")
        
        # 에피소드
        episodes = scenario.get("episodes", [])
        if episodes:
            print(f"\n📚 에피소드 ({len(episodes)}개):")
            for i, episode in enumerate(episodes, 1):
                if episode.get("title"):
                    print(f"  {i}. {episode['title']}")
                    if episode.get("objective"):
                        print(f"     └ 목표: {episode['objective']}")
        
        # NPC
        npcs = scenario.get("npcs", [])
        filled_npcs = [npc for npc in npcs if npc.get("name")]
        if filled_npcs:
            print(f"\n👥 주요 NPC ({len(filled_npcs)}명):")
            for npc in filled_npcs:
                print(f"  • {npc['name']}")
                if npc.get("relationship"):
                    print(f"    └ 관계: {npc['relationship']}")
        
        # 힌트
        hints = scenario.get("hints", [])
        filled_hints = [hint for hint in hints if hint.get("content")]
        if filled_hints:
            print(f"\n🔍 힌트 ({len(filled_hints)}개):")
            for i, hint in enumerate(filled_hints, 1):
                print(f"  {i}. {hint['content'][:50]}...")
        
        # 던전
        dungeons = scenario.get("dungeons", [])
        filled_dungeons = [dungeon for dungeon in dungeons if dungeon.get("name")]
        if filled_dungeons:
            print(f"\n🏰 던전/탐험지 ({len(filled_dungeons)}개):")
            for dungeon in filled_dungeons:
                print(f"  • {dungeon['name']}: {dungeon.get('type', '유형미정')}")
        
        print("="*80)
        print("🎉 시나리오 생성 완료!")
        print("="*80)
    
    def test_progressive_generation(self, user_id=99999):
        """점진적 생성 시스템 테스트"""
        user_request = """중세 판타지 시나리오를 만들어주세요. 다음 중 하나의 테마를 선택해서 예시를 참고해서 창의적으로 시나리오를 만들어주세요. 진행하되, 반드시 명확한 결말이 있는 핵심 사건으로 구성해주세요:

1. **미스터리 시나리오**: (예시) 마법사가 사라진 마을에서 일어나는 이상한 사건들을 조사하는 내용
2. **탐험 시나리오**: (예시) 고대 유적지에서 잃어버린 보물을 찾는 모험
3. **역사적 시나리오**: (예시) 왕국의 정치적 음모와 왕위 계승 문제를 해결하는 내용

선택한 테마에 맞춰 3-4개의 에피소드로 구성하고, 각 에피소드마다 명확한 목표와 결과가 있도록 만들어주세요. 최종적으로는 주인공들이 핵심 문제를 해결하고 보상을 받는 완전한 스토리로 완성해주세요."""
        
        # 완전한 시나리오 생성 실행
        result = self.generate_complete_scenario(user_id, user_request, max_iterations=20)
        
        print(f"\n📊 생성 결과:")
        print(f"  • 완료 여부: {'✅' if result['completed'] else '❌'}")
        print(f"  • 반복 횟수: {result['iterations']}")
        print(f"  • 생성된 필드: {len(result['completed_fields'])}개")
        
        return result

# 전역 인스턴스
scenario_manager = ScenarioManager()

def create_scenario_interactive():
    """대화형 시나리오 생성"""
    print("\n🎭 TRPG 시나리오 생성기")
    print("="*60)
    
    # 사용자 ID 입력
    user_id = input("👤 사용자 ID를 입력하세요 (기본값: 99999): ").strip()
    if not user_id:
        user_id = "99999"
    
    # 시나리오 요청 입력
    print("\n📝 어떤 시나리오를 만들고 싶으신가요?")
    print("예: '중세 판타지 미스터리', '현대 호러', '사이버펑크 액션' 등")
    user_request = input("🎯 시나리오 요청: ").strip()
    
    if not user_request:
        user_request = "중세 판타지 미스터리 시나리오"
        print(f"기본값 사용: {user_request}")
    
    # 최대 반복 횟수 입력
    max_iter = input("\n🔄 최대 생성 반복 횟수 (기본값: 30): ").strip()
    try:
        max_iterations = int(max_iter) if max_iter else 30
    except ValueError:
        max_iterations = 30
    
    print(f"\n🚀 시나리오 생성을 시작합니다...")
    print(f"   • 사용자 ID: {user_id}")
    print(f"   • 요청사항: {user_request}")
    print(f"   • 최대 반복: {max_iterations}회")
    
    # 시나리오 생성 실행
    result = scenario_manager.generate_complete_scenario(user_id, user_request, max_iterations)
    
    return result

# 테스트 실행 (직접 실행 시에만)
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # 대화형 모드
        create_scenario_interactive()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # 테스트 모드
        scenario_manager.test_progressive_generation()
    else:
        print("\n🎭 TRPG 시나리오 생성기")
        print("="*60)
        print("사용법:")
        print("  python scenario_manager.py interactive  # 대화형 생성")
        print("  python scenario_manager.py test        # 테스트 실행")
        print("\n또는 직접 함수 호출:")
        print("  scenario_manager.generate_complete_scenario(user_id, '요청사항')")
        
        # 기본 테스트 실행
        print("\n기본 테스트를 실행합니다...")
        scenario_manager.test_progressive_generation()