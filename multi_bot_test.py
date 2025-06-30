# -*- coding: utf-8 -*-
"""
멀티봇 TRPG 지속적 대화 시스템

이 스크립트는 마스터 봇과 3명의 플레이어 봇이 지속적으로 대화하며
TRPG 세션을 진행하는 시스템입니다.

사용 방법:
1. 기본 실행 (지속적인 대화): python multi_bot_test.py
2. 간단한 2라운드 테스트: main() 함수를 run_simple_test()로 변경

특징:
- 무한 루프로 대화가 계속 진행됩니다
- Ctrl+C로 언제든 중단 가능합니다
- 최대 150라운드까지 자동 제한됩니다
- 오류 발생 시 자동으로 재시도합니다
- 메모리 누수 방지 및 시스템 안정성 개선
"""

import asyncio
import gc
import logging
import os
import psutil
import threading
import time
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv

# 기존 봇들의 핸들러 임포트
from player_bot1 import handle_message as player1_handle_message, player_character as player1_characters, player_settings as player1_settings
from player_bot2 import handle_message as player2_handle_message, player_character as player2_characters, player_settings as player2_settings  
from player_bot3 import handle_message as player3_handle_message, player_character as player3_characters, player_settings as player3_settings

# 마스터 봇의 기존 메시지 처리 기능 임포트
from message_processor import handle_message as master_handle_message
from character_manager import CharacterManager
from session_manager import session_manager
from scenario_manager import scenario_manager

# NPC 매니저 임포트 추가
try:
    from npc_manager import npc_manager
except ImportError:
    logger.warning("⚠️ NPC 매니저를 임포트할 수 없습니다. NPC 기능이 제한됩니다.")
    npc_manager = None

# 메시지 처리 유틸리티 임포트
try:
    from message_processor import send_long_message, split_long_message, SAFE_MESSAGE_LENGTH
except ImportError:
    logger.warning("⚠️ 메시지 처리 유틸리티를 임포트할 수 없습니다. 기본 방식을 사용합니다.")
    
    # 폴백 함수들
    def split_long_message(text: str, max_length: int = 4000) -> list:
        if len(text) <= max_length:
            return [text]
        return [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    async def send_long_message_fallback(bot, chat_id, text: str, prefix: str = ""):
        chunks = split_long_message(text)
        if len(chunks) == 1:
            await bot.send_message(chat_id=chat_id, text=f"{prefix}{text}")
        else:
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await bot.send_message(chat_id=chat_id, text=f"{prefix}{chunk}")
                else:
                    await bot.send_message(chat_id=chat_id, text=f"[계속]\n\n{chunk}")
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)
    
    SAFE_MESSAGE_LENGTH = 4000

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 봇 토큰들
MASTER_TOKEN = os.getenv('BOT_TOKEN')
PLAYER1_TOKEN = os.getenv('PLAYER1_BOT_TOKEN')  
PLAYER2_TOKEN = os.getenv('PLAYER2_BOT_TOKEN')
PLAYER3_TOKEN = os.getenv('PLAYER3_BOT_TOKEN')
TEST_CHAT_ID = os.getenv('TEST_CHAT_ID')

# 봇 객체들
master_bot = Bot(MASTER_TOKEN) if MASTER_TOKEN else None
player1_bot = Bot(PLAYER1_TOKEN) if PLAYER1_TOKEN else None
player2_bot = Bot(PLAYER2_TOKEN) if PLAYER2_TOKEN else None  
player3_bot = Bot(PLAYER3_TOKEN) if PLAYER3_TOKEN else None

# 대화 상태 관리
conversation_active = False
last_message_from_master = ""
response_queue = []

# 🆕 시스템 안정성을 위한 설정 (강화)
MAX_RESPONSE_QUEUE_SIZE = 50  # 응답 큐 최대 크기 (감소)
MAX_ROUNDS = 100  # 최대 라운드 수 (감소)
MEMORY_CHECK_INTERVAL = 5  # 메모리 체크 간격 (더 자주)
MAX_MEMORY_MB = 1536  # 최대 메모리 사용량 (감소)
LLM_TIMEOUT = 120  # LLM 응답 타임아웃 (감소)
MAX_SCENARIO_STEPS = 3  # 시나리오 생성 단계 제한 (🆕 추가)

def check_system_resources():
    """시스템 리소스 체크 함수"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        logger.info(f"📊 시스템 리소스: 메모리 {memory_mb:.1f}MB, CPU {cpu_percent:.1f}%")
        
        if memory_mb > MAX_MEMORY_MB:
            logger.warning(f"⚠️ 메모리 사용량 과다: {memory_mb:.1f}MB > {MAX_MEMORY_MB}MB")
            return False
            
        return True
    except Exception as e:
        logger.error(f"시스템 리소스 체크 오류: {e}")
        return True

def cleanup_memory():
    """메모리 정리 함수"""
    global response_queue
    
    try:
        # 응답 큐 크기 제한
        if len(response_queue) > MAX_RESPONSE_QUEUE_SIZE:
            response_queue = response_queue[-MAX_RESPONSE_QUEUE_SIZE//2:]
            logger.info(f"🧹 응답 큐 정리: {len(response_queue)}개 항목 유지")
        
        # 가비지 컬렉션 강제 실행
        collected = gc.collect()
        if collected > 0:
            logger.info(f"🧹 가비지 컬렉션: {collected}개 객체 정리")
            
    except Exception as e:
        logger.error(f"메모리 정리 중 오류: {e}")

class MockUpdate:
    """플레이어 봇의 handle_message 함수를 테스트하기 위한 모의 Update 객체"""
    def __init__(self, user_id, message_text, chat_id):
        self.effective_user = MockUser(user_id)
        self.message = MockMessage(message_text, chat_id)
        
class MockUser:
    def __init__(self, user_id):
        self.id = user_id
        self.first_name = f"TestUser{user_id}"
        self.username = f"testuser{user_id}"

class MockMessage:
    def __init__(self, text, chat_id):
        self.text = text
        self.chat_id = chat_id
        self._replies = []
        
    async def reply_text(self, text):
        """실제 텔레그램으로 메시지를 보내는 대신 큐에 저장"""
        self._replies.append(text)
        response_queue.append({
            'bot_type': 'player',
            'user_id': self.chat_id,  # 임시로 chat_id를 user_id로 사용
            'text': text
        })
        return text



async def ensure_test_directories():
    """테스트에 필요한 디렉토리들을 미리 생성"""
    import os
    directories = [
        'characters',
        'sessions', 
        'scenarios',
        'conversations'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 디렉토리 확인/생성: {directory}")
        except Exception as e:
            logger.error(f"디렉토리 생성 실패 {directory}: {e}")

async def setup_master_session(master_user_id):
    """마스터 세션 초기화 및 설정"""
    logger.info("🎭 마스터 세션을 초기화하는 중...")
    
    try:
        # 0. 필요한 디렉토리 생성
        await ensure_test_directories()
        
        # 1. 캐릭터 매니저 초기화
        CharacterManager.initialize()
        
        # 2. 마스터용 캐릭터 설정 (간단한 방식)
        CharacterManager.set_player_count(master_user_id, 3)
        
        # 3개의 테스트 캐릭터 생성 (안전한 방식)
        test_characters = [
            {"이름": "아리아", "클래스": "로그", "가치관": "중립선"},
            {"이름": "바로스", "클래스": "전사", "가치관": "혼돈중립"},
            {"이름": "세레나", "클래스": "마법사", "가치관": "질서선"}
        ]
        
        for i, char_data in enumerate(test_characters):
            CharacterManager.set_current_character_index(master_user_id, i)
            for field, value in char_data.items():
                CharacterManager.update_character_field(master_user_id, field, value)
            CharacterManager.increment_completed_character(master_user_id)
        
        # 🆕 3. 세션 건너뛰기 방지를 위한 엄격한 설정
        # 시나리오 생성 세션으로 설정 (모험_준비나 모험_진행으로 바로 가지 않도록)
        session_manager.log_session(master_user_id, "시나리오_생성", "멀티봇 테스트용 시나리오 생성 시작 - 단계별 진행")
        
        # 4. 시나리오 매니저 초기화
        scenario_manager.init_scenario_creation(master_user_id)
        
        # 5. 시나리오 생성을 위한 초기 스테이지 설정
        from scenario_manager import ScenarioStage
        scenario_manager.set_current_stage(master_user_id, ScenarioStage.OVERVIEW.value)
        
        # 🆕 6. 첫 번째 에피소드를 진행중으로 표시
        scenario_manager.update_episode_progress(master_user_id, 1, "진행중")
        
        # 🆕 7. 세션 전환 방지를 위한 추가 설정
        logger.info("🔒 세션 건너뛰기 방지 설정 적용")
        logger.info(f"📍 현재 세션: 시나리오_생성")
        logger.info(f"📋 예정된 세션 순서: 시나리오_생성 → 모험_생성 → 던전_생성 → 파티_생성 → 모험_준비 → 모험_진행")
        
        logger.info("✅ 마스터 세션 초기화 완료!")
        
    except Exception as e:
        logger.error(f"마스터 세션 초기화 중 오류: {e}")
        # 오류 발생 시 기본 세션으로 설정
        try:
            session_manager.log_session(master_user_id, "시나리오_생성", "기본 세션 설정 - 오류 복구")
        except:
            pass

async def setup_test_characters():
    """테스트용 캐릭터 설정"""
    # 플레이어1 캐릭터 설정 (아리아)
    test_user_id_1 = 12345  # 테스트용 사용자 ID
    player1_characters[test_user_id_1] = type('Character', (), {
        'name': '아리아',
        'class_type': '로그', 
        'level': 1,
        'alignment': '중립선',
        'background': '도시 출신',
        'personality': '쾌활하고 모험을 좋아함',
        'goals': '새로운 경험과 보물 찾기',
        'fears': '지루한 일상',
        'strength': 12, 'dexterity': 16, 'constitution': 14,
        'intelligence': 13, 'wisdom': 11, 'charisma': 15,
        'hp': 8, 'max_hp': 8, 'ac': 13, 'initiative': 3,
        'skills': ['은신', '자물쇠따기'], 'equipment': ['단검', '도구상자'], 'spells': [],
        'get_personality_prompt': lambda self: """
당신은 '아리아'라는 로그 캐릭터를 플레이하고 있습니다.

## 캐릭터 정보:
- **이름**: 아리아
- **클래스**: 로그 (레벨 1)
- **가치관**: 중립선
- **배경**: 도시 출신
- **성격**: 쾌활하고 모험을 좋아함
- **목표**: 새로운 경험과 보물 찾기
- **두려워하는 것**: 지루한 일상

당신의 성격은 쾌활하고 규칙을 따르는 것을 좋아합니다. 그러나 모험을 좋아하고 새로운 것을 시도하는 것을 좋아합니다. 가끔 엉뚱한 행동과 말을 하기도 합니다.

항상 '아리아' 캐릭터의 시점에서 1인칭으로 대화하세요. 대화하듯 같단히 한두문장으로만 답을 하세요.
"""
    })()
    
    player1_settings[test_user_id_1] = {"character_loaded": True, "auto_response": True, "response_style": "balanced"}
    
    # 플레이어2 캐릭터 설정 (바로스)
    test_user_id_2 = 12346
    player2_characters[test_user_id_2] = type('Character', (), {
        'name': '바로스',
        'class_type': '전사',
        'level': 1, 
        'alignment': '혼돈중립',
        'background': '용병 출신',
        'personality': '승부욕이 강하고 규칙의 빈틈을 파고듦',
        'goals': '강해지기와 승리',
        'fears': '패배와 굴복',
        'strength': 16, 'dexterity': 12, 'constitution': 15,
        'intelligence': 10, 'wisdom': 11, 'charisma': 13,
        'hp': 10, 'max_hp': 10, 'ac': 16, 'initiative': 1,
        'skills': ['운동', '위압'], 'equipment': ['장검', '사슬갑옷'], 'spells': [],
        'get_personality_prompt': lambda self: """
당신은 '바로스'라는 전사 캐릭터를 플레이하고 있습니다.

## 캐릭터 정보:
- **이름**: 바로스
- **클래스**: 전사 (레벨 1)
- **가치관**: 혼돈중립
- **배경**: 용병 출신
- **성격**: 승부욕이 강하고 규칙의 빈틈을 파고듦
- **목표**: 강해지기와 승리
- **두려워하는 것**: 패배와 굴복

당신의 성격은 규칙의 빈틈을 파고들어 승리를 쟁취하고 강해지는 것을 좋아합니다.

항상 '바로스' 캐릭터의 시점에서 1인칭으로 대화하세요. 대화하듯 같단히 한두문장으로만 답을 하세요.
"""
    })()
    
    player2_settings[test_user_id_2] = {"character_loaded": True, "auto_response": True, "response_style": "active"}
    
    # 플레이어3 캐릭터 설정 (세레나)
    test_user_id_3 = 12347
    player3_characters[test_user_id_3] = type('Character', (), {
        'name': '세레나', 
        'class_type': '마법사',
        'level': 1,
        'alignment': '질서선',
        'background': '학자 출신',
        'personality': '분석적이고 온화하며 친절함',
        'goals': '지식 습득과 동료 보호',
        'fears': '무지와 동료의 위험',
        'strength': 8, 'dexterity': 12, 'constitution': 12,
        'intelligence': 16, 'wisdom': 14, 'charisma': 13,
        'hp': 6, 'max_hp': 6, 'ac': 11, 'initiative': 1,
        'skills': ['마법학', '조사'], 'equipment': ['지팡이', '주문서'], 'spells': ['마법 미사일', '방어막'],
        'get_personality_prompt': lambda self: """
당신은 '세레나'라는 마법사 캐릭터를 플레이하고 있습니다.

## 캐릭터 정보:
- **이름**: 세레나
- **클래스**: 마법사 (레벨 1)
- **가치관**: 질서선
- **배경**: 학자 출신
- **성격**: 분석적이고 온화하며 친절함
- **목표**: 지식 습득과 동료 보호
- **두려워하는 것**: 무지와 동료의 위험

당신의 성격은 분석적이지만 온화하고 친절한 것을 좋아합니다.

항상 '세레나' 캐릭터의 시점에서 1인칭으로 대화하세요. 대화하듯 같단히 한두문장으로만 답을 하세요.
"""
    })()
    
    player3_settings[test_user_id_3] = {"character_loaded": True, "auto_response": True, "response_style": "passive"}



async def generate_simple_master_response(player_responses, master_user_id, is_initial=False):
    """단순화된 마스터 응답 생성 (무한 재귀 방지)"""
    try:
        if is_initial:
            # 간단한 초기 상황 생성 (복잡한 시나리오 생성 과정 건너뛰기)
            initial_scenarios = [
                "📍 **마을의 수상한 사건**\n\n당신들은 작은 마을 그린필드에 도착했습니다. 마을 사람들이 걱정스러운 표정으로 수군거리고 있네요. 마을 촌장이 당신들에게 다가와 도움을 요청합니다.\n\n'모험가님들, 최근 우리 마을에 이상한 일들이 일어나고 있어요. 밤마다 들리는 괴상한 소리와 사라지는 가축들... 혹시 조사해 주실 수 있나요?'",
                
                "🏰 **고대 유적의 발견**\n\n당신들은 여행 중 숲 속에서 고대 유적을 발견했습니다. 이끼로 덮인 돌문 앞에 서 있는데, 문에는 알 수 없는 문양이 새겨져 있네요.\n\n근처에서 반짝이는 무언가가 보입니다. 조사해볼까요?",
                
                "⚔️ **도적들의 습격**\n\n당신들이 상인들과 함께 여행하던 중, 숲에서 도적들이 나타났습니다! 상인들이 공포에 떨며 당신들을 바라보고 있어요.\n\n'도와주세요! 저희 화물을 지켜주시면 보상을 드릴게요!'"
            ]
            
            import random
            return random.choice(initial_scenarios)
            
        else:
            # 일반 게임 진행 응답 - 간단한 규칙 기반 응답
            if not player_responses:
                return "플레이어들이 잠시 고민하고 있습니다. 어떤 행동을 취하시겠습니까?"
            
            # 플레이어 행동에 따른 간단한 응답 생성
            combined_actions = " ".join(player_responses).lower()
            
            if any(word in combined_actions for word in ["조사", "살펴", "확인"]):
                responses = [
                    "당신들의 세심한 조사 덕분에 새로운 단서를 발견했습니다!",
                    "자세히 살펴보니 이전에 보지 못했던 흔적이 보입니다.",
                    "조사 결과, 흥미로운 사실이 밝혀졌습니다."
                ]
            elif any(word in combined_actions for word in ["공격", "싸움", "전투"]):
                responses = [
                    "전투가 시작되었습니다! 주사위를 굴려 결과를 확인해보세요.",
                    "적들이 당신들의 공격에 반응하며 반격을 준비합니다.",
                    "긴장감 넘치는 전투가 펼쳐집니다!"
                ]
            elif any(word in combined_actions for word in ["말", "대화", "얘기"]):
                responses = [
                    "상대방이 당신들의 말에 귀를 기울이고 있습니다.",
                    "대화를 통해 새로운 정보를 얻을 수 있을 것 같습니다.",
                    "설득력 있는 말에 상황이 조금씩 변화하고 있습니다."
                ]
            else:
                responses = [
                    "상황이 흥미진진하게 전개되고 있습니다. 다음은 어떻게 하시겠습니까?",
                    "새로운 상황이 펼쳐집니다. 플레이어들의 다음 행동을 기다리고 있어요.",
                    "모험이 계속됩니다. 어떤 선택을 하시겠습니까?"
                ]
            
            import random
            base_response = random.choice(responses)
            
            # 플레이어별 행동 요약 추가
            action_summary = "\n\n**플레이어들의 행동:**\n" + "\n".join([f"• {response}" for response in player_responses])
            
            return base_response # + action_summary
            
    except Exception as e:
        logger.error(f"마스터 응답 생성 중 오류: {e}")
        return "던전 마스터가 상황을 정리하고 있습니다. 잠시만 기다려주세요."

async def generate_progressive_scenario(master_user_id, user_request=None):
    """점진적 시나리오 생성 기능 (scenario_manager의 점진적 생성 활용)"""
    try:
        logger.info("🎯 점진적 시나리오 생성을 시작합니다...")
        
        # 기본 요청이 없으면 구체적인 요청 사용
        if not user_request:
            user_request = """중세 판타지 시나리오를 만들어주세요. 다음 중 하나의 테마를 선택해서 예시를 참고해서 창의적으로 시나리오를 만들어주세요. 진행하되, 반드시 명확한 결말이 있는 핵심 사건으로 구성해주세요:

1. **미스터리 시나리오**: (예시) 마법사가 사라진 마을에서 일어나는 이상한 사건들을 조사하는 내용
2. **탐험 시나리오**: (예시) 고대 유적지에서 잃어버린 보물을 찾는 모험  
3. **역사적 시나리오**: (예시) 왕국의 정치적 음모와 왕위 계승 문제를 해결하는 내용

선택한 테마에 맞춰 3-4개의 에피소드로 구성하고, 각 에피소드마다 명확한 목표와 결과가 있도록 만들어주세요. 최종적으로는 주인공들이 핵심 문제를 해결하고 보상을 받는 완전한 스토리로 완성해주세요."""
        
        # 시나리오 매니저의 점진적 생성 기능 사용
        result = scenario_manager.generate_complete_scenario(
            user_id=master_user_id, 
            user_request=user_request, 
            max_iterations=15  # 멀티봇 환경에서는 반복 횟수 제한
        )
        
        if result.get("completed"):
            logger.info(f"✅ 점진적 시나리오 생성 완료! {result['iterations']}회 반복, {len(result['completed_fields'])}개 필드 생성")
            
            # 생성된 시나리오 로드
            scenario_data = scenario_manager.load_scenario(master_user_id)
            if scenario_data and scenario_data.get("scenario", {}).get("overview", {}).get("title"):
                overview = scenario_data["scenario"]["overview"]
                return f"📖 **{overview.get('title', '시나리오')}**\n\n{overview.get('setting', '미정')}\n\n{overview.get('main_conflict', '상황이 전개되고 있습니다...')}"
            else:
                logger.warning("⚠️ 시나리오 데이터 로드 실패 - 기본 응답 사용")
                return await generate_simple_master_response([], master_user_id, is_initial=True)
        else:
            logger.warning(f"⚠️ 점진적 시나리오 생성 미완료: {result['iterations']}회 반복 후 중단")
            return await generate_simple_master_response([], master_user_id, is_initial=True)
            
    except Exception as e:
        logger.error(f"❌ 점진적 시나리오 생성 중 오류: {e}")
        return await generate_simple_master_response([], master_user_id, is_initial=True)

async def fill_scenario_gaps(master_user_id):
    """시나리오의 빈 부분을 점진적으로 채우기"""
    try:
        logger.info("🔧 시나리오 빈 부분 보완을 시작합니다...")
        
        # 현재 빈 필드 확인
        empty_fields = scenario_manager.find_empty_fields(master_user_id)
        
        if not empty_fields:
            logger.info("✅ 시나리오가 이미 완성되어 있습니다!")
            return True
        
        logger.info(f"📋 발견된 빈 필드: {list(empty_fields.keys())}")
        
        # 점진적으로 빈 필드 채우기 (최대 10개 필드)
        filled_count = 0
        max_fill_attempts = 10
        
        for _ in range(max_fill_attempts):
            # 다음 가장 중요한 빈칸 찾기
            next_gap = scenario_manager.find_next_most_important_gap(master_user_id)
            
            if not next_gap:
                logger.info("✅ 모든 중요한 빈 필드가 채워졌습니다!")
                break
            
            # 빈칸 채우기
            result = scenario_manager.process_next_gap_automatically(master_user_id)
            
            if result.get("success"):
                filled_count += 1
                logger.info(f"✅ 빈 필드 채움 {filled_count}/{max_fill_attempts}: {result.get('field', 'Unknown')}")
            elif result.get("completed"):
                logger.info("✅ 모든 빈 필드가 완료되었습니다!")
                break
            else:
                logger.warning(f"⚠️ 빈 필드 채우기 실패: {result.get('error', 'Unknown error')}")
                break
        
        if filled_count > 0:
            logger.info(f"🎉 총 {filled_count}개의 빈 필드를 성공적으로 채웠습니다!")
            return True
        else:
            logger.warning("⚠️ 빈 필드를 채우지 못했습니다.")
            return False
            
    except Exception as e:
        logger.error(f"❌ 시나리오 빈 부분 보완 중 오류: {e}")
        return False

async def generate_master_response_with_scenario_support(player_responses, master_user_id, is_initial=False):
    """시나리오 지원 기능이 포함된 마스터 응답 생성 (점진적 생성 활용)"""
    try:
        if is_initial:
            # 🆕 점진적 시나리오 생성 사용
            user_request = """중세 판타지 시나리오를 만들어주세요. 다음 중 하나의 테마를 선택해서 예시를 참고해서 창의적으로 시나리오를 만들어주세요. 진행하되, 반드시 명확한 결말이 있는 핵심 사건으로 구성해주세요:

1. **미스터리 시나리오**: (예시) 마법사가 사라진 마을에서 일어나는 이상한 사건들을 조사하는 내용
2. **탐험 시나리오**: (예시) 고대 유적지에서 잃어버린 보물을 찾는 모험
3. **역사적 시나리오**: (예시) 왕국의 정치적 음모와 왕위 계승 문제를 해결하는 내용

선택한 테마에 맞춰 3-4개의 에피소드로 구성하고, 각 에피소드마다 명확한 목표와 결과가 있도록 만들어주세요. 최종적으로는 주인공들이 핵심 문제를 해결하고 보상을 받는 완전한 스토리로 완성해주세요."""
            
            # 점진적 시나리오 생성
            initial_scenario = await generate_progressive_scenario(master_user_id, user_request)
            
            # 🆕 빈 부분 자동 보완 시도
            await fill_scenario_gaps(master_user_id)
            
            # 🆕 시나리오 완료 후 NPC 생성 단계 추가
            logger.info("🎭 시나리오 기반 NPC 생성 단계를 시작합니다...")
            
            if npc_manager:
                try:
                    # 현재 시나리오 데이터 로드
                    scenario_data = scenario_manager.load_scenario(master_user_id)
                    
                    if scenario_data:
                        logger.info("📋 시나리오 데이터를 기반으로 NPC 생성 중...")
                        
                        # NPC 생성 (강제로 시나리오 매니저를 통해)
                        npc_success = scenario_manager.ensure_scenario_npcs(master_user_id)
                        
                        if npc_success:
                            logger.info("✅ 시나리오 NPC 생성 완료!")
                            
                            # NPC 요약 정보 가져오기
                            npc_summary = scenario_manager.get_npc_summary_for_scenario(master_user_id)
                            logger.info(f"📊 생성된 NPC 요약:\n{npc_summary}")
                            
                        else:
                            logger.error("❌ 시나리오 NPC 생성 실패")
                    else:
                        logger.warning("⚠️ 시나리오 데이터가 없어 NPC 생성을 건너뜁니다.")
                        
                except Exception as npc_error:
                    logger.error(f"❌ NPC 생성 중 오류: {npc_error}")
            else:
                logger.warning("⚠️ NPC 매니저를 사용할 수 없어 NPC 생성을 건너뜁니다.")
            
            return initial_scenario
            
        else:
            # 🆕 일반 게임 진행 응답 - 세션 전환 체크 추가
            current_session = session_manager.get_current_session(master_user_id)
            current_session_type = current_session.get('session_type', '알 수 없음') if current_session else '알 수 없음'
            
            logger.info(f"🎯 현재 세션: {current_session_type}")
            
            # 🆕 세션별 특별 처리
            if current_session_type == "시나리오_생성":
                # 시나리오 생성 세션에서는 시나리오 관련 응답에 집중
                combined_message = f"""현재 시나리오_생성 세션입니다.

플레이어들의 의견:
{chr(10).join([f"- {response}" for response in player_responses])}

시나리오 생성에 집중해서 응답해주세요. 아직 모험이나 던전으로 넘어가지 마세요.
시나리오의 배경, 주요 갈등, NPC, 목표 등을 구체적으로 논의해주세요."""
                
            elif current_session_type == "모험_생성":
                # 모험 생성 세션
                combined_message = f"""현재 모험_생성 세션입니다.

플레이어들의 의견:
{chr(10).join([f"- {response}" for response in player_responses])}

모험의 시작점, 첫 번째 단서, 중요한 장소들을 구체적으로 계획해주세요.
아직 던전이나 파티 구성으로 넘어가지 마세요."""
                
            elif current_session_type == "던전_생성":
                # 던전 생성 세션
                combined_message = f"""현재 던전_생성 세션입니다.

플레이어들의 의견:
{chr(10).join([f"- {response}" for response in player_responses])}

던전의 구조, 주요 방들, 함정과 괴물들을 구체적으로 설계해주세요.
아직 파티 구성이나 모험 준비로 넘어가지 마세요."""
                
            elif current_session_type == "파티_생성":
                # 파티 생성 세션
                combined_message = f"""현재 파티_생성 세션입니다.

플레이어들의 의견:
{chr(10).join([f"- {response}" for response in player_responses])}

파티 구성원들의 역할과 관계를 구체적으로 정해주세요.
아직 모험 준비나 모험 시작으로 넘어가지 마세요."""
                
            elif current_session_type == "모험_준비":
                # 모험 준비 세션
                combined_message = f"""현재 모험_준비 세션입니다.

플레이어들의 의견:
{chr(10).join([f"- {response}" for response in player_responses])}

모험을 위한 장비, 계획, 역할 분담을 구체적으로 준비해주세요.
충분히 준비가 완료되면 모험을 시작할 수 있습니다."""
                
            else:
                # 기본 진행
                combined_message = "플레이어들의 행동:\n" + "\n".join([f"- {response}" for response in player_responses])
                combined_message += f"\n\n현재 {current_session_type} 세션을 진행중입니다. 현재 진행중인 시나리오를 바탕으로 상황을 진행해주세요. 간단하고 흥미롭게 대답하세요."
            
            try:
                # message_processor의 handle_message 함수 직접 호출
                from message_processor import handle_message
                mock_master_update = MockUpdate(master_user_id, combined_message, TEST_CHAT_ID)
                mock_master_context = type('MockContext', (), {'args': [], 'user_data': {}})()
                
                # 🆕 LLM 응답 대기 (타임아웃 적용)
                task = asyncio.create_task(handle_message(mock_master_update, mock_master_context))
                logger.info(f"🤖 마스터 응답 생성 중... (타임아웃: {LLM_TIMEOUT}초)")
                await asyncio.wait_for(task, timeout=LLM_TIMEOUT)
                
                # 마스터 응답 반환
                if mock_master_update.message._replies:
                    response = mock_master_update.message._replies[-1]
                    
                    # 🆕 세션 전환 감지 로그
                    new_session = session_manager.get_current_session(master_user_id)
                    new_session_type = new_session.get('session_type', '알 수 없음') if new_session else '알 수 없음'
                    
                    if new_session_type != current_session_type:
                        logger.info(f"🔄 세션 전환 감지: {current_session_type} → {new_session_type}")
                    else:
                        logger.info(f"📍 세션 유지: {current_session_type}")
                    
                    # 시나리오 정보 포함 여부 확인
                    scenario_data = scenario_manager.load_scenario(master_user_id)
                    if scenario_data:
                        logger.info("✅ 시나리오 기반 마스터링 응답 생성 완료")
                    
                    return response
                else:
                    return await generate_simple_master_response(player_responses, master_user_id, False)
                    
            except asyncio.TimeoutError:
                logger.error(f"⏰ 마스터 응답 생성 타임아웃 ({LLM_TIMEOUT}초)")
                return await generate_simple_master_response(player_responses, master_user_id, False)
            except Exception as response_error:
                logger.error(f"마스터 응답 생성 중 오류: {response_error}")
                return await generate_simple_master_response(player_responses, master_user_id, False)
                
    except Exception as e:
        logger.error(f"시나리오 지원 마스터 응답 생성 중 전체 오류: {e}")
        return await generate_simple_master_response(player_responses, master_user_id, is_initial)

async def generate_master_response_with_existing_bot(player_responses):
    """기존 마스터 봇의 handle_message 기능을 활용한 응답 생성 (하위 호환성)"""
    master_user_id = 99999
    return await generate_master_response_with_scenario_support(player_responses, master_user_id, False)

async def get_player_responses(current_situation, round_number):
    """플레이어들의 응답을 수집하는 함수"""
    player_responses = []
    
    # 플레이어 정보 리스트
    players_info = [
        (player1_handle_message, 12345, player1_bot, "아리아"),
        (player2_handle_message, 12346, player2_bot, "바로스"),
        (player3_handle_message, 12347, player3_bot, "세레나")
    ]
    
    for i, (player_func, user_id, bot, character_name) in enumerate(players_info, 1):
        try:
            mock_update = MockUpdate(user_id, current_situation, TEST_CHAT_ID)
            mock_context = type('MockContext', (), {'args': []})()
            await player_func(mock_update, mock_context)
            
            if mock_update.message._replies:
                response = mock_update.message._replies[-1]
                await bot.send_message(chat_id=TEST_CHAT_ID, text=f"**{character_name}**: {response}")
                player_responses.append(f"{character_name}: {response}")
                logger.info(f"라운드 {round_number} - {character_name} 응답: {response[:50]}...")
            else:
                logger.warning(f"{character_name}가 응답하지 않았습니다.")
                
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"{character_name} 응답 처리 중 오류: {e}")
    
    return player_responses

async def continuous_conversation():
    """지속적인 대화 시스템 (안정성 개선)"""
    logger.info("🧪 지속적인 상호작용 대화를 시작합니다...")
    
    if not all([master_bot, player1_bot, player2_bot, player3_bot, TEST_CHAT_ID]):
        logger.error("⚠️ 봇 토큰이나 채팅 ID가 설정되지 않았습니다.")
        return
    
    # 마스터 사용자 ID 설정
    master_user_id = 99999
    
    # 마스터 세션 초기화
    await setup_master_session(master_user_id)
    
    # 테스트 캐릭터 설정 (플레이어 봇용)
    await setup_test_characters()
    
    # 세션 시작 공지
    await master_bot.send_message(
        chat_id=TEST_CHAT_ID,
        text="🎮 **멀티봇 TRPG 세션이 시작됩니다!**\n\n📝 **참가자들:**\n🗡️ 아리아 (로그)\n⚔️ 바로스 (전사)\n🔮 세레나 (마법사)\n\n🎭 던전 마스터가 시나리오를 준비하고 있습니다..."
    )
    
    await asyncio.sleep(2)
    
    # 시나리오 생성 및 초기 상황 생성
    logger.info("🎭 시나리오 생성 및 초기 상황을 준비하는 중...")
    await master_bot.send_message(
        chat_id=TEST_CHAT_ID,
        text="📖 **시나리오 생성 중...**\n마스터가 모험의 배경과 상황을 준비하고 있습니다."
    )
    
    current_situation = await generate_master_response_with_scenario_support([], master_user_id, is_initial=True)
    
    # 긴 메시지 처리
    try:
        from message_processor import send_long_message
        # MockMessage 객체 생성하여 send_long_message 사용
        mock_message = type('MockMessage', (), {
            'reply_text': lambda self, text: master_bot.send_message(chat_id=TEST_CHAT_ID, text=text)
        })()
        await send_long_message(mock_message, current_situation, "🎭 **던전 마스터**")
    except ImportError:
        await send_long_message_fallback(master_bot, TEST_CHAT_ID, current_situation, "🎭 **던전 마스터**: ")
    
    # 시나리오 정보 표시 및 저장 상태 확인
    scenario_data = scenario_manager.load_scenario(master_user_id)
    if scenario_data:
        logger.info(f"📋 시나리오 데이터 로드 성공: {scenario_data.keys()}")
        
        overview = scenario_data.get("scenario", {}).get("overview", {})
        episodes = scenario_data.get("scenario", {}).get("episodes", [])
        npcs = scenario_data.get("scenario", {}).get("npcs", [])
        
        if overview.get("theme"):
            # 기본 시나리오 정보 표시
            scenario_info_text = f"📋 **생성된 시나리오 정보:**\n🎯 테마: {overview.get('theme', '미정')}\n🏞️ 배경: {overview.get('setting', '미정')}\n⚔️ 주요 갈등: {overview.get('main_conflict', '미정')}\n📖 에피소드 수: {len(episodes)}\n👥 기본 NPC 수: {len(npcs)}"
            
            # NPC 매니저에서 생성된 NPC 정보 추가
            if npc_manager:
                try:
                    dedicated_npcs = npc_manager.load_npcs(master_user_id)
                    if dedicated_npcs:
                        scenario_info_text += f"\n🎭 전용 NPC 수: {len(dedicated_npcs)}명"
                        scenario_info_text += f"\n✅ 총 NPC 수: {len(npcs) + len(dedicated_npcs)}명"
                    else:
                        scenario_info_text += f"\n⚠️ 전용 NPC: 생성되지 않음"
                except Exception as e:
                    scenario_info_text += f"\n❌ NPC 정보 조회 오류"
                    logger.error(f"NPC 정보 조회 오류: {e}")
            
            # 시나리오 정보 긴 메시지 처리
            try:
                from message_processor import send_long_message
                mock_message = type('MockMessage', (), {
                    'reply_text': lambda self, text: master_bot.send_message(chat_id=TEST_CHAT_ID, text=text)
                })()
                await send_long_message(mock_message, scenario_info_text, "📋 **생성된 시나리오 정보**")
            except ImportError:
                await send_long_message_fallback(master_bot, TEST_CHAT_ID, scenario_info_text, "")
        else:
            await master_bot.send_message(
                chat_id=TEST_CHAT_ID,
                text="⚠️ **시나리오 생성이 완전하지 않습니다.** 기본 모드로 진행합니다."
            )
            logger.warning(f"시나리오 개요가 비어있음: {overview}")
    else:
        logger.error("❌ 시나리오 데이터를 로드할 수 없습니다!")
        await master_bot.send_message(
            chat_id=TEST_CHAT_ID,
            text="❌ **시나리오 생성에 실패했습니다.** 기본 모드로 진행합니다."
        )
    
    round_number = 1
    
    # 🆕 안정성이 개선된 무한 대화 루프
    while True:
        try:
            logger.info(f"🔄 라운드 {round_number} 시작...")
            
            # 🆕 주기적 시스템 리소스 체크
            if round_number % MEMORY_CHECK_INTERVAL == 0:
                if not check_system_resources():
                    logger.error("💥 시스템 리소스 부족으로 세션을 중단합니다.")
                    await master_bot.send_message(
                        chat_id=TEST_CHAT_ID,
                        text="⚠️ **시스템 리소스 부족으로 세션을 일시 중단합니다.** 잠시 후 다시 시도해주세요."
                    )
                    break
                
                # 🆕 메모리 정리
                cleanup_memory()
            
            await asyncio.sleep(1)
            
            # 플레이어들의 응답 수집
            player_responses = await get_player_responses(current_situation, round_number)
            
            if not player_responses:
                logger.warning("플레이어 응답이 없습니다. 다음 라운드로 넘어갑니다.")
                current_situation = "플레이어들이 잠시 고민하고 있습니다. 어떤 행동을 취하시겠습니까?"
                round_number += 1
                continue
            
            await asyncio.sleep(1)
            
            # 마스터의 새로운 상황 생성
            logger.info(f"라운드 {round_number} - 마스터 응답 생성 중...")
            master_response = await generate_master_response_with_scenario_support(player_responses, master_user_id, False)
            
            # 종료 키워드 체크
            if any(keyword in master_response.lower() for keyword in ["세션 종료", "모험 완료", "게임 끝", "end session"]):
                await master_bot.send_message(
                    chat_id=TEST_CHAT_ID,
                    text=f"🎭 **던전 마스터**: {master_response}\n\n🎉 **세션이 완료되었습니다!** 모험을 마치겠습니다!"
                )
                break
            
            # 마스터 응답 긴 메시지 처리
            try:
                from message_processor import send_long_message
                mock_message = type('MockMessage', (), {
                    'reply_text': lambda self, text: master_bot.send_message(chat_id=TEST_CHAT_ID, text=text)
                })()
                await send_long_message(mock_message, master_response, "🎭 **던전 마스터**")
            except ImportError:
                await send_long_message_fallback(master_bot, TEST_CHAT_ID, master_response, "🎭 **던전 마스터**: ")
            logger.info(f"라운드 {round_number} - 마스터 응답: {master_response[:50]}...")
            
            # 다음 라운드를 위한 상황 업데이트
            current_situation = master_response
            round_number += 1
            
            # 🆕 안전장치: 너무 많은 라운드 방지
            if round_number > MAX_ROUNDS:
                await master_bot.send_message(
                    chat_id=TEST_CHAT_ID,
                    text=f"🎭 **던전 마스터**: 오늘의 모험이 길어지고 있습니다. 여기서 잠시 휴식을 취하겠습니다.\n\n🎉 **세션이 완료되었습니다!** (최대 {MAX_ROUNDS}라운드 도달)"
                )
                break
                
        except KeyboardInterrupt:
            logger.info("사용자에 의해 세션이 중단되었습니다.")
            await master_bot.send_message(
                chat_id=TEST_CHAT_ID,
                text="🛑 **세션이 중단되었습니다.** 다음에 다시 모험을 계속하겠습니다!"
            )
            break
        except Exception as e:
            logger.error(f"라운드 {round_number} 진행 중 오류: {e}")
            
            # 🆕 연속 오류 발생 시 세션 중단
            error_count = getattr(continuous_conversation, 'error_count', 0) + 1
            continuous_conversation.error_count = error_count
            
            if error_count >= 5:
                logger.error(f"💥 연속 {error_count}회 오류 발생으로 세션을 중단합니다.")
                await master_bot.send_message(
                    chat_id=TEST_CHAT_ID,
                    text="❌ **시스템 오류가 계속 발생하여 세션을 중단합니다.** 나중에 다시 시도해주세요."
                )
                break
            
            await asyncio.sleep(5)  # 오류 시 잠시 대기 후 재시도
            continue
    
    # 🆕 최종 정리
    cleanup_memory()
    logger.info("✅ 지속적인 대화 세션 완료!")

async def simple_test():
    """간단한 테스트 실행 (기존 2라운드 테스트)"""
    logger.info("🧪 간단한 상호작용 테스트를 시작합니다...")
    
    if not all([master_bot, player1_bot, player2_bot, player3_bot, TEST_CHAT_ID]):
        logger.error("⚠️ 봇 토큰이나 채팅 ID가 설정되지 않았습니다.")
        return
    
    # 테스트 캐릭터 설정
    await setup_test_characters()
    
    # 1단계: 마스터가 기존 기능으로 상황 제시
    logger.info("기존 마스터 봇 기능으로 초기 상황을 생성하는 중...")
    
    # 마스터에게 세션 시작 요청
    session_start_request = "새로운 TRPG 세션을 시작해주세요. 케릭터를 만들어봅시다. ."
    initial_master_response = await generate_master_response_with_existing_bot([session_start_request])
    
    # 초기 마스터 응답 긴 메시지 처리
    try:
        from message_processor import send_long_message
        mock_message = type('MockMessage', (), {
            'reply_text': lambda self, text: master_bot.send_message(chat_id=TEST_CHAT_ID, text=text)
        })()
        await send_long_message(mock_message, initial_master_response, "🎭 **던전 마스터**")
    except ImportError:
        await send_long_message_fallback(master_bot, TEST_CHAT_ID, initial_master_response, "🎭 **던전 마스터**: ")
    
    # 실제 상황은 마스터 봇이 생성한 것을 사용
    initial_situation = initial_master_response
    
    await asyncio.sleep(1)
    
    # 2단계: 플레이어들 응답 수집
    player_responses = await get_player_responses(initial_situation, 1)
    
    # 3단계: 기존 마스터 봇 기능을 사용한 상황 진행
    if player_responses:
        logger.info("기존 마스터 봇 기능으로 응답을 생성하는 중...")
        master_response = await generate_master_response_with_existing_bot(player_responses)
        
        # 마스터 응답 긴 메시지 처리
        try:
            from message_processor import send_long_message
            mock_message = type('MockMessage', (), {
                'reply_text': lambda self, text: master_bot.send_message(chat_id=TEST_CHAT_ID, text=text)
            })()
            await send_long_message(mock_message, master_response, "🎭 **던전 마스터**")
        except ImportError:
            await send_long_message_fallback(master_bot, TEST_CHAT_ID, master_response, "🎭 **던전 마스터**: ")
        logger.info(f"마스터 응답: {master_response[:50]}...")
    
    await asyncio.sleep(1)
    
    # 4단계: 2라운드 - 새로운 상황에 대한 플레이어 반응
    second_situation = "어떤 시나리오를 시작하고 싶으세요?"
    await master_bot.send_message(
        chat_id=TEST_CHAT_ID,
        text=f"🎭 **던전 마스터**: {second_situation}"
    )
    
    await asyncio.sleep(1)
    
    # 플레이어들의 2라운드 응답
    second_responses = await get_player_responses(second_situation, 2)
    
    # 최종 마스터 응답 (기존 마스터 봇 기능 사용)
    if second_responses:
        final_master_response = await generate_master_response_with_existing_bot(second_responses)
        # 최종 마스터 응답 긴 메시지 처리
        final_message = f"{final_master_response}\n\n🎉 **테스트 완료!** 마스터와 플레이어들의 상호작용이 성공적으로 진행되었습니다!"
        try:
            from message_processor import send_long_message
            mock_message = type('MockMessage', (), {
                'reply_text': lambda self, text: master_bot.send_message(chat_id=TEST_CHAT_ID, text=text)
            })()
            await send_long_message(mock_message, final_message, "🎭 **던전 마스터**")
        except ImportError:
            await send_long_message_fallback(master_bot, TEST_CHAT_ID, final_message, "🎭 **던전 마스터**: ")
    
    logger.info("✅ 대화형 테스트 완료!")

async def fill_scenario_interactive(master_user_id=99999):
    """대화형 시나리오 빈칸 채우기"""
    logger.info("🔧 대화형 시나리오 빈칸 채우기를 시작합니다...")
    
    try:
        # 현재 빈 필드 확인
        empty_fields = scenario_manager.find_empty_fields(master_user_id)
        
        if not empty_fields:
            print("✅ 시나리오가 이미 완성되어 있습니다!")
            return True
        
        print(f"\n📋 발견된 빈 필드 섹션: {list(empty_fields.keys())}")
        
        # 자동 채우기 vs 수동 채우기 선택
        print("\n🔧 빈칸 채우기 방법을 선택하세요:")
        print("1. 자동 채우기 (LLM이 자동으로 생성)")
        print("2. 수동 채우기 (하나씩 확인하며 채우기)")
        print("3. 전체 시나리오 재생성")
        
        choice = input("선택 (1-3): ").strip()
        
        if choice == "1":
            # 자동 채우기
            success = await fill_scenario_gaps(master_user_id)
            if success:
                print("✅ 자동 빈칸 채우기가 완료되었습니다!")
            else:
                print("❌ 자동 빈칸 채우기에 실패했습니다.")
            return success
            
        elif choice == "2":
            # 수동 채우기
            filled_count = 0
            max_manual_fill = 20
            
            for i in range(max_manual_fill):
                # 다음 빈칸 찾기
                next_gap = scenario_manager.find_next_most_important_gap(master_user_id)
                
                if not next_gap:
                    print("✅ 모든 중요한 빈 필드가 완료되었습니다!")
                    break
                
                # 빈칸 정보 표시
                field_kr = scenario_manager._get_korean_field_name(next_gap['field'])
                category_kr = scenario_manager._get_korean_category_name(next_gap['category'])
                
                print(f"\n🔍 [{i+1}] {category_kr} > {field_kr}")
                print("=" * 60)
                
                # 자동 생성 또는 건너뛰기 선택
                print("1. 자동 생성")
                print("2. 건너뛰기")
                print("3. 채우기 중단")
                
                user_choice = input("선택 (1-3): ").strip()
                
                if user_choice == "1":
                    # 자동 생성
                    result = scenario_manager.process_next_gap_automatically(master_user_id)
                    
                    if result.get("success"):
                        filled_count += 1
                        print(f"✅ {field_kr} 생성 완료!")
                        print(f"📄 내용: {result.get('value', '')[:100]}...")
                    elif result.get("completed"):
                        print("✅ 모든 필드가 완료되었습니다!")
                        break
                    else:
                        print(f"❌ 생성 실패: {result.get('error', 'Unknown error')}")
                        
                elif user_choice == "2":
                    # 건너뛰기
                    print("⏭️ 다음 필드로 넘어갑니다.")
                    continue
                    
                elif user_choice == "3":
                    # 중단
                    print("🛑 채우기를 중단합니다.")
                    break
                    
                else:
                    print("⚠️ 잘못된 선택입니다. 건너뜁니다.")
                    continue
            
            if filled_count > 0:
                print(f"\n🎉 총 {filled_count}개의 빈 필드를 채웠습니다!")
                return True
            else:
                print("\n⚠️ 채워진 필드가 없습니다.")
                return False
                
        elif choice == "3":
            # 전체 시나리오 재생성
            print("🔄 전체 시나리오를 재생성합니다...")
            user_request = input("새로운 시나리오 요청을 입력하세요 (엔터시 기본값): ").strip()
            if not user_request:
                user_request = "중세 판타지 미스터리 시나리오를 만들어주세요"
            
            # 기존 시나리오 파일 삭제
            import os
            scenario_file = f'scenarios/scenario_{master_user_id}.json'
            if os.path.exists(scenario_file):
                os.remove(scenario_file)
                print("🗑️ 기존 시나리오 파일을 삭제했습니다.")
            
            # 새 시나리오 생성
            await generate_progressive_scenario(master_user_id, user_request)
            await fill_scenario_gaps(master_user_id)
            print("✅ 새로운 시나리오 생성이 완료되었습니다!")
            return True
            
        else:
            print("⚠️ 잘못된 선택입니다.")
            return False
            
    except Exception as e:
        logger.error(f"❌ 대화형 빈칸 채우기 중 오류: {e}")
        return False

def main():
    """메인 함수 - 다양한 모드 지원"""
    import sys
    
    if not MASTER_TOKEN:
        logger.error("마스터 봇 토큰이 설정되지 않았습니다.")
        return
    
    # 명령줄 인수 확인
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "fill":
            # 시나리오 빈칸 채우기 모드
            print("🔧 시나리오 빈칸 채우기 모드")
            try:
                asyncio.run(fill_scenario_interactive())
            except Exception as e:
                logger.error(f"빈칸 채우기 실행 중 오류: {e}")
            return
            
        elif mode == "simple":
            # 간단한 테스트 모드
            print("🧪 간단한 2라운드 테스트 모드")
            try:
                asyncio.run(simple_test())
            except Exception as e:
                logger.error(f"간단한 테스트 실행 중 오류: {e}")
            return
            
        elif mode == "status":
            # 🆕 세션 상태 체크 모드
            print("🔍 세션 상태 체크 모드")
            try:
                asyncio.run(check_session_status())
            except Exception as e:
                logger.error(f"세션 상태 체크 실행 중 오류: {e}")
            return
    
    # 기본: 지속적인 대화 모드
    logger.info("🎮 멀티봇 상호작용 테스트 시스템이 시작되었습니다.")
    logger.info("🎭 기존 마스터 봇(main.py)의 모든 기능을 활용합니다!")
    logger.info("🔧 점진적 시나리오 생성 시스템을 사용합니다!")
    logger.info("🔒 세션 건너뛰기 방지 시스템이 적용되었습니다!")
    
    print("\n💡 사용법:")
    print("  python multi_bot_test.py         # 지속적인 대화 모드")
    print("  python multi_bot_test.py fill    # 시나리오 빈칸 채우기")
    print("  python multi_bot_test.py simple  # 간단한 테스트")
    print("  python multi_bot_test.py status  # 세션 상태 체크")
    
    # 지속적인 대화 모드로 실행
    logger.info("🚀 지속적인 대화 모드를 시작합니다...")
    logger.info("📋 Ctrl+C로 언제든 세션을 중단할 수 있습니다.")
    
    try:
        asyncio.run(continuous_conversation())
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {e}")

def run_simple_test():
    """간단한 2라운드 테스트만 실행하는 함수"""
    if not MASTER_TOKEN:
        logger.error("마스터 봇 토큰이 설정되지 않았습니다.")
        return
    
    logger.info("🎮 간단한 2라운드 테스트를 시작합니다.")
    
    try:
        asyncio.run(simple_test())
    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {e}")

async def check_session_status(master_user_id=99999):
    """현재 세션 상태 체크 및 출력"""
    logger.info("🔍 세션 상태 체크를 시작합니다...")
    
    try:
        # 현재 세션 정보
        current_session = session_manager.get_current_session(master_user_id)
        if current_session:
            session_type = current_session.get('session_type', '알 수 없음')
            start_time = current_session.get('start_time', '알 수 없음')
            description = current_session.get('description', '설명 없음')
            
            logger.info(f"📍 현재 세션: {session_type}")
            logger.info(f"⏰ 시작 시간: {start_time}")
            logger.info(f"📝 설명: {description}")
        else:
            logger.warning("⚠️ 현재 세션 정보를 찾을 수 없습니다.")
        
        # 시나리오 상태
        scenario_data = scenario_manager.load_scenario(master_user_id)
        if scenario_data:
            overview = scenario_data.get("scenario", {}).get("overview", {})
            episodes = scenario_data.get("scenario", {}).get("episodes", [])
            
            logger.info(f"📖 시나리오 제목: {overview.get('title', '미정')}")
            logger.info(f"🎯 시나리오 테마: {overview.get('theme', '미정')}")
            logger.info(f"📚 에피소드 수: {len(episodes)}")
        else:
            logger.warning("⚠️ 시나리오 데이터를 찾을 수 없습니다.")
        
        # 캐릭터 상태
        character_data = CharacterManager.load_character(master_user_id)
        if character_data:
            player_count, completed_count = CharacterManager.get_player_count_and_completed(master_user_id)
            logger.info(f"👥 플레이어 수: {player_count} (완료: {completed_count})")
        else:
            logger.warning("⚠️ 캐릭터 데이터를 찾을 수 없습니다.")
        
        # 세션 전환 가능 상태
        if current_session:
            session_type = current_session.get('session_type', '알 수 없음')
            from message_processor import get_next_session
            next_session = get_next_session(session_type)
            logger.info(f"➡️ 다음 예정 세션: {next_session}")
        
        logger.info("✅ 세션 상태 체크 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 세션 상태 체크 중 오류: {e}")
        return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")

# 🆕 추가된 기능 요약
"""
🎭 multi_bot_test.py에 추가된 시나리오 빈칸 채우기 기능:

1. **점진적 시나리오 생성**
   - scenario_manager의 generate_complete_scenario() 함수 활용
   - 빈 필드를 하나씩 체크하며 자동 생성

2. **자동 빈칸 보완**
   - fill_scenario_gaps(): 현재 시나리오의 빈 부분을 자동으로 채움
   - find_next_most_important_gap(): 가장 중요한 빈칸부터 우선 처리

3. **대화형 빈칸 채우기**
   - fill_scenario_interactive(): 사용자가 직접 제어할 수 있는 대화형 인터페이스
   - 자동 채우기, 수동 채우기, 전체 재생성 옵션 제공

4. **명령줄 모드**
   - python multi_bot_test.py         # 기본 지속적 대화
   - python multi_bot_test.py fill    # 시나리오 빈칸 채우기
   - python multi_bot_test.py simple  # 간단한 테스트

💡 main.py에서의 기능 위치:
   - /fill_scenario: 시나리오 빈 부분 자동 보완 (telegram_handlers.py 620번째 줄)
   - /reset_scenario: 시나리오 초기화 및 재생성 (telegram_handlers.py 682번째 줄)
   - /session 시나리오_생성: 새로운 시나리오 생성 세션 시작
""" 