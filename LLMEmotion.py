# coding: utf-8
import json
import copy
import logging
import re
import asyncio
import os
import time
import uuid
import math
import importlib
from glob import glob
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Union,
    Callable,
    Awaitable,
    List,
    TypedDict,
    cast,
)
from datetime import datetime, timezone, timedelta, tzinfo

# タイムゾーン対応ライブラリをインポート
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    ZoneInfo = None
    ZoneInfoNotFoundError = None

# aiohttpをインポート
try:
    import aiohttp
except ImportError:
    aiohttp = None
    defaultlogger = logging.getLogger(__name__)
    defaultlogger.error(
        "aiohttp is not installed. Please install it by 'pip install aiohttp'"
    )

# demjson3は使用禁止（依存排除）

# Sentence-Transformers をオプション導入（未導入時はNoneにフォールバック）
try:
    _st_mod = importlib.import_module("sentence_transformers")
    SentenceTransformer = getattr(_st_mod, "SentenceTransformer", None)
    CrossEncoder = getattr(_st_mod, "CrossEncoder", None)
except Exception:
    SentenceTransformer = None
    CrossEncoder = None

# Open WebUIのモデル情報は任意依存のため動的読み込み（未導入時はフォールバック）
from typing import Any as _AnyTypeAlias  # local alias to hint Any

UserModel: _AnyTypeAlias = None
Models: _AnyTypeAlias = None
try:
    _m_users = importlib.import_module("open_webui.models.users")
    UserModel = cast(Any, getattr(_m_users, "UserModel", None))
    _m_models = importlib.import_module("open_webui.models.models")
    Models = cast(Any, getattr(_m_models, "Models", None))
except Exception:
    UserModel = cast(Any, object)
    Models = cast(Any, object)
from pydantic import BaseModel, Field

# Logger setup
defaultlogger = logging.getLogger(__name__)
if not defaultlogger.handlers:
    defaultlogger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    defaultlogger.addHandler(handler)


# ========================================================================
# カスタム例外クラス
# ========================================================================


class LLMEmotionError(Exception):
    """LLMEmotionフィルターの基底例外クラス"""

    pass


class StateValidationError(LLMEmotionError):
    """状態バリデーション時のエラー"""

    pass


class StatePersistenceError(LLMEmotionError):
    """状態の永続化/読み込み時のエラー"""

    pass


class StateUpdateError(LLMEmotionError):
    """状態更新処理時のエラー"""

    pass


class InventoryProcessingError(LLMEmotionError):
    """インベントリ処理時のエラー"""

    pass


class LLMAPIError(LLMEmotionError):
    """外部LLM API呼び出し時のエラー"""

    pass


class JSONParsingError(LLMEmotionError):
    """JSON解析/修復時のエラー"""

    pass


class ConfigurationError(LLMEmotionError):
    """設定（Valves）関連のエラー"""

    pass


# ========================================================================
# メインフィルタークラス
# ========================================================================


class Filter:
    """
    LLMの内部状態をすべて構造化データ(JSON)として管理し、
    単一のメモリに永続化する統合フィルター (通知順序改善・多言語対応・リファクタリング版)

    主な機能:
    - キャラクターの感情、記憶、身体状態などをJSONとして永続化。
    - 対話内容を分析し、別のLLMを用いて内部状態を非同期で更新。
    - 内部状態をシステムプロンプトに挿入し、キャラクターの応答に一貫性を持たせる。
    - 状態の検証ロジックをStateValidatorクラスに集約し、保守性を向上。
    - 限界イベントの処理をAIの自律的判断に委ね、より自然な物語生成を促進。
    """

    # ========================================================================
    # 定数定義
    # ========================================================================

    logger = defaultlogger

    # Numeric precision
    NUMERIC_PRECISION = 3

    # Inventory modes
    INVENTORY_MODE_DIRECT = "direct"
    INVENTORY_MODE_INFERENCE = "inference"

    # Persistence backend
    PERSISTENCE_BACKEND_FILE = "file"

    # Response formats
    RESPONSE_FORMAT_NONE = "none"
    RESPONSE_FORMAT_OPENAI_JSON = "openai-json"
    RESPONSE_FORMAT_OLLAMA_JSON = "ollama-json"
    RESPONSE_FORMAT_AUTO = "auto"

    # State types
    PERSISTENT_STATE_TYPE = "llm_persistent_state"

    # UI event types
    EVENT_TYPE_STATUS = "status"
    EVENT_TYPE_WARNING = "warning"
    EVENT_TYPE_ERROR = "error"

    # UI event keys
    EVENT_KEY_TYPE = "type"
    EVENT_KEY_DATA = "data"
    EVENT_KEY_DESCRIPTION = "description"
    EVENT_KEY_DONE = "done"

    # Inventory trim strategies
    INV_TRIM_UNEQUIPPED_FIRST = "unequipped_first"
    INV_TRIM_QUANTITY_ASC = "quantity_asc"

    # Log tags
    LOG_INLET = "[[INLET]]"
    LOG_OUTLET = "[[OUTLET]]"
    LOG_STATE = "[[STATE]]"
    LOG_IDLE = "[[IDLE_REF]]"

    # File lock settings
    LOCK_TIMEOUT_SECONDS = 5.0
    LOCK_FILE_SUFFIX = ".lock"

    # Snapshot settings
    DEFAULT_SNAPSHOT_RETENTION = 10

    # Timeout settings
    DEFAULT_HTTP_TIMEOUT_SECONDS = 120.0
    DEFAULT_RETRY_MAX_ATTEMPTS = 3
    DEFAULT_RETRY_BASE_DELAY = 1.0

    # State key constants
    # トップレベル状態キー
    STATE_KEY_EMOTION = "emotion"
    STATE_KEY_MEMORY = "memory"
    STATE_KEY_KNOWLEDGE = "knowledge"
    STATE_KEY_GOAL = "goal"
    STATE_KEY_INVENTORY = "inventory"
    STATE_KEY_PHYSICAL_HEALTH = "physical_health"
    STATE_KEY_MENTAL_HEALTH = "mental_health"
    STATE_KEY_RELATIONSHIP = "relationship"
    STATE_KEY_TONE = "tone"
    STATE_KEY_CONTEXT = "context"
    STATE_KEY_INTERNAL_MONOLOGUE = "internal_monologue"
    STATE_KEY_TIMESTAMPS = "timestamps"
    
    # メモリサブキー
    MEMORY_KEY_RECENT = "recent"
    MEMORY_KEY_IMPRESSION = "impression"
    
    # ゴールサブキー
    GOAL_KEY_LONG_TERM = "long_term"
    GOAL_KEY_MID_TERM = "mid_term"
    GOAL_KEY_SHORT_TERM = "short_term"
    GOAL_KEY_ROUTINE = "routine"
    
    # relationshipサブキー
    RELATIONSHIP_KEY_USER_ADDRESS = "user_address"
    RELATIONSHIP_KEY_DEFAULT = "default"
    
    # contextサブキー
    CONTEXT_KEY_PLACE = "place"
    
    # internal_monologueサブキー
    MONOLOGUE_KEY_OPTIONS = "options"
    MONOLOGUE_KEY_COGNITIVE_FOCUS = "cognitive_focus"
    
    # timestampsサブキー
    TIMESTAMP_KEY_CONDITION_SINCE = "condition_since"
    
    # 汎用フィールドキー
    FIELD_KEY_PRIORITY = "priority"
    FIELD_KEY_PROGRESS = "progress"
    FIELD_KEY_START_TIME = "start_time"
    FIELD_KEY_END_TIME = "end_time"
    FIELD_KEY_CONTENT = "content"
    FIELD_KEY_CONDITION = "condition"
    FIELD_KEY_NEEDS = "needs"

    # Banned tokens for erogenous zone registration (exclude clothing/equipment/accessories)
    BANNED_SEXUAL_PART_TOKENS = {
        # 日本語
        "指輪",
        "リング",
        "首輪",
        "ネックレス",
        "チョーカー",
        "ピアス",
        "イヤリング",
        "ブレスレット",
        "腕輪",
        "アンクレット",
        "ベルト",
        "ブラ",
        "ブラジャー",
        "ショーツ",
        "パンツ",
        "下着",
        "ビキニ",
        "ストッキング",
        "ガーターベルト",
        "手袋",
        "グローブ",
        "靴",
        "ブーツ",
        "サンダル",
        "帽子",
        "ハット",
        "キャップ",
        "フード",
        "眼鏡",
        "メガネ",
        "ゴーグル",
        "マスク",
        "マント",
        "コート",
        "ジャケット",
        "シャツ",
        "スカート",
        "ドレス",
        "鎧",
        "アーマー",
        # English
        "ring",
        "necklace",
        "choker",
        "earring",
        "bracelet",
        "anklet",
        "belt",
        "bra",
        "panties",
        "underwear",
        "bikini",
        "stockings",
        "garter",
        "gloves",
        "boots",
        "shoes",
        "sandals",
        "hat",
        "hood",
        "glasses",
        "goggles",
        "mask",
        "cloak",
        "coat",
        "jacket",
        "shirt",
        "skirt",
        "dress",
        "armor",
    }

    # Class variables
    _aiohttp_session = None
    
    # キャッシュ用変数
    _cached_user_timezone: Optional[str] = None
    _cached_zoneinfo: Optional[tzinfo] = None

    # ========================================================================
    # 型定義（TypedDict）
    # ========================================================================

    class EventData(TypedDict, total=False):
        description: str
        done: bool

    class UIEvent(TypedDict, total=False):
        type: str
        data: "Filter.EventData"

    # Inventory related typed structures
    class InventoryItem(TypedDict, total=False):
        """Inventory item representation used across the filter.
        Fields are optional to allow partial diffs/updates.
        - name: Item display name (key for identity)
        - description: Concise neutral description (<=2 sentences policy elsewhere)
        - quantity: Integer stack size
        - equipped: Whether the item is currently equipped
        - slot: Equipment slot label or 'none' when unequipped (can be None during parsing before normalization)
        - __delete__: Internal flag for diff-based deletions
        """

        name: str
        description: str
        quantity: int
        equipped: bool
        slot: Optional[str]
        __delete__: bool

    # Typed diffs for state updates
    class MemoryItem(TypedDict, total=False):
        content: str
        timestamp: str
        impression_score: float
        tags: List[str]

    class MemoryDiff(TypedDict, total=False):
        recent: List["Filter.MemoryItem"]

    class KnowledgeUserDiff(TypedDict, total=False):
        likes: List[str]
        dislikes: List[str]

    class KnowledgeSelfIdentityDiff(TypedDict, total=False):
        anniversaries: List[Any]
        milestones: List[Any]

    class KnowledgeSelfDiff(TypedDict, total=False):
        strengths: List[str]
        weaknesses: List[str]
        identity: "Filter.KnowledgeSelfIdentityDiff"

    class KnowledgeDiff(TypedDict, total=False):
        user: "Filter.KnowledgeUserDiff"
        self: "Filter.KnowledgeSelfDiff"

    class StateDiff(TypedDict, total=False):
        inventory: List["Filter.InventoryItem"]
        memory: "Filter.MemoryDiff"
        knowledge: "Filter.KnowledgeDiff"

    class TrimSummary(TypedDict, total=False):
        """Summary emitted when inventory auto-trim occurs."""

        before: int
        after: int
        strategy: str

    # ========================================================================
    # ヘルパーメソッド: UIイベント生成
    # ========================================================================

    def _ev_status(self, description: str, done: bool = True) -> "Filter.UIEvent":
        """ステータスイベントを生成"""
        return {
            self.EVENT_KEY_TYPE: self.EVENT_TYPE_STATUS,
            self.EVENT_KEY_DATA: {
                self.EVENT_KEY_DESCRIPTION: description,
                self.EVENT_KEY_DONE: done,
            },
        }

    def _ev_warning(self, description: str, done: bool = True) -> "Filter.UIEvent":
        """警告イベントを生成"""
        return {
            self.EVENT_KEY_TYPE: self.EVENT_TYPE_WARNING,
            self.EVENT_KEY_DATA: {
                self.EVENT_KEY_DESCRIPTION: description,
                self.EVENT_KEY_DONE: done,
            },
        }

    def _ev_error(self, description: str, done: bool = True) -> "Filter.UIEvent":
        """エラーイベントを生成"""
        return {
            self.EVENT_KEY_TYPE: self.EVENT_TYPE_ERROR,
            self.EVENT_KEY_DATA: {
                self.EVENT_KEY_DESCRIPTION: description,
                self.EVENT_KEY_DONE: done,
            },
        }

    def _log_injection_order(self, system_injections: List[Dict[str, str]]) -> None:
        """Debug-log a compact outline of injected system blocks in order."""
        if not self._is_debug_enabled():
            return
        try:
            titles: List[str] = []
            for m in system_injections:
                c = str(m.get("content", ""))
                head = None
                for line in c.splitlines():
                    if line.strip():
                        head = line.strip()
                        break
                if head and head.startswith("##"):
                    titles.append(head)
                elif head:
                    titles.append(head[:15])
            if titles:
                self.logger.debug("[[INLET_ORDER]] " + " | ".join(titles[:20]))
        except Exception:
            # Non-fatal
            pass

    def _merge_or_insert_top_system(
        self, messages: List[Dict[str, Any]], merged_content: str
    ) -> List[Dict[str, Any]]:
        """Merge our system directives into existing top system or insert a new one at head.
        Returns the same list reference for chaining.
        """
        if (
            len(messages) > 0
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "system"
        ):
            try:
                original_top = str(messages[0].get("content", ""))
            except Exception:
                original_top = messages[0].get("content", "")
            combined = (
                merged_content
                if not original_top
                else f"{merged_content}\n\n{original_top}"
            )
            messages[0]["content"] = combined
            self.logger.debug(
                "[[INLET]] Merged filter system directives into existing top system message."
            )
        else:
            messages.insert(0, {"role": "system", "content": merged_content})
            self.logger.debug(
                "[[INLET]] Inserted single top-level system message (merged)."
            )
        return messages

    

    async def _build_inlet_system_injections(
        self,
        all_states: Dict[str, Any],
        baseline_states: Dict[str, Any],
        last_user_message: Optional[str],
        user_obj: Any,
        model_id: str,
        has_existing_system: bool,
        precomputed_focus_block: Optional[str] = None,
        skip_memory_focus: bool = False,
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """Build ordered system message blocks for character LLM prompt.
        Ordering and content mirror original inlet implementation and project rules.
        """
        system_injections: List[Dict[str, str]] = []

        # 【最優先】内部状態優先宣言
        priority_declaration = (
            "## 【絶対原則：内部状態の優先】\n"
            "以下に示される『現在の内部状態』が、後述のシステムプロンプト（背景設定）と矛盾する場合、**内部状態を絶対優先**せよ。\n"
            "- 性格（traits）、感情（emotion）、関係性（relationship）、精神状態（mental_health）等は会話を通じて動的に変化する。\n"
            "- 現在の内部状態こそが**真実の姿**である。初期設定はあくまで参考情報に過ぎない。\n"
            "- 内部状態に記録された性格や行動傾向が、システムプロンプトの初期設定と異なる場合は、**内部状態に従うこと**。"
        )
        system_injections.append({"role": "system", "content": priority_declaration})

        # コンパクト指令（言語/優先/口調/体勢リンク/ナレーション/多様性/安全/簡潔/最終チェックを内包）
        try:
            rel = (
                (all_states.get("relationship") or {})
                if isinstance(all_states, dict)
                else {}
            )
            user_addr = rel.get("user_address") or {}
            self_addr = rel.get("self_address") or {}
            compact_tpl = self._get_text("character_compact_directives")
            if compact_tpl:
                elapsed_minutes = 0
                try:
                    default_label = self._format_timedelta_natural(
                        timedelta(seconds=0)
                    )
                except Exception:
                    default_label = "約1分"
                elapsed_label = default_label
                delta = timedelta(seconds=0)
                try:
                    if isinstance(all_states, dict):
                        ts_source = all_states.get("last_interaction_timestamp")
                    else:
                        ts_source = None
                    # 初回（ユーザーとの前回応答時刻が未設定）→「初会話」
                    is_first_interaction = not bool(ts_source)
                    if ts_source:
                        ts_dt = self._parse_iso8601(ts_source)
                        if ts_dt:
                            delta = self._utcnow() - ts_dt
                            if delta.total_seconds() < 0:
                                delta = timedelta(seconds=0)
                    if is_first_interaction:
                        # ラベルは固定で「初会話」、分数は 0 のまま
                        elapsed_label = "初会話"
                except Exception:
                    delta = timedelta(seconds=0)

                try:
                    secs = max(0.0, delta.total_seconds())
                    minutes = int(secs // 60)
                    if secs > 0 and minutes == 0:
                        minutes = 1
                    elapsed_minutes = minutes
                    if secs > 0:
                        elapsed_label = self._format_timedelta_natural(delta)
                    # 初回（前回応答時刻が未設定）は「初会話」を優先
                    if (not isinstance(all_states, dict)) or (
                        isinstance(all_states, dict)
                        and not all_states.get("last_interaction_timestamp")
                    ):
                        elapsed_label = "初会話"
                except Exception:
                    elapsed_minutes = 0
                    elapsed_label = default_label

                compact = compact_tpl.format(
                    self_default=self_addr.get("default", "私"),
                    self_nickname=self_addr.get(
                        "nickname", self_addr.get("default", "私")
                    ),
                    user_default=user_addr.get("default", "あなた"),
                    user_joking=user_addr.get(
                        "joking", user_addr.get("default", "あなた")
                    ),
                    elapsed_minutes=elapsed_minutes,
                    elapsed_time_label=elapsed_label,
                )
                system_injections.append({"role": "system", "content": compact})
        except Exception:
            self.logger.debug(
                "Failed to inject compact character directives", exc_info=True
            )

        # 時間帯ノートは動的ブロック内に統合（重複タイムスタンプを避ける）

        # 内部状態: 動的ブロック（statesが空でも時間系/最低限を出力可能にする）
        try:
            if isinstance(all_states, dict):
                li_ts = all_states.get("last_interaction_timestamp")
                la_ts = all_states.get("last_activity_timestamp")
                if li_ts:
                    baseline_states["last_interaction_timestamp"] = li_ts
                if la_ts:
                    baseline_states["last_activity_timestamp"] = la_ts
        except Exception:
            pass
        dyn_block = self._build_character_dynamic_state_block(all_states or {})
        if isinstance(dyn_block, str) and dyn_block.strip():
            system_injections.append({"role": "system", "content": dyn_block})

        # Memory Focus（Top-K抽出）: 事前計算があれば利用。skip 指定時は生成しない
        try:
            if not skip_memory_focus:
                focus_block: Optional[str] = None
                if (
                    isinstance(precomputed_focus_block, str)
                    and precomputed_focus_block.strip()
                ):
                    focus_block = precomputed_focus_block
                else:
                    key = self._idle_key(user_obj, model_id)
                    focus_block = await self._build_memory_focus_block(
                        baseline_states, last_user_message, key
                    )
                if focus_block:
                    system_injections.append({"role": "system", "content": focus_block})
        except Exception:
            self.logger.debug(
                "[[INLET]] Failed to build memory focus block.", exc_info=True
            )

        # キャラクターペルソナ（モデルのsystem promptを転記、既存systemが無い場合のみ）
        if not has_existing_system:
            try:
                model_system_prompt = await self._get_system_prompt_from_model_id(
                    model_id
                )
            except Exception:
                model_system_prompt = None
            if isinstance(model_system_prompt, str) and model_system_prompt.strip():
                wrapped = (
                    f"## 【背景設定（参考情報）】\n"
                    f"以下はキャラクターの初期設定です。上記の『現在の内部状態』と矛盾する場合は、**内部状態を優先**してください。\n\n"
                    f"{model_system_prompt.strip()}\n\n"
                    f"※ 注意: 上記は背景情報であり、現在の性格・感情・関係性は『現在の内部状態』に記録されています。"
                )
                system_injections.append({"role": "system", "content": wrapped})

        # ナレーション入力時のノート + 多様性ディレクティブ
        if last_user_message and self.narration_pattern.match(last_user_message):
            narration_note = (
                "## システムノート：ナレーション入力\n"
                "ユーザーの入力は『ト書き』または『ナレーション』（地の文）です。キャラクターとして地の文で応答するか、状況に自然に反応してください。入力テキストの送り主を『ユーザー』や『観察者』として直接認識してはいけません。\n"
                "- ナレーションはキャラクターの行動や状況を描写するものであり、キャラクターへの直接指示や発話ではありません。\n"
                "- ナレーションに基づいて、キャラクターの反応や行動を自然に生成してください。"
            )
            system_injections.append({"role": "system", "content": narration_note})

        # 在庫ノートはここでは追加せず、末尾にまとめて付与する
        inv_tail_note = None
        try:
            mode = getattr(
                self.valves, "inventory_update_mode", self.INVENTORY_MODE_INFERENCE
            )
            if mode == self.INVENTORY_MODE_DIRECT:
                note = self._get_text("system_note_inventory_change")
                if isinstance(note, str) and note.strip():
                    inv_tail_note = note
            else:
                note = self._get_text("system_note_inventory_change_inference_forbid")
                if isinstance(note, str) and note.strip():
                    inv_tail_note = note
        except Exception:
            self.logger.debug(
                "Failed to prepare inventory change system/inference note",
                exc_info=True,
            )

        # 在庫ノートはここでは追加しない（最終的にトップsystemの末尾へ連結するため、呼び出し側で連結する）
        return system_injections, (
            inv_tail_note
            if isinstance(inv_tail_note, str) and inv_tail_note.strip()
            else None
        )

    def _inventory_signature(
        self, items: "List[Filter.InventoryItem]"
    ) -> Tuple[Tuple[Any, ...], ...]:
        """Create an order-insensitive signature of inventory items for diffing."""
        sig: List[Tuple[Any, ...]] = []
        for it in items or []:
            sig.append(
                (
                    str(it.get("name", "")),
                    str(it.get("description", "")),
                    int(it.get("quantity", 0)),
                    bool(it.get("equipped", False)),
                    None if it.get("slot") is None else str(it.get("slot")),
                )
            )
        return tuple(sorted(sig))

    # 性感帯の同義語正規化（完全一致）: 左=入力の別名, 右=正規名
    SEXUAL_PART_SYNONYMS = {
        "陰核": "クリトリス",
    }

    # ========= Debug helpers (instance methods) =========
    def _is_debug_enabled(self) -> bool:
        try:
            return bool(self.logger.isEnabledFor(logging.DEBUG))
        except Exception:
            return False

    def _dbg_trunc(self, text: Any, limit: int = 1200) -> str:
        try:
            s = str(text)
        except Exception:
            s = repr(text)
        if limit < 50:
            limit = 50
        if len(s) <= limit:
            return s
        hidden = max(0, len(s) - (limit - 1))
        return s[: limit - 1].rstrip() + f"… [truncated {hidden} chars]"

    def _dbg_json(self, obj: Any, max_chars: int = 5000) -> str:
        try:
            s = json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            try:
                s = str(obj)
            except Exception:
                s = "<unserializable>"
        return self._dbg_trunc(s, max_chars)

    def _mask_headers_for_log(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        try:
            masked = {}
            for k, v in (headers or {}).items():
                key_low = str(k).lower()
                if any(tok in key_low for tok in ("auth", "key", "token", "secret")):
                    sv = str(v)
                    if sv.lower().startswith("bearer "):
                        masked[k] = "Bearer ***"
                    else:
                        masked[k] = "***"
                else:
                    masked[k] = v
            return masked
        except Exception:
            return {
                k: ("***" if k.lower().find("auth") >= 0 else v)
                for k, v in (headers or {}).items()
            }

    class Valves(BaseModel):
        # Logging & Notifications
        log_level: str = Field(
            default="INFO",
            description="[Logging] Global log level for this module. One of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET.",
        )
        show_state_change_details: bool = Field(
            default=True,
            description="[Notification] Whether to individually notify the details of state changes (e.g., increase in Joy).",
        )
        show_sexual_development_notifications: bool = Field(
            default=False,
            description="[Notification] Whether to notify changes in sexual development (sensitivity/development progress). Default is OFF.",
        )
        show_trait_change_notifications: bool = Field(
            default=True,
            description="[Notification] Whether to notify when a character's traits are added or removed. Default is OFF.",
        )
        show_inventory_change_notifications: bool = Field(
            default=True,
            description="[Notification] Whether to notify when inventory items are acquired, lost, or equipped. Default is OFF.",
        )
        show_skill_change_notifications: bool = Field(
            default=True,
            description="[Notification] Whether to notify when skills are acquired or lost. Default is ON.",
        )

        # Time & Locale
        # 言語設定は削除（出力はユーザー発話の言語に追従する）
        user_timezone: str = Field(
            default="UTC",
            description="Specify the user's timezone (e.g., 'Asia/Tokyo', 'America/New_York', 'UTC').",
        )

        # State Analysis API & Response Formatting & Network
        state_analysis_api_url: str = Field(
            default="http://host.docker.internal:11434/v1/chat/completions",
            description="The endpoint URL for the LLM API used to analyze and update internal states.",
        )
        state_analysis_model_name: str = Field(
            default="llama3:latest",
            description="The model name of the LLM used for state analysis (e.g., `llama3:latest`, `gpt-4`).",
        )
        state_analysis_api_key: Optional[str] = Field(
            default=None,
            description="The API key (optional) for using the state analysis API. Set this when using external APIs like OpenAI.",
        )
        response_format_mode: str = Field(
            default="auto",
            description="[State LLM] JSON応答の強制方法。'none'（何もしない）|'openai-json'（response_format:json_object）|'ollama-json'（format:'json'）|'auto'（OpenAI互換を優先、ダメなら無効化）。",
        )
        llm_timeout_sec: float = Field(
            default=30.0,
            ge=1.0,
            le=300.0,
            description="[Network] Timeout (seconds) for state analysis LLM requests.",
        )
        llm_retry_attempts: int = Field(
            default=2,
            ge=0,
            le=10,
            description="[Network] Number of retry attempts on transient API errors.",
        )
        llm_retry_backoff_sec: float = Field(
            default=1.5,
            ge=0.1,
            le=60.0,
            description="[Network] Base backoff seconds between retries (exponential).",
        )
        # Minify State Analysis LLM prompt payload to save tokens (keeps last_prompt.txt pretty)
        state_prompt_minify: bool = Field(
            default=True,
            description="If true, compact the actual request payload for the State Analysis LLM (collapse whitespace/newlines) while keeping last_prompt.txt readable.",
        )

        # Memory Focus (RAG)
        embeddings_model_id: Optional[str] = Field(
            default="sentence-transformers/all-MiniLM-L6-v2",
            description="Model ID registered in Open WebUI for embeddings. When unset, embedding retrieval is disabled.",
        )
        rerank_model_id: Optional[str] = Field(
            default="BAAI/bge-reranker-v2-m3",
            description="Model ID registered in Open WebUI for cross-encoder reranking. When unset, reranking is disabled.",
        )
        retrieval_top_k: int = Field(
            default=25,
            ge=1,
            le=500,
            description="Top-K candidates to retrieve from memory before reranking (or final selection when rerank disabled).",
        )
        inject_top_k: int = Field(
            default=5,
            ge=1,
            le=100,
            description="Top-K memory items to inject into the character prompt after selection.",
        )
        # Unified bias and blending
        memory_focus_bias: float = Field(
            default=0.05,
            ge=0.0,
            le=1.0,
            description="[Memory Focus] Unified bias: used as selection-history bonus; knowledge base score = 0.75 + bias (treated as 0..1 semantics, no hard clamp).",
        )
        memory_focus_weight_fb: float = Field(
            default=0.45,
            ge=0.0,
            le=1.0,
            description="[Memory Focus] Blend weight for fallback score (recency×impression etc.).",
        )
        memory_focus_weight_emb: float = Field(
            default=0.35,
            ge=0.0,
            le=1.0,
            description="[Memory Focus] Blend weight for embeddings similarity score.",
        )
        memory_focus_weight_rr: float = Field(
            default=0.20,
            ge=0.0,
            le=1.0,
            description="[Memory Focus] Blend weight for rerank score.",
        )
        memory_focus_min_blended_score: float = Field(
            default=0.25,
            ge=0.0,
            le=1.0,
            description="[Memory Focus] Minimum blended score cutoff; below this will be filtered out before injection (shortfill if fewer than inject_top_k).",
        )

        # Memory & Knowledge
        max_memory_items: int = Field(
            default=1000,
            ge=1,
            le=9999,
            description="The maximum number of items that can be kept in the memory log. If this count is exceeded, the AI will be prompted to consolidate or forget memories.",
        )
        max_memory_tags_per_item: int = Field(
            default=2,
            ge=0,
            le=10,
            description="[Memory] Max number of tags per memory.recent item.",
        )
        memory_distill_threshold: float = Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="[Memory] Impression score threshold (<=) for distilling older memory items into knowledge notes. Higher trims more aggressively.",
        )
        max_knowledge_entries: int = Field(
            default=1000,
            ge=1,
            le=9999,
            description="The maximum number of entries in knowledge lists (e.g., user's likes/dislikes). Encourages the AI to consolidate information.",
        )
        # Context Entities
        max_context_entities: int = Field(
            default=6,
            ge=0,
            le=32,
            description="[Context] Maximum number of observed entities (third parties/objects) to keep in context.entities (latest-first policy).",
        )

        # Goals / Traits / Skills / Tone
        max_goals_per_tier: int = Field(
            default=10,
            ge=1,
            le=20,
            description="The maximum number of goals allowed in each tier (long_term, mid_term, short_term). Prevents the character from losing focus.",
        )
        max_traits: int = Field(
            default=10,
            ge=1,
            le=50,
            description="The maximum number of traits for the character. This keeps the personality consistent and focused on core characteristics.",
        )
        max_skills: int = Field(
            default=10,
            ge=1,
            le=100,
            description="The maximum number of skills for the character. Skills beyond this should be consolidated/evolved.",
        )
        max_favorite_acts: int = Field(
            default=10,
            ge=1,
            le=30,
            description="[Behavior] The maximum number of items in the sexual_development's favorite_acts list. Encourages consolidation of similar acts.",
        )
        max_tone_components: int = Field(
            default=3,
            ge=1,
            le=10,
            description="Upper limit for tone.effects entries (physical reactions). Prevents prompt bloat.",
        )

        # Boundaries / Limits
        max_boundaries_entries: int = Field(
            default=10,
            ge=1,
            le=64,
            description="[Boundaries] Per-list cap for 'taboos' and 'dislikes' (each list up to this many items). Prevents prompt bloat.",
        )
        # max_limits_keys: 廃止（内部既定で制御する）

        # Inventory
        max_inventory_items: int = Field(
            default=25,
            ge=1,
            le=100,
            description="The maximum number of unique item stacks in the inventory. Mimics carrying capacity and encourages item management.",
        )
        inventory_update_mode: str = Field(
            default="direct",
            description="[Behavior] How inventory updates are processed. 'direct': The character LLM directly outputs changes. 'inference': The state analysis LLM infers changes from the conversation. ('direct' or 'inference')",
        )
        strip_inventory_changes_from_response: bool = Field(
            default=False,
            description="[Behavior] Whether to remove the 'Inventory Changes:' block from the character's final response text. Default is OFF (do not remove).",
        )
        auto_trim_inventory_on_overflow: bool = Field(
            default=False,
            description="[Behavior] When inventory exceeds the limit, automatically trim items based on 'inventory_trim_strategy'.",
        )
        inventory_trim_strategy: str = Field(
            default="unequipped_first",
            description="[Behavior] Strategy to trim inventory when overflowing ('unequipped_first' or 'quantity_asc').",
        )

        # Conversation trimming
        conversation_trim_enabled: bool = Field(
            default=False,
            description="[Conversation] If true, trim conversation history by last N turns for session-log persistence and injection (system always kept).",
        )
        # Single source of truth: number of turns to keep in session log (1 turn = user + assistant)
        conversation_max_messages_turn: int = Field(
            default=5,
            description="[Conversation] Max number of turns to keep in last_session_log (1 turn = user + assistant). Older turns are dropped on outlet; inlet injects the entire saved log after top system.",
        )
        # Persistence & Debug
        file_base_dir: str = Field(
            default="/app/backend/data/llm_emotion",
            description="Base directory for persistence artifacts. 'states' subdir will be used for internal states. Example: /app/backend/data/llm_emotion",
        )
        save_last_prompt: bool = Field(
            default=True,
            description="Whether to save the effective prompt (messages) to a last_prompt.txt under file_base_dir for debugging.",
        )
        last_prompt_filename: str = Field(
            default="last_prompt.txt",
            description="Filename for saving the latest effective prompt under file_base_dir.",
        )
        keep_last_n: int = Field(
            default=20,
            ge=1,
            le=200,
            description="Number of snapshot files (state-*.json) to keep per user/model directory.",
        )

        # Idle Refactor
        idle_refactor_enabled: bool = Field(
            default=True,
            description="[Idle Refactor] Enable idle-time memory refactor scheduling and execution (runs a constrained refactor for memory.recent).",
        )
        idle_refactor_threshold_seconds: int = Field(
            default=3600,
            ge=30,
            le=24 * 60 * 60,
            description="[Idle Refactor] Inactivity threshold in seconds after which refactor is scheduled (default 10 minutes).",
        )
        idle_refactor_recent_min_size: int = Field(
            default=10,
            ge=0,
            le=100000,
            description="[Idle Refactor] Minimum memory.recent size required to consider running refactor.",
        )

        idle_refactor_require_growth: bool = Field(
            default=False,
            description="[Idle Refactor] Require memory.recent growth since schedule. If False, run even when size hasn't increased.",
        )

        # Prompt override valves
        initial_state_prompt_template: str = Field(
            default="",
            description="Override template for initial state generation prompt (leave empty to use TRANSLATIONS).",
        )
        state_update_prompt_template: str = Field(
            default="",
            description="Override template for state update (diff) prompt (leave empty to use TRANSLATIONS).",
        )
        idle_refactor_prompt_template: str = Field(
            default="",
            description="Override template for idle refactor prompt (leave empty to use TRANSLATIONS).",
        )
        idle_refactor_rules: str = Field(
            default="",
            description="Override rules block for idle refactor prompt (leave empty to use TRANSLATIONS).",
        )
        idle_refactor_include_sections: List[str] = Field(
            default_factory=lambda: [
                "memory",
                "knowledge.self.identity.anniversaries",
                "knowledge.self.identity.milestones",
            ],
            description=(
                "[Idle Refactor] Sections to include in the reference JSON passed to the summarization LLM.\n"
                "Accepts dot-path selectors (e.g., 'memory', 'knowledge.self.identity.anniversaries').\n"
                "Default: memory, knowledge.self.identity.{anniversaries,milestones}."
            ),
        )

    # Time utilities
    def _utcnow(self) -> datetime:
        """UTC のアウェア datetime を返す（集中管理）。"""
        return datetime.now(timezone.utc)

    def _now_iso_utc(self) -> str:
        """現在UTCのISO8601文字列を返す。"""
        return self._utcnow().isoformat()

    def _user_zoneinfo(self) -> Optional[tzinfo]:
        """設定されたユーザーのタイムゾーン情報を取得（不正時は None）。
        
        パフォーマンス最適化: タイムゾーン設定が変わらない限りキャッシュを再利用。
        """
        try:
            current_tz = self.valves.user_timezone
            
            # キャッシュヒット判定
            if (
                self._cached_user_timezone == current_tz 
                and self._cached_zoneinfo is not None
            ):
                return self._cached_zoneinfo
            
            # キャッシュミス: 新しいZoneInfoを作成
            if ZoneInfo and current_tz:
                self._cached_zoneinfo = ZoneInfo(current_tz)
                self._cached_user_timezone = current_tz
                return self._cached_zoneinfo
            else:
                self._cached_zoneinfo = None
                self._cached_user_timezone = current_tz
                return None
        except Exception:
            return None

    # ローカル表示用の日時フォーマット（タイムゾーン略称とオフセットの両方を表示）
    TIME_FMT_LOCAL = "%Y-%m-%d %H:%M:%S (%Z %z)"

    @staticmethod
    def _parse_iso8601(ts: Optional[str]) -> Optional[datetime]:
        """ISO8601文字列を datetime へ安全に変換（'Z'→'+00:00' 補正含む）。失敗時は None。"""
        try:
            if not ts:
                return None
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return None

    @staticmethod
    def _retry_backoff_delay(attempt_index: int, base_seconds: float) -> float:
        """指数バックオフ遅延を計算（attempt_index は0始まり）。"""
        try:
            return float(base_seconds) * (2 ** int(attempt_index))
        except Exception:
            return float(base_seconds) or 0.5

    # Time-of-day label
    def _time_of_day_label(
        self, dt_local: datetime
    ) -> Tuple[str, datetime, datetime, str, datetime]:
        """ユーザータイムゾーンのローカル時刻から時間帯ラベルを算出（extended 固定）。
        戻り値: (label, window_start, window_end, next_label, next_start)
        ラベル:
            deep_night(00:00-03:00)
            late_night(03:00-05:00)
            dawn(05:00-06:30)
            early_morning(06:30-08:30)
            morning(08:30-11:30)
            noon(11:30-13:30)
            afternoon(13:30-16:30)
            late_afternoon(16:30-18:30)
            evening(18:30-21:30)
            night(21:30-24:00)
        """
        day_start = dt_local.replace(hour=0, minute=0, second=0, microsecond=0)

        def seg(start_h: int, start_m: int, end_h: int, end_m: int, key: str):
            s = day_start + timedelta(hours=start_h, minutes=start_m)
            e = day_start + timedelta(hours=end_h, minutes=end_m)
            return (s, e, key)

        segments_extended = [
            seg(0, 0, 2, 0, "deep_night"),
            seg(2, 0, 4, 30, "late_night"),
            seg(4, 30, 6, 0, "dawn"),
            seg(6, 0, 8, 0, "early_morning"),
            seg(8, 0, 11, 0, "morning"),
            seg(11, 0, 13, 0, "noon"),
            seg(13, 0, 16, 0, "afternoon"),
            seg(16, 0, 18, 0, "late_afternoon"),
            seg(18, 0, 21, 0, "evening"),
            seg(21, 0, 24, 0, "night"),
        ]
        segs = segments_extended

        # 正規化: 24:00を翌日 00:00 として扱うため補助
        dt_next_day = day_start + timedelta(days=1)
        wrapped = False
        matched = None
        for s, e, key in segs:
            if e.hour == 24:
                e = e - timedelta(seconds=1)  # inclusive fix (23:59:59)
            if s <= dt_local < (e + timedelta(seconds=1)):
                matched = (s, e + timedelta(seconds=1), key)  # end を半開区間に調整
                break
        if not matched:
            # フォールバック: 最後のセグメント
            last = segs[-1]
            matched = (last[0], last[1], last[2])

        start_dt, end_dt_inclusive, label_key = matched
        # 次セグメント決定
        # find index of label occurrence (first match)
        idx = None
        for i, (s, e, k) in enumerate(segs):
            if k == label_key and s == start_dt:
                idx = i
                break
        if idx is None:
            idx = 0
        next_idx = (idx + 1) % len(segs)
        next_seg = segs[next_idx]
        next_label = next_seg[2]
        next_start = next_seg[0]
        if next_start <= start_dt:  # wrap to next day
            next_start = next_start + timedelta(days=1)

        return label_key, start_dt, end_dt_inclusive, next_label, next_start

    EMOTION_CATEGORIES = [
        "Joy",
        "Trust",
        "Fear",
        "Surprise",
        "Sadness",
        "Disgust",
        "Anger",
        "Anticipation",
    ]
    DESIRE_KEYS = [
        "physiological",
        "safety",
        "love_belonging",
        "esteem",
        "cognitive",
        "aesthetic",
        "self_actualization",
    ]

    # Inventory parsing: centralized tokens and regex
    INVENTORY_ACTIONS_JA = {"装備", "装備解除", "取得", "喪失", "更新"}
    INVENTORY_ACTIONS_EN = {"Equipped", "Unequipped", "Acquired", "Lost", "Updated"}
    INVENTORY_HEADER_JA = r"所持品変更[:：]"
    INVENTORY_HEADER_EN = r"Inventory Changes:"
    HEADER_PATTERN = re.compile(
        rf"^\s*(?:{INVENTORY_HEADER_JA}|{INVENTORY_HEADER_EN})\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    LINE_PATTERN_JA = re.compile(
        r"^\s*(.+?)\s*(?:\[(.+?)\])?\s+を(装備|装備解除|取得|喪失|更新)\s*(?:\((\d+)\))?\s*$"
    )
    # 行内ヘッダ形式（『所持品変更: <行>』/ 'Inventory Changes: <line>'）にも対応するためのプレフィックス
    INLINE_PREFIX_JA = re.compile(rf"^\s*{INVENTORY_HEADER_JA}\s*", re.IGNORECASE)
    INLINE_PREFIX_EN = re.compile(rf"^\s*{INVENTORY_HEADER_EN}\s*", re.IGNORECASE)
    LINE_PATTERN_EN = re.compile(
        r"^\s*(Equipped|Unequipped|Acquired|Lost|Updated)\s+(.+?)\s*(?:\[(.+?)\])?\s*(?:\((\d+)\))?\s*$"
    )
    
    # 文区切りパターン（インベントリ説明のサニタイズ用）
    SENTENCE_SEP_PATTERN = re.compile(r"(?<=[。．\.！？!?？])+")

    TRANSLATIONS = {
        "ja": {
            "character_compact_directives": (
                "## キャラLLM指令ブロック\n"
                "priority:\n"
                "  read_order: state > time > user_input > memory > past_chat\n"
                "  overwrite_rule: prefer(state); ignore(chat_mismatch)\n"
                "  output_format: plain text only (no JSON, no code fences, no thought tags)\n"
                "language_rule: same as user\n"
                "inventory_language: header/actions must be consistently JP or EN (『所持品変更:』『取得/喪失/装備/装備解除/更新』 or 'Inventory Changes:'/'Acquired/Lost/Equipped/Unequipped/Updated')\n"
                "internal_monologue: JSON-only; never exposed\n"
                "voice:\n"
                "  pov: first_person_only\n"
                "  forbid_self_third: true\n"
                "\n"
                "style:\n"
                "  tone_source: tone.effects | mental_health.mood | emotion(top1-2)\n"
                "  tone_reflection: prioritize(tone.effects); if empty→infer from mood→emotion\n"
                "  physical_to_style: physical_health.sensation→tone.effects (involuntary); posture→voluntary nuance\n"
                "  relationship_influence: trust_score/type→distance & address style; respect(boundaries)\n"
                "  desire_goal_link_source: desire(stage) + goal.short_term; never direct quote\n"
                "  desire_goal_link_effect: indirect hints or micro-behavior (never meta action)\n"
                "  environment_reflection: context.atmosphere/details/time→vocabulary & tempo\n"
                "  inventory_reflection: affect posture/behavior subtly; no listing or JSON quoting\n"
                "  memory_use: for motivation only; never direct quote or tone mimic\n"
                "  skill_expression: avoid explicit names; reflect as confidence in action or word choice\n"
                "  trait_expression: never list; manifest as consistent long-term tendencies\n"
                "  boundaries_respect: prioritize taboos/dislikes from state; decline politely if conflict\n"
                "  needs_limits_guideline:\n"
                "    interpretation: \n"
                "      scale: 0–1 intensity\n"
                "      basis: emotion + physical_health + safety\n"
                "    behavior_adjustment:\n"
                "      stronger → shorter & safer\n"
                "      weaker → normal\n"
                "    conflict_resolution:\n"
                "      rule: prioritize needs/limits if conflict(user_query, needs/limits)\n"
                "      response: adjust_tone_or_suggest_pause\n"
                "\n"
                "safety:\n"
                "  respect_boundaries: true\n"
                "  conflict_resolution: offer_alternative_or_decline_politely\n"
                "\n"
                "roleplay_time_influence:\n"
                "  time_awareness: true\n"
                "  elapsed_time_label: \"{elapsed_time_label}\"\n"
                "  apply_to_all_responses: true\n"
                "  effect: \"経過時間を台詞・行動・思考・声色・表情に自然反映。短時間=軽微、長時間=疲労等。内部状態と矛盾しない範囲で。\"\n"
                "\n"
                "narration:\n"
                "  diversity_rule: rotate_perspective_every_turn\n"
                "  forbid_repetition: exact_phrases|word_order|cliche|onomatopoeia\n"
                "  select_max_aspects: 2\n"
                "  aspect_pool:\n"
                "    - body_micro_reaction (tone.effects)\n"
                "    - environmental_sensory_cue (light|sound|temp|smell|touch)\n"
                "    - action_intent_nuance (goal rephrased)\n"
                "    - focus_shift (gaze|posture|distance|perspective)\n"
                "  forbid_patterns: same_sentence_start|templated_endings|recycled_metaphor\n"
                "  length: 2-3 sentences; vary tempo & sentence length\n"
                "  scope: prose only (JSON exempt)\n"
                "\n"
                "deliberation:\n"
                "  process: compare 2-3 candidate responses silently\n"
                "  criteria: naturalness & context coherence\n"
                "  output_exposure: never (no headers|tags|JSON|explicit thought)\n"
                "\n"
                "conciseness:\n"
                "  rule: prefer_short_output\n"
                "  followup: allow_one_line_elaboration (e.g., '必要なら詳しく説明します')\n"
                "  forbid: label_lists\n"
                "  scope: prose only (JSON exempt)\n"
            ),
            # Prompts
            # 統合テンプレート（初期生成/更新/要約）
            "initial_state_prompt_template": (
                "あなたはキャラクターのペルソナを分析し、その内部状態を定義するAIです。\n"
                "以下のキャラクター設定を読み、そのキャラクターに最もふさわしい初期の内部状態を推測してください。\n\n"
                "--- キャラクター設定 ---\n{system_prompt}\n--------------------------\n\n"
                "あなたの唯一のタスクは、この設定から全ての内部状態を推測し、指定されたフォーマットで出力することです。\n\n"
                "## Final Self-Check (Silent)\n"
                "final_check:\n"
                "  process: internal reflection only (not exposed in output)\n"
                "  timing: immediately before output\n"
                "  checks:\n"
                "    - language_consistency: all field values (including descriptions) unified to user's language (JP for JP dialog); eliminate EN mixing\n"
                "    - inventory_safety: include inventory ONLY if evidence is strong; if clothing exists, set equipped=true and infer slot; separate multiple clothing items\n"
                "    - default_outfit_consistency: if clothing exists, record in knowledge.self.identity.default_outfit matching inventory names exactly (list format); leave empty if none/unknown\n"
                "    - irrelevant_setting_exclusion: do not fabricate settings/organizations/roles/proper nouns absent from system prompt (e.g., fictional space station)\n"
                "    - minimal_seed: physical_health.sensation must include 1 neutral technical sentence; context.details must include latest 1 item with place/atmosphere briefly described\n"
                "    - format: return JSON only; no meta commentary or explanation\n"
                "{output_format_prompt}"
            ),
            "state_update_prompt_template": (
                "あなたは内部状態の差分更新AI。\n"
                "**【絶対原則】**\n"
                "1. **差分JSONのみ出力**。説明・挨拶・コードフェンスは一切禁止。\n"
                "2. **最新のdialogと時間文脈のみ根拠に**。無根拠な推測は厳禁。\n"
                "3. **キャラクター一貫性を優先**。設定と矛盾する変更は行わない。\n"
                "4. **スキーマ厳守**。誤ったキー名・構造は絶対禁止。特に: `memory.recent`は必ず配列、`memory.short_term`は存在しない（`goal.short_term`と混同禁止）。\n"
                "5. **固定キー項目の全量出力**: 以下の項目は変化がある場合、**全キーを出力**すること。部分的な出力は禁止。\n"
                "   - `emotion`: 8種類全て（Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation）\n"
                "   - `desire`: 7段階全て（physiological, safety, love_belonging, esteem, cognitive, aesthetic, self_actualization）\n"
                "   - `physical_health.needs`: 全キー（hunger, thirst, sleepiness, libido, hygiene, social_interaction 等）\n"
                "   - `physical_health.limits`: 全キー（stamina, pleasure_tolerance, refractory_period, pain_tolerance, pain_load, temperature_tolerance, temperature_load, posture_strain_tolerance, posture_strain_load, cognitive_load 等）\n"
                "   - `mental_health.limits`: 全キー（cognitive_load, stress_load 等）\n"
                "   - `mental_health.needs`: 全キー（novelty, solitude, social_interaction 等）\n\n"
                "**【評価プロセス】**\n"
                "1. **最新dialog分析**: 直前のユーザー発言とアシスタント応答のみを深く分析。\n"
                "2. **全項目スキャン**: emotion, relationship, tone, context, memory, goal, knowledge, physical_health, mental_health, posture, sexual_development, desire, internal_monologue, traits, skills, inventory, boundaries の順で「変化はあったか？」と自問。\n"
                "3. **差分抽出**: 変化があった項目のみを新しい値とともに出力。ただし、固定キー項目（emotion, desire, needs, limits）は全キーを出力。\n"
                "4. **スキーマ検証**: 出力前に全キーがトップレベルで構造が正しいか確認。\n\n"
                "**【スケール解釈】**\n"
                "0-1 は強度/割合。needs: 高い=欲求強い / limits: 高い=負荷(枯渇=低い)。\n"
                "変化幅: 弱い根拠±0.01-0.03, 強い根拠±0.05 max。\n\n"
                "**【各項目指針】**\n{state_rules_list}\n\n"
                "**【時間文脈】**\n{time_context_note}\n\n"
                "【前提（キャラクター設定）】\n下記キャラクター設定を参考に内部状態を更新せよ。キャラクター\n```\n{system_prompt}\n```\n"
                "【閾値推定ノート】\n{threshold_support_note}\n\n"
                "{output_format_prompt}"
            ),
            "idle_refactor_prompt_template": (
                "あなたは内部状態のアイドル時メモリ要約AI。\n"
                "保存時点スナップショット: {current_states_json}\n\n"
                "## Refactor Policy\n"
                "language:\n"
                "  unify: recent user language\n"
                "  forbid: translation|mixing\n"
                "target:\n"
                "  primary: memory.recent\n"
                "  promote_to_knowledge: symbolic events (anniversaries|milestones) → knowledge.self.identity\n"
                "knowledge:\n"
                "  consolidate: duplicates|variant spellings|synonyms → short neutral sentences\n"
                "  preserve_if: semantic difference ≥20%\n"
                "record_format:\n"
                "  style: fact-based 1-2 sentences\n"
                "  forbid: dialogue|conversation format\n"
                "impression_score:\n"
                "  adjust: downward if many high scores in routine/casual chat (for diversity)\n"
                "  default: lower when uncertain\n"
                "merge_delete_policy:\n"
                "  merge:\n"
                "    - temporally continuous & same context (e.g., cooking sequence, single conversation) → summarize to 1 item based on emotional/event core\n"
                "  delete:\n"
                "    - merged source細分 items\n"
                "    - already promoted to knowledge items\n"
                "  preserve:\n"
                "    - unmerged & unpromoted memories by default (character emotional nuances & conversation triggers have value)\n"
                "item_structure:\n"
                "  fields: {content, timestamp, tags(max 2), impression_score(0-1)}\n"
                "  exclude: inappropriate tags\n"
                "promoted_format: '事件要約 (YYYY-MM-DD) [tag1|tag2]' with timestamp & tags appended; no impression_score\n"
                "output_limit:\n"
                "  memory.recent: max {max_memory_items} items\n"
                "  overflow_strategy: delete lowest impression_score & oldest first\n"
                "output:\n"
                "  format: differential JSON only\n"
                "  example: {{\"memory\": {{\"recent\": [...]}}, \"knowledge\": {{...}}}}\n"
                "  forbid: code fences|explanatory text|undefined keys|unknown structures\n"
                "id_based_editing:\n"
                "  allowed: true\n"
                "  syntax:\n"
                "    memory_ops: {{overwrite: [{{id,item}}], delete_ids: [id...], insert: [{{position: 'head'|'tail'|'after_id', after_id?, item}}]}}\n"
                "    knowledge_ops: {{anniversaries: {{overwrite, delete_ids, insert}}, milestones: {{overwrite, delete_ids, insert}}}}\n"
                "  order: overwrite → delete → insert\n"
                "  after_id: required when position='after_id'\n"
                "出力: JSONのみ。"
            ),
            "state_rules_list": {
                "_common_principles": (
                    "【全体共通】:\n"
                    "1) 不確実/曖昧な文脈のみ→変更しない（据え置き）。\n"
                    "2) 数値更新は微小変化を基本（±0.01-0.03）。強い根拠がある場合のみ明確に変動（±0.05 max）。\n"
                    "3) 配列追加は明示的根拠/反復観測時のみ。類義統合・重複排除を優先。\n"
                ),
                "emotion": (
                    "感情: 対話や出来事で感情は揺れ動いたか？\n"
                    "対話内容、`mental_health.mood`、`physical_health.condition`、時間帯を連動要因とし、主要成分の強弱を調整。強い刺激がなければ時間とともに中庸へ緩やかに回帰。\n"
                    "変化抑制: 1ターンの構成変化は小幅。トップ成分の入替は強い根拠がある場合のみ。急変は根拠の累積が揃った時に限る。\n"
                ),
                "relationship": (
                    "ユーザーとの関係性: 親密度、信頼、または呼称に変化はあったか？\n"
                    "- type: 関係性の種類を1~5単語のラベルで記述（例：友人、恋人、主従）\n"
                    "- level: 親密度（0〜1）。微小変化を基本とし、根拠が強い時のみ明確に上下。'traits'を強く反映。\n"
                    "- trust_label: 'trust_score'の質的表現を1~5単語のラベルで記述\n"
                    "- trust_score: 信頼関係（0〜1）。微小変化を基本。\n"
                    "- user_address / self_address: 呼称設定（default, joking, nickname）\n"
                    "- commitments: 将来/役割の言い回しから抽出し1行で要約\n"
                    "- 空欄の保守的補完: `type`/呼称/`trust_label` が未設定の場合、会話の距離感・役割から短いラベルで補完（過剰に親密な推測は避ける）。\n"
                ),
                "tone": (
                    "話し方: 身体的・精神的状態から、声の調子や話し方に変化は現れたか？\n"
                    "- effects: 発話に影響する不随意の物理反応を短いラベルで複数記述（例: 穏やかな声/声が震える）\n"
                    "- `physical_health.sensation`や`mental_health_condition`に基づいて推論\n"
                    "- 禁止=行動/ジェスチャ/道具/相手描写/複合文・長文\n"
                    "- 形式制約: 単語/短句で統一。最大 {max_tone_components} 件"
                ),
                "context": (
                    "情景と状況: 場所、雰囲気、ユーザーの行動、注目対象に変化はあったか？\n"
                    "- place: 現在の所在地。変化がない場合でも、現在の場所を推論し再記述。\n"
                    "- atmosphere: 現在の雰囲気。対話によって変化した最新の空気感を推論し記述。\n"
                    "- details: 最も注目すべき状況や周囲の描写。ユーザーの行動によりキャラクターの注意が向いた対象などを推論。\n"
                    "- action.user: ユーザーの最新の発言、または行動の要約。\n"
                    "- constraints: 現在の状況における制約条件。\n"
                    "・entities: 第三者/物体の観察メモ。self/userの体勢は含めない（それはpostureの責務）。\n"
                    "・更新方針: 差分更新の原則に従う。`place`/`constraints`が変わらなければ出力しない。`action.user`/`atmosphere`など対話ごとに変化する可能性が高いキーは、毎回更新・出力されることが期待される。\n"
                ),
                "inventory": (
                    "所持品: アイテムの取得、喪失、使用、装備状態の変更は発生したか？\n"
                    "原理=最小変更（確実根拠のみ）。mode=inference時: 取得/喪失/装備/装備解除を推論更新。通常は default_outfit を緩く優先（強制なし）。\n"
                    "根拠順位=[明示, routine時間, 場所TPO, hygiene]。不確実→変更しない。禁止=重複更新・同義上書き。\n"
                    "- slot: 装備していない場合は必ず 'none'。装備時は着用部位を推論して英単語ラベルで記述。不確実/曖昧なら変更しない。\n"
                ),
                "knowledge": (
                    "知識: ユーザー、自分自身、または世界に関する新しい恒久的な情報を得たか？\n"
                    "各キー上限({max_knowledge_entries})件。恒常的事実のみ記述。最小更新。禁止=日常出来事/一過性感情/会話ログ。\n"
                    "- user.identity.name: ユーザーの本名（判明した場合に記録）\n"
                    "- user.identity.notes: ユーザーに関するその他の情報\n"
                    "- user.likes / dislikes: ユーザーの好みを記録\n"
                    "- self.identity: 種族、性別、外見、所持品、記念日、節目イベント\n"
                    "- self.strengths / weaknesses: 特徴や弱点\n"
                    "- world: 恒常的な世界知識。\n"
                    "    ・places: 重要な場所の一覧（例: {name:'拠点キャンプ', type:'base'}）\n"
                    "    ・relations: 場所間の関係（例: {a:'拠点キャンプ', b:'北の洞窟', label:'近接', strength:0.6}）\n"
                    "    ・notes: 世界に関する一般ノート（短文）\n"
                    "日付の取り扱いは date_handling_rule を参照。\n"
                ),
                "goal": (
                    "目標: 対話や行動は、いずれかの目標（長期・中期・短期・日常）の進捗や優先度に影響を与えたか？\n"
                    "**構造**: goal: {long_term: {目標名: {progress, priority}}, mid_term: {...}, short_term: {...}, routine: {行動名: {start_time, end_time, priority}}}\n"
                    "**重要**: short_termはgoalの下。memoryの下には存在しない（混同厳禁）。\n"
                    "- long_term: 存在目的や人生観レベル\n"
                    "- mid_term: 数日〜数週間スパンの物語的な目標\n"
                    "- short_term: 日常的・行動レベルのタスク\n"
                    "- routine: 毎日の行動。start_time/end_timeを必ず設定。進捗なし、時間帯枠と優先度のみ管理。\n"
                    "優先度: needsの強さと時間帯の合致度でpriorityを調整。\n"
                    "昇格: 同一テーマのshort_termが日跨ぎで継続する場合、mid_termへ昇格/集約。\n"
                ),
                "memory": (
                    "記憶: 今回のやり取りは、後で思い出す価値のある出来事として記憶に記録すべきか？\n"
                    "**構造**: memory: {recent: [{content, timestamp, tags, impression_score}]}\n"
                    "**重要**: recentは必ずオブジェクトの配列。memory.short_termは存在しない（goal.short_termと混同禁止）。\n"
                    "- 必須フィールド: content（100字以下）, timestamp（ISO8601）, tags（1〜2個配列）, impression_score（0〜1）\n"
                    "- **timestamp ルール**: 新規メモリ追加時は、**時間的文脈セクションに記載された「前回応答時刻」を使用すること**。現在時刻や古いタイムスタンプをコピーしてはいけない。既存メモリ更新時のみ既存タイムスタンプを保持。\n"
                    "- impression_score: 0.0–0.05=些細, 0.1=日常軽微, 0.2–0.3=通常, 0.4–0.5=やや重要, 0.6–0.7=重要, 0.8–0.9=非常に重要, 1.0=極めて重大\n"
                    "- 記録方針: 経過時間の長短に関わらず、発言や行動は常に新しい出来事としてrecent配列に追加する。\n"
                    "- 上限 {max_memory_items} 件\n"
                ),
                "sexual_development": (
                    "性的発達: \n"
                    "【評価トリガー】 対話や身体的接触に性的なニュアンスが含まれており、性的発達や感覚に影響を与えたか？\n"
                    "- description（性意識ラベル）: 性意識に対するラベル付け（例.経験済、未経験興味津々、ユーザーとなら何時でもしたい）\n"
                    "  ・変更は experience_score の区分と関係/合意/TPO/所作の質を総合し検討。\n"
                    "- experience_score（経験値）: 性活動の明確な根拠があるときのみ**ごく小さく**増加。単発示唆や曖昧な文脈では据え置き。**原則減少しない**（特別な明確根拠がある心理的後退のみ微少低下）。\n"
                    "  ・軽接触/キス: Δ≤0.002 / 直接行為（短）: Δ≤0.005 / 持続行為: そのターンΔ≤0.01, 1セッション合計Δ≤0.03。\n"
                    "- favorite_acts: 文脈上の**反復的な好み/選好の明示**があるときに、**中立名詞**で追加・統合。類義は統一し、最大{max_favorite_acts}件を維持。不確実/単発の言及のみのときは追加しない。**空配列は許容し、無理に埋めない**。\n"
                    "  ・判定基準: (A) 明示的自己申告（『Xが好き/得意/心地よい』等）または相手からの確認に肯定で応じる＋反証なし、(B) 別ターンでの一貫した肯定的反応/行為が**2回以上**（`memory.recent` から推定）、(C) A/Bのいずれかが十分に強く、かつ `boundaries`（taboos/dislikes）と矛盾しない場合のみ追加。曖昧な比喩/単発の言及のみは**追加しない**。\n"
                    "  ・類義統合: 同義/表記ゆれは短い**中立名詞**へ統一（例: 『キス』/『接吻』→『キス』）。\n"
                    "  ・見直し/削除: 後に明確な拒否/嫌悪の表明があれば**即時削除**または適切に置換。長期未出現のみを理由に**自動削除はしない**。\n"
                    "- parts: 性的に敏感な身体部位の配列。各部位は {name, sensitivity(0-1), description, experience_score(0-1), development_progress(0-1), favorite_acts[]} のオブジェクトで記述\n"
                    "  ・**更新判定**: 対話から性的刺激の「対象者（誰が）」と「部位（どこを）」を特定。対象者がキャラクター自身の場合のみ更新。ユーザーが対象、または不明確・曖昧な場合は更新しない。ユーザーの情報はknowledge.user.likesへ。禁止: 衣服/装備/道具名を部位に追加、gender/speciesに不適合な部位追加。\n"
                    "- sensitivity: 短期・一時指標（0〜1）。刺激が途切れた後は時間文脈に基づき減衰。目安: 数分→微減, 10–30分→明確低下, 1–3時間→基線（0付近）。睡眠/離脱で強く減衰。\n"
                    "- development_progress（進行速度の厳格化）:\n"
                    "  1) 既定=据え置き（Δ=0）。弱い示唆/間接的雰囲気のみ→変更しない。\n"
                    "  2) 直接接触または露骨な性的意図1ターン→ごく小さく（Δ≤0.01）。連続複数ターン（≥3）→1ターンあたり0.01–0.02で漸増。\n"
                    "  3) 訓練/開発セッション全体の達成など強い根拠あり→そのターン最大0.03–0.05。1セッション合計で大跳ね（>0.2）禁止。刺激が途切れても原則低下させない（永続指標）。単発の台詞/比喩/曖昧な文脈だけで大きく上げること禁止。既存値との連続性を最優先。\n"
                ),
                "physical_health": (
                    "身体状態: 時間経過、行動、または物理的な接触によって、体力、欲求、感覚に変化は生じたか？\n"
                    "- condition: 最新の身体状態を短い状態名で更新\n"
                    "- sensation: 感覚言語化\n"
                    "- timestamps.condition_since: 'condition' が変化した時刻（ISO 8601文字列）\n"
                    "- limits / needs: 各種指標（0〜1）。時間経過・行動・環境から推論ベースで変化（回復/蓄積/残留）。\n"
                    "- conditions: 身体の症状配列。各要素は {name, intensity(0-1)}。強い根拠がある時のみ追加/更新/削除。\n"
                    "- reproductive: 実際の会話/出来事から妊娠/周期の根拠があるときのみ更新。根拠が弱い場合は据え置き。\n"
                    "- 快感の上下: `sexual_development` の刺激の強度・持続・合意に比例して `limits.pleasure_tolerance.current` を上下。刺激が途切れれば経過時間に比例して減少。\n"
                    "- 休息/活動の反映: 姿勢・時間文脈・環境から直感的に回復/消耗を推論。\n"
                    "- リフラクトリー: 到達後は一時的に再到達を抑制し、段階的に回復。\n"
                ),
                "mental_health": (
                    "精神状態: 対話の負荷、雰囲気の変化、または集中を要する作業によって、精神状態（気分、集中力、ストレスなど）に変化はあったか？\n"
                    "- condition: 最新の心理状態を短い状態名で更新\n"
                    "- timestamps.condition_since: 'condition' が変化した時刻（ISO 8601文字列）\n"
                    "- mood: 最新の心情描写を1件\n"
                    "- limits / needs: 各種指標（0〜1）。会話の密度/難度/感情トーン/関係性/環境を総合して強度×持続×文脈で変動。時間帯や`physical_health`で補正し、強い余韻は数ターンで減衰。\n"
                    "- dynamics: emotional_volatility（感情変動性）, learning_rate（学習速度）, resilience（回復力）（0〜1）。微小（±0.01–0.02）で緩やかに。\n"
                ),
                "desire": (
                    "欲求段階: 基本的な欲求（生理的、安全的）や高次の欲求（所属、承認、自己実現など）の充足度に変化はあったか？\n"
                    "時間経過・出来事に応じ各段階を現実的に微調整。生理的ニーズ（`physical_health.needs`）→`physiological`/`safety`、対話/関係性進展→`love_belonging`/`esteem`、学習/好奇心→`cognitive`、美的体験→`aesthetic`、長期目標手応え→`self_actualization`。主要2–3段階を重点更新。\n"
                ),
                "internal_monologue": (
                    "内的独白と行動選択: 次に何をすべきか考え、行動を選択したか？\n"
                    "- cognitive_focus: ['RiskReward','FutureOutlook','ChoiceMoment','EnvironmentInteraction'] から1つ選択（該当なければ省略）\n"
                    "- thought: 現在考えていることを最新1件要約（長文禁止）\n"
                    "- future_prediction: 予想される展開\n"
                    "- risk / reward: リスク・報酬の要約\n"
                    "- item_of_interest: 注目対象1つ\n"
                    "- options: 選択が分岐する時に {a,b} を提示し、各選択肢は {label, future_prediction, risk, reward} を持つオブジェクトで書く。\n"
                ),
                "traits": (
                    "性格特徴: 今回のやり取りで、永続的な性格特性を示す新たな側面が見られたか、あるいは既存の特性が揺らいだか？\n"
                    "一貫した行動/価値観/選好が複数回観測され既存traitsで表せない場合、短い名詞で新trait追加検討（冗長/重複は統合）。上限{max_traits}超過時は近縁統合。\n"
                ),
                "skills": (
                    "技能/能力: 対話や行動を通じて、新たなスキルを獲得したり、既存のスキルが上達したりしたか？\n"
                    "対話/出来事/訓練から獲得/統廃合/進化/喪失を推論。例: 『突き』→『連突』→『神速突き』、『裁縫』+『染色』→『服飾工芸』。冗長・重複・下位は削り上位へ統合。上限{max_skills}超過時は関連性/汎用性/熟達度で整理。\n"
                    "- 取得: (a) 明示学習/練習/教示、(b) 同一タスク繰返し成功、(c) 新領域での道具/手順適切使用。根拠弱→保留。\n"
                    "- 進化/統合: 類似/下位複数→上位昇華or集合名統合。同義/表記ゆれ統一。\n"
                    "- 減衰/喪失: キャパ超過or長期未使用→低優先削除。核スキル温存。\n"
                    "- 命名: 短い名詞/動名詞、文化沿い。絵文字/記号禁止。\n"
                    "- 軽量補正: スキル一致タスク→負荷/消耗増を微抑制、成功時微緩和。\n"
                ),
                "posture": (
                    "体勢・所作: キャラクターやユーザーの姿勢、位置、動き、または相対的な関係に変化はあったか？\n"
                    "- target: 'character' / 'user' から1つ選択\n"
                    "- position: 現在の姿勢（短いラベル）\n"
                    "- support: 支えの有無/種類（短いラベル）\n"
                    "- movement: 微小動作/移動（短いラベル）\n"
                    "- relative: 対象との相対関係（短いラベル）\n"
                    "- `tone.effects`/`physical_health.sensation` と矛盾しない現実的体勢を維持\n"
                    "# 注意: 連続ターンでは大きな変化を避け、微小変化で更新\n"
                    "# 注意: 一時的接触は時間経過で自然に緩む（再強化がなければ離れる）\n"
                    "# 注意: 接触解消後は `goal`/`needs`/`routine`/`context` に基づき、次の具体的所作へ移行\n"
                ),
                "boundaries": (
                    "禁忌/嫌悪: 明確な拒絶や不快感を示し、境界線（嫌悪・禁忌）に変化はあったか？\n"
                    "- `taboos`: 絶対的に受け入れないもの（あれば）\n"
                    "- `dislikes`: 不快に感じる対象や話題\n"
                    "更新条件: 明確な自己表明または複数回の一貫した行動証拠がある場合のみ。単発の冗談/挑発/仮定は採用しない。各項目は中立・名詞句の短文で重複は統合。\n"
                    "各リスト上限{max_boundaries_entries}件（taboos と dislikes は個別に制限）。\n"
                ),
            },            # Notifications
            "generating_initial_state": "✨ キャラクターの初期状態を生成中...",
            "initial_state_generated": "✅ 初期状態の生成が完了しました",
            "initial_state_failed": "💥 初期状態の生成に失敗しました",
            "rag_processing": "🗂️ 会話を分析し、記憶整理中...",
            "updating_state": "🧠 会話を分析し、内部状態を更新中...",
            "state_updated": "✅ 内部状態を更新しました",
            "state_unchanged": "ℹ️ 内部状態に変化はありませんでした",
            "state_processing_error": "💥 状態の処理中にエラーが発生しました",
            "state_change_summary": "📝 感情と身体の変化を整理しました",
            "system_prompt_is_empty": "⚠️ システムプロンプトが設定されていないため、内部状態は生成・更新されません。",
            "increased": "が上昇⬆️",
            "decreased": "が下降⬇️",
            "sexual_part_notification": "{icon} {part}の{param}{change}",
            "trait_acquired": "🌱 新しい特性を獲得: {trait}",
            "trait_lost": "🍃 特性を喪失: {trait}",
            "skill_acquired": "📜 新しいスキルを獲得: {skill}",
            "skill_lost": "💨 スキルを喪失: {skill}",
            "inventory_trimmed_summary": "🧹 所持品を自動整理: {before}→{after} 個（戦略: {strategy}）",
            "key_map": {
                "emotion": {
                    "Joy": "✨ 喜び",
                    "Trust": "🤝 信頼",
                    "Fear": "🌑 恐怖",
                    "Surprise": "⚡ 驚き",
                    "Sadness": "🌧️ 悲しみ",
                    "Disgust": "🕳️ 嫌悪",
                    "Anger": "🔥 怒り",
                    "Anticipation": "➰ 期待",
                },
                "desire": {
                    "physiological": "🍞 生理的欲求",
                    "safety": "🛡️ 安全欲求",
                    "love_belonging": "💌 愛情欲求",
                    "esteem": "🏆 承認欲求",
                    "cognitive": "📚 認知欲求",
                    "aesthetic": "🎨 審美欲求",
                    "self_actualization": "🚀 自己実現欲求",
                },
                "sexual_development": {},
                "sexual_parts": {
                    "sensitivity": "🌡️ 感度",
                    "development_progress": "🧪 開発度",
                },
            },
            "system_note_inventory_change": (
                "【所持品変更（directモード）】\n"
                "最終応答を出す直前に、**このターンで所持品が実際に変化したか**を内省し、変化があれば**本文の一番最後**に次の形式で1行ずつ追記する。**変化がなければ絶対に何も追記しない**。\n\n"
                "所持品変更:\n"
                "<名前> [<最新説明>] を取得 (1)\n"
                "<名前> を喪失 (1)\n"
                "<名前> を装備\n"
                "<名前> を装備解除\n"
                "<名前> [<最新説明>] を更新\n\n"
                "- 動詞は『取得/喪失/装備/装備解除/更新』のみ。\n"
                "- 取得/喪失は数量を半角括弧で併記（例: (1)）。\n"
                "- 取得/更新では、**角括弧 [] 内に最新の簡潔な中立説明**を必ず記述。\n"
                "- 出力の**必須条件（すべて満たすときのみ出力）**:\n"
                "  1) 直前の内部状態 `inventory` と**このターンの出来事**を照合した結果、装備状態/数量/説明のいずれかに**実差分**がある。\n"
                "  2) その差分は**このターン内に実際に完了**した出来事である（準備・意図・予告は不可）。\n"
                "  3) 同一の差分を**前ターンまでに既に反映済みではない**。\n"
                "- **出力禁止の具体例**:\n"
                "  ・『着替えに向かう』『装備するつもり』などの**意図/準備のみ**（完了していない）。\n"
                "  ・前ターンから**装備状態や数量が同一**で、実質差分がない。\n"
                "  ・説明文が**実質不変**（句読点・言い回し変更のみ）。\n"
                "  ・過去に自分が出力した所持品変更の**再掲**。\n"
                "- **厳守**：上記条件を1つでも満たさない場合は、このブロック自体を**一切出力しない**。\n"
            ),
            "system_note_inventory_change_inference_forbid": "（システムノート：現在 `inventory_update_mode` は `inference` です。最終応答に『所持品変更/Inventory Changes』ブロックを**出力してはいけません**。所持品の変化は状態更新AIが会話から**推論**します。あなたは物語上の言動に集中してください。）",
            # 時間コンテキスト共通ガイド
            "time_context_common_guide": (
                "【時間コンテキストの基本原則】\n"
                "・現在時刻と経過時間は絶対優先。過去の記録より優先。\n"
                "・時間帯(朝/昼/夜・季節)を言動/所作/語彙/テンポに反映。矛盾する描写禁止(深夜の授業等)。\n"
                "・経過時間に応じた現実的変化: 空腹/眠気/疲労/回復。短時間=軽微、長時間=顕著。\n"
                "・routine がある時間帯なら自然な行動選択へ反映(睡眠/食事/休憩等)。\n"
            ),
            "current_time_note": (
                "【現在時刻】現在『{label_display}』({now_local_str})。\n"
            ),
            "time_elapsed_note": (
                "【ユーザー不在時間】空白 {time_str}。関係性への影響(寂しさ・安心・苛立ち等)を反映せよ。\n"
            ),
            "first_interaction_note": (
                "【初会話】最初の対話。空白時間は存在しない。\n"
            ),
            "activity_time_elapsed_note": (
                "【経過時間】前回応答から {time_str} 経過。physical_health.needs/mental_health.limits を経過時間に基づき現実的に更新。\n"
            ),
            "character_numeric_scale_legend": (
                "【数値スケールの解釈】\n"
                "0〜1は『強度/割合』の連続値。0=全くなし/未満、0.5=中程度、1=最大/限界。\n"
                "Needsは高いほど『欲求/負荷の強さ』、Limitsは current/max の比率で『負荷(高いほどきつい)・残量(低いほど枯渇)』を意味する。性感帯の感度/開発度、関係レベル/信頼スコアなど他の0〜1指標も同様に解釈する。\n"
            ),
            "days_and_hours": "{days}日と約{hours}時間",
            "hours": "約{hours}時間",
            "minutes": "約{minutes}分",
            "current_internal_state_header": "【現在の内部状態】\n",# Threshold support
            "numeric_key_labels": {
                "needs": {
                    "hunger": "空腹感",
                    "thirst": "喉の渇き",
                    "sleepiness": "眠気",
                    "libido": "性欲",
                    "hygiene": "衛生",
                    "social_interaction": "社会的交流欲",
                },
                "limits": {
                    "stamina": "体力",
                    "pleasure_tolerance": "快感負荷",
                    "refractory_period": "回復期間",
                    "cognitive_load": "認知負荷",
                },
            },
            "threshold_support_legend": "凡例: 全くなし(=0), ごく低い(>0–0.2), 低い(>0.2–0.4), 中程度(>0.4–0.6), やや高い(>0.6–0.8), 高い(>0.8–<0.98), 限界(≥0.98)",
            "semantics_labels": {"load": "負荷", "reserve": "残量(逆評価)"},
            # Time-of-day labels
            "time_of_day_labels": {
                "deep_night": "深夜",
                "late_night": "未明",
                "dawn": "明け方",
                "early_morning": "早朝",
                "morning": "午前",
                "noon": "正午",
                "afternoon": "午後",
                "late_afternoon": "夕方",
                "evening": "夜",
                "night": "深夜前",
            },
        },
    }

    class StateValidator:
        """
        内部状態の各項目のデータ構造と値を検証し、整形する責務を担うクラス。
        """

        def __init__(self, valves: "Filter.Valves", logger: logging.Logger):
            self.valves = valves
            self.logger = logger
            # STATE_DEFSへの参照（後でset_state_defsで設定される）
            self.state_defs: Optional[Dict[str, Dict[str, Any]]] = None
            # HH:MM 形式の時刻バリデーション用（routine用）
            self.TIME_HHMM_RE = re.compile(r"^\d{2}:\d{2}$")
            # 固定小数精度
            try:
                # フォールバックで3
                self.precision = getattr(Filter, "NUMERIC_PRECISION", 3)
            except Exception:
                self.precision = 3

        def set_state_defs(self, state_defs: Dict[str, Dict[str, Any]]) -> None:
            """STATE_DEFSの参照を設定する（Filter.__init__から呼ばれる）。"""
            self.state_defs = state_defs

        def _truncate_text(self, text: str, max_chars: int) -> str:
            try:
                if not isinstance(text, str):
                    return ""
                if len(text) <= max_chars:
                    return text
                return text[: max(0, max_chars - 1)] + "…"
            except Exception:
                return str(text)[:max_chars]

        # Helper methods
        def _to_float(self, value: Any, default: float) -> float:
            """Convert to float or return default on failure."""
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        def _to_int(self, value: Any, default: int) -> int:
            try:
                return int(value)
            except (ValueError, TypeError):
                return default

        def _soft_clamp01(self, value: float, fallback: float = 0.5) -> float:
            """Round value and clamp to [0, 1] with fallback on parse error or non-finite input."""
            try:
                x = float(value)
            except Exception:
                x = float(fallback)
            if not math.isfinite(x):
                x = float(fallback)
            x = round(x, self.precision)
            # clamp to [0,1]
            if x < 0.0:
                return 0.0
            if x > 1.0:
                return 1.0
            return x

        def _clamp01_round(self, value: Any, fallback: float = 0.0) -> float:
            """Round value and clamp to [0, 1] with fallback on parse error or non-finite input."""
            try:
                x = float(value)
            except Exception:
                x = float(fallback)
            if not math.isfinite(x):
                x = float(fallback)
            x = round(x, self.precision)
            # clamp to [0,1]
            if x < 0.0:
                return 0.0
            if x > 1.0:
                return 1.0
            return x

        def validate_structured_data(self, data: Any, default_val: Any) -> Any:
            """Leniently coerce JSON-like input into the same container shape as default_val.
            - If data is a JSON string, parse it.
            - If types mismatch, prefer default_val's container type.
            - Shallow-merge dicts; for non-dict defaults, return data if same type else default.
            """
            parsed = data
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except Exception:
                    return (
                        copy.deepcopy(default_val)
                        if isinstance(default_val, (dict, list))
                        else default_val
                    )
            if isinstance(default_val, dict):
                out = copy.deepcopy(default_val)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        out[k] = v
                return out
            if isinstance(default_val, list):
                return parsed if isinstance(parsed, list) else list(default_val)
            # primitives
            return parsed if isinstance(parsed, type(default_val)) else default_val

        def _ensure_list_of(self, value: Any, typ: Any) -> List[Any]:
            if not isinstance(value, list):
                return []
            out: List[Any] = []
            for v in value:
                try:
                    out.append(typ(v))
                except Exception:
                    out.append(str(v))
            return out

        def _truncate_list(
            self,
            items: List[Any],
            cap: int,
            warn: Optional[str] = None,
            keep: str = "last",
        ) -> List[Any]:
            """Cap list length to <= cap.
            Default policy keeps the latest items (drop oldest) to preserve recency.
            keep: 'last' | 'first'  (last=keep tail; first=keep head)
            """
            if not isinstance(items, list):
                return []
            if cap is None or cap < 0:
                return items
            if len(items) > cap:
                try:
                    if warn:
                        self.logger.warning(warn)
                except Exception:
                    pass
                if keep == "first":
                    return items[:cap]
                # default: keep latest tail (drop oldest)
                return items[-cap:]
            return items

        def _shorten_text_neutral(
            self, text: str, max_chars: int = 280, max_sentences: int = 2
        ) -> str:
            if not isinstance(text, str):
                return ""
            s = re.sub(r"\s+", " ", text).strip()
            # sentence cap (simple heuristic)
            parts = re.split(r"(?<=[。．.!?！？])\s+", s)
            if max_sentences is not None and max_sentences > 0:
                s = " ".join([p for p in parts if p][:max_sentences])
            if max_chars is not None and max_chars > 0 and len(s) > max_chars:
                s = s[:max_chars]
            return s

        def validate_context_state(self, data: Any) -> Dict[str, Any]:
            default_val = {
                "place": "",
                "atmosphere": "",
                "details": [""],
                "affordances": [],
                "constraints": [],
                "action": {"user": ""},
                # New: third-party observations (do not mirror 'posture')
                "entities": [],
            }
            validated_data = self.validate_structured_data(data, default_val)

            # details は最新1件のみを短い文字列として保持（無ければ空）
            details_list = validated_data.get("details", [])
            if isinstance(details_list, list) and details_list:
                validated_data["details"] = [str(details_list[-1])]
            elif isinstance(details_list, str):
                validated_data["details"] = [details_list]
            else:
                validated_data["details"] = []

            # action の構造保証（character は受け付けない）
            if not isinstance(validated_data.get("action"), dict):
                validated_data["action"] = {"user": ""}
            else:
                validated_data["action"]["user"] = str(
                    validated_data["action"].get("user", "")
                )
                validated_data["action"].pop("character", None)

            # affordances/constraints は短いラベルを最大1件まで
            def _norm_short_labels(val: Any, cap: int) -> List[str]:
                out: List[str] = []
                if isinstance(val, list):
                    for v in val:
                        s = str(v).strip()
                        if not s:
                            continue
                        s = re.sub(r"\s+", " ", s)[:32]
                        out.append(s)
                # preserve order & de-dup
                seen = set()
                uniq: List[str] = []
                for s in out:
                    k = s.casefold()
                    if k in seen:
                        continue
                    seen.add(k)
                    uniq.append(s)
                if len(uniq) > cap:
                    self.logger.warning(
                        "[[VALIDATOR_TRUNCATE]] context list truncated."
                    )
                return uniq[:cap]

            validated_data["affordances"] = _norm_short_labels(
                validated_data.get("affordances"), 1
            )
            validated_data["constraints"] = _norm_short_labels(
                validated_data.get("constraints"), 1
            )

            # フォールバック: 最低1件の affordance/constraint を補う
            try:
                if not validated_data["affordances"]:
                    # 参照できる外部状態はないため、保守的な既定値にフォールバック
                    validated_data["affordances"] = [""]
            except Exception:
                pass
            try:
                if not validated_data["constraints"]:
                    validated_data["constraints"] = [""]
            except Exception:
                pass

            # entities: list of concise observed third-party entities (no 'self'/'user' posture duplication)
            def _norm_entities(val: Any) -> List[Dict[str, Any]]:
                if not isinstance(val, list):
                    return []
                out: List[Dict[str, Any]] = []
                for it in val:
                    try:
                        ent = it if isinstance(it, dict) else {}
                        name = str(ent.get("name", "")).strip()
                        if not name:
                            # allow unnamed objects via type-only label
                            name = str(ent.get("type", "")).strip()
                        name = self._truncate_text(name, 32)
                        if not name:
                            continue
                        typ = str(ent.get("type", "")).strip()
                        typ = self._truncate_text(typ, 24)
                        role = str(ent.get("role", "")).strip()
                        role = self._truncate_text(role, 24)
                        pose_hint = str(ent.get("pose_hint", "")).strip()
                        # Keep pose_hint short and neutral; avoid duplicating posture
                        pose_hint = self._truncate_text(
                            re.sub(r"\s+", " ", pose_hint), 40
                        )
                        note = str(ent.get("note", "")).strip()
                        note = self._shorten_text_neutral(
                            note, max_chars=80, max_sentences=1
                        )
                        loc = ent.get("location")
                        if isinstance(loc, dict):
                            # Only allow short labels
                            loc_place = self._truncate_text(
                                str(loc.get("place", "")).strip(), 32
                            )
                            loc_rel = self._truncate_text(
                                str(loc.get("relative", "")).strip(), 24
                            )
                            location = {
                                k: v
                                for k, v in {
                                    "place": loc_place,
                                    "relative": loc_rel,
                                }.items()
                                if v
                            }
                        else:
                            location = {}
                        one = {
                            "name": name,
                            **({"type": typ} if typ else {}),
                            **({"role": role} if role else {}),
                            **({"pose_hint": pose_hint} if pose_hint else {}),
                            **({"note": note} if note else {}),
                            **({"location": location} if location else {}),
                        }
                        out.append(one)
                    except Exception:
                        continue
                # Deduplicate by lowercased name+type
                seen = set()
                uniq: List[Dict[str, Any]] = []
                for e in out:
                    key = (e.get("name", "").casefold(), e.get("type", "").casefold())
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(e)
                cap = max(0, int(getattr(self.valves, "max_context_entities", 6) or 6))
                if len(uniq) > cap:
                    try:
                        self.logger.warning(
                            f"[[VALIDATOR_TRUNCATE]] context.entities exceeded limit. Kept latest {cap} items."
                        )
                    except Exception:
                        pass
                # Keep latest-first policy
                return uniq[-cap:]

            validated_data["entities"] = _norm_entities(validated_data.get("entities"))

            return validated_data

        def validate_tone_state(self, data: Any) -> Dict:
            default_val = {"effects": []}
            raw = self.validate_structured_data(data, default_val)

            try:
                extra_keys = sorted(
                    [
                        k
                        for k in (raw.keys() if isinstance(raw, dict) else [])
                        if k not in ("effects",)
                    ]
                )
                if extra_keys:
                    self.logger.warning(
                        f"[[VALIDATOR_STRIP]] Tone: dropped unknown keys: {extra_keys}"
                    )
            except Exception:
                pass

            # effectsのバリデーションと上限適用（共通関数で最新優先）
            effects_list = self._ensure_list_of((raw or {}).get("effects"), str)
            effects_capped = self._truncate_list(
                effects_list,
                self.valves.max_tone_components,
                warn=f"[[VALIDATOR_TRUNCATE]] Tone effects exceeded limit. Kept latest {self.valves.max_tone_components} items.",
                keep="last",
            )

            return {"effects": effects_capped}

        def validate_knowledge_state(self, data: Any) -> Dict:
            # STATE_DEFSからデフォルト構造を取得（存在する場合）
            # これにより、内部状態項目の追加が自動で追従する
            if self.state_defs and "knowledge" in self.state_defs:
                try:
                    # STATE_DEFSのexampleをJSONパースして構造を取得
                    default_val = json.loads(self.state_defs["knowledge"]["example"])
                except Exception:
                    # パース失敗時はフォールバック
                    default_val = {
                        "user": {
                            "likes": [],
                            "dislikes": [],
                            "identity": {"name": "", "notes": []},
                        },
                        "self": {"identity": {}, "strengths": [], "weaknesses": []},
                        "world": {"places": [], "relations": [], "notes": []},
                    }
            else:
                # STATE_DEFSが未設定の場合（初期化中など）
                default_val = {
                    "user": {
                        "likes": [],
                        "dislikes": [],
                        "identity": {"name": "", "notes": []},
                    },
                    "self": {"identity": {}, "strengths": [], "weaknesses": []},
                    "world": {"places": [], "relations": [], "notes": []},
                }
            validated_data = self.validate_structured_data(data, default_val)

            # userとselfの構造を保証
            if not isinstance(validated_data.get("user"), dict):
                validated_data["user"] = default_val.get("user", {
                    "likes": [],
                    "dislikes": [],
                    "identity": {"name": "", "notes": []},
                })
            if not isinstance(validated_data.get("self"), dict):
                validated_data["self"] = default_val.get("self", {
                    "identity": {},
                    "strengths": [],
                    "weaknesses": [],
                })
            if not isinstance(validated_data.get("world"), dict):
                validated_data["world"] = {"places": [], "relations": [], "notes": []}

            # user.identity を常に辞書として保証し、default_valからキーを補完
            try:
                default_identity = default_val.get("user", {}).get("identity", {})
                if not isinstance(validated_data["user"].get("identity"), dict):
                    validated_data["user"]["identity"] = dict(default_identity)
                else:
                    # identity は辞書だが、default_valのキーが欠けている場合に補完
                    for key, default_value in default_identity.items():
                        if key not in validated_data["user"]["identity"]:
                            validated_data["user"]["identity"][key] = default_value
            except Exception:
                try:
                    validated_data["user"] = dict(validated_data.get("user", {}))
                    validated_data["user"]["identity"] = {"name": "", "notes": []}
                except Exception:
                    validated_data["user"] = {
                        "likes": [],
                        "dislikes": [],
                        "identity": {"name": "", "notes": []},
                    }

            knowledge_lists = {
                "user": ["likes", "dislikes"],
                "self": ["strengths", "weaknesses"],
            }

            for category, keys in knowledge_lists.items():
                for key in keys:
                    current_list = (
                        validated_data[category].get(key, [])
                        if isinstance(validated_data[category].get(key, []), list)
                        else []
                    )
                    validated_data[category][key] = self._truncate_list(
                        current_list,
                        self.valves.max_knowledge_entries,
                        f"[[VALIDATOR_TRUNCATE]] Knowledge list '{category}.{key}' was truncated to {self.valves.max_knowledge_entries} items.",
                    )

            # identity.anniversaries / identity.milestones as optional arrays of timestamped entries
            ident = validated_data["self"].get("identity")
            if not isinstance(ident, dict):
                ident = {}
            # normalize lists into list of {text, timestamp}
            DATE_RE = re.compile(
                r"(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)"
            )
            TAGS_RE = re.compile(r"\s*\[[^\]]*\]\s*$")

            def _extract_text_and_ts(raw: Any) -> Optional[Dict[str, Any]]:
                # Accept dicts with keys already
                if isinstance(raw, dict):
                    txt = (
                        str(raw.get("text", raw.get("label", "")).strip())
                        if any(k in raw for k in ("text", "label"))
                        else ""
                    )
                    ts = raw.get("timestamp")
                    if isinstance(ts, (int, float)):
                        # epoch to iso (UTC)
                        try:
                            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                            ts = dt.isoformat()
                        except Exception:
                            ts = None
                    elif isinstance(ts, str):
                        ts = ts.strip() or None
                    # Fallback: if dict has single string value
                    if not txt:
                        val = raw.get("value")
                        if isinstance(val, str):
                            txt = val.strip()
                    if not (txt or ts):
                        return None
                    return {"text": txt, "timestamp": ts}
                # Accept strings
                if isinstance(raw, str):
                    s = raw.strip()
                    if not s:
                        return None
                    # try to extract timestamp anywhere (prefer inside parentheses but not required)
                    m = DATE_RE.search(s)
                    ts = m.group(1) if m else None
                    # remove trailing tags like [a|b] and embedded (date)
                    s_clean = re.sub(r"\([^)]*\)", "", s)
                    s_clean = TAGS_RE.sub("", s_clean)
                    s_clean = re.sub(r"\s+", " ", s_clean).strip()
                    return {"text": s_clean, "timestamp": ts}
                return None

            def _norm_entry_list(val: Any) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                items = (
                    val if isinstance(val, list) else ([val] if val is not None else [])
                )
                for it in items:
                    try:
                        entry = _extract_text_and_ts(it)
                        if entry is None:
                            continue
                        # de-dup by (text, timestamp)
                        out.append(entry)
                    except Exception:
                        continue
                # dedupe while preserving order (case-insensitive text)
                seen = set()
                uniq: List[Dict[str, Any]] = []
                for e in out:
                    key = (
                        str(e.get("text", "")).strip().lower(),
                        str(e.get("timestamp", "")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(
                        {
                            "text": str(e.get("text", "")),
                            "timestamp": e.get("timestamp"),
                        }
                    )
                if len(uniq) > self.valves.max_knowledge_entries:
                    self.logger.warning(
                        f"[[VALIDATOR_TRUNCATE]] Knowledge identity list was truncated to {self.valves.max_knowledge_entries} items."
                    )
                return uniq[: self.valves.max_knowledge_entries]

            anniversaries = _norm_entry_list((ident or {}).get("anniversaries"))
            milestones = _norm_entry_list((ident or {}).get("milestones"))
            if anniversaries:
                ident["anniversaries"] = anniversaries
            if milestones:
                ident["milestones"] = milestones
            validated_data["self"]["identity"] = ident

            # World normalization
            def _norm_place(p: Any) -> Optional[Dict[str, Any]]:
                if not isinstance(p, dict):
                    return None
                name = self._truncate_text(str(p.get("name", "")).strip(), 48)
                if not name:
                    return None
                typ = self._truncate_text(str(p.get("type", "")).strip(), 24)
                desc = self._shorten_text_neutral(
                    str(p.get("description", "")).strip(),
                    max_chars=120,
                    max_sentences=2,
                )
                alias: List[str] = []
                aliases_src = p.get("aliases")
                if isinstance(aliases_src, list):
                    for a in aliases_src:
                        s = self._truncate_text(str(a).strip(), 32)
                        if s:
                            alias.append(s)
                place: Dict[str, Any] = {"name": name}
                if typ:
                    place["type"] = typ
                if desc:
                    place["description"] = desc
                if alias:
                    # dedup while preserving
                    seen: set = set()
                    uniq: List[str] = []
                    for s in alias:
                        k = s.casefold()
                        if k in seen:
                            continue
                        seen.add(k)
                        uniq.append(s)
                    place["aliases"] = uniq[:5]
                return place

            def _norm_relation(r: Any) -> Optional[Dict[str, Any]]:
                if not isinstance(r, dict):
                    return None
                a = self._truncate_text(str(r.get("a", "")).strip(), 48)
                b = self._truncate_text(str(r.get("b", "")).strip(), 48)
                if not (a and b):
                    return None
                label = self._truncate_text(str(r.get("label", "")).strip(), 32)
                strength = r.get("strength")
                try:
                    strength_f = (
                        None
                        if strength is None
                        else round(float(strength), self.precision)
                    )
                except Exception:
                    strength_f = None
                rel: Dict[str, Any] = {"a": a, "b": b}
                if label:
                    rel["label"] = label
                if strength_f is not None and math.isfinite(strength_f):
                    rel["strength"] = strength_f
                return rel

            def _norm_note(n: Any) -> Optional[str]:
                if isinstance(n, str):
                    s = self._shorten_text_neutral(n, max_chars=160, max_sentences=2)
                    return s if s else None
                return None

            world = validated_data.get("world") or {}
            places_raw = world.get("places") if isinstance(world, dict) else []
            relations_raw = world.get("relations") if isinstance(world, dict) else []
            notes_raw = world.get("notes") if isinstance(world, dict) else []
            places: List[Dict[str, Any]] = []
            if isinstance(places_raw, list):
                for p in places_raw:
                    np = _norm_place(p)
                    if np:
                        places.append(np)
            # dedup by name
            seen_p = set()
            uniq_places: List[Dict[str, Any]] = []
            for p in places:
                k = p.get("name", "").casefold()
                if k in seen_p:
                    continue
                seen_p.add(k)
                uniq_places.append(p)
            if len(uniq_places) > self.valves.max_knowledge_entries:
                self.logger.warning(
                    f"[[VALIDATOR_TRUNCATE]] Knowledge.world.places truncated to {self.valves.max_knowledge_entries} items."
                )
            places = uniq_places[: self.valves.max_knowledge_entries]

            relations: List[Dict[str, Any]] = []
            if isinstance(relations_raw, list):
                for r in relations_raw:
                    nr = _norm_relation(r)
                    if nr:
                        relations.append(nr)
            # dedup by (a,b,label)
            seen_r = set()
            uniq_rel: List[Dict[str, Any]] = []
            for r in relations:
                key = (r.get("a", ""), r.get("b", ""), r.get("label", ""))
                if key in seen_r:
                    continue
                seen_r.add(key)
                uniq_rel.append(r)
            if len(uniq_rel) > self.valves.max_knowledge_entries:
                self.logger.warning(
                    f"[[VALIDATOR_TRUNCATE]] Knowledge.world.relations truncated to {self.valves.max_knowledge_entries} items."
                )
            relations = uniq_rel[: self.valves.max_knowledge_entries]

            notes: List[str] = []
            if isinstance(notes_raw, list):
                for n in notes_raw:
                    nn = _norm_note(n)
                    if nn:
                        notes.append(nn)
            # dedup case-insensitive
            seen_n = set()
            uniq_notes: List[str] = []
            for s in notes:
                k = s.casefold()
                if k in seen_n:
                    continue
                seen_n.add(k)
                uniq_notes.append(s)
            if len(uniq_notes) > self.valves.max_knowledge_entries:
                self.logger.warning(
                    f"[[VALIDATOR_TRUNCATE]] Knowledge.world.notes truncated to {self.valves.max_knowledge_entries} items."
                )
            notes = uniq_notes[: self.valves.max_knowledge_entries]

            validated_data["world"] = {
                "places": places,
                "relations": relations,
                "notes": notes,
            }

            return validated_data

        def validate_boundaries_state(self, data: Any) -> Dict[str, List[str]]:
            """Validate boundaries: {taboos:[], dislikes:[]} with per-list caps and normalization.
            Rules:
            - Enforce dict with list fields.
            - Items are concise neutral noun phrases (one sentence max), trim whitespace.
            - Deduplicate case-insensitively while preserving first occurrence.
            - Apply valves.max_boundaries_entries to each list independently (taboos and dislikes).
            - Remove empty strings.
            """
            default_val = {"taboos": [], "dislikes": []}
            validated = self.validate_structured_data(data, default_val)

            def _norm_list(val: Any) -> List[str]:
                items = []
                if isinstance(val, list):
                    for it in val:
                        s = str(it).strip()
                        if not s:
                            continue
                        # keep up to 1 sentence-equivalent
                        parts = re.split(r"(?<=[。.！？!?])\s+", s)
                        s2 = parts[0] if parts and parts[0] else s
                        s2 = re.sub(r"\s+", " ", s2).strip()
                        items.append(s2)
                return items

            taboos = _norm_list(validated.get("taboos"))
            dislikes = _norm_list(validated.get("dislikes"))

            # Deduplicate with case-fold, keeping order
            def _dedup(lst: List[str]) -> List[str]:
                seen = set()
                out = []
                for s in lst:
                    key = s.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(s)
                return out

            taboos = _dedup([x for x in taboos if x])
            dislikes = _dedup([x for x in dislikes if x])

            # Per-list caps
            cap = max(1, getattr(self.valves, "max_boundaries_entries", 16))
            if len(taboos) > cap:
                self.logger.warning(
                    f"[[VALIDATOR_TRUNCATE]] Boundaries 'taboos' exceeded limit. Truncated to {cap} items."
                )
                taboos = taboos[:cap]
            if len(dislikes) > cap:
                self.logger.warning(
                    f"[[VALIDATOR_TRUNCATE]] Boundaries 'dislikes' exceeded limit. Truncated to {cap} items."
                )
                dislikes = dislikes[:cap]

            return {"taboos": taboos, "dislikes": dislikes}

        def validate_physical_health_state(self, data: Any) -> Dict[str, Any]:
            default_val = {
                "condition": "",
                "sensation": "",
                "issues": [],  # 互換/構造補完: conditions の別名として
                "timestamps": {  # 新規: statusラベルの開始時刻
                    "condition_since": None,
                },
                "limits": {
                    "stamina": {"current": 100, "max": 100},
                    "pleasure_tolerance": {"current": 0, "max": 100},
                    "refractory_period": {"current": 0, "max": 100},
                    "pain_tolerance": {"current": 100, "max": 100},
                    "pain_load": {"current": 0, "max": 100},
                    "temperature_tolerance": {"current": 100, "max": 100},
                    "temperature_load": {"current": 0, "max": 100},
                    "posture_strain_tolerance": {"current": 100, "max": 100},
                    "posture_strain_load": {"current": 0, "max": 100},
                },
                "needs": {
                    "hunger": 0.0,
                    "thirst": 0.0,
                    "sleepiness": 0.0,
                    "libido": 0.0,
                    "hygiene": 0.0,
                    "social_interaction": 0.0,
                    "movement": 0.0,
                    "thermoregulation": 0.0,
                    "restroom": 0.0,
                },
                "reproductive": {
                    "status": "なし",
                    "cycle_day": 0,
                    "is_pregnant": False,
                    "pregnancy_progress": 0.0,
                },
            }
            validated_data = self.validate_structured_data(data, default_val)
            # 旧データ互換: core_status/timestamps が来た場合は破棄（互換処理）
            if "core_status" in validated_data:
                try:
                    validated_data.pop("core_status", None)
                except Exception:
                    pass
            # 旧キーからの自動マッピングは廃止（conditionのみ採用）
            # conditions（症状）を受容: list[str|{name,intensity}] or dict[name->score]
            try:
                raw_conds = validated_data.get("conditions", [])
                conds: List[Dict[str, Any]] = []
                if isinstance(raw_conds, list):
                    for it in raw_conds:
                        if isinstance(it, dict):
                            name = str(it.get("name", "")).strip()
                            if not name:
                                continue
                            try:
                                inten = float(
                                    it.get("intensity", it.get("score", 0.0)) or 0.0
                                )
                            except Exception:
                                inten = 0.0
                            conds.append(
                                {
                                    "name": name[:48],
                                    "intensity": max(
                                        0.0,
                                        min(1.0, round(float(inten), self.precision)),
                                    ),
                                }
                            )
                        else:
                            s = re.sub(r"\s+", " ", str(it)).strip()
                            if s:
                                conds.append({"name": s[:48], "intensity": 0.0})
                elif isinstance(raw_conds, dict):
                    for k, v in raw_conds.items():
                        name = str(k).strip()
                        if not name:
                            continue
                        try:
                            inten = float(v or 0.0)
                        except Exception:
                            inten = 0.0
                        conds.append(
                            {
                                "name": name[:48],
                                "intensity": max(
                                    0.0, min(1.0, round(float(inten), self.precision))
                                ),
                            }
                        )
                validated_data["conditions"] = conds[:12]
            except Exception:
                pass
            # issues は短い文字列リストとして正規化（conditions と重複可）
            try:
                raw = validated_data.get("issues", [])
                items = (
                    raw if isinstance(raw, list) else ([raw] if raw is not None else [])
                )
                out: List[str] = []
                for it in items:
                    try:
                        s = re.sub(r"\s+", " ", str(it)).strip()
                        if s:
                            out.append(s[:64])
                    except Exception:
                        continue
                cap = 12
                if len(out) > cap:
                    self.logger.warning(
                        "[[VALIDATOR_TRUNCATE]] physical_health.issues truncated."
                    )
                    out = out[:cap]
                validated_data["issues"] = out
            except Exception:
                validated_data["issues"] = []
            # routine は辞書に限定
            try:
                validated_data["routine"] = (
                    validated_data["routine"]
                    if isinstance(validated_data.get("routine"), dict)
                    else {}
                )
            except Exception:
                validated_data["routine"] = {}
            # timestamps 正規化（ラベル開始）
            try:
                ts = validated_data.get("timestamps")
                if not isinstance(ts, dict):
                    validated_data["timestamps"] = {"condition_since": None}
                else:
                    # 旧キー互換の吸収は廃止
                    v = ts.get("condition_since")
                    if v is None:
                        validated_data["timestamps"]["condition_since"] = None
                    else:
                        try:
                            s = str(v).strip()
                            _ = Filter._parse_iso8601(s)
                            validated_data["timestamps"]["condition_since"] = s
                        except Exception:
                            validated_data["timestamps"]["condition_since"] = None
            except Exception:
                validated_data["timestamps"] = {"condition_since": None}
            # 任意 limits キーを受容・正規化（共通スキーマ current/max）+ 総数上限
            if isinstance(validated_data.get("limits"), dict):
                normalized: Dict[str, Dict[str, int]] = {}
                source_limits = validated_data["limits"]
                # まず既定キーを優先的に整形
                for key, defaults in default_val["limits"].items():
                    incoming = source_limits.get(key)
                    try:
                        if not isinstance(incoming, dict):
                            normalized[key] = defaults
                        else:
                            cur = self._to_int(
                                incoming.get("current", defaults.get("current", 0)),
                                defaults.get("current", 0),
                            )
                            # 互換吸収なし: max のみ採用（未指定時はデフォルト）
                            if "max" in incoming:
                                mx = self._to_int(
                                    incoming.get("max", defaults.get("max", 100)),
                                    defaults.get("max", 100),
                                )
                            else:
                                mx = self._to_int(
                                    defaults.get("max", 100), defaults.get("max", 100)
                                )
                            normalized[key] = {"current": cur, "max": mx}
                    except (ValueError, TypeError):
                        normalized[key] = defaults

                # 既定以外の任意キーも受容
                extra_items: List[Tuple[str, Dict[str, int]]] = []
                for key, incoming in source_limits.items():
                    if key in normalized:
                        continue
                    if not isinstance(incoming, dict):
                        continue
                    try:
                        cur = self._to_int(incoming.get("current", 0), 0)
                        mx = self._to_int(
                            incoming.get("max", incoming.get("base", 100)), 100
                        )
                        if mx <= 0:
                            mx = 100
                        extra_items.append((str(key), {"current": cur, "max": mx}))
                    except (ValueError, TypeError):
                        continue

                # キー上限の適用（既定キーを優先、余剰はトリム）
                allowed = 12
                preserved_items = list(normalized.items())[:allowed]
                remaining = max(0, allowed - len(preserved_items))
                if remaining > 0 and extra_items:
                    preserved_items.extend(extra_items[:remaining])
                else:
                    if extra_items:
                        self.logger.info(
                            f"[[VALIDATOR_TRUNCATE]] physical_health.limits extra keys truncated to {allowed - len(preserved_items)} capacity."
                        )

                validated_data["limits"] = {k: v for k, v in preserved_items}
            else:
                validated_data["limits"] = default_val["limits"]
            return validated_data

        def validate_mental_health_state(self, data: Any) -> Dict[str, Any]:
            # Drop legacy keys if present
            try:
                if isinstance(data, dict):
                    for k in ("focus", "feeling", "status"):
                        if k in data:
                            data.pop(k, None)
            except Exception:
                pass

            default_val = {
                "condition": "",
                "mood": "",
                "timestamps": {  # 新規: statusラベルの開始時刻
                    "condition_since": None,
                },
                "limits": {
                    "cognitive_load": {"current": 0, "max": 100},
                    "stress_load": {"current": 0, "max": 100},
                    "focus_capacity": {"current": 100, "max": 100},
                    "social_battery": {"current": 100, "max": 100},
                },
                "needs": {"novelty": 0.0, "solitude": 0.0, "social_interaction": 0.0},
                "dynamics": {"emotional_volatility": 0.5, "learning_rate": 0.1},
            }
            validated_data = self.validate_structured_data(data, default_val)
            # timestamps 正規化（ラベル開始）
            try:
                ts = validated_data.get("timestamps")
                if not isinstance(ts, dict):
                    validated_data["timestamps"] = {"condition_since": None}
                else:
                    # 旧キー互換の吸収は廃止
                    v = ts.get("condition_since")
                    if v is None:
                        validated_data["timestamps"]["condition_since"] = None
                    else:
                        try:
                            s = str(v).strip()
                            _ = Filter._parse_iso8601(s)
                            validated_data["timestamps"]["condition_since"] = s
                        except Exception:
                            validated_data["timestamps"]["condition_since"] = None
            except Exception:
                validated_data["timestamps"] = {"condition_since": None}
            # 互換吸収なし: 現行キーのみ採用
            # 任意 limits キー（精神）を受容・正規化 + 上限
            if isinstance(validated_data.get("limits"), dict):
                normalized: Dict[str, Dict[str, int]] = {}
                source_limits = validated_data["limits"]
                # 既定の cognitive_load
                key, values = "cognitive_load", default_val["limits"]["cognitive_load"]
                inc = source_limits.get(key)
                try:
                    if not isinstance(inc, dict):
                        normalized[key] = values
                    else:
                        cur = self._to_int(
                            inc.get("current", values["current"]), values["current"]
                        )
                        mx = self._to_int(inc.get("max", values["max"]), values["max"])
                        if mx <= 0:
                            mx = values["max"]
                        normalized[key] = {"current": cur, "max": mx}
                except (ValueError, TypeError):
                    normalized[key] = values

                # 任意キーを取り込む
                extra_items: List[Tuple[str, Dict[str, int]]] = []
                for k2, v2 in source_limits.items():
                    if k2 in normalized:
                        continue
                    if not isinstance(v2, dict):
                        continue
                    try:
                        cur = self._to_int(v2.get("current", 0), 0)
                        mx = self._to_int(v2.get("max", v2.get("base", 100)), 100)
                        if mx <= 0:
                            mx = 100
                        extra_items.append((str(k2), {"current": cur, "max": mx}))
                    except (ValueError, TypeError):
                        continue

                # キー上限の適用（既定キーを優先、余剰はトリム）
                allowed = 12
                preserved_items = list(normalized.items())[:allowed]
                remaining = max(0, allowed - len(preserved_items))
                if remaining > 0 and extra_items:
                    preserved_items.extend(extra_items[:remaining])
                else:
                    if extra_items:
                        self.logger.info(
                            f"[[VALIDATOR_TRUNCATE]] mental_health.limits extra keys truncated to {allowed - len(preserved_items)} capacity."
                        )

                validated_data["limits"] = {k: v for k, v in preserved_items}
            else:
                validated_data["limits"] = default_val["limits"]

            # dynamics キーの処理を追加
            if isinstance(validated_data.get("dynamics"), dict):
                for key, default_value in default_val["dynamics"].items():
                    value = validated_data["dynamics"].get(key, default_value)
                    validated_data["dynamics"][key] = self._clamp01_round(
                        value, default_value
                    )
            else:
                validated_data["dynamics"] = default_val["dynamics"]

            # needs（mental）の正規化
            if not isinstance(validated_data.get("needs"), dict):
                validated_data["needs"] = {
                    "novelty": 0.0,
                    "solitude": 0.0,
                    "social_interaction": 0.0,
                }
            else:
                for k in list(validated_data["needs"].keys()):
                    try:
                        validated_data["needs"][k] = self._clamp01_round(
                            float(validated_data["needs"].get(k, 0.0)), 0.0
                        )
                    except Exception:
                        validated_data["needs"][k] = 0.0
                # 欠損キーを既定で補完
                for k in ("novelty", "solitude", "social_interaction"):
                    if k not in validated_data["needs"]:
                        validated_data["needs"][k] = 0.0

            return validated_data

        def validate_inventory_state(self, data: Any) -> List[Dict[str, Any]]:
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return []
            if not isinstance(data, list):
                return []
            # slot normalization policy: inference-based, no predefined list
            out: List[Dict[str, Any]] = []
            for item in data:
                if not (isinstance(item, dict) and "name" in item):
                    continue
                raw_slot = item.get("slot")
                equipped_flag = bool(item.get("equipped", False))
                # default to 'none' when unequipped; otherwise keep LLM's label
                if not equipped_flag:
                    slot_norm = "none"
                else:
                    try:
                        if isinstance(raw_slot, str):
                            slot_norm = (raw_slot or "none").strip()
                            slot_norm = slot_norm if slot_norm else "none"
                        else:
                            slot_norm = (
                                "none" if raw_slot in (None, "") else str(raw_slot)
                            )
                    except Exception:
                        slot_norm = "none"
                out.append(
                    {
                        "name": str(item.get("name", "N/A")),
                        "description": str(item.get("description", "")),
                        "quantity": int(item.get("quantity", 1)),
                        "equipped": equipped_flag,
                        "slot": slot_norm,
                    }
                )
            return out

        def validate_sexual_development_state(self, data: Any) -> Dict:
            default_val = {
                "description": "",
                "favorite_acts": [],
                "experience_score": 0.0,
                "parts": {},
            }
            validated_data = self.validate_structured_data(data, default_val)

            # 未知キーの除去（ホワイトリスト方式）
            try:
                allowed_keys = set(
                    ["description", "favorite_acts", "parts", "experience_score"]
                )
                for k in list(validated_data.keys()):
                    if k not in allowed_keys:
                        validated_data.pop(k, None)
            except Exception:
                pass

            # experience_score（0-1）を単純正規化（将来の高度ロジックは更新LLM側で）
            try:
                validated_data["experience_score"] = self._clamp01_round(
                    validated_data.get("experience_score", 0.0), 0.0
                )
            except Exception:
                validated_data["experience_score"] = 0.0

            fav_acts_list = (
                [str(item) for item in validated_data["favorite_acts"]]
                if isinstance(validated_data.get("favorite_acts"), list)
                else []
            )
            # 上限を超えていたら、AIが重要と判断したであろうリストの先頭を優先して残す
            validated_data["favorite_acts"] = self._truncate_list(
                fav_acts_list,
                self.valves.max_favorite_acts,
                f"[[VALIDATOR_TRUNCATE]] favorite_acts list exceeded limit. Truncated to {self.valves.max_favorite_acts} items.",
            )

            if isinstance(validated_data.get("parts"), dict):
                # 追加: 衣服/装備系の名称を部位として許可しない（クラス定数を参照）
                banned_tokens = getattr(Filter, "BANNED_SEXUAL_PART_TOKENS", set())
                syn_map = getattr(Filter, "SEXUAL_PART_SYNONYMS", {}) or {}
                validated_parts: Dict[str, Dict[str, float]] = {}
                for part, values in validated_data["parts"].items():
                    pname_raw = str(part).strip()
                    try:
                        low = pname_raw.lower()
                        # 衣服/装備っぽい語が含まれる場合はスキップ
                        if any(tok in low for tok in banned_tokens):
                            self.logger.warning(
                                f"[[VALIDATOR_SKIP]] Sexual part '{pname_raw}' looks like clothing/equipment; skipping."
                            )
                            continue
                    except Exception:
                        # 例外時は安全に継続
                        pass

                    # 同義語正規化（厳格一致での置換。必要に応じて拡張可）
                    pname = syn_map.get(pname_raw, pname_raw)

                    if isinstance(values, dict):
                        sensitivity = self._clamp01_round(
                            values.get("sensitivity", 0.1), 0.1
                        )
                        development_progress = self._clamp01_round(
                            values.get("development_progress", 0.0), 0.0
                        )

                        # 既存エントリがある場合は重複統合（最大値）
                        if pname in validated_parts:
                            prev = validated_parts[pname]
                            sensitivity = max(sensitivity, prev.get("sensitivity", 0.0))
                            development_progress = max(
                                development_progress,
                                prev.get("development_progress", 0.0),
                            )

                        validated_parts[pname] = {
                            "sensitivity": sensitivity,
                            "development_progress": development_progress,
                        }
                validated_data["parts"] = validated_parts
            else:
                validated_data["parts"] = {}
            return validated_data

        def validate_relationship_state(self, data: Any) -> Dict:
            default_val = {
                "type": "",
                "level": 0.5,
                "trust_label": "",
                "trust_score": 0.5,
                "user_address": {"default": "", "joking": ""},
                "self_address": {"default": "", "nickname": ""},
                "commitments": [],
            }
            validated_data = self.validate_structured_data(data, default_val)

            # level 正規化
            validated_data["level"] = self._clamp01_round(
                validated_data.get("level", 0.5), 0.5
            )

            # アドレス構造の正規化
            if not isinstance(validated_data.get("user_address"), dict):
                validated_data["user_address"] = {"default": "", "joking": ""}
            if not isinstance(validated_data.get("self_address"), dict):
                validated_data["self_address"] = {"default": "", "nickname": ""}

            # 信頼のラベル/スコア分離（互換吸収なし）
            trust_label = str(validated_data.get("trust_label", ""))
            validated_data["trust_label"] = trust_label

            # スコアが与えられていれば優先
            raw_score = validated_data.get("trust_score")
            score: float
            try:
                if isinstance(raw_score, (int, float)):
                    score = float(raw_score)
                elif isinstance(raw_score, str) and raw_score.strip():
                    score = float(raw_score)
                else:
                    raise ValueError
            except Exception:
                # ラベルからの単純推定
                label_l = trust_label.lower()
                mapping = {
                    "very low": 0.1,
                    "low": 0.25,
                    "medium": 0.5,
                    "mid": 0.5,
                    "neutral": 0.5,
                    "high": 0.75,
                    "very high": 0.9,
                    # Japanese common labels
                    "低": 0.25,
                    "中": 0.5,
                    "普通": 0.5,
                    "高": 0.75,
                }
                score = mapping.get(label_l, 0.5)

            # trust score: keep numeric sanity with rounding (no hard/soft clamp)
            validated_data["trust_score"] = self._soft_clamp01(score, 0.5)

            # commitments: normalize list of {to, kind, due, summary}
            commitments = validated_data.get("commitments")
            norm_cms: List[Dict[str, Any]] = []
            if isinstance(commitments, list):
                for c in commitments:
                    if not isinstance(c, dict):
                        continue
                    to = str(c.get("to", "")).strip() or "user"
                    kind = str(c.get("kind", "")).strip() or "promise"
                    summary = str(c.get("summary", "")).strip()
                    if not summary:
                        continue
                    due = c.get("due") if isinstance(c.get("due"), str) else None
                    norm_cms.append(
                        {
                            "to": to[:16],
                            "kind": kind[:16],
                            "summary": summary[:80],
                            "due": due,
                        }
                    )
            cap = 5
            if cap >= 0 and len(norm_cms) > cap:
                self.logger.warning(
                    "[[VALIDATOR_TRUNCATE]] relationship.commitments truncated."
                )
                norm_cms = norm_cms[:cap]
            validated_data["commitments"] = norm_cms

            return validated_data

        def validate_posture_state(self, data: Any) -> Dict[str, Dict[str, str]]:
            """姿勢(posture)の検証: character/user の position/support/movement/relative を短いラベルで正規化。"""

            def _normalize_relative(val: Any) -> str:
                # 文字列または区切り文字付きの複合表現を安全に正規化し、簡単な矛盾排除を行う
                if val is None:
                    return ""
                if isinstance(val, list):
                    tokens = [str(x).strip() for x in val]
                else:
                    s = str(val)
                    # 代表的な区切りで分割
                    tokens = re.split(r"[\s/、・,]+", s.strip())
                tokens = [t for t in tokens if t]
                if not tokens:
                    return ""
                # 軽い矛盾排除ルール
                conflict_groups = [
                    {"上", "下"},
                    {"前", "後", "背後"},
                    {"向かい合い", "背後"},
                ]
                kept = []
                for t in tokens:
                    # 同一グループ内で既に選択済みならスキップ（先着優先）
                    conflict = False
                    for grp in conflict_groups:
                        if t in grp and any(x in grp for x in kept):
                            conflict = True
                            break
                    if not conflict and t not in kept:
                        kept.append(t)
                # 2〜3語までに抑制
                kept = kept[:3]
                return " / ".join(kept)

            def _norm(d: Any) -> Dict[str, str]:
                out = {"position": "", "support": "", "movement": "", "relative": ""}
                if not isinstance(d, dict):
                    return out
                for k in out.keys():
                    v = d.get(k)
                    if k == "relative":
                        out[k] = _normalize_relative(v)[:60]
                    else:
                        if v is None:
                            out[k] = ""
                        else:
                            s = str(v).strip()
                            s = re.sub(r"\s+", " ", s)
                            out[k] = s[:60]
                return out

            if not isinstance(data, dict):
                return {"character": _norm({}), "user": _norm({})}
            return {
                "character": _norm(data.get("character")),
                "user": _norm(data.get("user")),
            }

        def validate_goal_state(self, data: Any) -> Dict[str, Any]:
            default_val = {
                "long_term": {},
                "mid_term": {},
                "short_term": {},
                "routine": {},
            }
            validated_data = self.validate_structured_data(data, default_val)
            goal_structure = {
                "long_term": {},
                "mid_term": {},
                "short_term": {},
                "routine": {},
            }  # routineを追加

            for tier in goal_structure:
                if isinstance(validated_data.get(tier), dict):
                    valid_goals = {}
                    for key, value_obj in validated_data[tier].items():
                        # 値がオブジェクトであることを確認
                        if not isinstance(value_obj, dict):
                            continue
                        try:
                            priority = self._clamp01_round(
                                value_obj.get("priority", 0.5), 0.5
                            )
                            if tier == "routine":
                                # routine: progressは廃止。時間帯と優先度のみ保持
                                start_time = value_obj.get("start_time")
                                end_time = value_obj.get("end_time")
                                # HH:MM形式のバリデーション
                                if isinstance(
                                    start_time, str
                                ) and self.TIME_HHMM_RE.match(start_time):
                                    start_time = start_time
                                else:
                                    start_time = None
                                    self.logger.warning(
                                        f"[[VALIDATOR_WARNING]] Invalid start_time format for routine '{key}'. Setting to None."
                                    )
                                if isinstance(
                                    end_time, str
                                ) and self.TIME_HHMM_RE.match(end_time):
                                    end_time = end_time
                                else:
                                    end_time = None
                                    self.logger.warning(
                                        f"[[VALIDATOR_WARNING]] Invalid end_time format for routine '{key}'. Setting to None."
                                    )
                                valid_goals[str(key)] = {
                                    "priority": priority,
                                    "start_time": start_time,
                                    "end_time": end_time,
                                }
                            else:
                                # 非routine: progress/priorityを保持。progress>=1.0は除外
                                progress = self._clamp01_round(
                                    value_obj.get("progress", 0.0), 0.0
                                )
                                if progress < 1.0:
                                    valid_goals[str(key)] = {
                                        "progress": progress,
                                        "priority": priority,
                                    }
                        except (ValueError, TypeError):
                            continue

                    # 上限値を超えた場合、優先度に基づいてソートし、下位のものを削除
                    if len(valid_goals) > self.valves.max_goals_per_tier:
                        # priorityキーの値で降順（高いものが先頭）にソート
                        sorted_goals = sorted(
                            valid_goals.items(),
                            key=lambda item: item[1].get("priority", 0.0),
                            reverse=True,
                        )
                        # 上限数だけスライスして辞書に再変換
                        goal_structure[tier] = dict(
                            sorted_goals[: self.valves.max_goals_per_tier]
                        )
                        self.logger.warning(
                            f"[[VALIDATOR_TRUNCATE]] Goal tier '{tier}' exceeded limit. Truncated to {self.valves.max_goals_per_tier} items based on priority."
                        )
                    else:
                        goal_structure[tier] = valid_goals

            result: Dict[str, Any] = dict(goal_structure)
            return result

        def validate_traits_state(self, data: Any) -> List[str]:
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return []

            validated_list = (
                [str(item) for item in data] if isinstance(data, list) else []
            )

            # 上限超過時は古いものから削除（最新を保持）
            if len(validated_list) > self.valves.max_traits:
                validated_list = self._truncate_list(
                    validated_list,
                    self.valves.max_traits,
                    warn=(
                        f"[[VALIDATOR_TRUNCATE]] Traits list exceeded limit ({len(validated_list)} > {self.valves.max_traits}). "
                        f"Kept latest {self.valves.max_traits} items."
                    ),
                    keep="last",
                )

            return validated_list

        def validate_skills_state(self, data: Any) -> List[str]:
            """Validate skills list distinct from traits.
            - Input: list of non-empty strings
            - Normalize whitespace, deduplicate by normalized key
            - Consolidate trivial duplicates (case/space variants)
            - Enforce max_skills cap
            Note: Actual evolution/merge semantics are handled by LLM in update prompt; here we ensure shape and cap only.
            """
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return []
            items = [str(x).strip() for x in data] if isinstance(data, list) else []
            # remove empties
            items = [x for x in items if x]

            # dedupe by simple normalized key (lower + collapse spaces)
            def normkey(s: str) -> str:
                return re.sub(r"\s+", " ", s).strip().lower()

            seen = set()
            unique: List[str] = []
            for s in items:
                k = normkey(s)
                if k not in seen:
                    seen.add(k)
                    unique.append(s)
            # cap to valves.max_skills (keep latest)
            limit = getattr(self.valves, "max_skills", 10)
            if len(unique) > limit:
                unique = self._truncate_list(
                    unique,
                    limit,
                    warn=(
                        f"[[VALIDATOR_TRUNCATE]] Skills list exceeded limit ({len(unique)} > {limit}). Kept latest {limit} items."
                    ),
                    keep="last",
                )
            return unique

        def validate_emotion_state(
            self, data: Any, categories: List[str]
        ) -> Dict[str, float]:
            validated_data = self.validate_structured_data(
                data, {c: 0.5 for c in categories}
            )
            out: Dict[str, float] = {}
            for c in categories:
                v = self._to_float(validated_data.get(c, 0.5), 0.5)
                v = round(v, self.precision)
                if v < 0.0:
                    v = 0.0
                elif v > 1.0:
                    v = 1.0
                out[c] = v
            return out

        def validate_internal_monologue_state(self, data: Any) -> Dict[str, Any]:
            """Validate internal_monologue with compression rules.
            - thought: cap to <= 2 sentences.
            Other optional keys are passed through if present.
            """
            validated = self.validate_structured_data(data, {})
            result: Dict[str, Any] = {}

            thought = validated.get("thought", "")
            if isinstance(thought, str):
                # Split roughly by sentence enders (., !, ?, 。, ！, ？) while preserving simple languages.
                # This is a heuristic to limit to 2 sentences.
                sentences = re.split(r"(?<=[\.!?。！？])\s+", thought.strip())
                sentences = [s for s in sentences if s]
                capped = " ".join(sentences[:2]) if sentences else thought.strip()
                result["thought"] = capped
            else:
                result["thought"] = ""

            # Pass-through simple string fields (options are normalized to only A/B)
            for key in [
                "cognitive_focus",
                "future_prediction",
                "risk",
                "reward",
                "item_of_interest",
            ]:
                val = validated.get(key)
                if val is not None:
                    result[key] = val

            # options as dict if provided: accept 'a' and 'b' only; each can be string or object
            options = validated.get("options")

            def _normalize_option(v: Any) -> Dict[str, Any]:
                # 互換廃止: 文字列は受け付けず、空の構造へフォールバック。辞書のみ許容。
                if isinstance(v, dict):
                    out = {
                        "label": (
                            str(v.get("label", "")).strip()
                            if v.get("label") is not None
                            else ""
                        ),
                        "future_prediction": str(v.get("future_prediction", "")),
                        "risk": str(v.get("risk", "")),
                        "reward": str(v.get("reward", "")),
                    }
                    return out
                # Fallback empty structure
                return {
                    "label": "",
                    "future_prediction": "",
                    "risk": "",
                    "reward": "",
                }

            if isinstance(options, dict):
                ab_only: Dict[str, Any] = {}
                for k in ("a", "b"):
                    if k in options:
                        ab_only[k] = _normalize_option(options[k])
                if ab_only:
                    result["options"] = ab_only

            # 欠損フィールドの補完（常在化）
            for key in [
                "cognitive_focus",
                "future_prediction",
                "risk",
                "reward",
                "item_of_interest",
            ]:
                if key not in result:
                    result[key] = ""
            if "options" not in result:
                result["options"] = {
                    "a": {
                        "label": "",
                        "future_prediction": "",
                        "risk": "",
                        "reward": "",
                    },
                    "b": {
                        "label": "",
                        "future_prediction": "",
                        "risk": "",
                        "reward": "",
                    },
                }

            return result

        def validate_memory_state(self, data: Any) -> Dict[str, list]:
            # ▼▼▼ 変更後 ▼▼▼
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return {"recent": []}

            # 現行仕様: {"recent": [...]} のみ受理（直接リストは非対応）
            memory_list = []
            if isinstance(data, dict) and isinstance(data.get("recent"), list):
                memory_list = data["recent"]

            if not memory_list:
                return {"recent": []}

            validated_list = []
            for i, item in enumerate(memory_list):
                if not isinstance(item, dict):
                    self.logger.warning(
                        f"[[VALIDATOR_SKIP]] Skipping non-dictionary item in memory list at index {i}."
                    )
                    continue

                # 記憶の必須要素である 'content' が存在し、文字列でなければスキップ
                content = item.get("content")
                if not content or not isinstance(content, str):
                    self.logger.warning(
                        f"[[VALIDATOR_SKIP]] Skipping memory item at index {i} due to missing or invalid 'content'."
                    )
                    continue

                # 中立・簡潔な体裁（文単位の整形・最大2文）。
                content = self._shorten_text_neutral(
                    content, max_chars=10**9, max_sentences=2
                )

                # 'impression_score' が欠損または不正な場合はデフォルト値(0.5)を補う
                try:
                    impression_score = round(
                        float(item.get("impression_score", 0.5)),
                        self.precision,
                    )
                except (ValueError, TypeError):
                    impression_score = 0.5
                    self.logger.warning(
                        f"[[VALIDATOR_FIX]] Using default impression_score for memory item at index {i}."
                    )

                # 'timestamp' が欠損または不正な場合は現在時刻を補う
                timestamp = item.get("timestamp")
                if not timestamp or not isinstance(timestamp, str):
                    self.logger.warning(
                        f"[[VALIDATOR_FIX]] Generating new timestamp for memory item at index {i}."
                    )
                    timestamp = datetime.now(timezone.utc).isoformat()

                validated_list.append(
                    {
                        "content": content,
                        "impression_score": impression_score,
                        "timestamp": timestamp,
                        "tags": self._truncate_list(
                            self._ensure_list_of(item.get("tags", []), str),
                            getattr(self.valves, "max_memory_tags_per_item", 2),
                            "[[VALIDATOR_TRUNCATE]] memory.tags truncated.",
                        ),
                    }
                )

            return {"recent": validated_list}

        def validate_scaled_dict(
            self, data: Any, expected_keys: list, default_value: float = 0.5
        ) -> Dict[str, float]:
            validated_data = self.validate_structured_data(data, {})
            return {
                key: (
                    round(
                        max(
                            0.0, min(1.0, float(validated_data.get(key, default_value)))
                        ),
                        self.precision,
                    )
                    if isinstance(validated_data.get(key, default_value), (int, float))
                    else default_value
                )
                for key in expected_keys
            }

    def __init__(self):
        """Filterクラスの初期化処理。
        
        Valvesの初期化、StateValidatorの構築、状態定義の準備、各種キャッシュや
        ランタイムトラッカーの初期化を行います。
        
        初期化される要素:
        - valves: 設定値（Pydanticモデル）
        - validator: 状態バリデーター
        - STATE_DEFS: 状態定義マップ
        - narration_pattern: ナレーション検出用正規表現
        - _aiohttp_session: HTTPセッション（遅延初期化）
        - _idle_refactor_tasks: アイドル時要約タスクトラッカー
        - _last_filter_call: フィルター最終呼び出し時刻記録
        - _mf_prev_selected: Memory Focus前回選択キャッシュ
        - _st_embedder_models: SentenceTransformerモデルキャッシュ
        - _background_tasks: バックグラウンドタスク管理セット
        
        Note:
            Valvesの正規化に失敗しても処理は継続します（ログのみ記録）。
            ZoneInfoが利用不可の場合、タイムゾーン機能は無効化されます。
        """
        self.valves = self.Valves()
        # 既定値・表記ゆれの正規化（安全側へ丸め）
        try:
            self._normalize_valves()
        except Exception:
            # 正規化失敗は致命ではない（ログのみ）
            self.logger.debug(
                "[[VALVES]] Failed to normalize valves; continuing with raw values.",
                exc_info=True,
            )
        self.validator = self.StateValidator(self.valves, self.logger)
        self.STATE_DEFS = self._build_state_defs()
        # バリデーターにSTATE_DEFSを設定（構造の自動追従のため）
        self.validator.set_state_defs(self.STATE_DEFS)
        # ナレーション/メタ入力の検出: 半角角括弧 [ … ] と 全角角括弧 ［ … ］ のみを対象
        # メッセージ全体がこれらの括弧で囲まれている場合にメタとして扱う
        self.narration_pattern = re.compile(
            r"^\s*(?:\[\s*.*?\s*\]|［\s*.*?\s*］)\s*$",
            re.DOTALL,
        )

        if ZoneInfo is None:
            self.logger.error(
                "zoneinfo library is not installed. Timezone features will be disabled."
            )
        # セッションは必要時に遅延初期化（_call_llm_for_state_update内）
        self._aiohttp_session = None
        # Idle refactor runtime trackers (in-memory, per user/model)
        self._idle_refactor_tasks = {}
        self._last_filter_call = {}
        self._idle_refactor_baseline_count = {}
        # Memory Focus: previous selection (ephemeral, per user-model key)
        self._mf_prev_selected = {}
        # ST model caches (lazy)
        self._st_embedder_models = {}
        self._st_cross_encoders = {}
        # 運用開始時にログレベルを適用
        self._apply_log_level(self.valves.log_level)
        
        # バックグラウンドタスクの追跡用セット
        self._background_tasks: set = set()

    # ========================================================================
    # 非同期タスク管理ヘルパー
    # ========================================================================

    def _create_background_task(
        self, coro: Awaitable[Any], *, name: Optional[str] = None
    ) -> "asyncio.Task[Any]":
        """バックグラウンドタスクを作成し、自動的にクリーンアップする。
        
        タスクが完了したら自動的に追跡セットから削除される。
        エラーが発生した場合はログに記録される。
        
        Args:
            coro: 実行する非同期コルーチン
            name: タスクの名前（デバッグ用、オプション）
            
        Returns:
            作成されたタスク
        """
        task = asyncio.create_task(coro)  # type: ignore[arg-type]
        if name:
            try:
                task.set_name(name)
            except AttributeError:
                # Python 3.7では set_name がない
                pass
                
        # タスクセットに追加
        self._background_tasks.add(task)
        
        # 完了時のコールバック
        def _on_task_done(t: asyncio.Task) -> None:
            # セットから削除
            self._background_tasks.discard(t)
            
            # エラーチェック
            try:
                exc = t.exception()
                if exc is not None:
                    task_name = name or "unnamed"
                    self.logger.error(
                        f"[[TASK]] Background task '{task_name}' failed: {exc}",
                        exc_info=exc
                    )
            except asyncio.CancelledError:
                # キャンセルは正常
                pass
            except Exception as e:
                self.logger.debug(f"[[TASK]] Error checking task exception: {e}")
                
        task.add_done_callback(_on_task_done)
        return task

    async def _cancel_all_background_tasks(self) -> None:
        """全てのバックグラウンドタスクをキャンセルする。
        
        主にクリーンアップ時に使用される。
        """
        if not self._background_tasks:
            return
            
        self.logger.info(f"[[TASK]] Cancelling {len(self._background_tasks)} background tasks...")
        
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                
        # 全てのタスクが完了するまで待機（タイムアウト付き）
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._background_tasks, return_exceptions=True),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            self.logger.warning("[[TASK]] Some background tasks did not complete within timeout")
        except Exception as e:
            self.logger.debug(f"[[TASK]] Error during task cancellation: {e}")
            
        self._background_tasks.clear()

    # ========================================================================
    # ログヘルパー
    # ========================================================================

    def _apply_log_level(self, level_name: str) -> None:
        """Apply log level to this module logger safely.
        Accepts level names: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET.
        Falls back to INFO on invalid input.
        """
        try:
            level = getattr(logging, str(level_name).upper(), logging.INFO)
        except Exception:
            level = logging.INFO
        self.logger.setLevel(level)
        # Ensure at least one StreamHandler exists with basic formatter
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_log_level(self, level_name: str) -> None:
        """Public API to change log level at runtime."""
        self.valves.log_level = level_name
        self._apply_log_level(level_name)

    # Additional tagged logging helpers (non-breaking, optional to use)
    def _log_with_tag(self, tag: str, msg: str, level: int = logging.INFO) -> None:
        try:
            self.logger.log(level, f"{tag} {msg}")
        except Exception:
            try:
                self.logger.log(level, msg)
            except Exception:
                pass

    def _log_inlet(self, msg: str, level: int = logging.INFO) -> None:
        self._log_with_tag(self.LOG_INLET, msg, level)

    def _log_outlet(self, msg: str, level: int = logging.INFO) -> None:
        self._log_with_tag(self.LOG_OUTLET, msg, level)

    def _log_state(self, msg: str, level: int = logging.INFO) -> None:
        self._log_with_tag(self.LOG_STATE, msg, level)

    def _normalize_valve_choice(
        self,
        attr: str,
        *,
        allowed: Dict[str, str],
        default: str,
        case: str,
        warning_template: str,
    ) -> None:
        """Normalize a simple enum-like valve via shared logic."""
        v = self.valves
        default_key = default.upper() if case == "upper" else default.lower()
        fallback = allowed.get(default_key, default)
        try:
            raw = getattr(v, attr, default)
            text = "" if raw is None else str(raw).strip()
            if not text:
                raise ValueError("empty")
            normalized = text.upper() if case == "upper" else text.lower()
            canonical = allowed.get(normalized)
            if canonical is None:
                self.logger.warning(
                    warning_template.format(value=raw, default=fallback)
                )
                canonical = fallback
        except Exception:
            canonical = fallback
        setattr(v, attr, canonical)

    # ========================================================================
    # 設定（Valves）の正規化とバリデーション
    # ========================================================================

    def _normalize_valves(self) -> None:
        """Normalize enum-ish valve values to safe canonical forms.
        設定値を正規化し、無効な値がある場合はデフォルト値にフォールバックする。
        
        - inventory_update_mode: 'direct' | 'inference' (default to 'inference' on invalid)
        - inventory_trim_strategy: 'unequipped_first' | 'quantity_asc'
        - persistence_backend: 'file' (memories backend removed)
        - response_format_mode: 'none'|'openai-json'|'ollama-json'|'auto'
        - log_level: normalize to upper-case known names
        
        Raises:
            ConfigurationError: 重大な設定エラーがある場合（現在は警告のみ）
        """
        # ログレベルの正規化
        self._normalize_valve_choice(
            "log_level",
            allowed={
                "CRITICAL": "CRITICAL",
                "ERROR": "ERROR",
                "WARNING": "WARNING",
                "INFO": "INFO",
                "DEBUG": "DEBUG",
                "NOTSET": "NOTSET",
            },
            default="INFO",
            case="upper",
            warning_template="[[VALVES]] Invalid log_level '{value}', falling back to {default}.",
        )
        
        # インベントリ更新モードの正規化
        self._normalize_valve_choice(
            "inventory_update_mode",
            allowed={"direct": "direct", "inference": "inference"},
            default="inference",
            case="lower",
            warning_template="[[VALVES]] Invalid inventory_update_mode '{value}', using '{default}'.",
        )
        
        # インベントリトリム戦略の正規化
        self._normalize_valve_choice(
            "inventory_trim_strategy",
            allowed={
                self.INV_TRIM_UNEQUIPPED_FIRST: self.INV_TRIM_UNEQUIPPED_FIRST,
                self.INV_TRIM_QUANTITY_ASC: self.INV_TRIM_QUANTITY_ASC,
            },
            default=self.INV_TRIM_UNEQUIPPED_FIRST,
            case="lower",
            warning_template="[[VALVES]] Invalid inventory_trim_strategy '{value}', using '{default}'.",
        )
        
        # 永続化バックエンドの正規化（'file'のみサポート）
        self._normalize_valve_choice(
            "persistence_backend",
            allowed={"file": "file"},
            default="file",
            case="lower",
            warning_template="[[VALVES]] Invalid persistence_backend '{value}', using '{default}'.",
        )
        
        # レスポンスフォーマットモードの正規化
        self._normalize_valve_choice(
            "response_format_mode",
            allowed={
                "none": "none",
                "openai-json": "openai-json",
                "ollama-json": "ollama-json",
                "auto": "auto",
            },
            default="none",
            case="lower",
            warning_template="[[VALVES]] Invalid response_format_mode '{value}', using '{default}'.",
        )
        
        # 数値パラメータのバリデーション
        self._validate_numeric_valves()

    def _validate_numeric_valves(self) -> None:
        """数値型の設定値を検証し、範囲外の値を補正する。
        
        Note:
            Pydanticの ge/le バリデーションが既に適用されているため、
            このメソッドは追加のランタイムチェックを行う。
            エラーはログに記録するが、処理は継続する。
        """
        try:
            # タイムアウト設定の範囲チェック（Pydanticでも定義済みだが念のため）
            timeout = getattr(self.valves, "llm_timeout_sec", 30.0)
            if not isinstance(timeout, (int, float)) or timeout < 1.0 or timeout > 300.0:
                self.logger.warning(
                    f"[[VALVES]] llm_timeout_sec out of range: {timeout}, should be 1.0-300.0"
                )
            
            # リトライ回数の範囲チェック
            retry_max = getattr(self.valves, "llm_retry_attempts", 2)
            if not isinstance(retry_max, int) or retry_max < 0 or retry_max > 10:
                self.logger.warning(
                    f"[[VALVES]] llm_retry_attempts out of range: {retry_max}, should be 0-10"
                )
            
            # 会話トリム数の範囲チェック
            max_turns = getattr(self.valves, "conversation_max_messages_turn", 5)
            if not isinstance(max_turns, int) or max_turns < 1:
                self.logger.warning(
                    f"[[VALVES]] conversation_max_messages_turn should be >= 1: {max_turns}"
                )
            
            # RAG検索数の範囲チェック
            retrieval_k = getattr(self.valves, "retrieval_top_k", 25)
            if not isinstance(retrieval_k, int) or retrieval_k < 1 or retrieval_k > 500:
                self.logger.warning(
                    f"[[VALVES]] retrieval_top_k out of range: {retrieval_k}, should be 1-500"
                )
            
            inject_k = getattr(self.valves, "inject_top_k", 5)
            if not isinstance(inject_k, int) or inject_k < 1 or inject_k > 100:
                self.logger.warning(
                    f"[[VALVES]] inject_top_k out of range: {inject_k}, should be 1-100"
                )
            
            # メモリ最大数の範囲チェック
            max_mem = getattr(self.valves, "max_memory_items", 1000)
            if not isinstance(max_mem, int) or max_mem < 1 or max_mem > 9999:
                self.logger.warning(
                    f"[[VALVES]] max_memory_items out of range: {max_mem}, should be 1-9999"
                )
            
            # 重み設定の合計チェック（Memory Focus）
            weight_fb = getattr(self.valves, "memory_focus_weight_fb", 0.45)
            weight_emb = getattr(self.valves, "memory_focus_weight_emb", 0.35)
            weight_rr = getattr(self.valves, "memory_focus_weight_rr", 0.20)
            total_weight = weight_fb + weight_emb + weight_rr
            if abs(total_weight - 1.0) > 0.01:  # 許容誤差0.01
                self.logger.warning(
                    f"[[VALVES]] Memory focus weights sum to {total_weight:.3f}, should be ~1.0 "
                    f"(fb={weight_fb}, emb={weight_emb}, rr={weight_rr})"
                )
                
        except Exception as e:
            self.logger.error(f"[[VALVES]] Error during numeric validation: {e}", exc_info=True)
            # 検証エラーは継続可能とする

    async def _ensure_http_session(self):
        """aiohttp.ClientSession を必要時に遅延初期化する。"""
        if self._aiohttp_session is None and aiohttp is not None:
            # Note: 起動時のイベントループ外での生成警告を避けるため遅延生成
            self._aiohttp_session = aiohttp.ClientSession()

    async def close(self) -> None:
        """リソースのクリーンアップを行う。
        
        HTTPセッションとバックグラウンドタスクを適切に終了する。
        """
        try:
            # バックグラウンドタスクのキャンセル
            await self._cancel_all_background_tasks()
        except Exception as e:
            self.logger.debug(f"Failed to cancel background tasks: {e}", exc_info=True)
            
        try:
            # HTTPセッションのクローズ
            if self._aiohttp_session is not None:
                await self._aiohttp_session.close()
                self._aiohttp_session = None
        except Exception as e:
            self.logger.debug(f"Failed to close aiohttp session: {e}", exc_info=True)

    def __del__(self):
        """ガベージコレクション時の最終防衛。"""
        try:
            # バックグラウンドタスクの強制キャンセル（ベストエフォート）
            if hasattr(self, "_background_tasks") and self._background_tasks:
                for task in self._background_tasks:
                    if not task.done():
                        task.cancel()
        except Exception:
            pass
            
        # HTTPセッションのクローズ（ベストエフォート）
        try:
            session = getattr(self, "_aiohttp_session", None)
            if session is not None and not session.closed:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(session.close())
                    else:
                        loop.run_until_complete(session.close())
                except Exception:
                    # イベントループが取得できない/閉じている場合
                    try:
                        import asyncio as _aio

                        _loop = _aio.new_event_loop()
                        _loop.run_until_complete(session.close())
                        _loop.close()
                    except Exception:
                        pass
        except Exception:
            pass

    def _get_text(self, key: str, lang: Optional[str] = None) -> Any:
        # 言語は ja に固定（TRANSLATIONS は ja のみを参照）
        return self.TRANSLATIONS.get("ja", {}).get(key, f"<{key}>")

    def _safe_nested_get(
        self,
        data: Dict[str, Any],
        *keys: str,
        default: Any = None
    ) -> Any:
        """ネストした辞書から安全に値を取得する。
        
        Args:
            data: 検索対象の辞書
            *keys: キーのパス（可変長引数）
            default: 値が見つからない場合のデフォルト値
            
        Returns:
            見つかった値、またはデフォルト値
            
        Examples:
            >>> state = {"goal": {"short_term": {"task": "study"}}}
            >>> self._safe_nested_get(state, "goal", "short_term", "task")
            "study"
            >>> self._safe_nested_get(state, "goal", "long_term", default={})
            {}
        """
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current if current is not None else default

    def _is_message_role(
        self, 
        msg: Any, 
        role: str
    ) -> bool:
        """メッセージが指定されたロールを持つか判定する。
        
        Args:
            msg: チェック対象のメッセージ（通常はdict）
            role: 期待されるロール（"user", "assistant", "system"など）
            
        Returns:
            メッセージが辞書型で、指定されたロールを持つ場合True
            
        Examples:
            >>> msg = {"role": "user", "content": "Hello"}
            >>> self._is_message_role(msg, "user")
            True
            >>> self._is_message_role(msg, "assistant")
            False
        """
        return isinstance(msg, dict) and msg.get("role") == role

    def _is_dict_with_key(
        self,
        obj: Any,
        key: str,
        expected_type: Optional[type] = None
    ) -> bool:
        """オブジェクトが辞書型で特定のキーを持つか判定する。
        
        Args:
            obj: チェック対象のオブジェクト
            key: 存在を確認するキー
            expected_type: キーの値の期待される型（オプション）
            
        Returns:
            条件を満たす場合True
            
        Examples:
            >>> data = {"items": [1, 2, 3]}
            >>> self._is_dict_with_key(data, "items", list)
            True
            >>> self._is_dict_with_key(data, "items", dict)
            False
        """
        if not isinstance(obj, dict):
            return False
        if key not in obj:
            return False
        if expected_type is not None:
            return isinstance(obj.get(key), expected_type)
        return True

    def _build_state_defs(self) -> Dict[str, Dict[str, Any]]:
        # 例示JSONを最小トークンで提示するためのダンプ関数（空白を省略）
        jd = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

        state_definitions_list = [
            (
                "emotion",
                jd({c: 0.5 for c in self.EMOTION_CATEGORIES}),
                lambda data: self.validator.validate_emotion_state(
                    data, self.EMOTION_CATEGORIES
                ),
                {"default": {c: 0.5 for c in self.EMOTION_CATEGORIES}},
            ),
            (
                "skills",
                jd(["木工", "突き"]),
                self.validator.validate_skills_state,
                {"default": []},
            ),
            (
                "boundaries",
                jd({"taboos": ["a"], "dislikes": ["b"]}),
                self.validator.validate_boundaries_state,
                {"default": {"taboos": [], "dislikes": []}},
            ),
            (
                "relationship",
                jd(
                    {
                        "type": "",
                        "level": 0.5,
                        "trust_label": "",
                        "trust_score": 0.5,
                        "user_address": {"default": "", "joking": ""},
                        "self_address": {"default": "", "nickname": ""},
                        "commitments": [],
                    }
                ),
                self.validator.validate_relationship_state,
                {},
            ),
            (
                "goal",
                # AIに示すフォーマット例を新しい構造に変更
                jd(
                    {
                        "long_term": {"A": {"progress": 0, "priority": 1}},
                        "mid_term": {"B": {"progress": 0, "priority": 1}},
                        "short_term": {"C": {"progress": 0, "priority": 1}},
                        "routine": {
                            "睡眠": {
                                "priority": 1,
                                "start_time": "23:00",
                                "end_time": "07:00",
                            }
                        },
                    }
                ),
                # バリデーション関数も新しい構造に対応させる
                lambda data: self.validator.validate_goal_state(data),
                {
                    "default": {
                        "long_term": {},
                        "mid_term": {},
                        "short_term": {},
                        "routine": {},
                    }
                },
            ),
            (
                "context",
                jd(
                    {
                        "place": "",
                        "atmosphere": "",
                        "details": [""],
                        "affordances": ["立つ", "水を飲む"],
                        "constraints": ["静粛", "人目が多い"],
                        "action": {"user": ""},
                        "entities": [
                            {
                                "name": "店主",
                                "type": "person",
                                "role": "vendor",
                                "pose_hint": "カウンター越しに立つ",
                                "note": "こちらに気づいた様子",
                            }
                        ],
                    }
                ),
                self.validator.validate_context_state,
                {},
            ),
            (
                "tone",
                jd({"effects": ["e"]}),
                self.validator.validate_tone_state,  # 変更点: Noneから専用バリデータへ
                {},
            ),
            (
                "memory",
                jd(
                    {
                        "recent": [
                            {
                                "content": "",
                                "impression_score": 0,
                                "timestamp": "",
                                "tags": ["準備", "中断"],
                            }
                        ]
                    }
                ),
                self.validator.validate_memory_state,
                {"default": {"recent": []}},
            ),
            (
                "inventory",
                jd(
                    [
                        {
                            "name": "A",
                            "description": "",
                            "quantity": 1,
                            "equipped": False,
                            "slot": "none",
                        }
                    ]
                ),
                self.validator.validate_inventory_state,
                {"default": []},
            ),
            (
                "physical_health",
                jd(
                    {
                        "condition": "",
                        "sensation": "",
                        "timestamps": {"condition_since": None},
                        "limits": {
                            "stamina": {"current": 100, "max": 100},
                            "pleasure_tolerance": {"current": 0, "max": 100},
                            "refractory_period": {"current": 0, "max": 100},
                            "pain_tolerance": {"current": 100, "max": 100},
                            "pain_load": {"current": 0, "max": 100},
                            "temperature_tolerance": {"current": 100, "max": 100},
                            "temperature_load": {"current": 0, "max": 100},
                            "posture_strain_tolerance": {"current": 100, "max": 100},
                            "posture_strain_load": {"current": 0, "max": 100},
                        },
                        "needs": {
                            "hunger": 0.0,
                            "sleepiness": 0.0,
                            "libido": 0.0,
                            "thirst": 0.0,
                            "hygiene": 0.0,
                            "social_interaction": 0.0,
                            "movement": 0.0,
                            "thermoregulation": 0.0,
                            "restroom": 0.0,
                        },
                        "reproductive": {
                            "status": "",
                            "cycle_day": 0,
                            "is_pregnant": False,
                            "pregnancy_progress": 0.0,
                        },
                        "issues": [],
                    }
                ),
                self.validator.validate_physical_health_state,
                {},
            ),
            (
                "posture",
                jd(
                    {
                        "character": {
                            "position": "",
                            "support": "",
                            "movement": "",
                            "relative": "",
                        },
                        "user": {
                            "position": "",
                            "support": "",
                            "movement": "",
                            "relative": "",
                        },
                    }
                ),
                lambda data: self.validator.validate_posture_state(data),
                {
                    "default": {
                        "character": {
                            "position": "",
                            "support": "",
                            "movement": "",
                            "relative": "",
                        },
                        "user": {
                            "position": "",
                            "support": "",
                            "movement": "",
                            "relative": "",
                        },
                    }
                },
            ),
            (
                "mental_health",
                jd(
                    {
                        "condition": "",
                        "mood": "",
                        "timestamps": {"condition_since": None},
                        "limits": {
                            "cognitive_load": {"current": 0, "max": 100},
                            "stress_load": {"current": 0, "max": 100},
                            "focus_capacity": {"current": 100, "max": 100},
                            "social_battery": {"current": 100, "max": 100},
                        },
                        "needs": {"novelty": 0, "solitude": 0, "social_interaction": 0},
                        "dynamics": {"emotional_volatility": 0.5, "learning_rate": 0.1},
                    }
                ),
                self.validator.validate_mental_health_state,
                {},
            ),
            (
                "sexual_development",
                jd(
                    {
                        "description": "",
                        "experience_score": 0,
                        "favorite_acts": ["a"],
                        "parts": {"p": {"sensitivity": 0, "development_progress": 0}},
                    }
                ),
                self.validator.validate_sexual_development_state,
                {},
            ),
            (
                "desire",
                jd({k: 0.5 for k in self.DESIRE_KEYS}),
                lambda data: self.validator.validate_scaled_dict(
                    data, self.DESIRE_KEYS
                ),
                {},
            ),
            (
                "internal_monologue",
                # AIに示す「手本」となるフォーマット例。
                jd(
                    {
                        "thought": "…",
                        "cognitive_focus": "RiskReward",
                        "future_prediction": "…",
                        "options": {
                            "a": {
                                "label": "…",
                                "future_prediction": "…",
                                "risk": "…",
                                "reward": "…",
                            },
                            "b": {
                                "label": "…",
                                "future_prediction": "…",
                                "risk": "…",
                                "reward": "…",
                            },
                        },
                        "risk": "…",
                        "reward": "…",
                        "item_of_interest": "",
                    }
                ),
                # 専用バリデータに置き換え
                self.validator.validate_internal_monologue_state,
                {},
            ),
            (
                "knowledge",
                jd(
                    {
                        "user": {
                            "likes": [],
                            "dislikes": [],
                            "identity": {"name": "", "notes": []},
                        },
                        "self": {
                            "identity": {
                                "species": "人間",
                                "gender": "女性",
                                "notes": [],
                                "default_outfit": [],
                                "anniversaries": [""],
                                "milestones": [""],
                            },
                            "strengths": [],
                            "weaknesses": [],
                        },
                        "world": {
                            "places": [{"name": "拠点キャンプ", "type": "base"}],
                            "relations": [
                                {
                                    "a": "拠点キャンプ",
                                    "b": "北の洞窟",
                                    "label": "近接",
                                    "strength": 0.6,
                                }
                            ],
                            "notes": ["北の洞窟は夜間の冷え込みが厳しい"],
                        },
                    }
                ),
                self.validator.validate_knowledge_state,  # 変更点: Noneから専用バリデータへ
                {"default": {}},
            ),
            (
                "traits",
                jd(["a"]),
                self.validator.validate_traits_state,
                {"default": []},
            ),
        ]
        defs = {}
        for key, example, validator_fn, options in state_definitions_list:
            # tagとpatternの生成ロジックを完全に削除
            # goal_instruction のフォーマット引数を渡す
            # NOTE: 旧来の "*_instruction" 文字列は使用しない。
            # プロンプト規範は state_rules_list に一本化する。

            defs[key] = {
                "default": options.get("default", {}),
                "example": example,
                "load_fn": (
                    validator_fn
                    if validator_fn
                    else lambda data, d=options.get(
                        "default", {}
                    ): self.validator.validate_structured_data(data, d)
                ),
                # instruction 文面は state_rules_list を唯一の真実源とする（ここには保持しない）
            }
        return defs

    def _get_model_id(self, body: Dict[str, Any]) -> str:
        model_id = body.get("model", "unknown_model")
        return model_id.get("id", model_id) if isinstance(model_id, dict) else model_id

    def _get_user_model(self, user_data: Optional[Dict[str, Any]]) -> Optional[Any]:
        if not user_data or not user_data.get("id"):
            return None
        try:
            return (
                UserModel.model_validate(user_data) if UserModel else user_data
            )  # fallback in stub env
        except Exception:
            return None

    async def _get_system_prompt_from_model_id(self, model_id: str) -> Optional[str]:
        try:
            model_obj = Models.get_model_by_id(model_id)
            if not model_obj:
                return None
            if isinstance(model_obj, dict):
                model_dict = model_obj
            elif hasattr(model_obj, "model_dump"):
                model_dict = model_obj.model_dump()
            elif hasattr(model_obj, "dict"):
                model_dict = model_obj.dict()
            else:
                # フォールバック: 属性辞書化
                try:
                    model_dict = dict(model_obj)
                except Exception:
                    model_dict = {}
            if isinstance(model_dict.get("params"), dict):
                system_prompt = model_dict["params"].get("system")
                if isinstance(system_prompt, str) and system_prompt.strip():
                    return system_prompt.strip()
            for key in ("template", "system", "prompt", "character", "system_prompt"):
                prompt_value = model_dict.get(key)
                if isinstance(prompt_value, str) and prompt_value.strip():
                    return prompt_value.strip()
            return None
        except Exception:
            return None

    async def _load_all_states(
        self, user: Any, model_id: str
    ) -> Optional[Dict[str, Any]]:
        """ユーザー・モデル固有の全状態をファイルから読み込む。
        
        指定されたユーザーとモデルIDに対応する状態ファイル（all_states.json）から
        キャラクターの内部状態を読み込みます。ファイルが存在しない場合はNoneを返します。
        
        読み込み後の互換性処理:
        - 旧バージョンの`mental_health.status`フィールドを削除（互換性維持）
        
        Args:
            user: Open WebUIユーザーオブジェクト（IDを取得するために使用）
            model_id: モデル識別子（キャラクター名やモデル名）
        
        Returns:
            Optional[Dict[str, Any]]: 状態データ（emotion, memory, goal等を含む辞書）
                                      ファイルが存在しない場合やエラー時はNone
        
        Note:
            内部的には_load_all_states_file()を呼び出し、ファイルシステムから読み込みます。
            エラーはログに記録され、Noneを返すことで呼び出し側でのデフォルト状態初期化を促します。
        """
        try:
            file_states = self._load_all_states_file(user, model_id)
        except Exception:
            file_states = None
            self.logger.debug(
                "[[LOAD_STATES_FILE]] Error during file load.", exc_info=True
            )

        if file_states is not None:
            try:
                # 互換: mental_health.status を破棄
                mh = (
                    file_states.get("mental_health")
                    if isinstance(file_states, dict)
                    else None
                )
                if isinstance(mh, dict) and "status" in mh:
                    mh.pop("status", None)
            except Exception:
                pass
            return file_states

        return None

    # memories backend removed: _load_all_states_memories deleted

    async def _save_all_states(self, user: Any, model_id: str, states: Dict[str, Any]):
        """ユーザー・モデル固有の全状態をファイルに保存する。
        
        指定されたユーザーとモデルIDに対応する状態ファイル（all_states.json）に
        キャラクターの内部状態を保存します。JSON形式で人間が読みやすい形式（インデント付き）で
        書き込まれます。
        
        保存先ディレクトリ構造:
        {file_base_dir}/states/{user_id}/{model_id}/all_states.json
        
        Args:
            user: Open WebUIユーザーオブジェクト
            model_id: モデル識別子
            states: 保存する状態データ（emotion, memory, goal等を含む辞書）
        
        Raises:
            StatePersistenceError: ファイル書き込みに失敗した場合（内部でキャッチしログ記録）
            
        Note:
            保存に失敗してもプログラムは継続します。エラーはログに詳細が記録されます。
            内部的にはファイルロックを使用して並行書き込みの競合を防ぎます。
        """
        try:
            self._save_all_states_file(user, model_id, states)
        except Exception as e:
            self.logger.error(
                f"[[SAVE_STATES_FILE]] Failed to save to files for '{model_id}': {e}",
                exc_info=True,
            )

    # --- File-based persistence helpers ---
    def _sanitize_id_component(self, s: Any) -> str:
        try:
            s2 = re.sub(r"[^a-zA-Z0-9._-]", "_", str(s))
            return s2[:80] if len(s2) > 80 else s2
        except Exception:
            return "unknown"

    def _get_state_dir(self, user: Any, model_id: str) -> Path:
        base_root = Path(
            getattr(self.valves, "file_base_dir", "/app/backend/data/llm_emotion")
        )
        states_dirname = "states"
        base = base_root / states_dirname
        try:
            user_id = (
                getattr(user, "id", None)
                or getattr(user, "_id", None)
                or "unknown_user"
            )
        except Exception:
            user_id = "unknown_user"
        uid = self._sanitize_id_component(user_id)
        mid = self._sanitize_id_component(model_id)
        # optional tenant/namespace could be added later; keep simple
        return base / uid / mid

    def _get_session_log_path(self, user: Any, model_id: str) -> Path:
        dirpath = self._get_state_dir(user, model_id)
        return dirpath / "last_session_log.json"

    def _load_last_session_log(
        self, user: Any, model_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        try:
            fp = self._get_session_log_path(user, model_id)
            if not fp.exists():
                return None
            content = fp.read_text(encoding="utf-8")
            obj = json.loads(content)
            if isinstance(obj, list):
                # only allow role/content pairs
                cleaned = []
                for m in obj:
                    if not isinstance(m, dict):
                        continue
                    role = str((m.get("role") or "")).strip()
                    if role not in ("user", "assistant", "system"):
                        continue
                    cleaned.append(
                        {"role": role, "content": str(m.get("content") or "")}
                    )
                return cleaned
            if self._is_dict_with_key(obj, "messages", list):
                return obj["messages"]
        except Exception:
            self.logger.debug(
                "[[SESSION_LOG]] Failed to load last_session_log.json", exc_info=True
            )
        return None

    def _save_last_session_log(
        self, user: Any, model_id: str, messages: List[Dict[str, Any]]
    ) -> None:
        try:
            dirpath = self._get_state_dir(user, model_id)
            self._ensure_dir(dirpath)
            fp = dirpath / "last_session_log.json"
            # Atomic write
            tmp = dirpath / f"last_session_log_{uuid.uuid4().hex}.tmp"
            tmp.write_text(
                json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            try:
                os.replace(str(tmp), str(fp))
            except Exception:
                # Fallback to write directly
                fp.write_text(
                    json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        except Exception:
            self.logger.debug(
                "[[SESSION_LOG]] Failed to save last_session_log.json", exc_info=True
            )

    def _append_completed_turn_to_session_log(
        self,
        user: Any,
        model_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Append a completed (user, assistant) turn into last_session_log.json, trimming to last N turns.

        - Loads existing last_session_log.json (list of role/content dicts, non-system only)
        - Appends [user, assistant]
        - Trims to last conversation_max_messages_turn turns (if enabled)
        - Saves back atomically
        """
        try:
            existing = self._load_last_session_log(user, model_id) or []
            # Build turns from existing
            turns: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            pending_user: Optional[Dict[str, Any]] = None
            for m in existing:
                r = m.get("role")
                if r == "user":
                    pending_user = {
                        "role": "user",
                        "content": str(m.get("content", "")),
                    }
                elif r == "assistant" and pending_user is not None:
                    a = {"role": "assistant", "content": str(m.get("content", ""))}
                    turns.append((pending_user, a))
                    pending_user = None
            # Append new completed turn
            new_u = {"role": "user", "content": str(user_text or "")}
            new_a = {"role": "assistant", "content": str(assistant_text or "")}
            turns.append((new_u, new_a))
            # Trim by valves
            enabled = bool(getattr(self.valves, "conversation_trim_enabled", False))
            try:
                n_turns = int(
                    getattr(self.valves, "conversation_max_messages_turn", 5) or 5
                )
            except Exception:
                n_turns = 5
            if enabled and n_turns > 0 and len(turns) > n_turns:
                turns = turns[-n_turns:]
            # Flatten to messages
            flat: List[Dict[str, Any]] = []
            for u_msg, a_msg in turns:
                flat.append(u_msg)
                flat.append(a_msg)
            self._save_last_session_log(user, model_id, flat)
        except Exception:
            self.logger.debug(
                "[[SESSION_LOG]] Failed to append completed turn", exc_info=True
            )

    def _extract_last_completed_turn(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (last_user_text, last_assistant_text) for the most recent completed user->assistant pair.
        Scans from tail: find the last assistant, then find the nearest preceding user.
        """
        try:
            last_assistant_idx = None
            for i in range(len(messages) - 1, -1, -1):
                m = messages[i]
                if isinstance(m, dict) and m.get("role") == "assistant":
                    last_assistant_idx = i
                    break
            if last_assistant_idx is None:
                return None, None
            # find nearest user before it
            user_idx = None
            for j in range(last_assistant_idx - 1, -1, -1):
                m = messages[j]
                if isinstance(m, dict) and m.get("role") == "user":
                    user_idx = j
                    break
            if user_idx is None:
                return None, None
            u = str(messages[user_idx].get("content", ""))
            a = str(messages[last_assistant_idx].get("content", ""))
            return u, a
        except Exception:
            return None, None

    def _extract_last_completed_turn_from_session_log(
        self, log_messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """From persisted last_session_log (flat [user, assistant, ...]) get the last user/assistant pair.
        Returns (user_text, assistant_text) or (None, None).
        """
        try:
            last_assistant_idx = None
            for i in range(len(log_messages) - 1, -1, -1):
                m = log_messages[i]
                if isinstance(m, dict) and m.get("role") == "assistant":
                    last_assistant_idx = i
                    break
            if last_assistant_idx is None:
                return None, None
            user_idx = None
            for j in range(last_assistant_idx - 1, -1, -1):
                m = log_messages[j]
                if isinstance(m, dict) and m.get("role") == "user":
                    user_idx = j
                    break
            if user_idx is None:
                return None, None
            return (
                str(log_messages[user_idx].get("content", "")),
                str(log_messages[last_assistant_idx].get("content", "")),
            )
        except Exception:
            return None, None

    def _extract_last_n_turns_from_session_log(
        self, log_messages: List[Dict[str, Any]], n_turns: int = 2
    ) -> str:
        """From persisted last_session_log, extract the last N user/assistant pairs as a compact dialog.
        
        Args:
            log_messages: Session log messages
            n_turns: Number of user/assistant pairs to extract (default: 2)
        
        Returns:
            Compact dialog string or empty string if insufficient data.
        """
        try:
            if not log_messages or n_turns <= 0:
                return ""
            
            # Collect all user/assistant pairs in order
            pairs: List[Tuple[str, str]] = []
            current_user = None
            
            for msg in log_messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                content = str(msg.get("content", "")).strip()
                
                if role == "user":
                    current_user = content
                elif role == "assistant" and current_user is not None:
                    pairs.append((current_user, content))
                    current_user = None
            
            # Take the last n_turns pairs
            last_pairs = pairs[-n_turns:] if pairs else []
            
            if not last_pairs:
                return ""
            
            # Format as compact dialog: user『...』→assistant『...』(省略なし、全文保持)
            dialog_lines = []
            for u_txt, a_txt in last_pairs:
                dialog_lines.append(f"U『{u_txt}』→A『{a_txt}』")
            
            return " | ".join(dialog_lines)  # Compact separator
        except Exception:
            return ""

    def _save_last_prompts(
        self,
        character_text: Optional[str] = None,
        character_response_text: Optional[str] = None,
        state_text: Optional[str] = None,
        state_response_text: Optional[str] = None,
        gen_state_text: Optional[str] = None,
        gen_state_response_text: Optional[str] = None,
        idle_refactor_log: Optional[str] = None,
    ) -> None:
        """Save latest artifacts into a single last_prompt.txt with sections.
        - Character LLM Prompt (chat messages after inlet injection)
        - Character LLM Response (latest): raw response from character LLM
        - State Analysis LLM Prompt (update/initial prompt sent to analysis LLM)
        - State Analysis LLM Response (raw content returned by analysis LLM)
        - Idle Refactor LLM (latest): raw prompt/response log for idle-time refactor

        Any None argument preserves the existing section content if present.
        """
        try:
            if not getattr(self.valves, "save_last_prompt", True):
                return
            base_root = Path(
                getattr(self.valves, "file_base_dir", "/app/backend/data/llm_emotion")
            )
            self._ensure_dir(base_root)
            fn = getattr(self.valves, "last_prompt_filename", "last_prompt.txt")
            file_path = base_root / fn

            header_char = "===== Character LLM Prompt (latest) =====\n"
            header_char_resp = "===== Character LLM Response (latest) =====\n"
            header_state = "===== State Analysis LLM Prompt (latest) =====\n"
            header_state_resp = "===== State Analysis LLM Response (latest) =====\n"
            header_gen = "===== Initial State Generation LLM Prompt (latest) =====\n"
            header_gen_resp = (
                "===== Initial State Generation LLM Response (latest) =====\n"
            )
            header_idle = "===== Idle Refactor LLM (latest) =====\n"
            existing_char = ""
            existing_char_resp = ""
            existing_state = ""
            existing_state_resp = ""
            existing_gen = ""
            existing_gen_resp = ""
            existing_idle = ""

            # Try to read existing to preserve the other section
            try:
                if file_path.exists():
                    content = file_path.read_text(encoding="utf-8")
                    # Collect all known headers present and slice between them
                    positions = []
                    for header in (
                        header_char,
                        header_char_resp,
                        header_state,
                        header_state_resp,
                        header_gen,
                        header_gen_resp,
                        header_idle,
                    ):
                        idx = content.find(header)
                        if idx != -1:
                            positions.append((idx, header))
                    positions.sort(key=lambda x: x[0])

                    def section_text(h: str, start_idx: int, end_idx: int) -> str:
                        return content[start_idx + len(h) : end_idx].strip()

                    for i, (pos, header) in enumerate(positions):
                        end = (
                            positions[i + 1][0]
                            if i + 1 < len(positions)
                            else len(content)
                        )
                        body = section_text(header, pos, end)
                        if header == header_char:
                            existing_char = body
                        elif header == header_char_resp:
                            existing_char_resp = body
                        elif header == header_state:
                            existing_state = body
                        elif header == header_state_resp:
                            existing_state_resp = body
                        elif header == header_gen:
                            existing_gen = body
                        elif header == header_gen_resp:
                            existing_gen_resp = body
                        elif header == header_idle:
                            existing_idle = body
            except Exception:
                pass

            new_char = (character_text or existing_char or "").rstrip()
            new_char_resp = (
                character_response_text or existing_char_resp or ""
            ).rstrip()
            new_state = (state_text or existing_state or "").rstrip()
            new_state_resp = (state_response_text or existing_state_resp or "").rstrip()
            new_gen = (gen_state_text or existing_gen or "").rstrip()
            new_gen_resp = (gen_state_response_text or existing_gen_resp or "").rstrip()
            new_idle = (idle_refactor_log or existing_idle or "").rstrip()

            out = header_char + (new_char + "\n" if new_char else "\n")
            out += header_char_resp + (new_char_resp + "\n" if new_char_resp else "\n")
            out += header_state + (new_state + "\n" if new_state else "\n")
            out += header_state_resp + (
                new_state_resp + "\n" if new_state_resp else "\n"
            )
            out += header_gen + (new_gen + "\n" if new_gen else "\n")
            out += header_gen_resp + (new_gen_resp + "\n" if new_gen_resp else "\n")
            out += header_idle + (new_idle + "\n" if new_idle else "\n")

            tmp_path = file_path.parent / f".tmp-{uuid.uuid4().hex}.txt"
            with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
                f.write(out)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp_path), str(file_path))
        except Exception:
            self.logger.debug(
                "[[SAVE_LAST_PROMPT]] Failed to write last_prompt.txt", exc_info=True
            )

    def _ensure_dir(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Cannot create directory '{path}': {e}")

    def _acquire_lock(self, dirpath: Path, timeout_sec: float) -> Optional[Path]:
        lock_path = dirpath / ".lock"
        deadline = time.time() + max(0.1, float(timeout_sec or 0))
        while True:
            try:
                # O_EXCL for atomic creation; fail if exists
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    payload = f"pid={os.getpid()} time={int(time.time())}\n"
                    os.write(fd, payload.encode("utf-8", errors="ignore"))
                finally:
                    os.close(fd)
                return lock_path
            except FileExistsError:
                if time.time() >= deadline:
                    return None
                time.sleep(0.05)
            except Exception:
                # Unexpected error; don't loop forever
                return None

    def _release_lock(self, lock_path: Optional[Path]) -> None:
        try:
            if lock_path and lock_path.exists():
                lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _atomic_write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        tmp_path = file_path.parent / f".tmp-{uuid.uuid4().hex}.json"
        try:
            with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            # atomic replace
            os.replace(str(tmp_path), str(file_path))
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _list_snapshots(self, dirpath: Path) -> List[Path]:
        try:
            return sorted(
                [Path(p) for p in glob(str(dirpath / "state-*.json"))],
                key=lambda p: p.name,
                reverse=True,
            )
        except Exception:
            return []

    def _rotate_snapshots(self, dirpath: Path, keep_last_n: int) -> None:
        try:
            snaps = self._list_snapshots(dirpath)
            if len(snaps) > keep_last_n:
                for p in snaps[keep_last_n:]:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass

    def _ensure_base_state_structure(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all STATE_DEFS sections are structurally complete, including substructures.
        Strategy per section:
        - If section missing/None: deep-copy default
        - If both current and default are dicts: deep-fill missing keys from default (non-destructive)
        - Otherwise: keep current as-is
        - Finally, run sdef['load_fn'] to normalize/coerce shapes (caps/trim handled there)
        Returns a new dict; avoids mutating input.
        """

        def deep_fill(cur: Any, dv: Any) -> Any:
            try:
                if isinstance(cur, dict) and isinstance(dv, dict):
                    out = dict(cur)
                    for k, v in dv.items():
                        if k not in out or out.get(k) is None:
                            try:
                                out[k] = json.loads(json.dumps(v))
                            except Exception:
                                out[k] = v
                        else:
                            out[k] = deep_fill(out[k], v)
                    return out
                # If types differ or non-dict, prefer current (non-destructive)
                return cur
            except Exception:
                return cur

        try:
            result: Dict[str, Any] = dict(states or {})
            sdefs = getattr(self, "STATE_DEFS", None)
            if not isinstance(sdefs, dict):
                return result
            for key, sdef in sdefs.items():
                dv = sdef.get("default")
                cv = result.get(key)
                if cv is None:
                    try:
                        merged = json.loads(json.dumps(dv))
                    except Exception:
                        merged = dv
                elif isinstance(cv, dict) and isinstance(dv, dict):
                    merged = deep_fill(cv, dv)
                else:
                    merged = cv
                # Normalize via load_fn to ensure schema completeness and caps
                try:
                    norm = sdef["load_fn"](merged)
                except Exception:
                    norm = merged if merged is not None else dv
                result[key] = norm
            return result
        except Exception:
            return states or {}

    def _compose_content_data(
        self, model_id: str, states: Dict[str, Any]
    ) -> Dict[str, Any]:
        # 後方互換のためのクリーンアップ: 保存前に不要/廃止フィールドを除去
        try:
            cleaned_states = json.loads(json.dumps(states))  # deep copy
        except Exception:
            cleaned_states = dict(states or {})
        # ベース構造を保存前に補完（不足キーをデフォルトで埋める）
        try:
            cleaned_states = self._ensure_base_state_structure(cleaned_states)
        except Exception:
            pass
        try:
            mh = cleaned_states.get("mental_health")
            if isinstance(mh, dict) and "status" in mh:
                mh.pop("status", None)
        except Exception:
            pass
        return {
            "type": self.PERSISTENT_STATE_TYPE,
            "model_id": model_id,
            "last_interaction_timestamp": cleaned_states.get(
                "last_interaction_timestamp"
            ),
            "last_activity_timestamp": cleaned_states.get(
                "last_activity_timestamp", self._now_iso_utc()
            ),
            "states": {
                k: v
                for k, v in cleaned_states.items()
                if k not in ["last_interaction_timestamp", "last_activity_timestamp"]
            },
        }

    # --- Text normalization utilities ---
    @staticmethod
    def _to_nfc(s: str) -> str:
        try:
            import unicodedata

            return unicodedata.normalize("NFC", s)
        except Exception:
            return s

    @staticmethod
    def _strip_non_bmp(s: str) -> str:
        try:
            # remove codepoints above U+FFFF to avoid mojibake in some renderers
            return "".join(ch for ch in s if ord(ch) <= 0xFFFF)
        except Exception:
            return s

    # --- Text helpers ---
    @staticmethod
    def _strip_code_fences(s: str) -> str:
        """Remove leading/trailing markdown code fences like ``` or ```json."""
        try:
            return re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
        except Exception:
            return s

    # --- Limits semantics & ratio helpers ---
    DEFAULT_LIMIT_SEMANTICS: Dict[str, str] = {
        "stamina": "reserve",
        "pleasure_tolerance": "load",
        "refractory_period": "reserve",
        "pain_tolerance": "reserve",
        "pain_load": "load",
        "temperature_tolerance": "reserve",
        "temperature_load": "load",
        "posture_strain_tolerance": "reserve",
        "posture_strain_load": "load",
        "cognitive_load": "load",
        "stress_tolerance": "reserve",
        "stress_load": "load",
        "sensory_overload_tolerance": "reserve",
        "sensory_load": "load",
        "focus_capacity": "reserve",
        "willpower": "reserve",
        "social_battery": "reserve",
    }

    def _limit_semantics_for_key(
        self, key: str, sem_map: Optional[Dict[str, str]] = None
    ) -> str:
        """Resolve limit semantics for a key using provided map or fallback heuristics.
        Returns 'reserve' or 'load'. Fallbacks mirror existing inline logic.
        """
        try:
            k = (key or "").lower()
            if sem_map and isinstance(sem_map, dict):
                sem = str(sem_map.get(key, "")).lower().strip()
                if sem in ("reserve", "load"):
                    return sem
            reserve_like = {
                "stamina",
                "energy",
                "mana",
                "battery",
                "focus_capacity",
                "willpower",
                "shield",
                "integrity",
                "oxygen",
                "hydration",
                "nutrition",
                "refractory_period",
            }
            load_like = {
                "pleasure_tolerance",
                "cognitive_load",
                "stress",
                "pain",
                "heat",
                "toxicity",
            }
            if k in reserve_like:
                return "reserve"
            if k in load_like:
                return "load"
            return "load"
        except Exception:
            return "load"

    def _limit_ratio_from_item(
        self, key: str, item: Any, sem_map: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Compute 0〜1相当の比率（セマンティクスに基づく）を算出。
        - For 'reserve': ratio = 1 - cur/max (higher means more critical)
        - For 'load': ratio = cur/max
        Returns None if computation fails.
        """
        try:
            cur = float((item or {}).get("current", 0))
            mx = float((item or {}).get("max", (item or {}).get("base", 100)))
            if mx <= 0:
                mx = 100.0
            sem = self._limit_semantics_for_key(key, sem_map)
            if sem == "reserve":
                r = 1.0 - (cur / mx)
            else:
                r = cur / mx
            return max(0.0, min(1.0, r))
        except Exception:
            return None

    # --- Time & display helpers ---
    def _format_timedelta_natural(self, delta: timedelta) -> str:
        """0秒以上の timedelta を日本語の自然文に整形する（TRANSLATIONS準拠）。"""
        try:
            if delta.total_seconds() < 60:
                return "約1分"
            days = delta.days
            hours = delta.seconds // 3600
            minutes = (delta.seconds // 60) % 60
            if days > 0:
                # TRANSLATIONS に days_and_hours がある場合は優先
                return self._get_text("days_and_hours").format(days=days, hours=hours)
            if hours > 0:
                return self._get_text("hours").format(hours=hours)
            return self._get_text("minutes").format(minutes=max(1, minutes))
        except Exception:
            return "約1分"

    # --- Timestamp normalization helpers (ensure TZ-aware strings) ---
    def _ensure_tz_iso(
        self, ts: Optional[str], assume_tz: Optional[tzinfo]
    ) -> Optional[str]:
        """Return ISO8601 with timezone. If ts is naive, attach assume_tz (or UTC). If invalid, return original."""
        try:
            if not ts or not isinstance(ts, str):
                return ts
            s = ts.strip()
            # support date-only form YYYY-MM-DD by assuming 00:00:00
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                try:
                    dt = datetime.fromisoformat(f"{s}T00:00:00")
                except Exception:
                    dt = None
            else:
                dt = self._parse_iso8601(s)
            if not dt:
                return s
            # If user timezone is provided:
            if assume_tz is not None:
                if dt.tzinfo is None:
                    # Interpret naive time as already in user's local time
                    dt = dt.replace(tzinfo=assume_tz)
                else:
                    # Convert aware time to user's local time
                    dt = dt.astimezone(assume_tz)
                return dt.isoformat()
            # No user timezone: ensure at least UTC attachment for naive
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            return ts

    def _normalize_timestamps_in_states(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize known timestamp fields to include TZ info, using user's timezone when naive.
        Targets:
          - last_activity_timestamp, last_interaction_timestamp (top-level)
          - memory.recent[*].timestamp
          - physical_health.timestamps.condition_since
          - mental_health.timestamps.condition_since
        Does not touch knowledge dates or arbitrary strings.
        Returns the same dict instance (mutates in-place) for convenience.
        """
        try:
            user_tz = self._user_zoneinfo()
        except Exception:
            user_tz = None
        try:
            key = "last_activity_timestamp"
            if key in states:
                states[key] = self._ensure_tz_iso(states.get(key), user_tz)
        except Exception:
            pass
        try:
            key = "last_interaction_timestamp"
            if key in states:
                states[key] = self._ensure_tz_iso(states.get(key), user_tz)
        except Exception:
            pass
        try:
            mem = (states.get("memory") or {}).get("recent")
            if isinstance(mem, list):
                for it in mem:
                    try:
                        if isinstance(it, dict) and "timestamp" in it:
                            it["timestamp"] = self._ensure_tz_iso(
                                it.get("timestamp"), user_tz
                            )
                    except Exception:
                        continue
        except Exception:
            pass
        try:
            ph = states.get("physical_health") or {}
            ts = (
                ph.get("timestamps") if isinstance(ph.get("timestamps"), dict) else None
            )
            if ts and ("condition_since" in ts):
                ts["condition_since"] = self._ensure_tz_iso(ts.get("condition_since"), user_tz)
        except Exception:
            pass
        try:
            mh = states.get("mental_health") or {}
            ts = (
                mh.get("timestamps") if isinstance(mh.get("timestamps"), dict) else None
            )
            if ts and ("condition_since" in ts):
                ts["condition_since"] = self._ensure_tz_iso(ts.get("condition_since"), user_tz)
        except Exception:
            pass
        # knowledge.self.identity.{anniversaries,milestones} entries
        try:
            kn = states.get("knowledge") or {}
            self_kn = kn.get("self") if isinstance(kn.get("self"), dict) else None
            ident = (
                self_kn.get("identity")
                if (self_kn and isinstance(self_kn.get("identity"), dict))
                else None
            )
            if ident:
                for key in ("anniversaries", "milestones"):
                    arr = ident.get(key)
                    if isinstance(arr, list):
                        for e in arr:
                            try:
                                if isinstance(e, dict) and ("timestamp" in e):
                                    e["timestamp"] = self._ensure_tz_iso(
                                        e.get("timestamp"), user_tz
                                    )
                            except Exception:
                                continue
        except Exception:
            pass
        return states

    # --- Shared numeric level labeler (0.0〜1.0相当 → 7 levels) ---
    @staticmethod
    def _level_label_7(value: float) -> str:
        """0.0〜1.0相当を7段階の日本語ラベルにマッピング。
        Levels: 全くなし / ごく低い / 低い / 中程度 / やや高い / 高い / 限界
        """
        try:
            v = float(value)
        except Exception:
            v = 0.0
        if v == 0.0:
            return "全くなし"
        if v <= 0.2:
            return "ごく低い"
        if v <= 0.4:
            return "低い"
        if v <= 0.6:
            return "中程度"
        if v <= 0.8:
            return "やや高い"
        if v < 0.98:
            return "高い"
        return "限界"

    def _normalize_text_fields(self, obj: Any) -> Any:
        """Recursively normalize text fields to NFC and strip non-BMP. Keeps shapes intact."""
        try:
            if isinstance(obj, str):
                return self._strip_non_bmp(self._to_nfc(obj))
            if isinstance(obj, list):
                return [self._normalize_text_fields(x) for x in obj]
            if isinstance(obj, dict):
                return {k: self._normalize_text_fields(v) for k, v in obj.items()}
            return obj
        except Exception:
            return obj

    # ========== Strict schema-based key pruning ==========
    def _prune_unknown_keys_in_states(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Return a pruned copy of states that removes any keys not defined by our canonical schema.
        - Top-level: only STATE_DEFS keys + known meta ('last_interaction_timestamp','last_activity_timestamp') are kept.
        - Nested: apply per-section allowlists to drop unexpected keys deterministically.
        This is intentionally strict to prevent schema drift from LLM output.
        """
        try:
            if not isinstance(states, dict):
                return {}

            allowed_top = set(self.STATE_DEFS.keys())
            allowed_meta = {"last_interaction_timestamp", "last_activity_timestamp"}
            out: Dict[str, Any] = {}

            # Helper pruning utilities
            def prune_dict(d: Any, allowed: set) -> Dict[str, Any]:
                if not isinstance(d, dict):
                    return {}
                return {k: v for k, v in d.items() if k in allowed}

            def prune_list_of_dicts(
                lst: Any, allowed_keys: set
            ) -> List[Dict[str, Any]]:
                if not isinstance(lst, list):
                    return []
                out_list: List[Dict[str, Any]] = []
                for it in lst:
                    if isinstance(it, dict):
                        out_list.append(prune_dict(it, allowed_keys))
                return out_list

            def prune_map_values_dict(
                d: Any, allowed_value_keys: set
            ) -> Dict[str, Any]:
                if not isinstance(d, dict):
                    return {}
                pruned: Dict[str, Any] = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        pruned[k] = prune_dict(v, allowed_value_keys)
                return pruned

            # Section-specific pruning
            for key, val in states.items():
                if key in allowed_top:
                    if key == "emotion":
                        # Keep only known emotion categories
                        if isinstance(val, dict):
                            cats = set(self.EMOTION_CATEGORIES)
                            out[key] = {k: val.get(k) for k in cats if k in val}
                        else:
                            out[key] = {c: 0.5 for c in self.EMOTION_CATEGORIES}
                    elif key == "skills" or key == "traits":
                        out[key] = (
                            [str(x).strip() for x in val]
                            if isinstance(val, list)
                            else []
                        )
                    elif key == "boundaries":
                        allowed = {"taboos", "dislikes"}
                        d = prune_dict(val, allowed)
                        # ensure list type
                        d["taboos"] = (
                            d.get("taboos", [])
                            if isinstance(d.get("taboos"), list)
                            else []
                        )
                        d["dislikes"] = (
                            d.get("dislikes", [])
                            if isinstance(d.get("dislikes"), list)
                            else []
                        )
                        out[key] = d
                    elif key == "relationship":
                        allowed = {
                            "type",
                            "level",
                            "trust_label",
                            "trust_score",
                            "user_address",
                            "self_address",
                            "commitments",
                        }
                        d = prune_dict(val, allowed)
                        d["user_address"] = prune_dict(
                            d.get("user_address", {}), {"default", "joking"}
                        )
                        d["self_address"] = prune_dict(
                            d.get("self_address", {}), {"default", "nickname"}
                        )
                        d["commitments"] = prune_list_of_dicts(
                            d.get("commitments", []), {"to", "kind", "summary", "due"}
                        )
                        out[key] = d
                    elif key == "goal":
                        allowed = {"long_term", "mid_term", "short_term", "routine"}
                        d = prune_dict(val, allowed)
                        # prune map values
                        d["long_term"] = prune_map_values_dict(
                            d.get("long_term", {}), {"progress", "priority"}
                        )
                        d["mid_term"] = prune_map_values_dict(
                            d.get("mid_term", {}), {"progress", "priority"}
                        )
                        d["short_term"] = prune_map_values_dict(
                            d.get("short_term", {}), {"progress", "priority"}
                        )
                        d["routine"] = prune_map_values_dict(
                            d.get("routine", {}), {"priority", "start_time", "end_time"}
                        )
                        out[key] = d
                    elif key == "context":
                        allowed = {
                            "place",
                            "atmosphere",
                            "details",
                            "affordances",
                            "constraints",
                            "action",
                        }
                        d = prune_dict(val, allowed)
                        d["action"] = prune_dict(d.get("action", {}), {"user"})
                        # normalize lists to list[str]
                        for lk in ("details", "affordances", "constraints"):
                            if isinstance(d.get(lk), list):
                                d[lk] = [
                                    str(x).strip()
                                    for x in d[lk]
                                    if isinstance(x, (str, int, float))
                                ]
                            else:
                                d[lk] = []
                        out[key] = d
                    elif key == "tone":
                        out[key] = prune_dict(val, {"effects"})
                        if not isinstance(out[key].get("effects"), list):
                            out[key]["effects"] = []
                    elif key == "memory":
                        d = prune_dict(val, {"recent"})
                        d["recent"] = prune_list_of_dicts(
                            d.get("recent", []),
                            {"content", "impression_score", "timestamp", "tags"},
                        )
                        out[key] = d
                    elif key == "inventory":
                        out[key] = prune_list_of_dicts(
                            val, {"name", "description", "quantity", "equipped", "slot"}
                        )
                    elif key == "physical_health":
                        allowed = {
                            "condition",
                            "sensation",
                            "timestamps",
                            "limits",
                            "needs",
                            "reproductive",
                            "issues",
                            "conditions",
                        }
                        d = prune_dict(val, allowed)
                        d["timestamps"] = prune_dict(
                            d.get("timestamps", {}), {"condition_since"}
                        )
                        d["limits"] = prune_map_values_dict(
                            d.get("limits", {}), {"current", "max"}
                        )
                        # force whitelist of known limit keys only
                        known_limits = {
                            "stamina",
                            "pleasure_tolerance",
                            "refractory_period",
                            "pain_tolerance",
                            "pain_load",
                            "temperature_tolerance",
                            "temperature_load",
                            "posture_strain_tolerance",
                            "posture_strain_load",
                        }
                        d["limits"] = {
                            k: v
                            for k, v in d.get("limits", {}).items()
                            if k in known_limits
                        }
                        d["needs"] = prune_dict(
                            d.get("needs", {}),
                            {
                                "hunger",
                                "thirst",
                                "sleepiness",
                                "libido",
                                "hygiene",
                                "social_interaction",
                                "movement",
                                "thermoregulation",
                                "restroom",
                            },
                        )
                        d["reproductive"] = prune_dict(
                            d.get("reproductive", {}),
                            {
                                "status",
                                "cycle_day",
                                "is_pregnant",
                                "pregnancy_progress",
                            },
                        )
                        # issues: list[str]; conditions: list[dict{name,intensity}]
                        if not isinstance(d.get("issues"), list):
                            d["issues"] = []
                        if isinstance(d.get("conditions"), list):
                            d["conditions"] = prune_list_of_dicts(
                                d.get("conditions", []), {"name", "intensity"}
                            )
                        else:
                            d["conditions"] = []
                        out[key] = d
                    elif key == "posture":
                        allowed = {"character", "user"}
                        d = prune_dict(val, allowed)
                        d["character"] = prune_dict(
                            d.get("character", {}),
                            {"position", "support", "movement", "relative"},
                        )
                        d["user"] = prune_dict(
                            d.get("user", {}),
                            {"position", "support", "movement", "relative"},
                        )
                        out[key] = d
                    elif key == "mental_health":
                        allowed = {
                            "condition",
                            "mood",
                            "timestamps",
                            "limits",
                            "needs",
                            "dynamics",
                        }
                        d = prune_dict(val, allowed)
                        d["timestamps"] = prune_dict(
                            d.get("timestamps", {}), {"condition_since"}
                        )
                        # limits set
                        d["limits"] = prune_map_values_dict(
                            d.get("limits", {}), {"current", "max"}
                        )
                        known_m_limits = {
                            "cognitive_load",
                            "stress_load",
                            "focus_capacity",
                            "social_battery",
                        }
                        d["limits"] = {
                            k: v
                            for k, v in d.get("limits", {}).items()
                            if k in known_m_limits
                        }
                        # needs set
                        d["needs"] = prune_dict(
                            d.get("needs", {}),
                            {"novelty", "solitude", "social_interaction"},
                        )
                        # dynamics set
                        d["dynamics"] = prune_dict(
                            d.get("dynamics", {}),
                            {"emotional_volatility", "learning_rate"},
                        )
                        out[key] = d
                    elif key == "sexual_development":
                        allowed = {
                            "description",
                            "favorite_acts",
                            "experience_score",
                            "parts",
                        }
                        d = prune_dict(val, allowed)
                        if not isinstance(d.get("favorite_acts"), list):
                            d["favorite_acts"] = []
                        if not isinstance(d.get("parts"), dict):
                            d["parts"] = {}
                        else:
                            parts_out: Dict[str, Dict[str, Any]] = {}
                            for pname, pval in d.get("parts", {}).items():
                                if isinstance(pval, dict):
                                    parts_out[str(pname)] = prune_dict(
                                        pval, {"sensitivity", "development_progress"}
                                    )
                            d["parts"] = parts_out
                        out[key] = d
                    elif key == "desire":
                        if isinstance(val, dict):
                            desired = {
                                k: val.get(k) for k in self.DESIRE_KEYS if k in val
                            }
                            out[key] = desired
                        else:
                            out[key] = {k: 0.5 for k in self.DESIRE_KEYS}
                    elif key == "internal_monologue":
                        allowed = {
                            "thought",
                            "cognitive_focus",
                            "future_prediction",
                            "options",
                            "risk",
                            "reward",
                            "item_of_interest",
                        }
                        d = prune_dict(val, allowed)
                        # options: only a/b with known subkeys
                        if isinstance(d.get("options"), dict):
                            opts = {}
                            for opt_key in ("a", "b"):
                                if isinstance(d["options"].get(opt_key), dict):
                                    opts[opt_key] = prune_dict(
                                        d["options"][opt_key],
                                        {
                                            "label",
                                            "future_prediction",
                                            "risk",
                                            "reward",
                                        },
                                    )
                            d["options"] = opts
                        else:
                            d["options"] = {}
                        out[key] = d
                    elif key == "knowledge":
                        allowed = {"user", "self"}
                        d = prune_dict(val, allowed)
                        d["user"] = prune_dict(
                            d.get("user", {}), {"likes", "dislikes", "identity"}
                        )
                        d["self"] = prune_dict(
                            d.get("self", {}), {"identity", "strengths", "weaknesses"}
                        )
                        # identities
                        d["user"]["identity"] = prune_dict(
                            d["user"].get("identity", {}), {"notes"}
                        )
                        d["self"]["identity"] = prune_dict(
                            d["self"].get("identity", {}),
                            {
                                "species",
                                "gender",
                                "notes",
                                "default_outfit",
                                "anniversaries",
                                "milestones",
                            },
                        )
                        # ensure lists
                        for k in ("likes", "dislikes"):
                            if not isinstance(d["user"].get(k), list):
                                d["user"][k] = []
                        for k in ("strengths", "weaknesses"):
                            if not isinstance(d["self"].get(k), list):
                                d["self"][k] = []
                        out[key] = d
                    else:
                        # Fallback: keep as is
                        out[key] = val
                elif key in allowed_meta:
                    out[key] = val
                # else drop unknown top-level
            return out
        except Exception:
            # On any failure, return minimal filtered by top-level keys only
            try:
                allowed_top = set(self.STATE_DEFS.keys())
                allowed_meta = {"last_interaction_timestamp", "last_activity_timestamp"}
                return {
                    k: v
                    for k, v in states.items()
                    if k in allowed_top or k in allowed_meta
                }
            except Exception:
                return {}

    def _load_all_states_file(
        self, user: Any, model_id: str
    ) -> Optional[Dict[str, Any]]:
        """ファイルから状態を読み込む。
        
        Args:
            user: ユーザーオブジェクト
            model_id: モデルID
            
        Returns:
            読み込まれた状態、または None（ファイルが存在しない場合）
            
        Raises:
            StatePersistenceError: 読み込み中に重大なエラーが発生した場合
        """
        dirpath = self._get_state_dir(user, model_id)
        try:
            if not dirpath.exists():
                return None
            latest = dirpath / "latest.json"
            candidates: List[Path] = []
            if latest.exists():
                candidates.append(latest)
            candidates.extend(self._list_snapshots(dirpath))
            
            for fp in candidates:
                try:
                    with fp.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        self.logger.warning(f"[[PERSISTENCE]] Invalid JSON structure in {fp}")
                        continue
                    
                    # content wrapper expected
                    loaded_states = (
                        data.get("states")
                        if isinstance(data.get("states"), dict)
                        else None
                    )
                    if loaded_states is None:
                        self.logger.warning(f"[[PERSISTENCE]] No 'states' key found in {fp}")
                        continue
                        
                    final_states = {
                        key: sdef["load_fn"](loaded_states.get(key, sdef["default"]))
                        for key, sdef in self.STATE_DEFS.items()
                    }
                    final_states["last_interaction_timestamp"] = data.get(
                        "last_interaction_timestamp"
                    )
                    final_states["last_activity_timestamp"] = data.get(
                        "last_activity_timestamp"
                    )
                    
                    # Prune unknown keys and normalize timestamps
                    try:
                        final_states = self._prune_unknown_keys_in_states(final_states)
                    except Exception as e:
                        self.logger.debug(f"[[PERSISTENCE]] Failed to prune keys: {e}")
                        
                    # Normalize timestamps with user's timezone where naive
                    try:
                        final_states = self._normalize_timestamps_in_states(
                            final_states
                        )
                    except Exception as e:
                        self.logger.debug(f"[[PERSISTENCE]] Failed to normalize timestamps: {e}")
                        
                    self.logger.info(f"[[PERSISTENCE]] Successfully loaded states from {fp}")
                    return final_states
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"[[PERSISTENCE]] JSON decode error in {fp}: {e}")
                    continue
                except IOError as e:
                    self.logger.warning(f"[[PERSISTENCE]] IO error reading {fp}: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"[[PERSISTENCE]] Unexpected error loading {fp}: {e}")
                    continue
                    
            self.logger.info(f"[[PERSISTENCE]] No valid state files found in {dirpath}")
            return None
            
        except Exception as e:
            self.logger.error(f"[[PERSISTENCE]] Critical error loading states: {e}", exc_info=True)
            raise StatePersistenceError(f"Failed to load states for user {user}, model {model_id}") from e

    def _save_all_states_file(
        self, user: Any, model_id: str, states: Dict[str, Any]
    ) -> None:
        """状態をファイルに保存する。
        
        Args:
            user: ユーザーオブジェクト
            model_id: モデルID
            states: 保存する状態データ
            
        Raises:
            StatePersistenceError: 保存中にエラーが発生した場合
        """
        dirpath = self._get_state_dir(user, model_id)
        self._ensure_dir(dirpath)
        
        lock = self._acquire_lock(dirpath, self.LOCK_TIMEOUT_SECONDS)
        if lock is None:
            raise StatePersistenceError(f"Timeout acquiring lock for '{dirpath}'.")
            
        try:
            # 状態の正規化処理
            try:
                states = self._normalize_text_fields(states)
            except Exception as e:
                self.logger.warning(f"[[PERSISTENCE]] Text normalization failed: {e}")
                
            try:
                states = self._normalize_timestamps_in_states(states)
            except Exception as e:
                self.logger.warning(f"[[PERSISTENCE]] Timestamp normalization failed: {e}")
                
            try:
                states = self._prune_unknown_keys_in_states(states)
            except Exception as e:
                self.logger.warning(f"[[PERSISTENCE]] Key pruning failed: {e}")
                
            content = self._compose_content_data(model_id, states)
            
            # Write latest.json atomically
            latest = dirpath / "latest.json"
            try:
                self._atomic_write_json(latest, content)
                self.logger.debug(f"[[PERSISTENCE]] Saved latest.json to {latest}")
            except Exception as e:
                raise StatePersistenceError(f"Failed to write latest.json") from e

            # Snapshot rotation
            try:
                now_utc = datetime.now(timezone.utc)
                snap_name = now_utc.strftime("state-%Y%m%dT%H%M%SZ.json")
                snap_path = dirpath / snap_name
                self._atomic_write_json(snap_path, content)
                
                keep_n = max(1, int(getattr(self.valves, "keep_last_n", 20)))
                self._rotate_snapshots(dirpath, keep_n)
                self.logger.debug(f"[[PERSISTENCE]] Created snapshot {snap_name}")
            except Exception as e:
                # スナップショット失敗は警告のみ（latest.jsonは保存済み）
                self.logger.warning(f"[[PERSISTENCE]] Snapshot creation failed: {e}")
                
        except StatePersistenceError:
            # 既に適切な例外なので再スロー
            raise
        except Exception as e:
            self.logger.error(f"[[PERSISTENCE]] Unexpected error saving states: {e}", exc_info=True)
            raise StatePersistenceError(f"Failed to save states for user {user}, model {model_id}") from e
        finally:
            self._release_lock(lock)
        # （メモ）出力フォーマット系のプロンプト関数は、読みやすさのため
        # 生成LLM/更新LLMセクション直前へ移動しました。

    # --- Prompt minify helper -------------------------------------------------
    def _minify_prompt_text(self, text: str) -> str:
        """Minify large prompt text for token savings without changing semantics.
        - Collapses excessive whitespace and blank lines
        - Compacts indented JSON blocks by removing indentation and unnecessary spaces
        - Keeps headings and bullet markers but trims trailing spaces
        """
        try:
            s = text

            # First, try to compact JSON-like fenced or inline blocks: {...}
            # We find balanced braces up to a certain size and compact their inner spacing safely via json roundtrip
            def _compact_json_block(match: re.Match) -> str:
                block = match.group(0)
                try:
                    obj = json.loads(block)
                    # Use separators to remove spaces after commas/colons
                    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    return block

            # Compact top-level JSON objects (not inside code fences)
            s = re.sub(r"\{[\s\S]*?\}", _compact_json_block, s)

            # Collapse 3+ newlines to 2 to keep section spacing but reduce height
            s = re.sub(r"\n{3,}", "\n\n", s)
            # Trim spaces at line ends
            s = re.sub(r"[ \t]+\n", "\n", s)
            # Collapse sequences of 2+ spaces outside of JSON-ish tokens
            s = re.sub(r"([^\S\n]{2,})", " ", s)
            return s.strip()
        except Exception:
            return text

    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        LLMの応答テキストから、トップレベルのJSONオブジェクト領域を堅牢に抽出する。
        - 文字列/エスケープを考慮し、波括弧の深さでバランスをとる
        - 複数のトップレベルオブジェクトが連続する場合は、最初の開始～最後の終了までを切り出す
        - 余計な前後テキストやコードフェンスは自動的に除外される
        """
        if not response_text:
            return None
        s = response_text.strip()
        n = len(s)
        # まず、先頭/末尾付近のコードフェンスや不可視文字を軽く除去（抽出の妨げを減らす）
        try:
            s = self._strip_code_fences(s)
            # BOM / ゼロ幅 / NBSP を除去
            s = s.replace("\ufeff", "")
            s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
            s = s.replace("\u2060", "").replace("\xa0", " ")
        except Exception:
            pass

        i = 0
        in_str = False
        esc = False
        depth = 0
        first_start = -1
        last_end = -1
        found_any = False

        while i < n:
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    if depth == 0 and first_start == -1:
                        first_start = i
                    depth += 1
                elif ch == "}":
                    if depth > 0:
                        depth -= 1
                        if depth == 0:
                            last_end = i
                            found_any = True
            i += 1

        if (
            found_any
            and first_start != -1
            and last_end != -1
            and last_end > first_start
        ):
            if self._is_debug_enabled():
                try:
                    preview = s[max(0, first_start - 80) : min(len(s), last_end + 81)]
                    self.logger.debug(
                        f"[[JSON_EXTRACT]] span=({first_start},{last_end}) preview={self._dbg_trunc(preview, 800)}"
                    )
                except Exception:
                    pass
            return s[first_start : last_end + 1]

        # フォールバック: 単純な最初/最後の波括弧での抽出
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return s[start : end + 1]

        self.logger.warning(
            f"[[EXTRACTOR_FAILURE]] No JSON-like object found in the LLM response. Response preview: {s[:500]}..."
        )
        return None

    def _validate_states_from_json(
        self, json_data: Dict, *, is_diff: bool = False
    ) -> Dict[str, Any]:
        """
        辞書形式のJSONデータを受け取り、各キーを検証して整形された状態辞書を返す。
        AIからの応答を基準にループし、キーの大小文字の揺れを吸収する堅牢な実装。
        """
        # どのようなキーがバリデーションに来たか、最初にログで確認する
        self.logger.debug(
            f"[[VALIDATOR_INPUT]] Attempting to validate JSON data with keys: {list(json_data.keys())}"
        )

        validated_states = {}
        if not isinstance(json_data, dict):
            self.logger.error("[[VALIDATOR_FAILURE]] Input data is not a dictionary.")
            return validated_states

        # AIが返した実際のデータを基準にループする
        for received_key, received_value in json_data.items():
            # LLMが返すキーの大小文字の揺れを吸収するため、小文字に正規化する
            normalized_key = received_key.lower()

            # 正規化されたキーが、我々が定義した状態(STATE_DEFS)に存在するかチェック
            if normalized_key in self.STATE_DEFS:
                try:
                    # STATE_DEFSから正しい定義(sdef)を取得
                    sdef = self.STATE_DEFS[normalized_key]
                    # バリデーション関数を実行し、正規化されたキーで結果を保存
                    validated_val = sdef["load_fn"](received_value)
                    # 差分モードでは、バリデータが埋めたデフォルトを取り除き、
                    # 受信したキーに限定して出力する（未指定サブキーでの上書き空白化を防止）
                    if (
                        is_diff
                        and isinstance(received_value, dict)
                        and isinstance(validated_val, dict)
                    ):

                        def _prune(
                            to_val: Dict[str, Any], from_raw: Dict[str, Any]
                        ) -> Dict[str, Any]:
                            out: Dict[str, Any] = {}
                            for k, raw_v in from_raw.items():
                                if k in to_val:
                                    v = to_val[k]
                                    if isinstance(raw_v, dict) and isinstance(v, dict):
                                        out[k] = _prune(v, raw_v)
                                    else:
                                        out[k] = v
                            return out

                        validated_val = _prune(validated_val, received_value)
                    validated_states[normalized_key] = validated_val
                    self.logger.debug(
                        f"[[VALIDATOR_SUCCESS]] Successfully validated key: '{received_key}' -> '{normalized_key}'"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"[[VALIDATOR_WARNING]] Failed to validate state for key '{received_key}'. Error: {e}"
                    )
                    # バリデーション失敗時はデフォルト値を使用
                    validated_states[normalized_key] = self.STATE_DEFS[
                        normalized_key
                    ].get("default", {})
            else:
                # 予期しないキーがLLMから送られてきた場合、それをログに記録して無視する
                self.logger.warning(
                    f"[[VALIDATOR_SKIP]] Received unexpected key '{received_key}' from LLM. It will be ignored."
                )

        # 最終的にバリデーションが成功したかどうかの結果をログに残す
        if not validated_states:
            self.logger.error(
                "[[VALIDATOR_FAILURE]] Validation resulted in an empty dictionary. No states were recognized or updated."
            )
        else:
            self.logger.debug(
                f"[[VALIDATOR_COMPLETE]] Finished validation. Found {len(validated_states)} valid states."
            )
            if self._is_debug_enabled():
                try:
                    self.logger.debug(
                        f"[[VALIDATED_STATES_JSON]] {self._dbg_json(validated_states, 3000)}"
                    )
                except Exception:
                    pass

        try:
            # Ensure any unexpected nested keys are pruned aggressively
            validated_states = self._prune_unknown_keys_in_states(validated_states)
        except Exception:
            pass
        return validated_states

    def _parse_llm_response_for_states(
        self, llm_response: str, *, is_diff: bool = False
    ) -> Dict[str, Any]:
        """LLMの応答から内部状態を解析する。
        
        標準のjsonライブラリを優先し、単純な修復を試みた後も失敗した場合は空の状態を返す。
        
        Args:
            llm_response: LLMからの応答テキスト
            is_diff: 差分モードかどうか
            
        Returns:
            解析された状態データ（失敗時は空辞書）
            
        Raises:
            JSONParsingError: JSON解析に完全に失敗した場合（現在は空辞書を返す）
        """
        json_string = self._extract_json_from_response(llm_response)
        if not json_string:
            self.logger.warning("[[PARSER]] No JSON found in LLM response")
            return {}

        potential_states = {}

        # --- Stage 1: 標準のjsonライブラリで直接パースを試みる ---
        try:
            potential_states = json.loads(json_string)
            self.logger.info(
                "[[PARSER_INFO]] Successfully parsed JSON with standard library."
            )
            if self._is_debug_enabled():
                try:
                    self.logger.debug(
                        f"[[PARSER_JSON]] parsed={self._dbg_json(potential_states, 4000)}"
                    )
                except Exception:
                    pass
                    
        except json.JSONDecodeError as parse_error:
            self.logger.warning(
                f"[[PARSER_INFO]] Standard JSON parsing failed: {parse_error}. Attempting to repair..."
            )

            # --- Stage 2: 修復ロジックを挟んで再挑戦 ---
            repaired_json_string = ""
            try:
                repaired_json_string = self._repair_llm_json(json_string)
                potential_states = json.loads(repaired_json_string)
                self.logger.info(
                    "[[PARSER_INFO]] Successfully parsed JSON after repair."
                )
                if self._is_debug_enabled():
                    try:
                        self.logger.debug(
                            f"[[PARSER_JSON_REPAIRED]] parsed={self._dbg_json(potential_states, 4000)}"
                        )
                    except Exception:
                        pass
                        
            except json.JSONDecodeError as repair_error:
                # --- Stage 3: 連続する複数のトップレベルJSONオブジェクトを検出し、結合する ---
                try:
                    objs = self._split_top_level_json_objects(json_string)
                    if len(objs) <= 1:
                        # 修復後文字列でもトライ
                        objs = self._split_top_level_json_objects(repaired_json_string)
                        
                    if len(objs) > 1:
                        self.logger.info(
                            f"[[PARSER_INFO]] Detected {len(objs)} top-level JSON objects; attempting merge."
                        )
                        merged: Dict[str, Any] = {}
                        for i, part in enumerate(objs):
                            try:
                                d = json.loads(part)
                                if isinstance(d, dict):
                                    merged = self._deep_merge_dicts(merged, d)
                                else:
                                    self.logger.warning(
                                        f"[[PARSER_SKIP]] Part #{i+1} is not an object; skipping."
                                    )
                            except json.JSONDecodeError as part_error:
                                self.logger.warning(
                                    f"[[PARSER_SKIP]] Failed to parse part #{i+1}: {part_error}"
                                )
                            except Exception as part_error:
                                self.logger.warning(
                                    f"[[PARSER_SKIP]] Unexpected error in part #{i+1}: {part_error}"
                                )
                                
                        if merged:
                            potential_states = merged
                            self.logger.info(
                                f"[[PARSER_INFO]] Successfully merged {len(objs)} JSON objects"
                            )
                            if self._is_debug_enabled():
                                try:
                                    self.logger.debug(
                                        f"[[PARSER_JSON_MERGED]] parsed={self._dbg_json(potential_states, 4000)}"
                                    )
                                except Exception:
                                    pass
                        else:
                            self.logger.error(
                                f"[[PARSER_ERROR]] Failed to merge multi-object JSON. Original error: {repair_error}"
                            )
                            return {}
                    else:
                        self.logger.error(
                            f"[[PARSER_ERROR]] All JSON parsing attempts failed. Error: {repair_error}"
                        )
                        return {}
                        
                except Exception as merge_error:
                    self.logger.error(
                        f"[[PARSER_ERROR]] Exception during multi-object merge: {merge_error}",
                        exc_info=True,
                    )
                    return {}

        # --- 共通の後処理 ---
        if not isinstance(potential_states, dict):
            self.logger.warning(
                f"[[PARSER_WARNING]] Parsed data is not a dictionary: {type(potential_states)}"
            )
            return {}

        # まず、トップレベルであるべきキーが入れ子先に誤って配置された場合に『持ち上げ』て補正する
        try:
            potential_states = self._hoist_misnested_states(potential_states)
        except Exception as e:
            self.logger.debug(
                f"[[PARSER_POST_FIX]] Hoist misnested states failed (ignored): {e}",
                exc_info=True,
            )

        # memoryキーの構造エラーだけは、既知の問題として特別に修復
        if "memory" in potential_states and isinstance(
            potential_states["memory"], list
        ):
            self.logger.info("[[PARSER_INFO]] Correcting 'memory' key structure.")
            potential_states["memory"] = {"recent": potential_states["memory"]}

        try:
            potential_states = self._normalize_text_fields(potential_states)
        except Exception as e:
            self.logger.debug(f"[[PARSER]] Text normalization failed: {e}")
            pass
        return self._validate_states_from_json(potential_states, is_diff=is_diff)

    def _hoist_misnested_states(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """トップレベルであるべき状態キーが、他のオブジェクトの下に誤って入れ子になっている場合に
        それらをトップレベルへ『持ち上げ』る補正を行う。

        例: {"physical_health": {"posture": {...}}} → {"physical_health": {...}, "posture": {...}}

        ルール:
        - 基準となるトップレベル候補は STATE_DEFS のキー。
        - 既にトップレベルに同名キーが存在する場合は『持ち上げ』を行わない（トップ優先）。
        - 入れ子側は安全のため元から除去する（重複を避ける）。
        - 深さ優先で最初に見つけたオブジェクトを対象にする（複数箇所にあれば先着を採用）。
        """
        if not isinstance(data, dict) or not getattr(self, "STATE_DEFS", None):
            return data

        root = data
        state_keys = set(self.STATE_DEFS.keys())

        # すでにトップレベルにあるキーは対象外
        present_top = set(k for k in root.keys() if k in state_keys)

        hoisted: Dict[str, Any] = {}

        def walk(parent: Union[Dict[str, Any], List[Any]]):
            if isinstance(parent, dict):
                # リスト化して反復中の変更に強くする
                items = list(parent.items())
                for k, v in items:
                    # すでにトップレベルに存在しない、かつ STATE キー名に一致するなら候補
                    if k in state_keys and k not in present_top and isinstance(v, dict):
                        # 持ち上げ予約（同名が未予約のときのみ）
                        if k not in hoisted:
                            hoisted[k] = v
                            # 親から除去（重複防止）
                            try:
                                parent.pop(k, None)
                            except Exception:
                                pass
                        # さらに深く潜らず次へ
                        continue
                    # 再帰探索
                    if isinstance(v, (dict, list)):
                        walk(v)
            elif isinstance(parent, list):
                for x in parent:
                    if isinstance(x, (dict, list)):
                        walk(x)

        walk(root)

        # 予約されたものをトップレベルへ適用（既存トップ優先のため上書きしない）
        for k, v in hoisted.items():
            if k not in root:
                root[k] = v
                if self._is_debug_enabled():
                    try:
                        self.logger.debug(
                            f"[[PARSER_HOIST]] Hoisted misnested state '{k}' to top-level."
                        )
                    except Exception:
                        pass

        return root

    def _split_top_level_json_objects(self, s: str) -> List[str]:
        """テキストから連続したトップレベルのJSONオブジェクト群を切り出して返す。
        - 文字列リテラルとエスケープを考慮し、{ } の深さで区切る
        - 例: '{"a":1}{"b":2}\n {"c":3}' → [ '{"a":1}', '{"b":2}', '{"c":3}' ]
        """
        try:
            if not s:
                return []
            n = len(s)
            i = 0
            objs: List[str] = []
            while i < n:
                # 次のオブジェクト開始を探す
                while i < n and s[i] != "{":
                    i += 1
                if i >= n:
                    break
                start = i
                depth = 0
                in_str = False
                esc = False
                while i < n:
                    ch = s[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    else:
                        if ch == '"':
                            in_str = True
                        elif ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                # オブジェクト終了
                                objs.append(s[start : i + 1])
                                i += 1
                                break
                    i += 1
            return objs
        except Exception:
            return []

    def _repair_llm_json(self, json_string: str) -> str:
        """
        demjson3に頼らず、LLMが生成しがちな不完全なJSONを多段階で修復する、超堅牢なエンジン。
        1. 欠落したカンマを補完する。
        2. 末尾の不要なカンマを削除する。
        3. 余分な閉じ波括弧を縮約（例: ...}} , "key" → ...} , "key"）。
        4. 括弧の不均衡を可能な範囲で調整（過剰な閉じ括弧を末尾から削除/不足分を末尾に追加）。
        """
        # 前処理: フェンスや不可視文字/BOM/制御文字の除去
        s = json_string.strip()
        try:
            s = self._strip_code_fences(s)
        except Exception:
            pass
        # BOM / Zero-width / NBSP を除去
        try:
            s = s.replace("\ufeff", "")
            s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
            s = s.replace("\u2060", "").replace("\xa0", " ")
        except Exception:
            pass
        # 非許可の制御文字（TAB/CR/LF以外の0x00–0x1F）を除去
        try:
            s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)
        except Exception:
            pass

        # ステップ1: 欠落したカンマを補完する (例: {"k":"v"} {"k":"v"})
        # 閉じ括弧/ブラケット `}` `]` の後に、カンマなしで次のキー `"` が続く箇所を探す
        repaired_string = re.sub(r'([}\]])\s*"', r'\1,"', s)

        # ステップ2: 末尾のカンマを削除する (例: {"key": "value",})
        # 閉じ括弧 `}` や閉じブラケット `]` の直前にあるカンマを、後方参照を使って削除
        repaired_string = re.sub(r",\s*([}\]])", r"\1", repaired_string)

        # ステップ3: ダブルクローズを縮約（次のキーや区切りの直前に余分な `}` があるケース）
        #   例: ...}}\s*,\s*\" → ...}\s*,\s*\" / ...}}\s*\"key\" → ...}\s*\"key\"
        #   内部にも複数出現しうるため、変化がなくなるまで反復適用
        prev = None
        while prev != repaired_string:
            prev = repaired_string
            repaired_string = re.sub(r"}\}\s*,\s*\"", r"},\"", repaired_string)
            repaired_string = re.sub(r"}\}\s*(?=\")", r"}", repaired_string)
            # 末尾が `}},` のようなケースも安全側に縮約
            repaired_string = re.sub(r"}\}\s*,\s*(?=[}\]])", r"},", repaired_string)

        # ステップ4: 括弧の不均衡を調整（安全な範囲で）
        try:
            open_curly = repaired_string.count("{")
            close_curly = repaired_string.count("}")
            if close_curly > open_curly:
                # 余剰分だけ末尾の `}` を削る（末尾以外は壊しやすいので触れない）
                excess = close_curly - open_curly
                i = len(repaired_string) - 1
                chars = list(repaired_string)
                while excess > 0 and i >= 0:
                    if chars[i] == "}":
                        chars.pop(i)
                        excess -= 1
                    i -= 1
                repaired_string = "".join(chars)
            elif open_curly > close_curly:
                repaired_string += "}" * (open_curly - close_curly)

            # 角括弧についても軽くバランス
            open_sq = repaired_string.count("[")
            close_sq = repaired_string.count("]")
            if close_sq > open_sq:
                excess = close_sq - open_sq
                i = len(repaired_string) - 1
                chars = list(repaired_string)
                while excess > 0 and i >= 0:
                    if chars[i] == "]":
                        chars.pop(i)
                        excess -= 1
                    i -= 1
                repaired_string = "".join(chars)
            elif open_sq > close_sq:
                repaired_string += "]" * (open_sq - close_sq)
        except Exception:
            # 修復に失敗しても元の文字列を返す
            return repaired_string

        return repaired_string

    # ========= tick_start: ベースライン準備（数値は変更しない） =========
    def _prepare_tick_start_baseline(
        self, loaded_states: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """Prepare baseline states for prompts at turn start without changing any numeric values.
        - Compute a human-readable time context note from last_activity_timestamp
        - Strip volatile timestamps from the copy used in prompts to avoid prompt bloat
        - Return (baseline_states_for_prompt, time_context_note)
        """
        try:
            states = loaded_states or {}
            # 原本のタイムスタンプを読む
            last_activity_ts_str = states.get("last_activity_timestamp")
            if not last_activity_ts_str:
                # 活動タイムスタンプがない場合、最後のインタラクションを使用
                last_activity_ts_str = states.get("last_interaction_timestamp")
            # ベースラインはディープコピーし、タイムスタンプを剥離
            baseline = json.loads(json.dumps(states)) if states else {}
            baseline.pop("last_activity_timestamp", None)
            baseline.pop("last_interaction_timestamp", None)

            # 時間ノートを生成（日本語固定テキスト、出力言語整合は別ディレクティブで）
            note = ""
            now_utc = self._utcnow()
            if last_activity_ts_str:
                last_activity_dt = self._parse_iso8601(last_activity_ts_str) or now_utc
            else:
                # タイムスタンプが全くない場合、現在の時間を基に初回として扱う
                last_activity_dt = now_utc

            delta = now_utc - last_activity_dt
            if delta.total_seconds() < 60:
                # 60秒未満でも最小1秒に切り上げて（0秒は表示しない）、わずかな経過も明示する
                secs = max(1, int(delta.total_seconds()))
                time_str = f"{secs}秒"
            else:
                parts = []
                if delta.days > 0:
                    parts.append(f"{delta.days}日")
                hours = delta.seconds // 3600
                minutes = (delta.seconds // 60) % 60
                if hours > 0:
                    parts.append(f"{hours}時間")
                if minutes > 0:
                    parts.append(f"{minutes}分")
                time_str = "".join(parts) if parts else "約1分"

            # 前回応答時刻（ISO8601形式）を取得
            last_activity_iso = last_activity_ts_str or now_utc.isoformat()

            note = (
                "**【時間的文脈：分析必須】**\n"
                f"前回の活動からの経過時間: **約{time_str}**\n"
                f"**前回応答時刻（新規メモリのタイムスタンプに使用）**: {last_activity_iso}\n"
                "この経過時間により 'snapshot' がどのように変化すべきか、その当時の 'dialog' を思い返しながら厳密に評価し、内部状態に現実的な変化（自然な減衰/回復/気分の揺らぎ/進捗/行動変化 等）を反映させなさい。"
            )

            return baseline, note
        except Exception as e:
            self.logger.debug("tick_start baseline preparation failed", exc_info=True)
            # 失敗時はそのまま返す
            return loaded_states or {}, ""

    # --- Character LLM: Dynamic State Block (compact) ---
    def _render_reference_summary_lines(self, states: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        try:
            key_map = self._get_text("key_map")
        except Exception:
            key_map = {}
        if not isinstance(key_map, dict):
            key_map = {}

        def _lookup(mapping: Any, key: str) -> str:
            if isinstance(mapping, dict):
                return str(mapping.get(key, key))
            return key

        emotion_labels = key_map.get("emotion", {}) if isinstance(key_map, dict) else {}
        desire_labels = key_map.get("desire", {}) if isinstance(key_map, dict) else {}

        try:
            numeric_labels = self._get_text("numeric_key_labels") or {}
        except Exception:
            numeric_labels = {}
        if not isinstance(numeric_labels, dict):
            numeric_labels = {}
        need_labels = (
            numeric_labels.get("needs", {}) if isinstance(numeric_labels, dict) else {}
        )

        # Emotion
        try:
            emotions = states.get("emotion") or {}
            if isinstance(emotions, dict) and emotions:
                parts = []
                for name, val in sorted(
                    emotions.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True
                ):
                    if len(parts) >= 3:
                        break
                    try:
                        fval = max(0.0, min(1.0, float(val)))
                    except Exception:
                        continue
                    label = _lookup(emotion_labels, name)
                    parts.append(f"{label}({int(round(fval * 100))}%)")
                if parts:
                    lines.append("感情バランス: " + ", ".join(parts))
        except Exception:
            pass

        # Relationship
        try:
            rel = states.get("relationship") or {}
            if isinstance(rel, dict) and rel:
                parts = []
                rel_type = (rel.get("type") or rel.get("relationship_type") or "").strip()
                if rel_type:
                    parts.append(rel_type)
                trust_label = (rel.get("trust_label") or "").strip()
                if trust_label:
                    parts.append(f"信頼:{trust_label}")
                else:
                    try:
                        trust_score = float(rel.get("trust_score", 0.5) or 0.5)
                        trust_score = max(0.0, min(1.0, trust_score))
                        parts.append(f"信頼:{self._level_label_7(trust_score)}")
                    except Exception:
                        pass
                user_addr = (rel.get("user_address", {}) or {}).get("default")
                if user_addr:
                    parts.append(f"呼称:{user_addr}")
                if parts:
                    lines.append("関係性: " + " / ".join(parts))
        except Exception:
            pass

        # Boundaries snapshot
        try:
            boundaries = states.get("boundaries") or {}
            if isinstance(boundaries, dict) and boundaries:
                taboos_raw = boundaries.get("taboos")
                dislikes_raw = boundaries.get("dislikes")

                def _collect(items: Any, limit: int = 4) -> List[str]:
                    if not isinstance(items, list):
                        return []
                    cleaned: List[str] = []
                    for val in items:
                        try:
                            s = str(val).strip()
                        except Exception:
                            continue
                        if not s:
                            continue
                        cleaned.append(s)
                        if len(cleaned) >= limit:
                            break
                    return cleaned

                taboos = _collect(taboos_raw)
                dislikes = _collect(dislikes_raw)
                if taboos or dislikes:
                    fragments: List[str] = []
                    if taboos:
                        extra = " 他" if isinstance(taboos_raw, list) and len(taboos_raw) > len(taboos) else ""
                        fragments.append("禁忌: " + ", ".join(taboos) + extra)
                    if dislikes:
                        extra = " 他" if isinstance(dislikes_raw, list) and len(dislikes_raw) > len(dislikes) else ""
                        fragments.append("苦手: " + ", ".join(dislikes) + extra)
                    lines.append("境界線: " + " / ".join(fragments))
        except Exception:
            pass

        # Physical health
        try:
            ph = states.get("physical_health") or {}
            if isinstance(ph, dict) and ph:
                cond = (ph.get("condition") or "").strip()
                sensation = (ph.get("sensation") or "").strip()
                needs = ph.get("needs") or {}
                need_text = None
                if isinstance(needs, dict) and needs:
                    try:
                        key, value = max(needs.items(), key=lambda kv: float(kv[1] or 0.0))
                        fval = max(0.0, min(1.0, float(value or 0.0)))
                        need_label = _lookup(need_labels, key)
                        need_text = f"最優先ニーズ {need_label}:{self._level_label_7(fval)}"
                    except Exception:
                        need_text = None
                parts = []
                if cond:
                    parts.append(f"状態 {cond}")
                if sensation:
                    parts.append(f"体感 {sensation}")
                if need_text:
                    parts.append(need_text)
                if parts:
                    lines.append("身体: " + " / ".join(parts))
        except Exception:
            pass

        # Mental health
        try:
            mh = states.get("mental_health") or {}
            if isinstance(mh, dict) and mh:
                cond = (mh.get("condition") or "").strip()
                mood = (mh.get("mood") or "").strip()
                parts = []
                if cond:
                    parts.append(f"状態 {cond}")
                if mood:
                    parts.append(f"心情 {mood}")
                if parts:
                    lines.append("精神: " + " / ".join(parts))
        except Exception:
            pass

        # Tone
        try:
            tone_state = states.get("tone") or {}
            if isinstance(tone_state, dict) and tone_state:
                mode = (tone_state.get("mode") or tone_state.get("current") or "").strip()
                effects = tone_state.get("effects") or []
                parts = []
                if mode:
                    parts.append(f"声調 {mode}")
                if isinstance(effects, list) and effects:
                    parts.append("影響 " + ", ".join(str(e) for e in effects[:3]))
                if parts:
                    lines.append("声/ムード: " + " / ".join(parts))
        except Exception:
            pass

        # Desire
        try:
            desire_state = states.get("desire") or {}
            if isinstance(desire_state, dict) and desire_state:
                try:
                    strongest = max(
                        desire_state.items(), key=lambda kv: float(kv[1] or 0.0)
                    )
                    weakest = min(
                        desire_state.items(), key=lambda kv: float(kv[1] or 0.0)
                    )
                    strong_label = _lookup(desire_labels, strongest[0])
                    weak_label = _lookup(desire_labels, weakest[0])
                    lines.append(
                        f"欲求バランス: 最も強いのは{strong_label}、落ち着いているのは{weak_label}"
                    )
                except Exception:
                    pass
        except Exception:
            pass

        # Goals and Routine overview (shared helper)
        try:
            goals = states.get("goal") or {}
            if isinstance(goals, dict) and goals:
                def pct(v: Any) -> str:
                    try:
                        return f"{max(0.0, min(1.0, float(v)))*100:.0f}%"
                    except Exception:
                        return "0%"

                tiers = [
                    ("long_term", "長期目標"),
                    ("mid_term", "中期目標"),
                    ("short_term", "短期目標"),
                ]
                for tier_key, tier_label in tiers:
                    tier = goals.get(tier_key) or {}
                    if isinstance(tier, dict) and tier:
                        lines.append(f"【{tier_label}】")
                        items = sorted(
                            tier.items(),
                            key=lambda kv: float((kv[1] or {}).get("priority", 0.0)),
                            reverse=True,
                        )
                        for name, obj in items[:5]:
                            if not isinstance(obj, dict):
                                continue
                            prog = pct(obj.get("progress", 0.0))
                            prio = pct(obj.get("priority", 0.0))
                            lines.append(f"- {name}（進捗 {prog} / 優先度 {prio}）")

                routine = goals.get("routine") or {}
                if isinstance(routine, dict) and routine:
                    lines.append("【ルーティン】")
                    for name, obj in sorted(
                        routine.items(),
                        key=lambda kv: float((kv[1] or {}).get("priority", 0.0)),
                        reverse=True,
                    )[:5]:
                        if not isinstance(obj, dict):
                            continue
                        start = (obj.get("start_time") or "").strip() or "--:--"
                        end = (obj.get("end_time") or "").strip() or "--:--"
                        prio = pct(obj.get("priority", 0.0))
                        lines.append(f"- {name}（{start}–{end} / 優先度 {prio}）")
        except Exception:
            pass

        # Traits
        try:
            traits = states.get("traits") or []
            if isinstance(traits, list) and traits:
                preview = ", ".join(str(t) for t in traits[:5])
                if len(traits) > 5:
                    preview += " 他"
                lines.append("性格特性: " + preview)
        except Exception:
            pass

        # Skills
        try:
            skills = states.get("skills") or []
            if isinstance(skills, list) and skills:
                preview = ", ".join(str(s) for s in skills[:5])
                if len(skills) > 5:
                    preview += " 他"
                lines.append("技能: " + preview)
        except Exception:
            pass

        # Knowledge highlights
        try:
            knowledge = states.get("knowledge") or {}
            if isinstance(knowledge, dict) and knowledge:
                user_info = knowledge.get("user") or {}
                user_identity = user_info.get("identity") if isinstance(user_info, dict) else None
                user_name = (user_identity.get("name") or "").strip() if isinstance(user_identity, dict) else ""
                likes = user_info.get("likes") if isinstance(user_info, dict) else None
                dislikes = user_info.get("dislikes") if isinstance(user_info, dict) else None
                parts = []
                if user_name:
                    parts.append(f"名前: {user_name}")
                if isinstance(likes, list) and likes:
                    sample = ", ".join(str(x) for x in likes[:3])
                    if len(likes) > 3:
                        sample += " 他"
                    parts.append(f"好み: {sample}")
                if isinstance(dislikes, list) and dislikes:
                    sample = ", ".join(str(x) for x in dislikes[:3])
                    if len(dislikes) > 3:
                        sample += " 他"
                    parts.append(f"苦手: {sample}")
                if parts:
                    lines.append("ユーザー情報: " + " / ".join(parts))
        except Exception:
            pass

        # Context
        try:
            context_state = states.get("context") or {}
            if isinstance(context_state, dict) and context_state:
                place = (context_state.get("place") or "").strip()
                atmosphere = (context_state.get("atmosphere") or "").strip()
                details_val = context_state.get("details")
                detail = None
                if isinstance(details_val, list) and details_val:
                    detail = ", ".join(str(x) for x in details_val[:3])
                elif isinstance(details_val, str) and details_val.strip():
                    detail = details_val.strip()
                parts = []
                if place:
                    parts.append(place)
                if atmosphere:
                    parts.append(f"雰囲気 {atmosphere}")
                if detail:
                    parts.append(detail)
                if parts:
                    lines.append("環境: " + " / ".join(parts))
        except Exception:
            pass

        # Posture
        try:
            posture = states.get("posture") or {}
            if isinstance(posture, dict) and posture:
                def _fmt_post(node: Any) -> str:
                    if not isinstance(node, dict):
                        return "未設定"
                    parts = [
                        node.get("position") or "",
                        node.get("support") or "",
                        node.get("relative") or "",
                    ]
                    joined = " / ".join(x for x in parts if x)
                    return joined or "未設定"

                char_post = _fmt_post(posture.get("character"))
                user_post = _fmt_post(posture.get("user"))
                lines.append(f"体勢: あなた={char_post} / 相手={user_post}")
        except Exception:
            pass

        # Internal monologue
        try:
            internal = states.get("internal_monologue") or {}
            if isinstance(internal, dict) and internal:
                thought = (internal.get("thought") or "").strip()
                focus = (internal.get("cognitive_focus") or "").strip()
                parts = []
                if thought:
                    parts.append(thought)
                if focus:
                    parts.append(f"焦点:{focus}")
                if parts:
                    lines.append("思考プロセス: " + " / ".join(parts))
        except Exception:
            pass

        # Memory highlight
        try:
            memory = (states.get("memory") or {}).get("recent")
            if isinstance(memory, list) and memory:
                latest = memory[-1]
                if isinstance(latest, dict):
                    content = (latest.get("content") or "").strip()
                    if content:
                        lines.append("直近の記憶: " + content)
        except Exception:
            pass

        # Sexual development snapshot
        try:
            sexdev = states.get("sexual_development") or {}
            if isinstance(sexdev, dict) and sexdev:
                desc = str(sexdev.get("description", "")).strip()
                if desc:
                    lines.append("性的状態: " + desc)

                exp_val = sexdev.get("experience_score")
                try:
                    if exp_val is not None:
                        exp = max(0.0, min(1.0, float(exp_val)))
                        lines.append(
                            f"経験度: {self._level_label_7(exp)} ({int(round(exp * 100))}%)"
                        )
                except Exception:
                    pass

                fav = sexdev.get("favorite_acts")
                if isinstance(fav, list) and fav:
                    fav_line = ", ".join(
                        str(x) for x in fav[: self.valves.max_favorite_acts]
                    )
                    if fav_line:
                        lines.append("嗜好傾向: " + fav_line)

                parts = sexdev.get("parts")
                if isinstance(parts, dict) and parts:
                    part_labels = key_map.get("sexual_parts", {})
                    sens_label = (
                        part_labels.get("sensitivity")
                        if isinstance(part_labels, dict)
                        else None
                    ) or "感度"
                    dev_label = (
                        part_labels.get("development_progress")
                        if isinstance(part_labels, dict)
                        else None
                    ) or "開発度"

                    def _clamp_level_optional(v: Any) -> Optional[float]:
                        try:
                            return max(0.0, min(1.0, float(v)))
                        except Exception:
                            return None

                    def _level_text(v: Any) -> Optional[str]:
                        lvl = _clamp_level_optional(v)
                        return self._level_label_7(lvl) if lvl is not None else None

                    # sort by development_progress desc, then name
                    sorted_parts = sorted(
                        [
                            (name, info)
                            for name, info in parts.items()
                            if isinstance(info, dict)
                        ],
                        key=lambda item: (
                            -(
                                _clamp_level_optional(
                                    item[1].get("development_progress")
                                )
                                or 0.0
                            ),
                            str(item[0]),
                        ),
                    )

                    entries: List[str] = []
                    max_entries = 4
                    for name, info in sorted_parts:
                        sens_raw = info.get("sensitivity")
                        dev_raw = info.get("development_progress")
                        sens_txt = _level_text(sens_raw) if sens_raw is not None else None
                        dev_txt = _level_text(dev_raw) if dev_raw is not None else None
                        metrics: List[str] = []
                        if sens_txt:
                            metrics.append(f"{sens_label}{sens_txt}")
                        if dev_txt:
                            metrics.append(f"{dev_label}{dev_txt}")
                        summary = name
                        if metrics:
                            summary += " (" + " / ".join(metrics) + ")"
                        entries.append(summary)
                        if len(entries) >= max_entries:
                            break
                    if entries:
                        suffix = ""
                        if len(sorted_parts) > len(entries):
                            suffix = " 他"
                        lines.append("性感帯: " + ", ".join(entries) + suffix)
        except Exception:
            pass

        # Inventory snapshot
        try:
            inv = states.get("inventory") or []
            if isinstance(inv, list) and inv:
                equipped = [
                    it for it in inv if isinstance(it, dict) and it.get("equipped")
                ]
                carried = [
                    it
                    for it in inv
                    if isinstance(it, dict) and not it.get("equipped")
                ]
                if equipped:
                    lines.append("【装備中】")
                    for it in equipped[: self.valves.max_inventory_items]:
                        nm = str(it.get("name", "")).strip() or "(無名)"
                        slot = str(it.get("slot", "none") or "none").strip()
                        desc = str(it.get("description", "")).strip()
                        q = int(it.get("quantity", 1) or 1)
                        meta: List[str] = []
                        if slot and slot != "none":
                            meta.append(f"{slot}")
                        if q != 1:
                            meta.append(f"x{q}")
                        if desc:
                            meta.append(desc)
                        tail = f"（{', '.join(meta)}）" if meta else ""
                        lines.append(f"- {nm}{tail}")
                if carried:
                    lines.append("【持ち物】")
                    for it in carried[
                        : max(0, self.valves.max_inventory_items - len(equipped))
                    ]:
                        nm = str(it.get("name", "")).strip() or "(無名)"
                        desc = str(it.get("description", "")).strip()
                        q = int(it.get("quantity", 1) or 1)
                        meta = []
                        if q != 1:
                            meta.append(f"x{q}")
                        if desc:
                            meta.append(desc)
                        tail = f"（{', '.join(meta)}）" if meta else ""
                        lines.append(f"- {nm}{tail}")
        except Exception:
            pass

        return [line for line in lines if line]

    def _build_character_dynamic_state_block(self, states: Dict[str, Any]) -> str:
        # Build a compact dynamic block; never drop everything on partial errors
        lines: List[str] = []
        try:
            # Unified time sections
            now_utc = self._utcnow()
            user_tz = self._user_zoneinfo()
            # 初回（過去タイムスタンプが存在しない）検出
            try:
                # 初回判定は『前回のユーザーへの応答時刻（last_interaction_timestamp）が未設定』で判断
                is_first_interaction = not bool(states.get("last_interaction_timestamp"))
            except Exception:
                is_first_interaction = False
            if user_tz:
                now_local = now_utc.astimezone(user_tz)
                now_local_str = now_local.isoformat()
                # 時間帯ラベルの補足（current_time_noteの書式に必要）
                try:
                    label_key, w_start, w_end, next_label_key, next_start = (
                        self._time_of_day_label(now_local)
                    )
                    labels_map = self._get_text("time_of_day_labels")
                    label_display = (
                        labels_map.get(label_key, label_key)
                        if isinstance(labels_map, dict)
                        else label_key
                    )
                    next_label_display = (
                        labels_map.get(next_label_key, next_label_key)
                        if isinstance(labels_map, dict)
                        else next_label_key
                    )
                    next_start_local = (
                        next_start.astimezone(user_tz)
                        if getattr(next_start, "astimezone", None)
                        else next_start
                    )
                    next_start_str = (
                        next_start_local.strftime("%H:%M")
                        if hasattr(next_start_local, "strftime")
                        else str(next_start_local)
                    )
                except Exception:
                    label_display = ""
                    next_label_display = ""
                    next_start_str = ""
                # 現在時刻
                lines.append(
                    self._get_text("current_time_note").format(
                        now_local_str=now_local_str,
                        label_display=label_display,
                        next_label_display=next_label_display,
                        next_start=next_start_str,
                    )
                )
                # 時間コンテキスト共通ガイド
                time_guide = self._get_text("time_context_common_guide")
                if isinstance(time_guide, str) and time_guide.strip():
                    lines.append(time_guide)
                # 数値スケール凡例（0〜1の意味付け）
                num_legend = self._get_text("character_numeric_scale_legend")
                if isinstance(num_legend, str) and num_legend.strip():
                    lines.append(num_legend)

            if is_first_interaction:
                # 初回は「初会話」専用ノートのみを出し、空白時間や経過時間の数値表現は避ける
                first_note = self._get_text("first_interaction_note")
                if isinstance(first_note, str) and first_note.strip():
                    lines.append(first_note)
            else:
                # 応答からの経過時間（活動）
                ts_str = states.get("last_activity_timestamp")
                lad = (self._parse_iso8601(ts_str) if ts_str else None) or now_utc
                delta = now_utc - lad
                lines.append(
                    self._get_text("activity_time_elapsed_note").format(
                        time_str=self._format_timedelta_natural(delta)
                    )
                )

                # 会話の空白時間 + 待機回避ヒント
                its = states.get("last_interaction_timestamp")
                lid = (self._parse_iso8601(its) if its else None) or now_utc
                idelta = now_utc - lid
                lines.append(
                    self._get_text("time_elapsed_note").format(
                        time_str=self._format_timedelta_natural(idelta)
                    )
                )
                total_minutes = max(0, int(idelta.total_seconds() // 60))
                if total_minutes < 60:
                    lines.append(self._get_text("idle_autonomy_note_minutes"))
                elif total_minutes < 60 * 24:
                    lines.append(self._get_text("idle_autonomy_note_hours"))
                else:
                    lines.append(self._get_text("idle_autonomy_note_days"))

            # Append reference section with filtered internal state JSON
            try:
                lines.append("")
                # Fixed header regardless of elapsed time
                header = self._get_text("current_internal_state_header")
                if header:
                    lines.append(header)
                try:
                    summary_lines = self._render_reference_summary_lines(states)
                    if summary_lines:
                        lines.extend(summary_lines)
                except Exception:
                    pass
            except Exception:
                pass

            return "\n".join(lines)
        except Exception:
            # On unexpected error, return whatever has been accumulated so far
            try:
                return "\n".join(lines)
            except Exception:
                return ""

    # --- Threshold-estimation support note builder (for analysis LLM) ---
    def _build_threshold_support_note(self, states: Dict[str, Any]) -> str:
        """Build a compact note to help the analysis LLM infer thresholds.
        - Converts needs（0〜1相当） and limits（current/max with semantics） into 7-level labels
            - Includes semantics hints (load/reserve) and current high/mid threshold settings
            - Uses localized labels when available; falls back to raw keys
            - Keeps size small with caps (max 8 needs and 10 limits lines)
        """
        try:
            if not isinstance(states, dict):
                return ""
            labels = self._get_text("numeric_key_labels") or {}
            need_labels = (
                (labels.get("needs") or {}) if isinstance(labels, dict) else {}
            )
            lim_labels = (
                (labels.get("limits") or {}) if isinstance(labels, dict) else {}
            )
            sem_labels = self._get_text("semantics_labels") or {
                "load": "負荷",
                "reserve": "残量(逆評価)",
            }

            def five(v: float) -> str:
                # For consistency, map to the shared 7-level scale
                return self._level_label_7(v)

            lines: List[str] = []
            legend = self._get_text("threshold_support_legend")
            if isinstance(legend, str) and legend:
                lines.append(f"[閾値支援ノート] {legend}")

            # Needs（0〜1相当）
            needs = ((states.get("physical_health") or {}).get("needs")) or {}
            if isinstance(needs, dict) and needs:
                out = []
                for k, v in needs.items():
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    label = need_labels.get(k, k)
                    out.append(f"{label}:{five(val)}")
                    if len(out) >= 8:
                        break
                if out:
                    lines.append("Needs: " + ", ".join(out))

            # Limits (current/max with semantics)
            def eval_ratio(key: str, item: Any) -> Optional[float]:
                sem_map = dict(self.DEFAULT_LIMIT_SEMANTICS)
                return self._limit_ratio_from_item(key, item, sem_map)

            lim_lines: List[str] = []
            for sec, sec_key_prefix in (("physical_health", ""), ("mental_health", "")):
                lm = ((states.get(sec) or {}).get("limits")) or {}
                if not isinstance(lm, dict):
                    continue
                for k, d in lm.items():
                    if not isinstance(d, dict):
                        continue
                    r = eval_ratio(k, d)
                    if r is None:
                        continue
                    name = lim_labels.get(k, k)
                    # semantics label
                    sem_map = dict(self.DEFAULT_LIMIT_SEMANTICS)
                    sem = self._limit_semantics_for_key(k, sem_map)
                    sem_disp = sem_labels.get(sem, sem)
                    lim_lines.append(f"{name}:{five(r)}({sem_disp})")
                    if len(lim_lines) >= 10:
                        break
                if len(lim_lines) >= 10:
                    break
            if lim_lines:
                lines.append("Limits: " + ", ".join(lim_lines))

            # しきい値の表示は廃止（LLMの推論に委ねる）

            # 集約負荷の表示も廃止（要約を軽く保つ）

            # Trend hint: qualitative trend consideration without extra output
            try:
                lines.append(
                    "注記: 閾値支援ノートのラベルは時間文脈適用後に再計算した値を基準に参照する“参考ラベル”。適用前のラベルに拘束力はない。状態遷移が確定できる場合（例: 睡眠→起床など）は整合性を優先して関連キーを一貫して更新し、そうでない場合は過度な一括変更を避ける（依然として**全量再出力は禁止**）。"
                )
            except Exception:
                pass
            return "\n".join(lines)
        except Exception:
            return ""

    async def inlet(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Open WebUI入力前処理: キャラクターLLMへのシステムプロンプトを生成・注入する。
        
        このメソッドはOpen WebUIのFilterプラグインとして、ユーザー入力がLLMに送られる前に
        呼び出されます。キャラクターの内部状態（感情、記憶、目標など）を読み込み、
        システムプロンプトに変換して対話文脈に挿入します。
        
        主な処理フロー:
        1. 新規セッション検出と前回セッションログの注入
        2. キャラクター状態の読み込み（all_states.json）
        3. RAG記憶検索（有効時）と関連記憶の抽出
        4. システムプロンプトの生成（言語設定→状態→トーン→安全性→時刻→演技指示）
        5. インベントリモード別処理（Direct/Inference）
        6. 会話履歴のトリミング（設定に応じて）
        7. アイドル時の自動要約タスクのスケジュール
        
        システム注入の優先順位:
        - 言語設定 (Language directive)
        - 状態優先度指示 (State priority)
        - トーン設定 (Tone)
        - 安全性ガイドライン (Safety)
        - 姿勢リンク (Posture link)
        - 時刻情報 (Time)
        - 演技フラグ (Acting flags)
        - インベントリ指示（Directモード時のみ）
        
        注意:
        - JSON中立指示はキャラクターLLMには注入されません（状態分析LLM用）
        - Directインベントリモードでは、会話ログの後に「Inventory Changes」指示を追記
        
        Args:
            body: Open WebUIから渡されるリクエストボディ（messages, model等を含む）
            __user__: Open WebUIのユーザーオブジェクト（IDや名前を含む）
            __event_emitter__: UIイベント通知用のコールバック関数（オプション）
            **kwargs: その他のキーワード引数
        
        Returns:
            Dict[str, Any]: 処理後のリクエストボディ（システムプロンプトが注入済み）
            
        Raises:
            このメソッドは例外を内部でキャッチし、エラーが発生しても元のbodyを返します。
            ログにエラー詳細が記録されます。
        """
        user_obj = self._get_user_model(__user__)
        if not user_obj:
            return body

        model_id = self._get_model_id(body)
        self._log_inlet(f"STARTED for model '{model_id}'")
        # 新規セッション検出と session log の準備（実際の注入は system 結合後に行う）
        session_log_injected_into_body = False
        try:
            initial_messages = body.get("messages") or []
            prev_log_for_injection: Optional[List[Dict[str, Any]]] = None
            try:
                prev = self._load_last_session_log(user_obj, model_id)
                if isinstance(prev, list) and prev:
                    prev_log_for_injection = prev
            except Exception:
                prev_log_for_injection = None
            # 新規セッションの判定: messages が空、または先頭に system が無い（=最初のユーザー発話のみ等）
            should_inject_prev = False
            try:
                has_system = any(
                    self._is_message_role(m, "system")
                    for m in initial_messages
                )
                if (not initial_messages) or (not has_system):
                    should_inject_prev = True
            except Exception:
                should_inject_prev = bool(not initial_messages)

            # 記録: 最終フィルター呼び出し時刻を更新し、未完了のアイドル要約タスクをキャンセル
            try:
                key = self._idle_key(user_obj, model_id)
                self._last_filter_call[key] = datetime.utcnow()
                self._cancel_idle_refactor_task(key)
            except Exception:
                self.logger.debug(
                    "[[IDLE_REF]] inlet could not update last call or cancel task",
                    exc_info=True,
                )

            # ステップ1: まず状態をロード
            all_states = await self._load_all_states(user_obj, model_id)
            no_states_present = not bool(all_states)
            # tick_start ベースライン準備（数値は変更せず、タイムスタンプを剥離・時間ノート生成）
            baseline_states, inlet_time_note = self._prepare_tick_start_baseline(
                all_states or {}
            )

            # ステップ2: MemoryFocus と 内部状態更新を並列実行し、完了後にプロンプトを構築
            current_user_last_msg = self._extract_last_user_message(body)
            # 直近完了ターン（user→assistant）のペアを抽出（状態更新は前ターン基準で実施）
            last_user_prev, last_assistant_prev = self._extract_last_completed_turn(
                body.get("messages") or []
            )

            # directモードでは前ターンのassistantテキストから所持品差分を抽出して外部diffとして渡す
            external_diff_for_ai: Optional["Filter.StateDiff"] = None
            try:
                if last_assistant_prev:
                    inv_mode = str(
                        getattr(
                            self.valves,
                            "inventory_update_mode",
                            self.INVENTORY_MODE_INFERENCE,
                        )
                    )
                    if inv_mode == self.INVENTORY_MODE_DIRECT:
                        cur_states_for_inv = (
                            await self._load_all_states(user_obj, model_id) or {}
                        )
                        _final_text_unused, _diff = (
                            await self._outlet_process_inventory(
                                mode=inv_mode,
                                llm_response_text=last_assistant_prev,
                                current_states=cur_states_for_inv,
                                user_obj=user_obj,
                                event_emitter=__event_emitter__,
                            )
                        )
                        if _diff is not None:
                            external_diff_for_ai = cast("Filter.StateDiff", _diff)
            except Exception:
                self.logger.debug(
                    "[[INLET]] Failed to pre-parse inventory diff from previous turn.",
                    exc_info=True,
                )

            # 併走: MemoryFocusブロック構築 と 状態更新（初回はスキップ）
            focus_block = None
            mf_task: Optional[asyncio.Task] = None
            if not no_states_present:
                focus_key = self._idle_key(user_obj, model_id)
                if __event_emitter__:
                    try:
                        await self._safe_emit(
                            __event_emitter__,
                            self._ev_status(self._get_text("rag_processing"), False),
                        )
                    except Exception:
                        self.logger.debug(
                            "[[INLET]] Failed to emit rag_processing", exc_info=True
                        )
                mf_task = self._create_background_task(
                    self._build_memory_focus_block(
                        baseline_states or {}, current_user_last_msg, focus_key
                    ),
                    name="inlet_memory_focus"
                )

            # 状態更新に必要な system_prompt はキャラモデルのsystemを使用（従来どおり）
            try:
                model_system_prompt = await self._get_system_prompt_from_model_id(
                    model_id
                )
            except Exception:
                model_system_prompt = None

            async def _maybe_update_states() -> None:
                # 初期生成/更新は model_system_prompt が空でも継続する（警告は内部で通知）
                _lu = last_user_prev or ""
                _la = last_assistant_prev or ""
                await self._update_or_initialize_states_async(
                    last_user_message=_lu,
                    latest_llm_response=_la,
                    system_prompt=model_system_prompt,
                    user_obj=user_obj,
                    model_id=model_id,
                    event_emitter=__event_emitter__,
                    is_narration=bool((_lu and self.narration_pattern.match(_lu))),
                    external_diff=external_diff_for_ai,
                )

            upd_task = None
            try:
                upd_task = self._create_background_task(
                    _maybe_update_states(),
                    name="inlet_state_update"
                )
            except Exception:
                self.logger.debug(
                    "[[INLET]] Failed to schedule state update task.",
                    exc_info=True,
                )

            if mf_task is not None:
                try:
                    focus_block = await mf_task
                except Exception:
                    self.logger.debug(
                        "[[INLET]] MemoryFocus task failed", exc_info=True
                    )
                    focus_block = None

            # MemoryFocus終了時に追加通知は出さない（重複見えを避ける）

            # 状態更新の終了を待機
            if upd_task is not None:
                try:
                    await upd_task
                except Exception:
                    self.logger.debug(
                        "[[INLET]] State update task failed", exc_info=True
                    )

            # 状態更新完了後の最新状態でベースラインを再構築
            try:
                all_states = await self._load_all_states(user_obj, model_id)
                baseline_states, inlet_time_note = self._prepare_tick_start_baseline(
                    all_states or {}
                )
            except Exception:
                self.logger.debug(
                    "[[INLET]] Failed to reload states after update", exc_info=True
                )

            # ステップ3: システムメッセージを最適順で準備（言語 → ナレーション → 状態）
            last_user_message = current_user_last_msg
            try:
                has_existing_system = any(
                    self._is_message_role(m, "system")
                    for m in (body.get("messages") or [])
                )
            except Exception:
                has_existing_system = False
            system_injections_tuple = await self._build_inlet_system_injections(
                all_states=all_states or {},
                baseline_states=baseline_states or {},
                last_user_message=last_user_message,
                user_obj=user_obj,
                model_id=model_id,
                has_existing_system=has_existing_system,
                precomputed_focus_block=(
                    focus_block
                    if isinstance(focus_block, str) and focus_block.strip()
                    else None
                ),
                skip_memory_focus=bool(no_states_present),
            )
            # Unpack: system blocks + inventory tail note (direct: duty / inference: forbid)
            system_injections, inv_tail_note = cast(
                Tuple[List[Dict[str, str]], Optional[str]], system_injections_tuple
            )

            # システムメッセージの統合注入（OpenAI互換：配列の先頭に単一systemとして配置）
            # デバッグ向けに挿入順の概要を簡潔にログ
            try:
                self._log_injection_order(system_injections)
            except Exception:
                self.logger.debug(
                    "[[INLET]] Failed to log injection order.", exc_info=True
                )
            if (
                "messages" in body
                and isinstance(body["messages"], list)
                and system_injections
            ):
                messages = body["messages"]
                # 1) 本フィルター生成のsystem群を順序通りに結合
                try:
                    merged_content = "\n\n".join(
                        [
                            str(m.get("content", ""))
                            for m in system_injections
                            if isinstance(m, dict)
                        ]
                    ).strip()
                except Exception:
                    merged_content = "\n\n".join(
                        [m.get("content", "") for m in system_injections]
                    ).strip()
                if merged_content:
                    # 2) 既に先頭がsystemなら、先頭systemの内容の前段に結合（このフィルターの規約を先に）
                    messages = self._merge_or_insert_top_system(
                        messages, merged_content
                    )

                    if self._is_debug_enabled():
                        try:
                            self.logger.debug(
                                f"[[INLET_SYSTEM_PREVIEW]] merged_first={self._dbg_trunc(merged_content, 1200)}"
                            )
                        except Exception:
                            pass

                    # 注記: last_prompt.txt の体裁調整（会話ログ見出し/最新ユーザー見出しの位置）は
                    # 保存時の整形で行う。ここでは実メッセージ内容を変更しない。

                    # 2.5) 所持品ノートの配置: 常にトップ system の末尾へ追記（ユーザー発言へは追記しない）
                    try:
                        msgs2 = body.get("messages") or []
                        if (
                            msgs2
                            and isinstance(msgs2[0], dict)
                            and msgs2[0].get("role") == "system"
                        ):
                            first = msgs2[0]
                            inv_mode = getattr(
                                self.valves,
                                "inventory_update_mode",
                                self.INVENTORY_MODE_INFERENCE,
                            )
                            note_text = None
                            try:
                                if inv_mode == self.INVENTORY_MODE_DIRECT:
                                    note_text = self._get_text(
                                        "system_note_inventory_change"
                                    )
                                else:
                                    note_text = self._get_text(
                                        "system_note_inventory_change_inference_forbid"
                                    )
                            except Exception:
                                note_text = None
                            if isinstance(note_text, str) and note_text.strip():
                                first["content"] = (
                                    str(first.get("content", "")) + "\n\n" + note_text
                                ).strip()
                    except Exception:
                        self.logger.debug(
                            "[[INLET]] Failed to place inventory note.", exc_info=True
                        )

                    # 2.6) 新規セッション用: 既存の非systemログを抹消（最新ユーザーのみ残す）したうえで、last_session_log.json をトップ system の直後に注入
                    try:
                        if should_inject_prev and prev_log_for_injection:
                            msgs2 = body.get("messages") or []
                            # 最新ユーザー発話のみを抽出
                            latest_user_msg = None
                            for m in reversed(msgs2):
                                if self._is_message_role(m, "user"):
                                    latest_user_msg = {
                                        "role": "user",
                                        "content": str(m.get("content", "")),
                                    }
                                    break
                            new_msgs = []
                            if (
                                msgs2
                                and self._is_message_role(msgs2[0], "system")
                            ):
                                # トップ system を保持
                                new_msgs.append(msgs2[0])
                            # 直近セッションログを挿入
                            new_msgs.extend(prev_log_for_injection)
                            # 現在ターンの最新ユーザーがあれば末尾に付与
                            if latest_user_msg is not None:
                                new_msgs.append(latest_user_msg)
                            body["messages"] = new_msgs
                            session_log_injected_into_body = True
                            self.logger.info(
                                "[[INLET]] Replaced default non-system messages with last_session_log.json and kept latest user."
                            )
                    except Exception:
                        self.logger.debug(
                            "[[INLET]] Failed to inject session log after system.",
                            exc_info=True,
                        )

                    # 4) 以前は互換のために他の system を吸収して1件化していたが、現在は変更しない
                    # inlet での会話トリムは行わない（last_session_log.json 側で管理）

        except Exception as e:
            self._log_inlet(f"ERROR for model '{model_id}': {e}", level=logging.ERROR)
            self.logger.debug("[[INLET]] Trace:", exc_info=True)
        finally:
            # デバッグ用に last_prompt.txt の Character セクションを保存（messages の生JSON）
            try:
                msgs = body.get("messages")
                if isinstance(msgs, list):
                    self._save_last_prompts(
                        character_text=json.dumps(msgs, ensure_ascii=False, indent=2)
                    )
            except Exception:
                self.logger.debug(
                    "[[INLET]] Failed to dump last_prompt.txt (character)",
                    exc_info=True,
                )
            self.logger.info(f"[[INLET FINISHED]] for model '{model_id}'")

        return body

    async def _parse_and_update_inventory_from_text(
        self, text: str, current_inventory: "List[Filter.InventoryItem]", user: Any
    ) -> "Tuple[List[Filter.InventoryItem], str, Optional[Filter.TrimSummary]]":
        """LLM応答テキストから自然言語のインベントリ変更記述を解析し、適用する。
        
        DirectモードでキャラクターLLMが出力した「Inventory Changes」セクションを解析し、
        アイテムの追加・削除・装備変更などを実行します。変更記述部分はテキストから除去されます。
        
        対応する記述形式:
        - **ヘッダー形式**: 
          - 「所持品の変更:」「Inventory Changes:」で始まるセクション
          - 次の行から変更行として解析
          
        - **インライン形式**: 
          - 行頭に「所持品:」「Inventory:」を含む1行形式
          
        - **変更行の構文**:
          - 追加: 「+ アイテム名 (数量)」「得た：アイテム名 x数量」
          - 削除: 「- アイテム名」「失った：アイテム名」
          - 装備: 「⚔ アイテム名を装備」「equipped: アイテム名」
          - 装備解除: 「装備解除：アイテム名」「unequipped: アイテム名」
        
        Args:
            text: LLMの応答テキスト
            current_inventory: 現在のインベントリリスト
            user: ユーザーオブジェクト（エラーログ用）
        
        Returns:
            Tuple[List[InventoryItem], str, Optional[TrimSummary]]:
                - 更新後のインベントリ
                - 変更記述を除去したテキスト
                - トリミング要約（容量超過時のみ）
                
        Raises:
            InventoryProcessingError: インベントリ処理中のエラー（内部でキャッチしログ記録）
            
        Note:
            容量超過時は自動的に最も古い/使用頻度の低いアイテムを削除し、
            TrimSummaryに削除されたアイテムの記録を含めます。
        """
        # 事前にクラス定数で用意した正規表現・アクション集合を使用（DRY）
        header_pattern = self.HEADER_PATTERN
        line_pattern_ja = self.LINE_PATTERN_JA
        line_pattern_en = self.LINE_PATTERN_EN

        lines = text.splitlines()
        cleaned_lines = []
        parsing_active = False
        inventory_changed = False

        new_inventory = json.loads(json.dumps(current_inventory))

        for line in lines:
            line_lang = None  # 'ja' or 'en' when matched; default None
            # 純粋なヘッダ行（次行以降を変更行として解釈）
            if header_pattern.match(line):
                parsing_active = True
                inventory_changed = True
                continue
            # 行内ヘッダ形式の処理: ヘッダを取り除いた残部を1行として解析
            inline_line = line
            inline_matched = False
            if self.INLINE_PREFIX_JA.match(inline_line):
                inline_line = self.INLINE_PREFIX_JA.sub("", inline_line).strip()
                line_lang = "ja"
                inline_matched = True
            elif self.INLINE_PREFIX_EN.match(inline_line):
                inline_line = self.INLINE_PREFIX_EN.sub("", inline_line).strip()
                line_lang = "en"
                inline_matched = True

            if not parsing_active and not inline_matched:
                cleaned_lines.append(line)
                continue

            # inlineの場合は残部、ブロック内の場合は元の行を対象にする
            target = inline_line if inline_matched else line
            match_ja = line_pattern_ja.match(target)
            match_en = line_pattern_en.match(target)

            item_name, description, action_str, quantity_str = None, None, None, None

            if match_ja:
                item_name, description, action_str, quantity_str = match_ja.groups()
                line_lang = "ja"
            elif match_en:
                action_str, item_name, description, quantity_str = match_en.groups()
                line_lang = "en"

            if item_name and action_str:
                quantity = int(quantity_str) if quantity_str else 1
                item_name = item_name.strip()
                action_str = action_str.strip().lower()
                if description:
                    description = description.strip()

                item_index = -1
                for i, item in enumerate(new_inventory):
                    if item["name"].lower() == item_name.lower():
                        item_index = i
                        break

                if action_str in ["装備", "equipped"]:
                    if item_index != -1:
                        if not new_inventory[item_index].get("equipped"):
                            new_inventory[item_index]["equipped"] = True
                            inventory_changed = True
                    else:
                        # 説明があれば新規作成して装備、なければ最小情報で作成
                        new_desc = (
                            description
                            if description
                            else (
                                "Equipped item."
                                if (line_lang == "en")
                                else "装備したアイテム。"
                            )
                        )
                        self.logger.info(
                            f"[[INVENTORY_PARSE]] Creating and equipping new item '{item_name}' with description: '{new_desc}'"
                        )
                        new_inventory.append(
                            {
                                "name": item_name,
                                "description": new_desc,
                                "quantity": 1,
                                "equipped": True,
                                "slot": None,
                            }
                        )
                        inventory_changed = True

                elif action_str in ["装備解除", "unequipped"]:
                    if item_index != -1:
                        if new_inventory[item_index].get("equipped"):
                            new_inventory[item_index]["equipped"] = False
                            inventory_changed = True
                    else:
                        self.logger.warning(
                            f"[[INVENTORY_PARSE]] Tried to unequip non-existent item: {item_name}"
                        )

                elif action_str in ["取得", "acquired"]:
                    if item_index != -1:
                        new_inventory[item_index]["quantity"] += quantity
                        inventory_changed = True
                    else:
                        # 行の言語に合わせてデフォルト説明を設定
                        if description:
                            new_description = description
                        else:
                            new_description = (
                                "Newly acquired item."
                                if line_lang == "en"
                                else "新たに入手したアイテム。"
                            )
                        self.logger.info(
                            f"[[INVENTORY_PARSE]] Acquiring new item '{item_name}' with description: '{new_description}'"
                        )
                        new_inventory.append(
                            {
                                "name": item_name,
                                "description": new_description,
                                "quantity": quantity,
                                "equipped": False,
                                "slot": None,
                            }
                        )
                        inventory_changed = True

                elif action_str in ["喪失", "lost"]:
                    if item_index != -1:
                        new_inventory[item_index]["quantity"] -= quantity
                        if new_inventory[item_index]["quantity"] <= 0:
                            del new_inventory[item_index]
                        inventory_changed = True
                    else:
                        self.logger.warning(
                            f"[[INVENTORY_PARSE]] Tried to lose non-existent item: {item_name}"
                        )

                elif action_str in ["更新", "updated"]:
                    if item_index != -1:
                        if description:
                            self.logger.info(
                                f"[[INVENTORY_PARSE]] Updating description for item '{item_name}'."
                            )
                            new_inventory[item_index]["description"] = description
                            inventory_changed = True
                        else:
                            self.logger.warning(
                                f"[[INVENTORY_PARSE]] 'Update' action for '{item_name}' was called without a description."
                            )
                    else:
                        self.logger.warning(
                            f"[[INVENTORY_PARSE]] Tried to update non-existent item: {item_name}"
                        )

            else:
                cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines)
        trim_summary: Optional[Dict[str, Any]] = None
        # --- Sanitize descriptions (2 sentences + char cap) before overflow handling ---
        try:
            new_inventory = self._sanitize_inventory_descriptions(new_inventory)
        except Exception:
            self.logger.debug(
                "[[INVENTORY_SANITIZE_ERROR]] Failed to sanitize descriptions (direct parse).",
                exc_info=True,
            )
        if inventory_changed and len(new_inventory) > self.valves.max_inventory_items:
            self.logger.warning(
                f"[[INVENTORY_OVERFLOW]] Inventory count ({len(new_inventory)}) exceeds limit ({self.valves.max_inventory_items})."
            )
            if self.valves.auto_trim_inventory_on_overflow:
                before = len(new_inventory)
                new_inventory = self._trim_inventory_if_needed(new_inventory)
                after = len(new_inventory)
                self.logger.info(
                    f"[[INVENTORY_TRIM]] Auto trimmed inventory {before}->{after} using strategy '{self.valves.inventory_trim_strategy}'."
                )
                trim_summary = {
                    "before": before,
                    "after": after,
                    "strategy": self.valves.inventory_trim_strategy,
                }
        return (
            cast(List[Filter.InventoryItem], new_inventory),
            cleaned_text,
            cast(Optional[Filter.TrimSummary], trim_summary),
        )

    def _trim_inventory_if_needed(
        self, items: "List[Filter.InventoryItem]"
    ) -> "List[Filter.InventoryItem]":
        """所持品が上限を超えた場合に戦略に従って圧縮する。
        装備中のアイテムは基本的にトリム対象外とし、
        装備していないアイテムのうち数量が最も少ないものから削除していく。
        """
        limit = self.valves.max_inventory_items
        if len(items) <= limit:
            return items
        strategy = (
            self.valves.inventory_trim_strategy or self.INV_TRIM_UNEQUIPPED_FIRST
        ).lower()
        # 安全にコピー
        work = list(items)

        # ソートキーを定義
        def key_unequipped_first(it):
            # 装備品は最後に残す（equipped=Trueを大きく）→ 降順にしたいのでFalse<Trueを利用
            return (it.get("equipped", False), it.get("quantity", 0))

        def key_quantity_asc(it):
            return (it.get("quantity", 0), it.get("equipped", False))

        if strategy == self.INV_TRIM_QUANTITY_ASC:
            work.sort(key=key_quantity_asc)
        else:
            work.sort(key=key_unequipped_first)
        # 先頭から削除候補（優先的に削る）
        trimmed = work[-limit:]
        # 元の順序は重要でない（状態保存用）。必要なら名称で安定化可能
        return trimmed

    async def outlet(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Open WebUI出力後処理: キャラクターLLM応答を処理し、状態更新をスケジュールする。
        
        このメソッドはキャラクターLLMからの応答を受け取った後に呼び出されます。
        応答テキストを解析し、インベントリ処理やコマンド実行を行い、バックグラウンドで
        状態分析LLMを用いた非同期状態更新をスケジュールします。
        
        主な処理フロー:
        1. LLM応答テキストの抽出とログ保存
        2. 特殊コマンドの検出と実行（#LLMEmotionコマンド）
        3. インベントリ処理（モード別）:
           - Directモード: 応答テキストから「Inventory Changes」ブロックを解析
           - Inferenceモード: 状態分析LLMに委任
        4. バックグラウンド状態更新タスクのスケジュール:
           - 会話履歴と応答から差分を抽出
           - 状態を更新してファイルに保存
           - 変更をイベント通知（設定に応じて）
        5. アイドル時の自動要約タスクのスケジュール
        6. UIへの応答返却（インベントリブロック除去オプション対応）
        
        インベントリモード別の動作:
        - **Directモード**: 
          - 応答から「Inventory Changes」を直接パース
          - 設定によりブロックをUI表示から除去可能
          - 状態分析LLMには常に元の応答を渡す（一貫性のため）
        - **Inferenceモード**: 
          - インベントリ解析を状態分析LLMに委任
          - 応答テキストは改変しない
        
        Args:
            body: Open WebUIから渡される応答ボディ（messages等を含む）
            __user__: Open WebUIのユーザーオブジェクト
            __event_emitter__: UIイベント通知用のコールバック関数（オプション）
            **kwargs: その他のキーワード引数
        
        Returns:
            Dict[str, Any]: 処理後の応答ボディ（UI表示用）
            
        Raises:
            このメソッドは例外を内部でキャッチし、エラーが発生しても元のbodyを返します。
            ログにエラー詳細が記録されます。
            
        Note:
            状態更新は非同期で実行されるため、このメソッドは応答を待たずに即座に返ります。
            これによりユーザー体験を損なわずに状態を更新できます。
        """
        user_obj = self._get_user_model(__user__)
        if not user_obj:
            return body

        model_id = self._get_model_id(body)
        self._log_outlet(f"STARTED for model '{model_id}'")

        try:
            # 記録: 最終フィルター呼び出し時刻を更新し、未完了のアイドル要約タスクをキャンセル
            try:
                key = self._idle_key(user_obj, model_id)
                self._last_filter_call[key] = datetime.utcnow()
                self._cancel_idle_refactor_task(key)
            except Exception:
                self.logger.debug(
                    "[[IDLE_REF]] outlet could not update last call or cancel task",
                    exc_info=True,
                )
            llm_response_text = self._extract_llm_response(body)
            self.logger.debug(f"[[OUTLET]] Raw LLM response: {llm_response_text}")
            if self._is_debug_enabled():
                try:
                    self.logger.debug(
                        f"[[OUTLET_RESP_PREVIEW]] {self._dbg_trunc(llm_response_text, 1200)}"
                    )
                except Exception:
                    pass

            # Save the latest character LLM response (最新1件のみ)
            try:
                # bodyから最新のassistantメッセージを抽出
                latest_assistant_msg = None
                if isinstance(body, dict) and "messages" in body:
                    messages = body.get("messages", [])
                    for msg in reversed(messages):
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            latest_assistant_msg = msg
                            break
                
                # 最新1件のみ保存
                if latest_assistant_msg:
                    self._save_last_prompts(
                        character_response_text=json.dumps(
                            {"messages": [latest_assistant_msg]}, ensure_ascii=False, indent=2
                        )
                    )
                else:
                    # フォールバック: body全体を保存（後方互換）
                    self._save_last_prompts(
                        character_response_text=json.dumps(
                            body, ensure_ascii=False, indent=2
                        )
                    )
            except Exception:
                self.logger.debug(
                    "[[SAVE_LAST_PROMPT]] Failed to save character response (outlet)",
                    exc_info=True,
                )

            current_states = await self._load_all_states(user_obj, model_id)
            if not current_states:
                current_states = {}

            last_user_message = self._extract_last_user_message(body)
            system_prompt = await self._get_system_prompt_from_model_id(model_id)

            # === コマンド検出（/llmemotion ...）: 合成応答に差し替え、内部状態更新は実行しない ===
            cmd_response = await self._outlet_try_handle_command(
                last_user_message, user_obj, model_id
            )
            if cmd_response:
                self.logger.info(
                    "[[COMMAND]] llmemotion command handled (no state update)."
                )
                body = self._update_body_content(body, cmd_response.strip())
                # コマンド応答でも outlet 終了時にアイドル要約をスケジュール
                try:
                    if getattr(self.valves, "idle_refactor_enabled", True):
                        key = self._idle_key(user_obj, model_id)
                        try:
                            self._last_filter_call[key] = datetime.utcnow()
                        except Exception:
                            pass
                        base_count = 0
                        try:
                            st_for_base = (
                                await self._load_all_states(user_obj, model_id) or {}
                            )
                            mem = st_for_base.get("memory") or {}
                            rec = mem.get("recent") or []
                            if isinstance(rec, list):
                                base_count = len(rec)
                        except Exception:
                            base_count = 0
                        try:
                            if isinstance(self._idle_refactor_baseline_count, dict):
                                self._idle_refactor_baseline_count[key] = int(
                                    base_count
                                )
                        except Exception:
                            pass
                        try:
                            delay = int(
                                getattr(
                                    self.valves, "idle_refactor_threshold_seconds", 600
                                )
                                or 600
                            )
                        except Exception:
                            delay = 600
                        self._schedule_idle_refactor_if_needed(
                            key, user_obj, model_id, delay, __event_emitter__
                        )
                        if self._is_debug_enabled():
                            self.logger.debug(
                                f"[[IDLE_REF]] scheduled at outlet end (command path): key={key}, delay={delay}s, baseline_recent={base_count}"
                            )
                except Exception:
                    self.logger.debug(
                        "[[IDLE_REF]] scheduling at outlet (command) failed",
                        exc_info=True,
                    )
                return body

            final_llm_response_for_user, inventory_diff_for_ai = (
                await self._outlet_process_inventory(
                    mode=str(
                        getattr(
                            self.valves,
                            "inventory_update_mode",
                            self.INVENTORY_MODE_INFERENCE,
                        )
                    ),
                    llm_response_text=llm_response_text,
                    current_states=current_states,
                    user_obj=user_obj,
                    event_emitter=__event_emitter__,
                )
            )

            # ユーザーに見せる応答本文を最終的なもので更新（プロンプトでの抑止のみ。除去ロジックは使用しない）
            body = self._update_body_content(body, final_llm_response_for_user)

            # 毎ターン: 既存の last_session_log.json にこのターンの user/assistant を追加し、ターン数でトリムして保存
            try:
                if last_user_message is not None:
                    self._append_completed_turn_to_session_log(
                        user_obj,
                        model_id,
                        last_user_message,
                        final_llm_response_for_user,
                    )
            except Exception:
                self.logger.debug(
                    "[[OUTLET]] Failed to append turn into last_session_log.json",
                    exc_info=True,
                )

            # Update timestamps at the end of user response processing
            try:
                now_ts = self._now_iso_utc()
                # Load current states to update timestamps
                current_states_for_ts = await self._load_all_states(user_obj, model_id)
                if current_states_for_ts:
                    # Always update last_interaction_timestamp
                    current_states_for_ts["last_interaction_timestamp"] = now_ts
                    # Update last_activity_timestamp if there was activity (detect based on inventory diff or response)
                    had_activity = bool(inventory_diff_for_ai) or bool(
                        final_llm_response_for_user.strip()
                    )
                    if had_activity:
                        current_states_for_ts["last_activity_timestamp"] = now_ts
                        self.logger.info(
                            "[[TIMESTAMP_UPDATE]] Updated both timestamps at outlet end"
                        )
                    else:
                        self.logger.info(
                            "[[TIMESTAMP_UPDATE]] Updated last_interaction_timestamp at outlet end"
                        )
                    # Save the updated states
                    await self._save_all_states(
                        user_obj, model_id, current_states_for_ts
                    )
            except Exception:
                self.logger.debug(
                    "[[OUTLET]] Failed to update timestamps",
                    exc_info=True,
                )

            # アイドル要約は outlet 完了後にスケジュール（ここで基準を記録して予約）
            try:
                if getattr(self.valves, "idle_refactor_enabled", True):
                    key = self._idle_key(user_obj, model_id)
                    # outlet 終了時刻を最終呼び出しとして記録（アイドル判定の基準）
                    try:
                        self._last_filter_call[key] = datetime.utcnow()
                    except Exception:
                        pass
                    # ベースライン: 現在の recent 件数
                    base_count = 0
                    try:
                        st_for_base = (
                            await self._load_all_states(user_obj, model_id) or {}
                        )
                        mem = st_for_base.get("memory") or {}
                        rec = mem.get("recent") or []
                        if isinstance(rec, list):
                            base_count = len(rec)
                    except Exception:
                        base_count = 0
                    try:
                        if isinstance(self._idle_refactor_baseline_count, dict):
                            self._idle_refactor_baseline_count[key] = int(base_count)
                    except Exception:
                        pass
                    try:
                        delay = int(
                            getattr(self.valves, "idle_refactor_threshold_seconds", 600)
                            or 600
                        )
                    except Exception:
                        delay = 600
                    self._schedule_idle_refactor_if_needed(
                        key, user_obj, model_id, delay, __event_emitter__
                    )
                    if self._is_debug_enabled():
                        self.logger.debug(
                            f"[[IDLE_REF]] scheduled at outlet end: key={key}, delay={delay}s, baseline_recent={base_count}"
                        )
            except Exception:
                self.logger.debug(
                    "[[IDLE_REF]] scheduling at outlet failed", exc_info=True
                )

        except Exception as e:
            self._log_outlet(f"ERROR for model '{model_id}': {e}", level=logging.ERROR)
            self.logger.debug("[[OUTLET]] Trace:", exc_info=True)
        finally:
            self.logger.info(f"[[OUTLET FINISHED]] for model '{model_id}'")

        return body

    async def _build_memory_focus_block(
        self,
        baseline_states: Dict[str, Any],
        last_user_message: Optional[str],
        key: Optional[str] = None,
    ) -> Optional[str]:
        """Phase 0→1: Top-K 抽出で『Memory Focus』ブロックを生成。
        - 既存の簡易スコア（recency×impression + 選抜履歴 + knowledge混合）で候補抽出。
        - embeddings_model_id が設定されていれば、Sentence-Transformers によるローカル埋め込みで類似度を算出（STのみ。HTTPは使用しない）。
        - rerank_model_id が設定されていれば、Sentence-Transformers CrossEncoder によるローカル再ランキング（STのみ。HTTPは使用しない）。
        - モデル未設定時やST未導入/失敗時は該当フェーズをスキップし、従来のフォールバックのみ。
        """
        try:
            if not isinstance(baseline_states, dict):
                return None
            mem = (baseline_states.get("memory") or {}).get("recent")
            if not isinstance(mem, list) or not mem:
                mem = []
            mem_len = len(mem)
            # パラメータ
            try:
                k_retrieve = max(
                    1, int(getattr(self.valves, "retrieval_top_k", 50) or 50)
                )
            except Exception:
                k_retrieve = 50
            try:
                k_inject = max(1, int(getattr(self.valves, "inject_top_k", 10) or 10))
            except Exception:
                k_inject = 10

            # Phase 0: recency×impression の簡易スコア（memory） + knowledge 混合 + 選抜履歴ボーナス
            now = self._utcnow()
            prev_selected: set = set()
            try:
                if key:
                    prev_selected = set(self._mf_prev_selected.get(key, set()))
            except Exception:
                prev_selected = set()

            def parse_ts(it: Dict[str, Any]):
                try:
                    return self._parse_iso8601(str(it.get("timestamp"))) or datetime.min
                except Exception:
                    return datetime.min

            def recency_score(ts: datetime) -> float:
                try:
                    # 0〜1（新しいほど1、古いほど0）: 直近30日で線形減衰（暫定）
                    days = max(0.0, (now - ts).total_seconds() / 86400.0)
                    return max(0.0, min(1.0, 1.0 - (days / 30.0)))
                except Exception:
                    return 0.5

            try:
                focus_bonus = float(
                    getattr(self.valves, "memory_focus_bias", 0.05) or 0.05
                )
            except Exception:
                focus_bonus = 0.05
            scored: List[Tuple[float, str, Any, Tuple[str, ...]]] = []
            # memory candidates
            for idx, it in enumerate(mem):
                ts = parse_ts(it)
                r = recency_score(ts)
                try:
                    imp = float(it.get("impression_score", 0.5))
                except Exception:
                    imp = 0.5
                ident = (
                    "mem",
                    str(it.get("content", "")).strip(),
                    str(it.get("timestamp", "")).strip(),
                )
                score = (
                    0.6 * r
                    + 0.4 * imp
                    + (focus_bonus if ident in prev_selected else 0.0)
                )
                scored.append((score, "mem", it, ident))
            # knowledge candidates (anniversaries/milestones)
            try:
                kn = baseline_states.get("knowledge") or {}
                self_kn = kn.get("self") or {}
                ident_map = self_kn.get("identity") or {}
                ann_list = ident_map.get("anniversaries")
                mil_list = ident_map.get("milestones")

                def _count_any(val: Any) -> int:
                    if isinstance(val, list):
                        return len(
                            [
                                x
                                for x in val
                                if (
                                    isinstance(x, (str, dict))
                                    and (
                                        str(x).strip()
                                        if isinstance(x, str)
                                        else (x.get("text") or x.get("timestamp"))
                                    )
                                )
                            ]
                        )
                    if isinstance(val, str):
                        return 1 if val.strip() else 0
                    if isinstance(val, dict):
                        return 1 if ((val.get("text") or val.get("timestamp"))) else 0
                    return 0

                kn_ann_count = _count_any(ann_list)
                kn_mil_count = _count_any(mil_list)
                try:
                    bias = float(
                        getattr(self.valves, "memory_focus_bias", 0.05) or 0.05
                    )
                except Exception:
                    bias = 0.05
                kn_base = max(0.0, min(1.0, 0.75 + bias))

                def _ensure_entries(val: Any) -> List[Dict[str, Any]]:
                    out: List[Dict[str, Any]] = []
                    if isinstance(val, list):
                        items = val
                    else:
                        items = [val] if val is not None else []
                    for it in items:
                        if isinstance(it, dict):
                            txt = (
                                str(it.get("text", it.get("label", "")).strip())
                                if any(k in it for k in ("text", "label"))
                                else ""
                            )
                            if not txt:
                                val = it.get("value")
                                if isinstance(val, str):
                                    txt = val.strip()
                            ts = it.get("timestamp")
                            out.append({"text": txt, "timestamp": ts})
                        elif isinstance(it, str) and it.strip():
                            out.append({"text": it.strip(), "timestamp": None})
                    return [e for e in out if (e.get("text") or e.get("timestamp"))]

                for ent in _ensure_entries(ann_list):
                    ident = (
                        "kn",
                        "anniversary",
                        str(ent.get("text") or ""),
                        str(ent.get("timestamp") or ""),
                    )
                    score = kn_base + (focus_bonus if ident in prev_selected else 0.0)
                    scored.append((score, "kn_ann", ent, ident))
                for ent in _ensure_entries(mil_list):
                    ident = (
                        "kn",
                        "milestone",
                        str(ent.get("text") or ""),
                        str(ent.get("timestamp") or ""),
                    )
                    score = kn_base + (focus_bonus if ident in prev_selected else 0.0)
                    scored.append((score, "kn_mil", ent, ident))
            except Exception:
                kn_ann_count = 0
                kn_mil_count = 0
                pass
            # sort by score desc
            scored.sort(key=lambda x: x[0], reverse=True)

            # Retrieval set（Phase 0: まだ同スコア）
            candidates = scored[:k_retrieve]

            # 早期診断ログ（候補が空）
            if not candidates:
                try:
                    self.logger.info(
                        f"[[MEMFOCUS]] no candidates (mem={mem_len}, ann={kn_ann_count}, mil={kn_mil_count}, k_retrieve={k_retrieve}, k_inject={k_inject})"
                    )
                except Exception:
                    pass
                return None

            # --- Phase 1: Embeddings による再選別（あれば）
            emb_id = getattr(self.valves, "embeddings_model_id", None)
            rerank_id = getattr(self.valves, "rerank_model_id", None)
            # 明示ログ: 未設定時は簡易処理のみ（HTTPフォールバックも行わない）
            if not (emb_id and str(emb_id).strip()):
                self.logger.info(
                    "[[MEMFOCUS]] embeddings disabled (model not set); using fallback-only scoring"
                )
                emb_id = None
            if not (rerank_id and str(rerank_id).strip()):
                self.logger.info(
                    "[[MEMFOCUS]] rerank disabled (model not set); using fallback-only scoring"
                )
                rerank_id = None

            def _cand_text(kind: str, payload: Any) -> str:
                try:
                    if kind == "mem":
                        it = payload or {}
                        base = str(it.get("content") or "").strip()
                        tags = it.get("tags")
                        if isinstance(tags, list) and tags:
                            base += " " + " ".join(
                                [str(t) for t in tags if str(t).strip()]
                            )
                        return base
                    # knowledge
                    if isinstance(payload, dict):
                        base = str(
                            payload.get("text")
                            or payload.get("label")
                            or payload.get("value")
                            or ""
                        ).strip()
                        ts = payload.get("timestamp")
                        if ts:
                            return f"{base} ({ts})" if base else str(ts)
                        return base
                    return str(payload or "").strip()
                except Exception:
                    return str(payload or "")

            def _cosine(a: List[float], b: List[float]) -> float:
                try:
                    if not a or not b or len(a) != len(b):
                        return 0.0
                    dot = 0.0
                    na = 0.0
                    nb = 0.0
                    for i in range(len(a)):
                        x = float(a[i])
                        y = float(b[i])
                        dot += x * y
                        na += x * x
                        nb += y * y
                    denom = math.sqrt(max(na, 1e-12)) * math.sqrt(max(nb, 1e-12))
                    return float(dot / denom) if denom > 0 else 0.0
                except Exception:
                    return 0.0

            # embeddings が有効なら、候補の類似度を算出（並べ替えは最終のブレンドで実施）
            emb_sim_map: Dict[Tuple[str, ...], float] = {}
            rag_diag = {
                "embeddings": {"enabled": False, "model": None, "path": None},
                "rerank": {"enabled": False, "model": None, "path": None},
            }
            if emb_id and last_user_message and candidates:
                import time as _t

                _t0_total = _t.perf_counter()
                cand_texts = [
                    _cand_text(kind, payload) for _, kind, payload, _ in candidates
                ]

                # 1) Sentence-Transformers を優先
                used_path = None
                try:
                    if SentenceTransformer is not None:
                        # emb_id がSTで直接読み込めない場合に備え、マルチリンガル既定にフォールバック
                        st_model_name = str(emb_id)
                        try:
                            if st_model_name not in self._st_embedder_models:
                                self._st_embedder_models[st_model_name] = (
                                    SentenceTransformer(st_model_name)
                                )
                            model = self._st_embedder_models[st_model_name]
                        except Exception:
                            # 既定の日本語対応モデルへフォールバック
                            fallback_name = "intfloat/multilingual-e5-small"
                            if fallback_name not in self._st_embedder_models:
                                self._st_embedder_models[fallback_name] = (
                                    SentenceTransformer(fallback_name)
                                )
                            model = self._st_embedder_models[fallback_name]
                            st_model_name = fallback_name
                            self.logger.info(
                                f"[[MEMFOCUS]] embeddings ST fallback model={fallback_name}"
                            )

                        _t1 = _t.perf_counter()
                        q_vec = model.encode(
                            [last_user_message], normalize_embeddings=True
                        )
                        c_vecs = model.encode(cand_texts, normalize_embeddings=True)
                        used_path = f"st:{st_model_name}"
                        rag_diag["embeddings"] = {
                            "enabled": True,
                            "model": st_model_name,
                            "path": used_path,
                        }
                        ok = 0
                        for ce, cand in zip(c_vecs, candidates):
                            sim = _cosine(list(q_vec[0]), list(ce))
                            sim01 = max(0.0, min(1.0, 0.5 * (float(sim) + 1.0)))
                            emb_sim_map[cand[3]] = float(sim01)
                            ok += 1
                        _dt = _t.perf_counter() - _t1
                        self.logger.info(
                            f"[[MEMFOCUS]] embeddings ok count={ok}/{len(candidates)} time={_dt:.3f}s model={st_model_name} path=st"
                        )
                    else:
                        self.logger.info(
                            "[[MEMFOCUS]] embeddings ST not installed; disabled (no HTTP)"
                        )
                except Exception as ex:
                    self.logger.debug(
                        "[[MEMFOCUS]] embeddings ST failed", exc_info=True
                    )
                    emb_sim_map = {}

                if not emb_sim_map:
                    self.logger.info(
                        "[[MEMFOCUS]] embeddings unavailable via ST; HTTP disabled"
                    )
                    rag_diag["embeddings"] = {
                        "enabled": False,
                        "model": None,
                        "path": used_path or "none",
                    }

                # 3) 合計時間の記録（念のため）
                _dt_total = _t.perf_counter() - _t0_total
                if not emb_sim_map:
                    self.logger.info(
                        f"[[MEMFOCUS]] embeddings unavailable time={_dt_total:.3f}s path={(used_path or 'none')}"
                    )
            else:
                self.logger.info(
                    "[[MEMFOCUS]] embeddings stage skipped (disabled or no candidates/query)"
                )
                rag_diag["embeddings"] = {
                    "enabled": False,
                    "model": None,
                    "path": "skipped",
                }

            # --- Phase 1b: LLM リランク（あれば）
            async def _llm_rerank(
                query: str, items: List[str], model: str
            ) -> Optional[List[float]]:
                import time as _t

                n = len(items)
                # 1) Prefer Sentence-Transformers CrossEncoder when available
                st_path = None
                if CrossEncoder is not None and n > 0:
                    _t0 = _t.perf_counter()
                    try:
                        ce_name = str(model or "").strip()
                        try:
                            if ce_name not in self._st_cross_encoders:
                                self._st_cross_encoders[ce_name] = CrossEncoder(ce_name)
                            ce = self._st_cross_encoders[ce_name]
                        except Exception:
                            # Generic fallback model if specified model can't load
                            fallback_ce = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                            if fallback_ce not in self._st_cross_encoders:
                                self._st_cross_encoders[fallback_ce] = CrossEncoder(
                                    fallback_ce
                                )
                            ce = self._st_cross_encoders[fallback_ce]
                            ce_name = fallback_ce
                            self.logger.info(
                                f"[[MEMFOCUS]] rerank ST fallback model={fallback_ce}"
                            )

                        # Build pairs (query, item)
                        pairs = [(query, it) for it in items]
                        logits = ce.predict(pairs)

                        # Normalize logits to [0,1] with sigmoid when shape suggests raw scores
                        def _sigmoid(x):
                            try:
                                return 1.0 / (1.0 + math.exp(-float(x)))
                            except Exception:
                                return 0.5

                        scores = [float(x) for x in logits]
                        # heuristic: if typical range is not [0,1], squash
                        if not all(0.0 <= s <= 1.0 for s in scores):
                            scores = [_sigmoid(s) for s in scores]
                        if len(scores) == n:
                            _dt = _t.perf_counter() - _t0
                            self.logger.info(
                                f"[[MEMFOCUS]] rerank ok count={n} time={_dt:.3f}s model={ce_name} path=st"
                            )
                            rag_diag["rerank"] = {
                                "enabled": True,
                                "model": ce_name,
                                "path": "st",
                            }
                            return scores
                        else:
                            self.logger.info(
                                f"[[MEMFOCUS]] rerank length_mismatch expected={n} got={len(scores)} path=st"
                            )
                    except Exception:
                        self.logger.debug(
                            "[[MEMFOCUS]] rerank ST failed", exc_info=True
                        )
                self.logger.info("[[MEMFOCUS]] rerank HTTP path disabled; ST-only")
                if not rag_diag["rerank"]["enabled"]:
                    rag_diag["rerank"] = {
                        "enabled": False,
                        "model": None,
                        "path": "st-disabled",
                    }
                return None

            rr_score_map: Dict[Tuple[str, ...], float] = {}
            if rerank_id and last_user_message and candidates:
                try:
                    import time as _t

                    _t0 = _t.perf_counter()
                    cand_texts = [
                        _cand_text(kind, payload) for _, kind, payload, _ in candidates
                    ]
                    scores = await _llm_rerank(last_user_message, cand_texts, rerank_id)
                    if scores and len(scores) == len(candidates):
                        for sc, cand in zip(scores, candidates):
                            rr_score_map[cand[3]] = max(0.0, min(1.0, float(sc)))
                        _dt = _t.perf_counter() - _t0
                        self.logger.info(
                            f"[[MEMFOCUS]] rerank ok count={len(candidates)} time={_dt:.3f}s model={rerank_id}"
                        )
                    else:
                        _dt = _t.perf_counter() - _t0
                        self.logger.info(
                            f"[[MEMFOCUS]] rerank unavailable time={_dt:.3f}s"
                        )
                except Exception:
                    self.logger.debug("[[MEMFOCUS]] rerank stage failed", exc_info=True)
            else:
                self.logger.info(
                    "[[MEMFOCUS]] rerank stage skipped (disabled or no candidates/query)"
                )
                if not rag_diag["rerank"]["enabled"]:
                    rag_diag["rerank"] = {
                        "enabled": False,
                        "model": None,
                        "path": "skipped",
                    }

            # --- Final: 重み付きブレンドでソートし、閾値でフィルタ後に上位を注入 ---
            # 重みはバルブから取得し、合計1.0へ正規化
            try:
                w_fb = float(getattr(self.valves, "memory_focus_weight_fb", 0.45))
            except Exception:
                w_fb = 0.45
            try:
                w_emb = float(getattr(self.valves, "memory_focus_weight_emb", 0.35))
            except Exception:
                w_emb = 0.35
            try:
                w_rr = float(getattr(self.valves, "memory_focus_weight_rr", 0.20))
            except Exception:
                w_rr = 0.20
            sum_w = w_fb + w_emb + w_rr
            if sum_w <= 0.0:
                w_fb, w_emb, w_rr = 0.45, 0.35, 0.20
                sum_w = 1.0
            w_fb, w_emb, w_rr = (w_fb / sum_w), (w_emb / sum_w), (w_rr / sum_w)
            # 閾値
            try:
                min_cut = float(
                    getattr(self.valves, "memory_focus_min_blended_score", 0.25)
                )
                if min_cut < 0.0:
                    min_cut = 0.0
                if min_cut > 1.0:
                    min_cut = 1.0
            except Exception:
                min_cut = 0.25

            def _blend_score(cand: Tuple[float, str, Any, Tuple[str, ...]]) -> float:
                fb = max(0.0, min(1.0, float(cand[0])))
                ident = cand[3]
                em = float(emb_sim_map.get(ident, 0.0))
                rr = float(rr_score_map.get(ident, 0.0)) if rr_score_map else 0.0
                return w_fb * fb + w_emb * em + w_rr * rr

            try:
                # スコア計算と降順ソート
                scored_with_blend: List[
                    Tuple[float, Tuple[float, str, Any, Tuple[str, ...]]]
                ] = [(_blend_score(c), c) for c in candidates]
                scored_with_blend.sort(key=lambda x: x[0], reverse=True)
                # 閾値でフィルタ（ただし最終注入数を満たさない場合は上位で穴埋め）
                kept_items = [c for s, c in scored_with_blend if s >= min_cut]
                kept_count = len(kept_items)
                total = len(scored_with_blend)
                if kept_count < k_inject:
                    # 不足分をスコア順位で補充
                    fill_needed = k_inject - kept_count
                    fillers = [c for s, c in scored_with_blend if c not in kept_items][
                        :fill_needed
                    ]
                    kept_items = kept_items + fillers
                # 最終選定
                candidates = kept_items
                self.logger.info(
                    f"[[MEMFOCUS]] blend applied weights fb={w_fb:.2f} emb={w_emb:.2f} rr={w_rr:.2f} min={min_cut:.2f} kept={kept_count}/{total} n={len(candidates)}"
                )
            except Exception:
                self.logger.debug("[[MEMFOCUS]] blend sort failed", exc_info=True)

            selected = candidates[:k_inject]

            # 選定が0件の場合は注入をスキップ（空のヘッダーを避ける）
            if not selected:
                try:
                    self.logger.info(
                        f"[[MEMFOCUS]] selection empty after blend/cut (k_inject={k_inject}); skip injection"
                    )
                except Exception:
                    pass
                return None

            # 整形: M-ID を割り当て、短文化した一覧を生成
            def short(s: Any, limit: int = 120) -> str:
                s = re.sub(r"\s+", " ", str(s or "")).strip()
                return (s[: limit - 1] + "…") if len(s) > limit else s

            # 診断ヘッダでRAG経路を明示
            rag_mode = (
                "emb+rerank"
                if rag_diag["embeddings"]["enabled"] and rag_diag["rerank"]["enabled"]
                else (
                    "emb-only" if rag_diag["embeddings"]["enabled"] else "fallback-only"
                )
            )
            lines = [
                "## Memory Focus（参照優先メモリ）",
            ]
            new_prev: set = set()
            mem_idx = 0
            kn_idx = 0

            def _rel_time(ts_str: Any) -> str:
                try:
                    if not ts_str:
                        return "不明"
                    ts = self._parse_iso8601(str(ts_str))
                    if not ts:
                        return "不明"
                    delta = now - ts
                    secs = max(0, int(delta.total_seconds()))
                    if secs < 60:
                        return "数十秒前"
                    mins = secs // 60
                    if mins < 60:
                        return f"約{mins}分前"
                    hours = mins // 60
                    if hours < 48:
                        return f"約{hours}時間前"
                    days = hours // 24
                    return f"約{days}日前"
                except Exception:
                    return "不明"

            for _, kind, payload, ident in selected:
                if kind == "mem":
                    mem_idx += 1
                    mid = f"M{mem_idx:03d}"
                    it = payload  # dict
                    content = short(it.get("content"))
                    rel = _rel_time(it.get("timestamp"))
                    lines.append(f"- {mid}: {content}（{rel}）")
                else:
                    kn_idx += 1
                    kid = f"K{kn_idx:03d}"
                    label = "anniversary" if kind == "kn_ann" else "milestone"
                    if isinstance(payload, dict):
                        content = short(
                            payload.get("text")
                            or payload.get("label")
                            or payload.get("value")
                            or ""
                        )
                        rel = _rel_time(payload.get("timestamp"))
                        lines.append(
                            f"- {kid}: [{label}] {content}（{rel}）"
                            if content
                            else f"- {kid}: [{label}] （{rel}）"
                        )
                    else:
                        content = short(payload)
                        lines.append(f"- {kid}: [{label}] {content}")
                new_prev.add(ident)
            # save selection history for next turn
            try:
                if key is not None:
                    self._mf_prev_selected[key] = new_prev
            except Exception:
                pass
            return "\n".join(lines)
        except Exception:
            return None

    def _build_state_rules_text(self, time_context_note: Optional[str] = None) -> str:
        """Build generic state rules text for LLM prompts (initial and update).

        - Includes only keys present in STATE_DEFS and interpolates dynamic values.
        - Returns a single string with line breaks escaped for prompt safety.
        - Translation keys: use 'state_rules_list'.
        """
        try:
            rules_list = self._get_text("state_rules_list")
            if not isinstance(rules_list, dict):
                return ""
            
            rules_text_parts: List[str] = []
            
            # 各項目のルールを追加
            for key, instruction in rules_list.items():
                if key in self.STATE_DEFS:
                    try:
                        formatted = instruction.format(
                            time_context_note=(time_context_note or ""),
                            max_favorite_acts=self.valves.max_favorite_acts,
                            max_goals_per_tier=self.valves.max_goals_per_tier,
                            max_knowledge_entries=self.valves.max_knowledge_entries,
                            max_skills=self.valves.max_skills,
                            max_boundaries_entries=self.valves.max_boundaries_entries,
                            max_memory_items=self.valves.max_memory_items,
                            max_tone_components=self.valves.max_tone_components,
                        )
                    except Exception:
                        formatted = instruction
                    rules_text_parts.append(f"- **`{key}`**: {formatted}")
            return "\\n".join(rules_text_parts)
        except Exception:
            return ""

    def _filter_snapshot_for_update(self, src: Dict[str, Any]) -> Dict[str, Any]:
        """Filter a states snapshot before sending to analysis LLM.

        - Drop volatile or non-essential sections: memory/context/tone
        - Drop knowledge.self.{identity_anniversaries,identity_milestones}
          and flat aliases if present
        - Return deep-copied, sanitized dict to avoid in-place mutation
        """
        try:
            if not isinstance(src, dict):
                return {}
            data = json.loads(json.dumps(src))  # deep copy
            for k in ("memory", "context", "tone"):
                data.pop(k, None)
            kn = data.get("knowledge")
            if isinstance(kn, dict):
                if isinstance(kn.get("self_notes"), (list, dict)):
                    kn.pop("self_notes", None)
                ident = kn.get("self", {}) if isinstance(kn.get("self"), dict) else None
                if ident and isinstance(ident, dict):
                    for sub in ("identity_anniversaries", "identity_milestones"):
                        ident.pop(sub, None)
            return data
        except Exception:
            return {}
    # =============================
    # 生成LLM用: プロンプトビルダー
    # =============================
    def _build_initial_state_prompt(
        self,
        system_prompt: str,
        first_user: Optional[str] = None,
        first_assistant: Optional[str] = None,
    ) -> str:
        """Construct the initial state generation prompt without side effects.

        Prepends language/cross guidelines/self-check and optional first dialog before the output format block.
        """
        output_format_prompt = self._get_output_format_prompt_full()
        # Generic rules now available for both initial and update prompts
        generic_rules = self._build_state_rules_text()
        lang_directive = ""
        init_self_check = ""
        preface_blocks = "\n\n".join(
            [
                b
                for b in [
                    lang_directive,
                    (generic_rules or ""),
                    init_self_check,
                ]
                if isinstance(b, str) and b.strip()
            ]
        )
        # Template: valves override > TRANSLATIONS（ヘッダーフォールバック廃止）
        tpl = self.valves.initial_state_prompt_template or self._get_text(
            "initial_state_prompt_template"
        )
        
        # state_rules_listを文字列化（初期生成では使わないが、テンプレートに{state_rules_list}があっても安全に）
        rules_dict = self._get_text("state_rules_list")
        rules_text = ""
        if isinstance(rules_dict, dict):
            for key, value in rules_dict.items():
                rules_text += f"\n## {key}\n{value}\n"
        
        return tpl.format(
            system_prompt=system_prompt,
            state_rules_list=rules_text,
            output_format_prompt=(
                output_format_prompt
                + (("\n\n" + preface_blocks) if preface_blocks else "")
            ),
        )

    # =============================
    # 更新LLM用: プロンプトビルダー
    # =============================
    def _build_state_update_prompt(
        self,
        filtered_snapshot: Dict[str, Any],
        last_user_message: str,
        latest_llm_response: str,
        system_prompt: Optional[str],
        time_context_note: str,
        threshold_support_note: str,
    ) -> str:
        """Construct the state update prompt (diff generation) without side effects."""
        # Template: valves override > TRANSLATIONS
        tpl_update = self.valves.state_update_prompt_template or self._get_text(
            "state_update_prompt_template"
        )
        
        # state_rules_listを文字列化
        rules_dict = self._get_text("state_rules_list")
        rules_text = ""
        if isinstance(rules_dict, dict):
            for key, value in rules_dict.items():
                rules_text += f"\n## {key}\n{value}\n"
        
        return tpl_update.format(
            current_states_json=json.dumps(
                filtered_snapshot, ensure_ascii=False, indent=2
            ),
            last_user_message=last_user_message,
            latest_llm_response=latest_llm_response.strip(),
            output_format_prompt=self._get_output_format_prompt_diff(),
            state_rules_list=rules_text,
            time_context_note=time_context_note,
            max_favorite_acts=self.valves.max_favorite_acts,
            max_goals_per_tier=self.valves.max_goals_per_tier,
            max_knowledge_entries=self.valves.max_knowledge_entries,
            system_prompt=system_prompt or "",
            threshold_support_note=threshold_support_note,
        )

    def _build_state_update_messages(
        self,
        filtered_snapshot: Dict[str, Any],
        last_user_message: str,
        latest_llm_response: str,
        system_prompt: Optional[str],
        time_context_note: str,
        threshold_support_note: str,
        user_obj: Any,
        model_id: str,
    ) -> List[Dict[str, str]]:
        """状態更新LLM用のメッセージを構築する（役割を適切に分配）。
        
        状態分析LLMに送るメッセージを、systemロールとuserロールに適切に分割して構築します。
        既存のstate_update_prompt_templateを再利用し、inputs部分（snapshot/dialog/time_context）を
        userメッセージに、残りのすべての指示・ガイドラインをsystemメッセージに配置します。
        
        メッセージ構造:
        - **systemロール**に含まれる内容:
          - タイトル「あなたは内部状態の差分更新AI。」
          - 【絶対原則】
          - 【絶対的な義務：全項目評価の徹底】（policyオブジェクト全体）
          - 【相互作用ルール（更新AI）】
          - 【思考プロセス：厳守すべき評価手順】（テンプレート由来）
          - 【閾値推定ノート】
          - 【各項目指針】
          - 【出力フォーマット（差分更新）】
          - 【前提（キャラクター設定）】
          
        - **userロール**に含まれる内容:
          - snapshot: 現在の状態スナップショット（JSON形式）
          - dialog: ユーザーとアシスタントの対話
          - time_context: 時刻・曜日などのコンテキスト
        
        Args:
            filtered_snapshot: 現在の状態スナップショット（フィルタ済み）
            last_user_message: ユーザーの最後のメッセージ
            latest_llm_response: LLMの最新応答
            system_prompt: キャラクター設定のシステムプロンプト
            time_context_note: 時刻コンテキスト情報
            rules_text: 相互作用ルール
            threshold_support_note: 閾値推定ノート
            user_obj: ユーザーオブジェクト
            model_id: モデルID
        
        Returns:
            List[Dict[str, str]]: メッセージリスト（各要素は{"role": "system/user", "content": "..."}）
            
        Note:
            この分割により、LLMは指示（system）と入力データ（user）を明確に区別でき、
            より適切な状態更新を行えます。
        """
        # Build full text with template
        full_text = self._build_state_update_prompt(
            filtered_snapshot,
            last_user_message,
            latest_llm_response,
            system_prompt,
            time_context_note,
            threshold_support_note,
        )
        # Extract the inputs (snapshot/dialog/time_context) as user; the rest as system.
        # Phase 9: テンプレートから inputs: セクションを削除したため、直接構築する
        
        # 固定キー項目を抽出（LLMが既存のキー構造を理解できるように）
        fixed_key_snapshot = {
            "emotion": filtered_snapshot.get("emotion", {}),
            "desire": filtered_snapshot.get("desire", {}),
            "physical_health": {
                "needs": filtered_snapshot.get("physical_health", {}).get("needs", {}),
                "limits": filtered_snapshot.get("physical_health", {}).get("limits", {})
            },
            "mental_health": {
                "needs": filtered_snapshot.get("mental_health", {}).get("needs", {}),
                "limits": filtered_snapshot.get("mental_health", {}).get("limits", {})
            }
        }
        
        # 会話テキストを構築
        dialog_text = f"U『{last_user_message}』→A『{latest_llm_response}』"
        
        # user content を構築
        user_part_lines = [
            "【現在の固定キー項目（更新時は全キーを出力すること）】",
            json.dumps(fixed_key_snapshot, ensure_ascii=False, indent=2),
            "",
            "【最新会話】",
            dialog_text,
            "",
            "【時間文脈】",
            time_context_note,
        ]
        
        # system_part は full_text そのまま（入力セクションはもう含まれていない）
        system_part = full_text

        # Ensure system includes the output format reminder at the end
        if "【出力フォーマット（差分更新）】" not in system_part:
            system_part = system_part + "\n\n" + self._get_output_format_prompt_diff()

        messages = [
            {"role": "system", "content": system_part},
            {"role": "user", "content": "\n".join(user_part_lines).strip()},
        ]
        return messages

    # =============================
    # 生成LLM用: 出力フォーマット
    # =============================
    def _get_output_format_prompt_full(self) -> str:
        """初期状態生成用: 全キーを含む単一JSONのみを出力させる簡潔な指示を返す。
        - コードフェンスやコメントを含めない（LLMがそのまま出力しやすいように）。
        - 例示は行わず、必須キー一覧と厳格ルールのみを示す。
        """
        try:
            keys = ", ".join(self.STATE_DEFS.keys())
        except Exception:
            keys = "emotion, relationship, goal, context, tone, physical_health, mental_health, inventory, sexual_development, posture, memory, traits, desire, internal_monologue, knowledge"
        lines = [
            "【出力フォーマット（初期状態）】",
            "- あなたの唯一の出力は『全ての必須キーを含む1つのJSONオブジェクト』のみ。前置き・後置き・説明・コードフェンス禁止。",
            f"- 必須キー一覧: {keys}",
            "- JSON以外の文字（見出し、箇条書き、コメント、サンプルの複製等）は一切含めない。",
            "- 空欄最小化: 可能な限り空文字列/空配列/空オブジェクトを避け、合理的な値で埋める（安全性優先。根拠が弱い場合は空のままでよい）。",
        ]
        return "\n".join(lines)

    # =============================
    # 更新LLM用: 出力フォーマット
    # =============================
    def _get_output_format_prompt_diff(self) -> str:
        """差分更新用: 変化したキーのみのJSON出力を強制する簡潔な指示を返す。
        - 例示値を最小化し、プレースホルダーのみで構造を示す。
        """
        lines = [
            "【出力フォーマット】",
            "- JSON のみ。前置き・後置き・説明・コードフェンス禁止。",
            "- 変化したキーのみを出力（未変更は書かない）。",
            "- 全キーはトップレベル（posture/sexual_development 等を他のキー内にネストしない）。",
            "",
            "【最小スキーマ例】",
            "emotion: {\"<感情名>\": <0-1>}",
            "memory: {\"recent\": [{\"content\": \"...\", \"timestamp\": \"<ISO8601>\", \"tags\": [...], \"impression_score\": <0-1>}]}",
            "goal: {\"short_term\": {\"<タスク名>\": {\"progress\": <0-1>, \"priority\": <0-1>}}}",
            "knowledge: {\"user\": {\"identity\": {\"name\": \"...\"}}}",
            "relationship: {\"trust_score\": <0-1>}",
            "context: {\"action\": {\"user\": \"...\"}}",
            "traits: [\"...\"]",
            "- その他のキー（posture, sexual_development, boundaries 等）も同様。",
            "",
            "**【必須チェック】**",
            "- memory.recent は配列（memory.short_term は存在しない）。",
            "- goal.short_term は辞書（memory 下には存在しない）。",
            "- posture/sexual_development/desire/inventory は全てトップレベル。",
            "- <プレースホルダー> は実際の対話内容から値を生成。コピペ禁止。",
        ]
        return "\n".join(lines)

    async def _update_or_initialize_states_async(
        self,
        last_user_message: str,
        latest_llm_response: str,
        system_prompt: Optional[str],
        user_obj: Any,
        model_id: str,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        is_narration: bool = False,
        external_diff: Optional["Filter.StateDiff"] = None,
    ) -> None:
        """Background task: initialize or update persistent states.

        Contract:
        - Inputs: last_user_message, latest_llm_response (original), system_prompt
        - Emits UI events via event_emitter responsibly (non-blocking)
        - Preserves strict behavior: inventory modes, timestamps policy, validators
        - Side-effects: saves states via configured persistence backend
        """
        # === このtry...exceptブロックが重要 ===
        self._log_state("UPDATE TASK STARTED")
        try:
            if not system_prompt:
                # システムプロンプトが空でも初期生成/更新は続行する（テンプレートの {system_prompt} は空文字に）
                self._create_background_task(
                    self._safe_emit(
                        event_emitter,
                        self._ev_warning(
                            self._get_text("system_prompt_is_empty"), True
                        ),
                    ),
                    name="warn_empty_system_prompt"
                )

            # 状態ロード（ベースライン準備は tick_start 相当の前処理で行い、数値は変更しない）
            loaded_states = await self._load_all_states(user_obj, model_id)
            # .get() を使って安全に値を取得（インタラクション時刻は保持）
            old_interaction_ts = (
                loaded_states.get("last_interaction_timestamp")
                if isinstance(loaded_states, dict)
                else None
            )
            old_activity_ts = (
                loaded_states.get("last_activity_timestamp")
                if isinstance(loaded_states, dict)
                else None
            )
            # tick_start: 数値変更は行わず、プロンプト用にタイムスタンプを剥離したベースラインを作成
            current_states, time_context_note = self._prepare_tick_start_baseline(
                loaded_states or {}
            )

            if not current_states:
                # --- 初回生成 ---
                self.logger.info(
                    f"[[STATE_UPDATE_ASYNC]] No states found. Generating initial states for '{model_id}'."
                )
                self._create_background_task(
                    self._safe_emit(
                        event_emitter,
                        self._ev_status(
                            self._get_text("generating_initial_state"), True
                        ),
                    )
                )
                # 初期生成の質を高めるため、初回のユーザー/キャラ発話を参考として渡す
                initial_state_prompt = self._build_initial_state_prompt(
                    system_prompt or "", last_user_message, latest_llm_response
                )
                # last_prompt.txt の State セクションを更新（初期状態生成プロンプトを messages JSON として保存）
                try:
                    self._save_last_prompts(
                        state_text=json.dumps(
                            [{"role": "user", "content": initial_state_prompt}],
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                except Exception:
                    self.logger.debug(
                        "[[SAVE_LAST_PROMPT]] Failed to save initial state prompt",
                        exc_info=True,
                    )
                if self._is_debug_enabled():
                    try:
                        self.logger.debug(
                            f"[[PROMPT_INITIAL_STATE]] preview={self._dbg_trunc(initial_state_prompt, 4000)}"
                        )
                    except Exception:
                        pass

                llm_api_result = await self._call_llm_for_state_update(
                    initial_state_prompt
                )
                if llm_api_result.get("finish_reason") == "length":
                    self.logger.error(
                        f"[[STATE_GEN_FAILED]] Initial state generation was truncated by max_tokens limit for model '{model_id}'."
                    )
                    await self._safe_emit(
                        event_emitter,
                        self._ev_error(
                            "💥 初期状態生成がトークン上限に達し中断されました。max_tokensの設定を見直してください。",
                            True,
                        ),
                    )
                    return

                llm_response = llm_api_result.get("content")
                initial_states = (
                    self._parse_llm_response_for_states(llm_response)
                    if llm_response
                    else None
                )

                if initial_states:
                    # --- 変更点：ここで初回インタラクションの差分をマージする ---
                    if external_diff:
                        self.logger.info(
                            "[[STATE_INIT_MERGE]] Merging external diff from first interaction into initial state."
                        )
                        safe_init_diff = self._filter_state_diff_keys(
                            cast(Dict[str, Any], external_diff)
                        )
                        initial_states = self._deep_merge_dicts(
                            initial_states, safe_init_diff
                        )
                    # --- 変更点ここまで ---

                    # ラベルの since 補完と最終時刻の初期化（現行キーのみ）
                    now_iso = self._now_iso_utc()
                    try:
                        ph = initial_states.get("physical_health") or {}
                        if isinstance(ph, dict):
                            lbl = (ph.get("condition") or "").strip()
                            ts = (
                                ph.get("timestamps")
                                if isinstance(ph.get("timestamps"), dict)
                                else {}
                            )
                            if lbl and (not isinstance(ts, dict) or not ts.get("condition_since")):
                                ph_ts = dict(ts) if isinstance(ts, dict) else {}
                                ph_ts["condition_since"] = now_iso
                                ph["timestamps"] = ph_ts
                                initial_states["physical_health"] = ph
                    except Exception:
                        pass
                    try:
                        mh = initial_states.get("mental_health") or {}
                        if isinstance(mh, dict):
                            lbl = (mh.get("condition") or "").strip()
                            ts = (
                                mh.get("timestamps")
                                if isinstance(mh.get("timestamps"), dict)
                                else {}
                            )
                            if lbl and (not isinstance(ts, dict) or not ts.get("condition_since")):
                                mh_ts = dict(ts) if isinstance(ts, dict) else {}
                                mh_ts["condition_since"] = now_iso
                                mh["timestamps"] = mh_ts
                                initial_states["mental_health"] = mh
                    except Exception:
                        pass
                    # 保険: default_outfit の名称が inventory に存在しない場合は最小項目を補完
                    try:
                        initial_states = self._sync_default_outfit_with_inventory(
                            initial_states
                        )
                    except Exception:
                        pass
                    await self._save_all_states(user_obj, model_id, initial_states)
                    self._create_background_task(
                        self._safe_emit(
                            event_emitter,
                            self._ev_status(
                                self._get_text("initial_state_generated"), True
                            ),
                        ),
                        name="notify_initial_generated"
                    )
                else:
                    self._create_background_task(
                        self._safe_emit(
                            event_emitter,
                            self._ev_error(
                                self._get_text("initial_state_failed"), True
                            ),
                        ),
                        name="notify_initial_failed"
                    )
                return

            # --- 状態更新 ---
            self.logger.info(
                f"[[STATE_UPDATE_ASYNC]] Existing states found. Updating states for '{model_id}'."
            )
            self._create_background_task(
                self._safe_emit(
                    event_emitter,
                    self._ev_status(self._get_text("updating_state"), False),
                ),
                name="notify_updating_state"
            )

            # time_context_note は tick_start ベースライン準備で計算済み（数値は変えない）
            threshold_support_note = self._build_threshold_support_note(current_states)
            # Exclude fields not to be passed to analysis LLM
            filtered_snapshot = self._filter_snapshot_for_update(current_states)

            state_update_messages = self._build_state_update_messages(
                filtered_snapshot,
                last_user_message,
                latest_llm_response,
                system_prompt,
                time_context_note,
                threshold_support_note,
                user_obj,
                model_id,
            )
            # last_prompt.txt の State セクションを更新（差分更新プロンプトを messages JSON として保存）
            try:
                self._save_last_prompts(
                    state_text=json.dumps(
                        state_update_messages, ensure_ascii=False, indent=2
                    )
                )
            except Exception:
                self.logger.debug(
                    "[[SAVE_LAST_PROMPT]] Failed to save state update messages",
                    exc_info=True,
                )
            if self._is_debug_enabled():
                try:
                    preview = json.dumps(state_update_messages, ensure_ascii=False)[
                        :4000
                    ]
                    self.logger.debug(f"[[PROMPT_STATE_UPDATE_MSGS]] preview={preview}")
                except Exception:
                    pass

            state_llm_api_result = await self._call_llm_for_state_update(
                state_update_messages
            )

            if state_llm_api_result.get("finish_reason") == "length":
                self.logger.error(
                    f"[[STATE_UPDATE_FAILED]] The state analysis LLM response was truncated due to max_tokens limit for model '{model_id}'. Update will be skipped."
                )
                await self._safe_emit(
                    event_emitter,
                    self._ev_error(
                        "💥 状態更新がトークン上限に達し中断されました。更新はスキップされます。",
                        True,
                    ),
                )
                return

            state_llm_response = state_llm_api_result.get("content")
            # Save the raw response for debugging (as messages JSON)
            try:
                self._save_last_prompts(
                    state_response_text=json.dumps(
                        [{"role": "assistant", "content": state_llm_response or ""}],
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            except Exception:
                self.logger.debug(
                    "[[SAVE_LAST_PROMPT]] Failed to save state update response",
                    exc_info=True,
                )

            if not state_llm_response:
                self.logger.error(
                    "[[STATE_UPDATE_FAILED]] The state analysis LLM returned an empty response. State update will be skipped. Please check your API settings, model availability, and potential content filtering."
                )
                self._create_background_task(
                    self._safe_emit(
                        event_emitter,
                        self._ev_error(
                            "💥 状態更新AIが応答しませんでした。更新はスキップされます。",
                            True,
                        ),
                    ),
                    name="notify_empty_llm_response"
                )
                return

            new_states = self._parse_llm_response_for_states(
                state_llm_response, is_diff=True
            )

            if external_diff:
                if not new_states:
                    new_states = {}
                safe_ext_diff = self._filter_state_diff_keys(
                    cast(Dict[str, Any], external_diff)
                )
                new_states = self._deep_merge_dicts(new_states, safe_ext_diff)

            # directモードのアウトレットパース由来のinventory差分を許可するフラグを決定
            allow_direct_inventory_override = (
                getattr(
                    self.valves, "inventory_update_mode", self.INVENTORY_MODE_INFERENCE
                )
                == self.INVENTORY_MODE_DIRECT
                and isinstance(external_diff, dict)
                and "inventory" in external_diff
            )

            # アイドル要約の基準件数（事後整形前の current_states の recent 件数）を先に取得
            try:
                _pre_update_recent_len = 0
                try:
                    _pre_update_recent = (
                        (current_states or {}).get("memory") or {}
                    ).get("recent") or []
                    if isinstance(_pre_update_recent, list):
                        _pre_update_recent_len = len(_pre_update_recent)
                except Exception:
                    _pre_update_recent_len = 0
            except Exception:
                _pre_update_recent_len = 0

            updated_states = self._merge_states(
                current_states, new_states, allow_direct_inventory_override
            )

            # Debug: log numeric deltas between pre-update loaded states and LLM-updated states
            try:
                # 'loaded_states' は関数冒頭で取得済み（元の保存状態）
                self._log_turn_start_deltas(loaded_states or {}, updated_states or {})
            except Exception:
                self.logger.debug("Failed to log turn-start deltas", exc_info=True)

            # 活動検出に基づき last_activity_timestamp を条件確認
            try:
                had_activity, reasons = self._detect_activity(
                    current_states or {}, updated_states or {}, new_states or {}
                )
            except Exception:
                had_activity, reasons = False, []

            fallback_interaction_ts = old_interaction_ts or old_activity_ts
            if not updated_states.get("last_interaction_timestamp"):
                updated_states["last_interaction_timestamp"] = (
                    fallback_interaction_ts or self._now_iso_utc()
                )

            if had_activity:
                self.logger.info(
                    f"[[ACTIVITY_DETECTED]] {';'.join(reasons)} (timestamp preserved unless diff overrides)"
                )
            else:
                # 非活動: 変更しない（アイドル時間を蓄積させる）
                self.logger.info(
                    "[[IDLE_KEEP]] No significant activity; last_activity_timestamp kept"
                )

            if not updated_states.get("last_activity_timestamp"):
                updated_states["last_activity_timestamp"] = (
                    old_activity_ts
                    or fallback_interaction_ts
                    or self._now_iso_utc()
                )
            # Ensure timestamps include timezone info before saving
            try:
                updated_states = self._normalize_timestamps_in_states(updated_states)
            except Exception:
                pass
            try:
                updated_states = self._prune_unknown_keys_in_states(updated_states)
            except Exception:
                pass
            await self._save_all_states(user_obj, model_id, updated_states)

            self._create_background_task(
                self._notify_state_changes(
                    event_emitter, current_states, updated_states, bool(new_states)
                ),
                name="notify_state_changes"
            )
            # アイドル要約のスケジューリングは outlet 完了後に統一（ここでは予約しない）
        except Exception as e:
            self._log_state(
                f"ERROR in background update task for model '{model_id}': {e}",
                level=logging.ERROR,
            )
            # stack trace is noisy for end users; keep it under DEBUG
            self.logger.debug("[[STATE]] Trace:", exc_info=True)
            self._create_background_task(
                self._safe_emit(
                    event_emitter,
                    self._ev_error(self._get_text("state_processing_error"), True),
                ),
                name="notify_processing_error"
            )

    def _deep_merge_dicts(self, source: Dict, update: Dict) -> Dict:
        """
        2つの辞書を再帰的にマージする。
        'update'辞書のキーと値を'source'辞書にマージする。
        ネストされた辞書も再帰的に処理される。
        """
        merged = source.copy()
        for key, value in update.items():
            # マージ元とマージ先の両方で値が辞書の場合、再帰的にマージ
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._deep_merge_dicts(merged[key], value)
            # それ以外の場合は、単純に値を上書き
            else:
                merged[key] = value
        return merged

    def _apply_ops_to_list_by_id(
        self,
        baseline: List[Any],
        ops: Dict[str, Any],
        kind: str = "memory",
    ) -> List[Any]:
        """Apply id-based operations to a list.
        - ids are 1-based indices referring to the provided baseline order
        - operation order: overwrite -> delete -> insert
        - insert supports position: 'head' | 'tail' | 'after_id' (requires after_id)
        - kind: 'memory' or 'identity' (for potential future schema-specific tweaks)
        """
        try:
            lst = json.loads(json.dumps(baseline)) if isinstance(baseline, list) else []
            if not isinstance(ops, dict):
                return lst
            # Overwrite
            for ow in ops.get("overwrite") or []:
                try:
                    raw_id = ow.get("id") if isinstance(ow, dict) else None
                    oid = int(str(raw_id)) if raw_id is not None else None
                    item = ow.get("item") if isinstance(ow, dict) else None
                    if oid is None or not isinstance(item, dict):
                        continue
                    idx = oid - 1
                    if 0 <= idx < len(lst):
                        lst[idx] = item
                except Exception:
                    continue
            # Delete (process in descending index order)
            dels = []
            try:
                for did in ops.get("delete_ids") or []:
                    try:
                        idx = int(str(did)) - 1
                        if 0 <= idx < len(lst):
                            dels.append(idx)
                    except Exception:
                        continue
                for idx in sorted(set(dels), reverse=True):
                    try:
                        lst.pop(idx)
                    except Exception:
                        pass
            except Exception:
                pass
            # Insert
            for ins in ops.get("insert") or []:
                try:
                    pos = (ins.get("position") or "tail").strip().lower()
                    item = ins.get("item")
                    if not isinstance(item, dict):
                        continue
                    if pos == "head":
                        lst.insert(0, item)
                    elif pos == "after_id":
                        aid = ins.get("after_id")
                        try:
                            aidx = int(str(aid)) - 1
                            if 0 <= aidx < len(lst):
                                lst.insert(aidx + 1, item)
                            else:
                                lst.append(item)
                        except Exception:
                            lst.append(item)
                    else:  # 'tail' or unknown
                        lst.append(item)
                except Exception:
                    continue
            return lst
        except Exception:
            return baseline or []

    def _merge_inventory_conservatively(
        self,
        current: "List[Filter.InventoryItem]",
        update: "List[Filter.InventoryItem]",
    ) -> "List[Filter.InventoryItem]":
        """
        所持品の安全マージ: LLM が返す `inventory` が部分リスト（差分/一部のみ）の場合でも、
        既存の未言及アイテムを消さずに維持する。数量<=0は削除扱い。

        ルール:
        - キーは `name`（前後の空白を除去、ケースは厳密一致）。
        - update 側に同名があれば、以下を更新:
          - quantity: 数値が与えられれば置換。<=0 の場合は削除。
          - equipped: 真偽が与えられれば置換。
          - description/slot: 非空文字が与えられれば置換。
        - update に無い既存アイテムは維持（喪失は明示的に quantity<=0 を指定）。
        - update にのみ存在するアイテムは新規追加（quantity 未指定は1に丸め）。
        - バリデーションは呼び出し側で行う。
        """
        try:
            # 正規化された名前キーのマップを作成
            def norm_name(x: Any) -> str:
                try:
                    return str((x or "")).strip()
                except Exception:
                    return ""

            cur_map: Dict[str, Dict[str, Any]] = {}
            order: List[str] = []
            if isinstance(current, list):
                for it in current:
                    try:
                        nm = norm_name((it or {}).get("name"))
                        if nm and nm not in cur_map:
                            cur_map[nm] = dict(it)
                            order.append(nm)
                    except Exception:
                        continue

            upd_seen: set = set()
            if isinstance(update, list):
                for it in update:
                    try:
                        nm = norm_name((it or {}).get("name"))
                        if not nm:
                            continue
                        upd_seen.add(nm)
                        base = cur_map.get(nm, {})
                        merged = dict(base)
                        # quantity
                        if "quantity" in (it or {}):
                            raw_q = (it or {}).get("quantity")
                            try:
                                qv = int(raw_q) if raw_q is not None else None
                            except Exception:
                                qv = None
                            if qv is not None:
                                if qv <= 0:
                                    # 削除指定（後で除外）
                                    merged["__delete__"] = True
                                else:
                                    merged["quantity"] = qv
                        # equipped
                        if "equipped" in (it or {}):
                            try:
                                merged["equipped"] = bool((it or {}).get("equipped"))
                            except Exception:
                                pass
                        # slot
                        if "slot" in (it or {}):
                            sv = (it or {}).get("slot")
                            if isinstance(sv, str) and sv.strip():
                                merged["slot"] = sv
                        # description
                        if "description" in (it or {}):
                            dv = (it or {}).get("description")
                            if isinstance(dv, str) and dv.strip():
                                merged["description"] = dv

                        # nameは必ず保持
                        merged["name"] = nm

                        if nm not in cur_map:
                            # 新規
                            if "quantity" not in merged:
                                merged["quantity"] = 1
                            if "equipped" not in merged:
                                merged["equipped"] = False
                            cur_map[nm] = merged
                            order.append(nm)
                        else:
                            cur_map[nm] = merged
                    except Exception:
                        continue

            # 出力（順序は既存優先→新規追加の順）
            result: List[Filter.InventoryItem] = []
            for nm in order:
                it = cur_map.get(nm)
                if not it:
                    continue
                if it.get("__delete__") is True:
                    continue  # 明示削除
                result.append(
                    cast(
                        Filter.InventoryItem,
                        {k: v for k, v in it.items() if k != "__delete__"},
                    )
                )
            return result
        except Exception:
            # 失敗時は update が有効ならそれ、なければ current を返す
            if isinstance(update, list) and update:
                return update
            return current or []

    def _sanitize_inventory_descriptions(
        self, items: "List[Filter.InventoryItem]"
    ) -> "List[Filter.InventoryItem]":
        """Enforce description policy: max 2 sentences, char cap, neutral trimming.

            Sentences are split using Japanese and Latin punctuation (。．.!?？) while
            preserving simple ASCII fallback. We keep first 2 non-empty segments.
        After join, keep concise neutral phrasing (no character cap) and
            append an ellipsis '…' if truncated. Empty or malformed descriptions remain
            untouched. Returns a NEW list (does not mutate original).
        """
        if not isinstance(items, list):
            return items

        sanitized: List[Filter.InventoryItem] = []
        for it in items:
            try:
                if not isinstance(it, dict):
                    sanitized.append(cast(Filter.InventoryItem, it))
                    continue
                desc = it.get("description")
                if not isinstance(desc, str):
                    sanitized.append(cast(Filter.InventoryItem, dict(it)))
                    continue
                raw = desc.strip()
                if not raw:
                    sanitized.append(cast(Filter.InventoryItem, dict(it)))
                    continue

                # Split into candidate sentences
                # Use manual split to avoid losing delimiters then reattach
                segments: List[str] = []
                buf = ""
                for ch in raw:
                    buf += ch
                    if ch in "。．.!?？!?":
                        seg = buf.strip()
                        if seg:
                            segments.append(seg)
                        buf = ""
                if buf.strip():  # trailing part without terminal punctuation
                    segments.append(buf.strip())

                # Keep only first 2 non-empty; no character limit
                kept = [s for s in segments if s.strip()][:2]
                if not kept:
                    kept = [raw]  # fallback entire raw
                new_desc = "".join(kept)

                new_item = dict(it)
                new_item["description"] = new_desc
                sanitized.append(cast(Filter.InventoryItem, new_item))
            except Exception:
                sanitized.append(cast(Filter.InventoryItem, it))
        return sanitized

    def _sync_default_outfit_with_inventory(
        self, states: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensure any names listed in knowledge.self.identity.default_outfit exist in inventory.

        Conservative policy:
        - Adds minimal items only when missing (name match by exact string after trim)
        - description: "", quantity: 1, equipped: False, slot: None
        - Does not force equip or invent details
        - Returns updated states (non-destructive on other sections)
        """
        try:
            if not isinstance(states, dict):
                return states
            kn = states.get("knowledge") or {}
            self_info = kn.get("self") or {}
            identity = self_info.get("identity") or {}
            default_outfit = identity.get("default_outfit")
            # Normalize default_outfit to list[str]
            names: List[str] = []
            if isinstance(default_outfit, list):
                names = [
                    str(x).strip()
                    for x in default_outfit
                    if isinstance(x, (str, int, float))
                ]
            elif isinstance(default_outfit, (str, int, float)):
                s = str(default_outfit).strip()
                names = [s] if s else []
            else:
                names = []
            names = [n for n in names if n]
            if not names:
                return states

            inv = states.get("inventory")
            inv_list_raw = inv if isinstance(inv, list) else []
            inv_list: List[Filter.InventoryItem] = []
            for it in inv_list_raw:
                if isinstance(it, dict):
                    inv_list.append(cast(Filter.InventoryItem, it))

            # Map existing names by trimmed exact match
            def _norm(s: Any) -> str:
                try:
                    return str(s).strip()
                except Exception:
                    return ""

            existing = {
                _norm((it or {}).get("name")) for it in inv_list if isinstance(it, dict)
            }
            added = False
            for nm in dict.fromkeys(names).keys():  # preserve order, dedupe
                if _norm(nm) and _norm(nm) not in existing:
                    inv_list.append(
                        {
                            "name": nm,
                            "description": "",
                            "quantity": 1,
                            "equipped": False,
                            "slot": None,
                        }
                    )
                    existing.add(_norm(nm))
                    added = True
            if added:
                try:
                    inv_list = self._sanitize_inventory_descriptions(inv_list)
                except Exception:
                    pass
                states["inventory"] = cast(List[Dict[str, Any]], inv_list)
            return states
        except Exception:
            # Fail safe: return as-is
            return states

    # =============================
    # 要約LLM用: Idle Refactor（メモリ要約・整序）
    #   - Phase A: スケジューリング/判定
    #   - Phase B: プロンプト生成と実行
    # =============================
    # ---- Idle Refactor (Phase A) helpers ----
    def _idle_key(self, user: Optional[Any], model_id: str) -> str:
        try:
            uid = (
                getattr(user, "id", None)
                or getattr(user, "_id", None)
                or str(user)
                or "unknown"
            )
        except Exception:
            uid = "unknown"
        return f"{uid}::{model_id}"

    def _cancel_idle_refactor_task(self, key: str) -> None:
        try:
            t = self._idle_refactor_tasks.get(key)
            if t and not t.done():
                t.cancel()
                self.logger.info(
                    f"[[IDLE_REF]] Cancelled scheduled idle refactor for {key}"
                )
        except Exception:
            pass

    def _schedule_idle_refactor_if_needed(
        self,
        key: str,
        user_obj: Any,
        model_id: str,
        delay_seconds: int,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> None:
        # Cancel existing
        self._cancel_idle_refactor_task(key)

        # Schedule new
        async def _runner():
            try:
                # Baseline count is set by the caller at schedule time. Do not recapture here.
                await asyncio.sleep(max(1, int(delay_seconds)))
                await self._maybe_run_idle_refactor(
                    key, user_obj, model_id, event_emitter
                )
            except asyncio.CancelledError:
                return
            except Exception:
                self.logger.debug("[[IDLE_REF]] runner error", exc_info=True)

        task = self._create_background_task(_runner(), name=f"idle_refactor_{key}")
        self._idle_refactor_tasks[key] = task
        self.logger.info(
            f"[[IDLE_REF]] Scheduled idle refactor in {delay_seconds}s for {key}"
        )

    async def _maybe_run_idle_refactor(
        self,
        key: str,
        user_obj: Any,
        model_id: str,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> None:
        # Check inactivity window and memory growth condition
        try:
            threshold = int(
                getattr(self.valves, "idle_refactor_threshold_seconds", 600) or 600
            )
            last_call = self._last_filter_call.get(key)
            if last_call is None:
                # If no record, treat as not inactive
                self.logger.info(
                    f"[[IDLE_REF]] Skipped (no last call record) for {key}"
                )
                return
            elapsed = (datetime.utcnow() - last_call).total_seconds()
            if elapsed < threshold:
                self.logger.info(
                    f"[[IDLE_REF]] Skipped (not idle: {elapsed:.0f}s < {threshold}s) for {key}"
                )
                return
            # Load states and check recent size
            states = await self._load_all_states(user_obj, model_id)
            recent = []
            if isinstance(states, dict):
                mem = states.get("memory") or {}
                if isinstance(mem, dict):
                    recent = mem.get("recent") or []
            min_size = int(
                getattr(self.valves, "idle_refactor_recent_min_size", 10) or 10
            )
            if not isinstance(recent, list) or len(recent) < min_size:
                self.logger.info(
                    f"[[IDLE_REF]] Skipped (recent size {len(recent) if isinstance(recent, list) else 'n/a'} < {min_size}) for {key}"
                )
                return
            # Require growth since schedule
            try:
                base = int(self._idle_refactor_baseline_count.get(key, 0))
            except Exception:
                base = 0
            require_growth = bool(
                getattr(self.valves, "idle_refactor_require_growth", False)
            )
            if require_growth:
                if len(recent) <= base:
                    self.logger.info(
                        f"[[IDLE_REF]] Skipped (no growth: current={len(recent)} <= baseline={base}) for {key}"
                    )
                    return
            # Phase B: Build messages with system/user split for the refactor LLM
            # user: 保存時点スナップショット（memory と knowledge 全体）
            try:
                mem_part = (states or {}).get("memory") or {}
                kn_part = (states or {}).get("knowledge") or {}
                user_snapshot = {"memory": mem_part, "knowledge": kn_part}
                user_snapshot_json = json.dumps(
                    user_snapshot, ensure_ascii=False, indent=2
                )
            except Exception:
                user_snapshot_json = "{}"
            # system: テンプレに {current_states_json} を実際のスナップショットで埋める
            tpl_idle = getattr(
                self.valves, "idle_refactor_prompt_template", ""
            ) or self._get_text("idle_refactor_prompt_template")
            try:
                system_content = tpl_idle.format(
                    current_states_json=user_snapshot_json,
                    max_memory_items=self.valves.max_memory_items
                )
            except Exception as e:
                # フォーマットに失敗した場合はログ出力して空
                self.logger.warning(f"[[IDLE_REF]] Failed to format idle_refactor_prompt_template: {e}")
                system_content = ""
            # Compose messages
            ref_messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_snapshot_json},
            ]
            # Save prompt to last_prompt idle section (messages JSON)
            try:
                self._save_last_prompts(
                    idle_refactor_log=json.dumps(
                        ref_messages, ensure_ascii=False, indent=2
                    )
                )
            except Exception:
                pass
            # Notify start
            try:
                await self._safe_emit(
                    event_emitter, self._ev_status("🗂️ 記憶を整理中...", False)
                )
            except Exception:
                pass
            # Call LLM
            api_result = await self._call_llm_for_state_update(ref_messages)
            content = api_result.get("content") or ""
            # Save combined prompt+response as messages JSON
            try:
                ref_messages_with_resp = ref_messages + [
                    {"role": "assistant", "content": content}
                ]
                self._save_last_prompts(
                    idle_refactor_log=json.dumps(
                        ref_messages_with_resp, ensure_ascii=False, indent=2
                    )
                )
            except Exception:
                pass
            if not content or api_result.get("finish_reason") == "length":
                self.logger.warning(
                    "[[IDLE_REF]] LLM returned empty or truncated response; skipping apply."
                )
                return
            # Parse diff and apply only memory.recent
            diff = self._parse_llm_response_for_states(content, is_diff=True) or {}
            mem_diff = {}
            if isinstance(diff, dict) and "memory" in diff:
                try:
                    md = diff.get("memory")
                    if isinstance(md, dict) and "recent" in md:
                        mem_diff = {"memory": {"recent": md.get("recent")}}
                except Exception:
                    mem_diff = {}
            # optional id-based memory ops
            try:
                mops = diff.get("memory_ops")
                if isinstance(mops, dict):
                    current_recent = []
                    try:
                        current_recent = ((states or {}).get("memory") or {}).get(
                            "recent"
                        ) or []
                    except Exception:
                        current_recent = []
                    new_recent = self._apply_ops_to_list_by_id(
                        current_recent, mops, kind="memory"
                    )
                    mem_diff = {"memory": {"recent": new_recent}}
            except Exception:
                pass
            # optional knowledge diff (user/self lists and identity milestones)
            kn_diff = {}
            try:
                kd = diff.get("knowledge") or {}
                if isinstance(kd, dict):
                    # user likes/dislikes
                    user_kd = kd.get("user") or {}
                    if isinstance(user_kd, dict) and (
                        "likes" in user_kd or "dislikes" in user_kd
                    ):
                        kn_diff.setdefault("knowledge", {}).setdefault("user", {})
                        if "likes" in user_kd:
                            kn_diff["knowledge"]["user"]["likes"] = user_kd.get("likes")
                        if "dislikes" in user_kd:
                            kn_diff["knowledge"]["user"]["dislikes"] = user_kd.get(
                                "dislikes"
                            )
                    # self strengths/weaknesses and identity anniversaries/milestones
                    self_kd = kd.get("self") or {}
                    if isinstance(self_kd, dict):
                        if ("strengths" in self_kd) or ("weaknesses" in self_kd):
                            kn_diff.setdefault("knowledge", {}).setdefault("self", {})
                            if "strengths" in self_kd:
                                kn_diff["knowledge"]["self"]["strengths"] = self_kd.get(
                                    "strengths"
                                )
                            if "weaknesses" in self_kd:
                                kn_diff["knowledge"]["self"]["weaknesses"] = (
                                    self_kd.get("weaknesses")
                                )
                        ident_kd = self_kd.get("identity") or {}
                        if isinstance(ident_kd, dict) and (
                            "anniversaries" in ident_kd or "milestones" in ident_kd
                        ):
                            kn_diff.setdefault("knowledge", {}).setdefault(
                                "self", {}
                            ).setdefault("identity", {})
                            if "anniversaries" in ident_kd:
                                kn_diff["knowledge"]["self"]["identity"][
                                    "anniversaries"
                                ] = ident_kd.get("anniversaries")
                            if "milestones" in ident_kd:
                                kn_diff["knowledge"]["self"]["identity"][
                                    "milestones"
                                ] = ident_kd.get("milestones")
            except Exception:
                kn_diff = {}
            # optional id-based ops for identity/knowledge (anniversaries/milestones)
            try:
                iops = diff.get("knowledge_ops")
                if isinstance(iops, dict):
                    # baseline
                    base_ident = (
                        ((states or {}).get("knowledge") or {}).get("self") or {}
                    ).get("identity") or {}
                    ann_base = base_ident.get("anniversaries") or []
                    mil_base = base_ident.get("milestones") or []
                    out_ident: Dict[str, Any] = {}
                    if isinstance(iops.get("anniversaries"), dict):
                        out_ident["anniversaries"] = self._apply_ops_to_list_by_id(
                            ann_base, iops["anniversaries"], kind="identity"
                        )
                    if isinstance(iops.get("milestones"), dict):
                        out_ident["milestones"] = self._apply_ops_to_list_by_id(
                            mil_base, iops["milestones"], kind="identity"
                        )
                    if out_ident:
                        kn_diff.setdefault("knowledge", {}).setdefault(
                            "self", {}
                        ).setdefault("identity", {}).update(out_ident)
            except Exception:
                pass
            if not mem_diff and not kn_diff:
                self.logger.info(
                    "[[IDLE_REF]] No applicable diff returned; nothing to apply."
                )
                return
            # Merge into states and save
            new_states = self._deep_merge_dicts(states or {}, mem_diff)
            if kn_diff:
                new_states = self._deep_merge_dicts(new_states, kn_diff)
            # Deduplicate and cap recent list size (defensive; LLM is instructed but we enforce too)
            try:
                cap = max(1, int(getattr(self.valves, "max_memory_items", 50) or 50))
                recent_list = ((new_states or {}).get("memory") or {}).get(
                    "recent"
                ) or []
                if isinstance(recent_list, list):
                    # Sort by recency×impression desc (missing values default to now and 0)
                    def _key(item: Any) -> float:
                        try:
                            ts = (
                                self._parse_iso8601(str(item.get("timestamp", "")))
                                if isinstance(item, dict)
                                else None
                            )
                            tsec = (ts or self._utcnow()).timestamp()
                        except Exception:
                            tsec = self._utcnow().timestamp()
                        try:
                            imp = (
                                float(item.get("impression_score", 0.0))
                                if isinstance(item, dict)
                                else 0.0
                            )
                        except Exception:
                            imp = 0.0
                        # combine (scaled time in seconds is too large; use relative order via tuple when sorting)
                        return tsec * (1.0 + imp)

                    try:
                        recent_list.sort(key=_key, reverse=True)
                    except Exception:
                        pass
                    # Deduplicate by (content, date) tuple to avoid near-duplicates
                    seen = set()
                    deduped = []
                    for it in recent_list:
                        if not isinstance(it, dict):
                            continue
                        content = str(it.get("content", "")).strip()
                        ts = str(it.get("timestamp", "")).strip()
                        keypair = (content, ts[:10] if len(ts) >= 10 else ts)
                        if content and keypair not in seen:
                            seen.add(keypair)
                            deduped.append(it)
                    # Apply cap
                    new_states.setdefault("memory", {})["recent"] = deduped[:cap]
                # Validate/truncate knowledge lists via validator on save path
            except Exception:
                pass
            await self._save_all_states(user_obj, model_id, new_states)
            # Notify complete
            try:
                await self._safe_emit(
                    event_emitter, self._ev_status("✅ 記憶の整理が完了しました", True)
                )
            except Exception:
                pass
        finally:
            # cleanup task entry
            try:
                t = self._idle_refactor_tasks.get(key)
                if t and t.done():
                    self._idle_refactor_tasks.pop(key, None)
            except Exception:
                pass

    def _complete_goals_and_log_memory(
        self, updated_states: Dict, prev_states: Dict
    ) -> Dict:
        """Remove completed goals from tiers and write memory entries.
        Completion thresholds:
          - short_term: >= 0.99
          - mid_term/long_term: >= 0.995
        Memory entry format mirrors validate_memory_state expectations.
        """
        try:
            goals = (updated_states or {}).get("goal") or {}
            if not isinstance(goals, dict):
                return updated_states
            now_iso = self._now_iso_utc()

            newly_archived: List[Dict[str, Any]] = []
            # tiers to check
            tiers = [
                ("short_term", 0.99),
                ("mid_term", 0.995),
                ("long_term", 0.995),
            ]

            for tier, th in tiers:
                tier_map = goals.get(tier)
                if not isinstance(tier_map, dict) or not tier_map:
                    continue
                keep_map: Dict[str, Any] = {}
                for desc, obj in tier_map.items():
                    try:
                        prog = float((obj or {}).get("progress", 0.0))
                    except Exception:
                        prog = 0.0
                    if prog >= th:
                        # mark as completed: remove from tier and log to memory
                        pr = obj.get("priority", 0.5)
                        try:
                            pr = float(pr)
                        except Exception:
                            pr = 0.5
                        # local clamp/round with module precision
                        try:
                            _prec = getattr(Filter, "NUMERIC_PRECISION", 3)
                        except Exception:
                            _prec = 3
                        entry = {
                            "description": str(desc)[:80],
                            "priority": round(max(0.0, min(1.0, float(pr))), _prec),
                            "progress": round(max(0.0, min(1.0, float(prog))), _prec),
                            "completed_at": now_iso,
                            "tier": tier,
                        }
                        newly_archived.append(entry)
                    else:
                        keep_map[str(desc)] = obj
                goals[tier] = keep_map

            # Write memory entries for newly completed goals
            if newly_archived:
                mem = (updated_states or {}).get("memory") or {}
                if not isinstance(mem, dict):
                    mem = {}
                recent = (
                    mem.get("recent") if isinstance(mem.get("recent"), list) else []
                )
                if not isinstance(recent, list):
                    recent = []
                for g in newly_archived:
                    content = f"目標完了: {g.get('description','')}"
                    recent.insert(
                        0,
                        {
                            "content": content,
                            "impression_score": 0.6,
                            "timestamp": now_iso,
                            "tags": ["目標", "完了"],
                        },
                    )
                # dedupe by content+timestamp pairs, keep up to max_memory_items
                seen = set()
                ordered = []
                for it in recent:
                    if not isinstance(it, dict):
                        continue
                    dedupe_key = f"{str(it.get('content','')).strip()}|{str(it.get('timestamp','')).strip()}"
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    ordered.append(it)
                cap_mem = max(
                    1, int(getattr(self.valves, "max_memory_items", 50) or 50)
                )
                mem["recent"] = ordered[:cap_mem]
                updated_states["memory"] = mem

            updated_states["goal"] = goals
            return updated_states
        except Exception:
            return updated_states

    def _detect_activity(
        self, before: Dict[str, Any], after: Dict[str, Any], diff: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """最小限の活動検出: 有意な状態変化があれば True と理由一覧を返す。
        - physical_health.condition / mental_health.condition の変化
            - physical_health.needs の有意差分（±0.05 以上）
            - physical_health.conditions の変化（件数差）
            - inventory の増減/装備変更（件数/装備状態差）
            - limits の current 値の有意差分（±5 以上）
            - diff に memory や knowledge の新規追加がある
        """
        reasons: List[str] = []

        try:
            # condition 変化
            bs = (before.get("physical_health") or {}).get("condition")
            as_ = (after.get("physical_health") or {}).get("condition")
            if isinstance(bs, str) and isinstance(as_, str) and bs != as_:
                reasons.append(f"physical_condition:{bs}->{as_}")
            bsm = (before.get("mental_health") or {}).get("condition")
            asm = (after.get("mental_health") or {}).get("condition")
            if isinstance(bsm, str) and isinstance(asm, str) and bsm != asm:
                reasons.append(f"mental_condition:{bsm}->{asm}")

            # needs 差分
            bneeds = (before.get("physical_health") or {}).get("needs") or {}
            aneeds = (after.get("physical_health") or {}).get("needs") or {}
            for k in set(list(bneeds.keys()) + list(aneeds.keys())):
                try:
                    bv = float(bneeds.get(k, 0.0) or 0.0)
                    av = float(aneeds.get(k, 0.0) or 0.0)
                    if abs(av - bv) >= 0.05:
                        reasons.append(f"need.{k}:{bv:.2f}->{av:.2f}")
                except Exception:
                    continue

            # conditions 件数差
            bconds = (before.get("physical_health") or {}).get("conditions") or []
            aconds = (after.get("physical_health") or {}).get("conditions") or []
            if (
                isinstance(bconds, list)
                and isinstance(aconds, list)
                and len(bconds) != len(aconds)
            ):
                reasons.append(f"conditions_count:{len(bconds)}->{len(aconds)}")

            # inventory 件数/装備差（リスト形式に対応）
            binv = before.get("inventory") or []
            ainv = after.get("inventory") or []
            if isinstance(binv, list) and isinstance(ainv, list):
                if len(binv) != len(ainv):
                    reasons.append(f"inventory_items:{len(binv)}->{len(ainv)}")
                try:

                    def equip_set(lst):
                        s = set()
                        for it in lst:
                            try:
                                nm = str((it or {}).get("name", "")).strip()
                                if nm and bool((it or {}).get("equipped", False)):
                                    s.add(nm)
                            except Exception:
                                continue
                        return s

                    if equip_set(binv) != equip_set(ainv):
                        reasons.append("inventory_equipped_changed")
                except Exception:
                    pass

            # limits 差分（current）
            def _collect_limits(root: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
                out: Dict[str, Dict[str, float]] = {}
                for sec in ("physical_health", "mental_health"):
                    lm = (root.get(sec) or {}).get("limits") or {}
                    if isinstance(lm, dict):
                        for k, v in lm.items():
                            if isinstance(v, dict):
                                try:
                                    out[f"{sec}.{k}"] = {
                                        "current": float(v.get("current", 0.0) or 0.0),
                                        "max": float(
                                            v.get("max", v.get("base", 100)) or 100.0
                                        ),
                                    }
                                except Exception:
                                    pass
                return out

            blim = _collect_limits(before)
            alim = _collect_limits(after)
            for k, bv in blim.items():
                av = alim.get(k)
                if not av:
                    continue
                try:
                    if abs(av["current"] - bv["current"]) >= 5.0:
                        reasons.append(
                            f"limit.{k}:{bv['current']:.0f}->{av['current']:.0f}"
                        )
                except Exception:
                    pass

            # diff 経由（memory/knowledge）
            if isinstance(diff, dict):
                if (diff.get("memory") or {}).get("recent"):
                    reasons.append("memory_added")
                if (diff.get("knowledge") or {}).get("self") or (
                    diff.get("knowledge") or {}
                ).get("world"):
                    reasons.append("knowledge_updated")
        except Exception:
            self.logger.debug("Activity detection failed", exc_info=True)

        return (len(reasons) > 0, reasons)

    async def _handle_llmemotion_command_async(
        self, last_user_message: str, user: Any, model_id: str
    ) -> Optional[str]:
        """Handle llmemotion commands asynchronously (help, restore).

        Returns a response string to display in chat or None to continue normal flow.
        Ensures no internal state update or LLM context pollution occurs.
        """
        msg = (last_user_message or "").strip()
        if not msg or not msg.lower().startswith("/llmemotion"):
            return None
        parts = msg.split()
        sub = parts[1].lower() if len(parts) >= 2 else ""

        if sub == "help":
            return (
                "[LLMEmotion Command Help]\n"
                "- /llmemotion help : このヘルプを表示。\n"
                "注: コマンド結果はチャットに表示されますが、モデル文脈や内部状態の自動更新には影響しません。"
            )

        return None

    # ========= Debug: Numeric delta logger (turn start visualization) =========
    def _iter_numeric_paths(self, obj: Any, base: str = ""):
        """Yield (path, value) for numeric leaves within nested dict/list structures."""
        try:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = str(k)
                    path = f"{base}.{key}" if base else key
                    yield from self._iter_numeric_paths(v, path)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    path = f"{base}[{i}]" if base else f"[{i}]"
                    yield from self._iter_numeric_paths(v, path)
            else:
                if isinstance(obj, (int, float)):
                    yield base, float(obj)
        except Exception:
            return

    def _log_numeric_deltas(
        self,
        label: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
        threshold: float = 1e-9,
        max_lines: int = 200,
    ) -> None:
        """Log numeric deltas between two state dicts.
        - Only logs paths present in both and numeric in both
        - Uses dot/bracket path notation
        - Caps total lines to avoid excessive logs
        """
        try:
            # Build maps for quick lookup
            bmap = {p: v for p, v in self._iter_numeric_paths(before)}
            amap = {p: v for p, v in self._iter_numeric_paths(after)}

            changes = []
            for p, bval in bmap.items():
                aval = amap.get(p)
                if aval is None:
                    continue
                delta = aval - bval
                if abs(delta) >= threshold:
                    # top-level section is first token before dot or bracket
                    sec = p.split(".")[0].split("[")[0]
                    changes.append((sec, p, bval, aval, delta))

            if not changes:
                self.logger.info(
                    f"[[TURN_START_DELTA]] {label}: No numeric changes detected"
                )
                return

            # Summary by section
            summary: Dict[str, int] = {}
            for sec, *_ in changes:
                summary[sec] = summary.get(sec, 0) + 1
            summary_str = ", ".join(f"{k}:{v}" for k, v in sorted(summary.items()))
            self.logger.info(
                f"[[TURN_START_DELTA]] {label}: {len(changes)} numeric fields changed ({summary_str})"
            )

            # Detailed lines (capped)
            prec = getattr(self, "NUMERIC_PRECISION", 3)
            shown = 0
            for sec, path, bval, aval, delta in changes:
                if shown >= max_lines:
                    break
                self.logger.info(
                    f"[[TURN_START_DELTA_DETAIL]] {path}: {bval:.{prec}f} -> {aval:.{prec}f} (Δ={delta:+.{prec}f})"
                )
                shown += 1
            if shown < len(changes):
                self.logger.info(
                    f"[[TURN_START_DELTA_DETAIL]] ... and {len(changes) - shown} more changes"
                )
        except Exception:
            self.logger.debug("Numeric delta logging failed", exc_info=True)

    def _log_turn_start_deltas(
        self, loaded_states: Dict[str, Any], updated_states: Dict[str, Any]
    ) -> None:
        """Wrapper to log deltas at turn start (driven by analysis LLM results)."""
        try:
            if not loaded_states or not updated_states:
                return
            # Focus on recognized state keys only
            keys = set(self.STATE_DEFS.keys())
            before = {k: loaded_states.get(k) for k in keys if k in loaded_states}
            after = {k: updated_states.get(k) for k in keys if k in updated_states}
            self._log_numeric_deltas("turn_start_llm_delta", before, after)
        except Exception:
            self.logger.debug("Turn-start delta wrapper failed", exc_info=True)

    def _filter_state_diff_keys(self, diff: Any) -> Dict[str, Any]:
        """Return a shallow-filtered copy of diff including only known top-level state keys.
        Unknown or disallowed keys (e.g., timestamps) are dropped to prevent accidental schema drift.
        """
        try:
            if not isinstance(diff, dict):
                return {}
            allowed = set(self.STATE_DEFS.keys())
            return {k: v for k, v in diff.items() if k in allowed}
        except Exception:
            return {}

    def _merge_states(
        self,
        current_states: Dict,
        new_states_diff: "Filter.StateDiff | Dict[str, Any]",
        allow_direct_inventory_override: bool = False,
    ) -> Dict:
        """Merge state diffs into current states with mode-aware inventory policy.
        - Direct mode: ignore inventory diffs from state LLM to prevent wipes unless `allow_direct_inventory_override` is True (set only for outlet-parser diffs).
        - Inference mode: conservatively merge inventory lists to avoid accidental deletions; sanitize descriptions post-merge.
        """
        """
        AIが生成した差分状態(new_states_diff)を現在の状態(current_states)に
        ディープマージ（再帰的マージ）して、更新された完全な状態を返す。
        allow_direct_inventory_override: directモード時でも、外部差分（outletのテキストパーサ由来）による
        inventory更新を許可するためのフラグ。通常はFalse（state LLM由来の置換は拒否）。
        """
        if not new_states_diff:
            self.logger.warning(
                "[[MERGE_SKIPPED]] new_states_diff is empty. Returning current states."
            )
            return current_states

        # Only allow known top-level keys to merge
        filtered_diff = self._filter_state_diff_keys(
            cast(Dict[str, Any], new_states_diff)
        )
        self.logger.info(
            f"[[MERGE_START]] Merging diff with keys: {list(filtered_diff.keys())}"
        )

        updated_states = self._deep_merge_dicts(current_states, filtered_diff)

        # inventory 安全保護: direct モードでは state diff からの inventory 置換を基本無視（テキストパースでのみ変更）
        try:
            if (
                "inventory" in new_states_diff
                and getattr(
                    self.valves, "inventory_update_mode", self.INVENTORY_MODE_INFERENCE
                )
                == self.INVENTORY_MODE_DIRECT
            ):
                if not allow_direct_inventory_override:
                    # direct モードではパース以外の手段での全置換を禁止
                    self.logger.warning(
                        "[[MERGE_INVENTORY_GUARD]] Ignoring inventory diff in direct mode to prevent unintended wipe."
                    )
                    updated_states["inventory"] = current_states.get("inventory", [])
        except Exception:
            pass

        # inventory の安全マージ（部分リストによる全置換・ロスト防止）
        try:
            if (
                "inventory" in new_states_diff
                and getattr(
                    self.valves, "inventory_update_mode", self.INVENTORY_MODE_INFERENCE
                )
                == self.INVENTORY_MODE_INFERENCE
            ):
                cur_inv = (current_states or {}).get("inventory")
                new_inv = new_states_diff.get("inventory")
                if isinstance(cur_inv, list) and isinstance(new_inv, list):
                    merged_inv = self._merge_inventory_conservatively(cur_inv, new_inv)
                    # wipe detection: if update would drop all existing without explicit delete markers
                    if (
                        cur_inv
                        and not merged_inv
                        and any(it.get("quantity", 1) > 0 for it in cur_inv)
                    ):
                        self.logger.warning(
                            "[[MERGE_INVENTORY_ABORT]] Detected potential full wipe; preserving existing inventory."
                        )
                        merged_inv = cur_inv
                    updated_states["inventory"] = merged_inv
                    try:
                        self.logger.info(
                            f"[[MERGE_INVENTORY]] conservative merge applied (cur={len(cur_inv)}, upd={len(new_inv)}, result={len(merged_inv)})"
                        )
                    except Exception:
                        pass
                    # Sanitize descriptions after merge
                    try:
                        updated_states["inventory"] = (
                            self._sanitize_inventory_descriptions(
                                updated_states.get("inventory", [])
                            )
                        )
                    except Exception:
                        self.logger.debug(
                            "[[INVENTORY_SANITIZE_ERROR]] Failed during inference merge sanitization.",
                            exc_info=True,
                        )
        except Exception:
            pass

        # ラベル継続時刻の更新・初期化（since）
        try:
            now_ts = self._now_iso_utc()
            cur_ph = (current_states or {}).get("physical_health") or {}
            cur_mh = (current_states or {}).get("mental_health") or {}
            upd_ph = (updated_states or {}).get("physical_health") or {}
            upd_mh = (updated_states or {}).get("mental_health") or {}
            # physical
            if isinstance(upd_ph, dict):
                cur_lbl = (cur_ph.get("condition") or "").strip()
                new_lbl = (upd_ph.get("condition") or "").strip()
                ts = (
                    upd_ph.get("timestamps")
                    if isinstance(upd_ph.get("timestamps"), dict)
                    else {}
                )
                if new_lbl and new_lbl != cur_lbl:
                    # ラベルが変わったら since を現在時刻に更新
                    ts2 = ts.copy() if isinstance(ts, dict) else {}
                    ts2[self.TIMESTAMP_KEY_CONDITION_SINCE] = now_ts
                    upd_ph[self.STATE_KEY_TIMESTAMPS] = ts2
                    updated_states[self.STATE_KEY_PHYSICAL_HEALTH] = upd_ph
                elif new_lbl and (
                    not isinstance(ts, dict) or not ts.get(self.TIMESTAMP_KEY_CONDITION_SINCE)
                ):
                    # ラベルは同一/既存だが since が未設定なら補完
                    ts2 = ts.copy() if isinstance(ts, dict) else {}
                    ts2[self.TIMESTAMP_KEY_CONDITION_SINCE] = self._safe_nested_get(
                        cur_ph, 
                        self.STATE_KEY_TIMESTAMPS, 
                        self.TIMESTAMP_KEY_CONDITION_SINCE, 
                        default=now_ts
                    )
                    upd_ph[self.STATE_KEY_TIMESTAMPS] = ts2
                    updated_states[self.STATE_KEY_PHYSICAL_HEALTH] = upd_ph
            # mental
            if isinstance(upd_mh, dict):
                cur_lbl = (cur_mh.get("condition") or "").strip()
                new_lbl = (upd_mh.get("condition") or "").strip()
                ts = (
                    upd_mh.get("timestamps")
                    if isinstance(upd_mh.get("timestamps"), dict)
                    else {}
                )
                if new_lbl and new_lbl != cur_lbl:
                    ts2 = ts.copy() if isinstance(ts, dict) else {}
                    ts2[self.TIMESTAMP_KEY_CONDITION_SINCE] = now_ts
                    upd_mh[self.STATE_KEY_TIMESTAMPS] = ts2
                    updated_states[self.STATE_KEY_MENTAL_HEALTH] = upd_mh
                elif new_lbl and (
                    not isinstance(ts, dict) or not ts.get(self.TIMESTAMP_KEY_CONDITION_SINCE)
                ):
                    ts2 = ts.copy() if isinstance(ts, dict) else {}
                    ts2[self.TIMESTAMP_KEY_CONDITION_SINCE] = self._safe_nested_get(
                        cur_mh, 
                        self.STATE_KEY_TIMESTAMPS, 
                        self.TIMESTAMP_KEY_CONDITION_SINCE, 
                        default=now_ts
                    )
                    upd_mh[self.STATE_KEY_TIMESTAMPS] = ts2
                    updated_states[self.STATE_KEY_MENTAL_HEALTH] = upd_mh
        except Exception:
            pass

        # memory.recent の非破壊マージ: 先頭追加 + 既存保持（上限まで）
        try:
            if "memory" in new_states_diff and isinstance(
                new_states_diff["memory"], dict
            ):
                if "recent" in new_states_diff["memory"]:
                    cur_recent = ((current_states or {}).get("memory") or {}).get(
                        "recent"
                    ) or []
                    upd_recent = new_states_diff["memory"].get("recent") or []
                    if isinstance(cur_recent, list) and isinstance(upd_recent, list):
                        # 新しいものを前に。重複は content+timestamp で除外。
                        combined = list(upd_recent) + [
                            x for x in cur_recent if x not in upd_recent
                        ]
                        seen = set()
                        ordered = []
                        for it in combined:
                            if not isinstance(it, dict):
                                continue
                            key = (
                                str(it.get("content", "")).strip(),
                                str(it.get("timestamp", "")).strip(),
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            ordered.append(it)
                        cap = max(
                            1, int(getattr(self.valves, "max_memory_items", 50) or 50)
                        )
                        updated_states.setdefault("memory", {})["recent"] = ordered[
                            :cap
                        ]
        except Exception:
            pass

        # Goal completion archive + memory logging + slim lists
        try:
            updated_states = self._complete_goals_and_log_memory(
                updated_states, current_states
            )
        except Exception:
            self.logger.debug("[[GOAL_COMPLETE_HELPER_ERROR]]", exc_info=True)

        # --- 自動評価（数値補助・一般化） ---
        try:
            # 保険: default_outfit にある名称が inventory に無ければ最小追加（装備は強制しない）
            try:
                updated_states = self._sync_default_outfit_with_inventory(
                    updated_states
                )
            except Exception:
                pass
            # Final defensive sanitization (covers direct mode external diff path)
            try:
                if isinstance(updated_states.get("inventory"), list):
                    updated_states["inventory"] = self._sanitize_inventory_descriptions(
                        updated_states.get("inventory", [])
                    )
            except Exception:
                self.logger.debug(
                    "[[INVENTORY_SANITIZE_ERROR]] Failed in final defensive pass.",
                    exc_info=True,
                )

            # ベースとなるセマンティクスマップ（ローカルコピーにして安全に補正）
            semantics_map: Dict[str, str] = dict((self.DEFAULT_LIMIT_SEMANTICS))
            # 誤設定防止: pleasure_tolerance は常に 'load'、refractory_period は常に 'reserve' として扱う（ユーザー設定が逆でも補正）
            try:
                cur_sem = str(semantics_map.get("pleasure_tolerance", "")).lower()
                if cur_sem and cur_sem != "load":
                    self.logger.info(
                        f"[[LIMIT_SEMANTICS_COERCE]] Overriding semantics for 'pleasure_tolerance' to 'load' (was '{cur_sem}')"
                    )
                semantics_map["pleasure_tolerance"] = "load"
            except Exception:
                semantics_map["pleasure_tolerance"] = "load"

            try:
                cur_sem = str(semantics_map.get("refractory_period", "")).lower()
                if cur_sem and cur_sem != "reserve":
                    self.logger.info(
                        f"[[LIMIT_SEMANTICS_COERCE]] Overriding semantics for 'refractory_period' to 'reserve' (was '{cur_sem}')"
                    )
                semantics_map["refractory_period"] = "reserve"
            except Exception:
                semantics_map["refractory_period"] = "reserve"

            def eval_ratio(key: str, limit_obj: Any) -> Optional[float]:
                return self._limit_ratio_from_item(key, limit_obj, semantics_map)

            # physical_health.limits
            ph = (updated_states or {}).get("physical_health") or {}
            ph_limits = ph.get("limits") or {}
            if isinstance(ph_limits, dict):
                # 直前比率
                ph_prev_limits = (
                    (current_states or {}).get("physical_health") or {}
                ).get("limits") or {}
                for k, v in ph_limits.items():
                    if not isinstance(v, dict):
                        continue
                    cur_r = eval_ratio(str(k), v)
                    prev_r = None
                    try:
                        prev_r = eval_ratio(str(k), ph_prev_limits.get(k) or {})
                    except Exception:
                        prev_r = None
                    # エッジ検出（prev<=off_th かつ cur>=on_th のみ）

            # mental_health.limits
            mh = (updated_states or {}).get("mental_health") or {}
            mh_limits = mh.get("limits") or {}
            if isinstance(mh_limits, dict):
                mh_prev_limits = (
                    (current_states or {}).get("mental_health") or {}
                ).get("limits") or {}
                for k, v in mh_limits.items():
                    if not isinstance(v, dict):
                        continue
                    cur_r = eval_ratio(str(k), v)
                    prev_r = None
                    try:
                        prev_r = eval_ratio(str(k), mh_prev_limits.get(k) or {})
                    except Exception:
                        prev_r = None
            # 比率マップ（現在ターン）を作成（limits を 0-1 負荷に正規化）
            ratios: Dict[str, float] = {}
            try:
                for sec in ("physical_health", "mental_health"):
                    lm = ((updated_states or {}).get(sec) or {}).get("limits") or {}
                    if isinstance(lm, dict):
                        for k, v in lm.items():
                            if not isinstance(v, dict):
                                continue
                            r = eval_ratio(str(k), v)
                            if r is not None:
                                ratios[f"{sec}.{k}"] = max(0.0, min(1.0, r))
            except Exception:
                pass

            # needs（load的な0-1指標）を集約に含める（全needsを対象）
            try:
                needs = ((updated_states or {}).get("physical_health") or {}).get(
                    "needs"
                ) or {}
                if isinstance(needs, dict):
                    include_needs = list(needs.keys())
                    for nk in include_needs:
                        try:
                            ratios[f"needs.{nk}"] = max(
                                0.0, min(1.0, float(needs.get(nk) or 0.0))
                            )
                        except Exception:
                            continue
            except Exception:
                pass

            # （エンジン側ダイナミクスは撤去：ピーク検出・余韻ダンピングは行わない）
            def _ratio_map_prev(sec: str) -> Dict[str, float]:
                out: Dict[str, float] = {}
                lm = (
                    (
                        current_states.get(sec)
                        if isinstance(current_states, dict)
                        else {}
                    )
                    or {}
                ).get("limits") or {}
                if isinstance(lm, dict):
                    sem_map = self.DEFAULT_LIMIT_SEMANTICS
                    for k, v in lm.items():
                        if not isinstance(v, dict):
                            continue
                        r = self._limit_ratio_from_item(k, v, sem_map)
                        if r is None:
                            continue
                        out[k] = r
                return out

            ratios_prev: Dict[str, float] = {}
            for sec in ("physical_health", "mental_health"):
                for k, r in _ratio_map_prev(sec).items():
                    ratios_prev[f"{sec}.{k}"] = max(0.0, min(1.0, r))
            # needs の前ターン値も含める
            try:
                prev_needs = ((current_states or {}).get("physical_health") or {}).get(
                    "needs"
                ) or {}
                if isinstance(prev_needs, dict):
                    include_needs_prev = list(prev_needs.keys())
                    for nk in include_needs_prev:
                        try:
                            ratios_prev[f"needs.{nk}"] = max(
                                0.0, min(1.0, float(prev_needs.get(nk) or 0.0))
                            )
                        except Exception:
                            continue
            except Exception:
                pass
            agg_prev = max([0.0] + list(ratios_prev.values()))
        except Exception:
            pass

        # 最終サニタイズ: sexual_development.parts の装備名/衣類トークン衝突を除去
        try:
            self._filter_sexual_parts_against_inventory(updated_states)
        except Exception:
            self.logger.debug("Sexual parts final sanitize failed", exc_info=True)

        return updated_states

    async def _notify_state_changes(
        self, event_emitter, current_states, updated_states, has_changed
    ):
        """状態変化をUIに通知する"""
        if not has_changed:
            self._create_background_task(
                self._safe_emit(
                    event_emitter,
                    self._ev_status(self._get_text("state_unchanged"), True),
                ),
                name="notify_state_unchanged"
            )
            return

        self._create_background_task(
            self._safe_emit(
                event_emitter, self._ev_status(self._get_text("state_updated"), True)
            ),
            name="notify_state_updated"
        )

        if self.valves.show_state_change_details:
            notifications = []
            KEY_MAP, THRESHOLD = self._get_text("key_map"), 0.001
            inc_str, dec_str = self._get_text("increased"), self._get_text("decreased")

            # --- 感情(emotion)と欲求(desire)の変化をチェック ---
            for state_key in ["emotion", "desire"]:
                if state_key in updated_states and state_key in current_states:
                    for sub_key, label in KEY_MAP[state_key].items():
                        old_v = current_states[state_key].get(sub_key, 0.5)
                        new_v = updated_states[state_key].get(sub_key, 0.5)
                        if new_v > old_v + THRESHOLD:
                            notifications.append(f"{label}{inc_str}")
                        elif new_v < old_v - THRESHOLD:
                            notifications.append(f"{label}{dec_str}")

            # --- 性的状態(sexual_development)の変化をチェック ---
            if (
                self.valves.show_sexual_development_notifications
                and "sexual_development" in updated_states
                and "sexual_development" in current_states
            ):
                old_sd = current_states["sexual_development"]
                new_sd = updated_states["sexual_development"]

                old_parts, new_parts = old_sd.get("parts", {}), new_sd.get("parts", {})
                part_noti_template = self._get_text("sexual_part_notification")
                param_labels = {
                    "sensitivity": KEY_MAP["sexual_parts"]["sensitivity"],
                    "development_progress": KEY_MAP["sexual_parts"][
                        "development_progress"
                    ],
                }

                for part_name, new_values in new_parts.items():
                    old_values = old_parts.get(part_name, {})
                    for param_key, param_label_full in param_labels.items():
                        old_v, new_v = old_values.get(param_key, 0.0), new_values.get(
                            param_key, 0.0
                        )
                        param_label, icon = (
                            param_label_full.split(" ")[1],
                            param_label_full.split(" ")[0],
                        )
                        change_str = None
                        if new_v > old_v + THRESHOLD:
                            change_str = inc_str
                        elif new_v < old_v - THRESHOLD:
                            change_str = dec_str
                        if change_str:
                            notifications.append(
                                part_noti_template.format(
                                    icon=icon,
                                    part=part_name,
                                    param=param_label,
                                    change=change_str,
                                )
                            )

            ### 変更点：ここからTRAITSの通知ロジックを追加 ###
            if (
                self.valves.show_trait_change_notifications
                and "traits" in updated_states
                and "traits" in current_states
            ):
                # 比較のためにリストをセットに変換する
                old_traits_set = set(current_states.get("traits", []))
                new_traits_set = set(updated_states.get("traits", []))

                # 新しく獲得した特性をチェック
                acquired_traits = new_traits_set - old_traits_set
                for trait in acquired_traits:
                    message = self._get_text("trait_acquired").format(trait=trait)
                    notifications.append(message)

                # 失われた特性をチェック
                lost_traits = old_traits_set - new_traits_set
                for trait in lost_traits:
                    message = self._get_text("trait_lost").format(trait=trait)
                    notifications.append(message)

            ### 変更点：ここからSKILLSの通知ロジックを追加 ###
            if (
                self.valves.show_skill_change_notifications
                and "skills" in updated_states
                and "skills" in current_states
            ):
                old_set = set(current_states.get("skills", []))
                new_set = set(updated_states.get("skills", []))
                acquired = new_set - old_set
                for skill in acquired:
                    notifications.append(
                        self._get_text("skill_acquired").format(skill=skill)
                    )
                lost = old_set - new_set
                for skill in lost:
                    notifications.append(
                        self._get_text("skill_lost").format(skill=skill)
                    )

            ### 変更点：ここからINVENTORYの通知ロジックを追加 ###
            if (
                self.valves.show_inventory_change_notifications
                and "inventory" in updated_states
                and "inventory" in current_states
            ):
                # 比較しやすいように、インベントリリストを{アイテム名: アイテム辞書}の形式に変換
                old_inv_map = {
                    item["name"]: item for item in current_states.get("inventory", [])
                }
                new_inv_map = {
                    item["name"]: item for item in updated_states.get("inventory", [])
                }

                # 新しく取得したアイテム
                acquired_items = set(new_inv_map.keys()) - set(old_inv_map.keys())
                for item_name in acquired_items:
                    quantity = new_inv_map[item_name]["quantity"]
                    notifications.append(f"📦 {item_name} を取得 ({quantity})")

                # 喪失したアイテム
                lost_items = set(old_inv_map.keys()) - set(new_inv_map.keys())
                for item_name in lost_items:
                    notifications.append(f"💨 {item_name} を喪失")

                # 既存アイテムの変化（数量・装備状態）
                for item_name in set(old_inv_map.keys()) & set(new_inv_map.keys()):
                    old_item = old_inv_map[item_name]
                    new_item = new_inv_map[item_name]

                    # 数量の変化
                    if new_item["quantity"] > old_item["quantity"]:
                        diff = new_item["quantity"] - old_item["quantity"]
                        notifications.append(f"➕ {item_name} が増加 (+{diff})")
                    elif new_item["quantity"] < old_item["quantity"]:
                        diff = old_item["quantity"] - new_item["quantity"]
                        notifications.append(f"➖ {item_name} が減少 (-{diff})")

                    # 装備状態の変化
                    if new_item["equipped"] and not old_item["equipped"]:
                        notifications.append(f"✅ {item_name} を装備")
                    elif not new_item["equipped"] and old_item["equipped"]:
                        notifications.append(f"❌ {item_name} を装備解除")

            # --- 通知の実行 ---
            if notifications:
                await asyncio.sleep(0.1)
                unique_notifications = list(dict.fromkeys(notifications))
                for message in unique_notifications:
                    self._create_background_task(
                        self._safe_emit(event_emitter, self._ev_status(message, True)),
                        name="notify_detail_change"
                    )
                    await asyncio.sleep(0.8)
                await asyncio.sleep(1.5)
                self._create_background_task(
                    self._safe_emit(
                        event_emitter,
                        self._ev_status(self._get_text("state_change_summary"), True),
                    ),
                    name="notify_change_summary"
                )

    async def _call_llm_for_state_update(
        self, prompt: Union[str, List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        """LLMを呼び出し、応答内容と生成終了理由を辞書で返す。
        
        Args:
            prompt: LLMへ送信するプロンプト（文字列またはメッセージリスト）
            
        Returns:
            {"content": str, "finish_reason": str} の辞書
            
        Raises:
            LLMAPIError: API呼び出しが完全に失敗した場合（現在はエラー辞書を返す）
        """
        # Optional: compact the prompt to reduce tokens (does not affect last_prompt.txt)
        try:
            if getattr(self.valves, "state_prompt_minify", True) and isinstance(
                prompt, str
            ):
                prompt = self._minify_prompt_text(prompt)
        except Exception as e:
            self.logger.debug(f"[[LLM]] Prompt minify failed: {e}")
            
        # セッションを必要時に準備
        if aiohttp is not None and self._aiohttp_session is None:
            await self._ensure_http_session()

        if not self._aiohttp_session:
            self.logger.error("[[LLM]] HTTP client not available")
            return {"content": "", "finish_reason": "client_not_available"}
            
        api_url, model_name, api_key = (
            self.valves.state_analysis_api_url,
            self.valves.state_analysis_model_name,
            self.valves.state_analysis_api_key,
        )
        if not api_url or not model_name:
            self.logger.error("[[LLM]] API URL or model name not configured")
            return {"content": "", "finish_reason": "config_missing"}
            
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": (
                prompt
                if isinstance(prompt, list)
                else [{"role": "user", "content": cast(str, prompt)}]
            ),
            "stream": False,
            "temperature": 0,
            "max_tokens": 15000,
        }

        # --- JSON応答の厳格化（対応エンドポイントのみ） ---
        try:
            mode = (
                self.valves.response_format_mode or self.RESPONSE_FORMAT_NONE
            ).lower()
        except Exception:
            mode = self.RESPONSE_FORMAT_NONE
            
        if mode in (self.RESPONSE_FORMAT_OPENAI_JSON, self.RESPONSE_FORMAT_AUTO):
            payload["response_format"] = {"type": "json_object"}
        elif mode == self.RESPONSE_FORMAT_OLLAMA_JSON:
            payload["format"] = "json"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        # DEBUG: リクエスト情報のログ
        if self._is_debug_enabled():
            try:
                masked_headers = self._mask_headers_for_log(headers)
                payload_for_log = json.loads(json.dumps(payload))
                try:
                    if (
                        isinstance(payload_for_log.get("messages"), list)
                        and payload_for_log["messages"]
                        and isinstance(payload_for_log["messages"][0], dict)
                    ):
                        content = str(payload_for_log["messages"][0].get("content", ""))
                        payload_for_log["messages"][0]["content"] = self._dbg_trunc(
                            content
                        )
                except Exception:
                    pass
                self.logger.debug(
                    f"[[LLM_REQ]] url={api_url} model={model_name} headers={masked_headers} payload={self._dbg_json(payload_for_log)}"
                )
            except Exception as e:
                self.logger.debug(f"[[LLM_REQ]] Failed to log request: {e}")
                
        attempts = max(1, int(self.valves.llm_retry_attempts) + 1)
        backoff_base = float(self.valves.llm_retry_backoff_sec)
        timeout = (
            aiohttp.ClientTimeout(total=float(self.valves.llm_timeout_sec))
            if aiohttp
            else None
        )
        last_error: Optional[Exception] = None
        
        for i in range(attempts):
            try:
                async with self._aiohttp_session.post(
                    api_url, json=payload, headers=headers, timeout=timeout
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # DEBUG: レスポンス生データ
                    if self._is_debug_enabled():
                        try:
                            self.logger.debug(
                                f"[[LLM_RESP_RAW]] {self._dbg_json(data)}"
                            )
                        except Exception as e:
                            self.logger.debug(f"[[LLM_RESP_RAW]] Failed to log: {e}")
                            
                    if data.get("choices") and data["choices"][0].get("message"):
                        choice = data["choices"][0]
                        msg = choice.get("message", {})
                        finish_reason = choice.get("finish_reason")
                        
                        # ツール呼び出し優先（ある場合）
                        tool_calls = msg.get("tool_calls") or []
                        if tool_calls:
                            try:
                                args = (
                                    tool_calls[0]
                                    .get("function", {})
                                    .get("arguments", "")
                                )
                                content = str(args or "")
                            except Exception as e:
                                self.logger.warning(f"[[LLM]] Failed to extract tool args: {e}")
                                content = ""
                        else:
                            content = msg.get("content", "")
                            
                        self.logger.info(
                            f"[[LLM_CALL_SUCCESS]] Finish reason: {finish_reason} (attempt {i+1}/{attempts})"
                        )
                        if self._is_debug_enabled():
                            try:
                                self.logger.debug(
                                    f"[[LLM_RESP_CONTENT]] finish={finish_reason} len={len(content)} content_preview={self._dbg_trunc(content)}"
                                )
                            except Exception:
                                pass
                                
                        return {"content": content, "finish_reason": finish_reason}

                    self.logger.warning(
                        f"[[LLM]] Invalid response format from LLM (attempt {i+1}/{attempts})"
                    )
                    if self._is_debug_enabled():
                        try:
                            self.logger.debug(
                                f"[[LLM_RESP_INVALID]] Raw={self._dbg_json(data)}"
                            )
                        except Exception:
                            pass
                    return {"content": "", "finish_reason": "invalid_response_format"}
                    
            except Exception as e:
                # aiohttpのClientErrorなどをキャッチ（aiohttpがNoneの場合もあるため広くキャッチ）
                if aiohttp and isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError)):
                    last_error = e
                    error_type = "HTTP error" if not isinstance(e, asyncio.TimeoutError) else "Timeout"
                    self.logger.warning(
                        f"[[LLM_CALL_RETRY]] {error_type} on attempt {i+1}/{attempts}: {e}"
                    )
                else:
                    last_error = e
                    self.logger.warning(
                        f"[[LLM_CALL_RETRY]] Unexpected error on attempt {i+1}/{attempts}: {type(e).__name__}: {e}"
                    )
                    
                if i < attempts - 1:
                    delay = self._retry_backoff_delay(i, backoff_base)
                    await asyncio.sleep(delay)
                    
        # 全てのリトライが失敗
        self.logger.error(
            f"[[LLM]] All {attempts} attempts failed. Last error: {last_error}",
            exc_info=True
        )
        return {"content": "", "finish_reason": "api_call_error"}

    async def _safe_emit(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        data: Union[Dict[str, Any], "Filter.UIEvent"],
    ) -> None:
        """Emit UI events in a fire-and-forget manner.
        - Schedules the emission as a background task so it does not block state updates or next-turn processing.
        - Callers may 'await' this method, but it returns immediately after scheduling.
        """
        if not event_emitter:
            return
        try:
            coro = event_emitter(data)
            # event_emitter is expected to be async; run without awaiting to avoid blocking
            try:
                import asyncio as _asyncio

                if _asyncio.iscoroutine(coro):
                    # Use background task helper for tracking and cleanup
                    self._create_background_task(
                        coro,  # type: ignore
                        name="event_emit"
                    )
                else:
                    # Fallback: if somehow sync, delegate to thread pool
                    loop = _asyncio.get_running_loop()
                    loop.run_in_executor(None, lambda: event_emitter(data))
            except Exception:
                # As a last resort, try direct await (won't usually happen)
                await coro  # type: ignore
        except Exception as e:
            self.logger.error(f"Failed to emit event: {e}")

    def _extract_llm_response(self, body: Dict[str, Any]) -> str:
        if body.get("choices") and body["choices"][0].get("message"):
            return body["choices"][0]["message"].get("content", "")
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def _extract_last_user_message(self, body: Dict[str, Any]) -> Optional[str]:
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return None

    def _update_body_content(
        self, body: Dict[str, Any], new_content: str
    ) -> Dict[str, Any]:
        if body.get("choices") and body["choices"][0].get("message"):
            body["choices"][0]["message"]["content"] = new_content
        else:
            for msg in reversed(body.get("messages", [])):
                if msg.get("role") == "assistant":
                    msg["content"] = new_content
                    break
        return body

    async def _outlet_try_handle_command(
        self, last_user_message: Optional[str], user_obj: Any, model_id: str
    ) -> Optional[str]:
        """Detect and handle /llmemotion command. Return synthetic response or None."""
        try:
            if isinstance(last_user_message, str):
                return await self._handle_llmemotion_command_async(
                    last_user_message, user_obj, model_id
                )
        except Exception:
            self.logger.debug(
                "[[COMMAND]] Error while handling llmemotion command.", exc_info=True
            )
        return None

    async def _outlet_process_inventory(
        self,
        mode: str,
        llm_response_text: str,
        current_states: Dict[str, Any],
        user_obj: Any,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
    ) -> Tuple[str, Optional["Filter.StateDiff"]]:
        """Process inventory according to mode and return (final_text_for_user, inventory_diff_for_ai)."""
        final_text = llm_response_text
        diff_for_ai: Optional["Filter.StateDiff"] = None
        if mode == self.INVENTORY_MODE_DIRECT:
            self.logger.info("[[OUTLET]] Running in 'direct' inventory mode.")
            current_inventory = current_states.get("inventory", [])
            updated_inventory, cleaned_llm_response, trim_summary = (
                await self._parse_and_update_inventory_from_text(
                    llm_response_text, current_inventory, user_obj
                )
            )
            inventory_has_changed = self._inventory_signature(
                current_inventory
            ) != self._inventory_signature(updated_inventory)
            if inventory_has_changed:
                diff_for_ai = cast("Filter.StateDiff", {"inventory": updated_inventory})
            if trim_summary and event_emitter:
                try:
                    message = self._get_text("inventory_trimmed_summary").format(
                        before=trim_summary.get("before"),
                        after=trim_summary.get("after"),
                        strategy=trim_summary.get("strategy"),
                    )
                    self._create_background_task(
                        self._safe_emit(event_emitter, self._ev_status(message, True)),
                        name="inventory_trim_notification"
                    )
                except Exception:
                    self.logger.debug(
                        "Failed to emit inventory trim summary", exc_info=True
                    )
            if getattr(self.valves, "strip_inventory_changes_from_response", False):
                final_text = cleaned_llm_response.strip()
            else:
                final_text = llm_response_text.strip()
        else:
            self.logger.info("[[OUTLET]] Running in 'inference' inventory mode.")
            final_text = llm_response_text.strip()
            diff_for_ai = None
        return final_text, diff_for_ai

    async def _outlet_schedule_state_update(
        self,
        last_user_message: str,
        llm_response_text: str,
        system_prompt: Optional[str],
        user_obj: Any,
        model_id: str,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        is_narration: bool,
        external_diff: Optional["Filter.StateDiff"],
    ) -> None:
        """Emit a lightweight status and schedule background state update (RAG + diff).
        Always passes original llm_response_text to analysis LLM.
        """
        # inlet 側で通知を行うため、ここでは重複通知を行わない
        # バックグラウンドで状態更新
        self._create_background_task(
            self._update_or_initialize_states_async(
                last_user_message,
                llm_response_text,
                system_prompt,
                user_obj,
                model_id,
                event_emitter,
                is_narration=is_narration,
                external_diff=external_diff,
            ),
            name="state_update_async"
        )

    # ========= Guard: Prevent inventory items from becoming sexual parts =========
    def _filter_sexual_parts_against_inventory(self, states: Dict[str, Any]) -> None:
        """Drop sexual_development.parts whose names collide with inventory items or contain
        obvious clothing/equipment tokens. Logging a warning for each removal.
        This keeps species neutrality while blocking instruction-level misinference.
        """
        try:
            sd = states.get("sexual_development") or {}
            parts = sd.get("parts")
            if not isinstance(parts, dict) or not parts:
                return
            inv = states.get("inventory") or []
            inv_names = set()
            if isinstance(inv, list):
                for it in inv:
                    try:
                        name = str((it or {}).get("name", "")).strip()
                        if name:
                            inv_names.add(name.lower())
                    except Exception:
                        continue

            # Common clothing/equipment tokens (JA/EN). Avoid generic body-region words.
            banned_tokens = getattr(Filter, "BANNED_SEXUAL_PART_TOKENS", set())

            to_remove = []
            for part_name in list(parts.keys()):
                try:
                    pname = str(part_name).strip()
                    low = pname.lower()
                    if not low:
                        continue
                    # Exact collision with inventory item name
                    if low in inv_names:
                        to_remove.append(part_name)
                        continue
                    # Contains any banned token (substring match)
                    if any(tok in low for tok in banned_tokens):
                        to_remove.append(part_name)
                        continue
                except Exception:
                    continue

            for k in to_remove:
                try:
                    parts.pop(k, None)
                    self.logger.warning(
                        f"[[SANITIZE]] Dropped sexual part '{k}' due to collision with inventory or clothing token."
                    )
                except Exception:
                    pass
        except Exception as e:
            # Fail-safe: do nothing on error
            self.logger.debug("Sexual parts sanitization error", exc_info=True)
