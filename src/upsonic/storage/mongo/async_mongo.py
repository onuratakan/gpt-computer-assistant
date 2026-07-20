"""Async MongoDB storage implementation for Upsonic agent framework."""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from upsonic.session.base import Session, SessionType
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.storage.schemas import KnowledgeRow

try:
    from motor.motor_asyncio import (
        AsyncIOMotorClient,
        AsyncIOMotorCollection,
        AsyncIOMotorDatabase,
    )

    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    AsyncIOMotorClient = None  # type: ignore
    AsyncIOMotorCollection = None  # type: ignore
    AsyncIOMotorDatabase = None  # type: ignore

try:
    from pymongo import AsyncMongoClient  # type: ignore
    from pymongo.collection import AsyncCollection  # type: ignore
    from pymongo.database import AsyncDatabase  # type: ignore
    from pymongo.driver_info import DriverInfo as _DriverInfo

    PYMONGO_ASYNC_AVAILABLE = True
except ImportError:
    PYMONGO_ASYNC_AVAILABLE = False
    AsyncMongoClient = None  # type: ignore
    AsyncDatabase = None  # type: ignore
    AsyncCollection = None  # type: ignore
    _DriverInfo = None  # type: ignore

try:
    from pymongo import ReturnDocument, ReplaceOne
except ImportError:
    ReturnDocument = None  # type: ignore
    ReplaceOne = None  # type: ignore

from upsonic.storage.base import AsyncStorage
from upsonic.storage.mongo.utils import (
    apply_pagination,
    apply_sorting,
    create_collection_indexes_async,
    remove_mongo_id,
)
from upsonic.storage.schemas import UserMemory
from upsonic.storage.utils import deserialize_session, deserialize_session_json_fields
from upsonic.utils.logging_config import get_logger


_logger = get_logger("upsonic.storage.mongo")

try:
    from importlib.metadata import version as _get_version
    _UPSONIC_VERSION: str | None = _get_version("upsonic")
except Exception:
    _UPSONIC_VERSION = None

_ASYNC_DRIVER_INFO = _DriverInfo(name="Upsonic", version=_UPSONIC_VERSION) if _DriverInfo is not None else None


# Client type constants
_CLIENT_TYPE_MOTOR = "motor"
_CLIENT_TYPE_PYMONGO_ASYNC = "pymongo_async"
_CLIENT_TYPE_UNKNOWN = "unknown"


# Create union types for client, database, and collection
if TYPE_CHECKING:
    if MOTOR_AVAILABLE and PYMONGO_ASYNC_AVAILABLE:
        AsyncMongoClientType = Union[AsyncIOMotorClient, AsyncMongoClient]  # type: ignore
        AsyncMongoDatabaseType = Union[AsyncIOMotorDatabase, AsyncDatabase]  # type: ignore
        AsyncMongoCollectionType = Union[AsyncIOMotorCollection, AsyncCollection]  # type: ignore
    elif MOTOR_AVAILABLE:
        AsyncMongoClientType = AsyncIOMotorClient  # type: ignore
        AsyncMongoDatabaseType = AsyncIOMotorDatabase  # type: ignore
        AsyncMongoCollectionType = AsyncIOMotorCollection  # type: ignore
    else:
        AsyncMongoClientType = AsyncMongoClient  # type: ignore
        AsyncMongoDatabaseType = AsyncDatabase  # type: ignore
        AsyncMongoCollectionType = AsyncCollection  # type: ignore
else:
    # Runtime type - use Any to avoid import issues
    AsyncMongoClientType = Any
    AsyncMongoDatabaseType = Any
    AsyncMongoCollectionType = Any


def _detect_client_type(client: Any) -> str:
    """
    Detect whether a client is Motor or PyMongo async.
    
    Args:
        client: MongoDB async client instance
    
    Returns:
        Client type string constant
    """
    if client is None:
        return _CLIENT_TYPE_UNKNOWN

    # Check PyMongo async
    if PYMONGO_ASYNC_AVAILABLE and AsyncMongoClient is not None:
        try:
            if isinstance(client, AsyncMongoClient):
                return _CLIENT_TYPE_PYMONGO_ASYNC
        except (TypeError, AttributeError):
            pass

    if MOTOR_AVAILABLE and AsyncIOMotorClient is not None:
        try:
            if isinstance(client, AsyncIOMotorClient):
                return _CLIENT_TYPE_MOTOR
        except (TypeError, AttributeError):
            pass

    # Fallback to string matching
    client_type_name = type(client).__name__
    if "Motor" in client_type_name or "AsyncIOMotor" in client_type_name:
        return _CLIENT_TYPE_MOTOR
    elif "AsyncMongo" in client_type_name:
        return _CLIENT_TYPE_PYMONGO_ASYNC

    # Last resort: check module name
    module_name = type(client).__module__
    if "motor" in module_name:
        return _CLIENT_TYPE_MOTOR
    elif "pymongo" in module_name:
        return _CLIENT_TYPE_PYMONGO_ASYNC

    return _CLIENT_TYPE_UNKNOWN


class AsyncMongoStorage(AsyncStorage):
    """
    Async MongoDB storage implementation for Upsonic agent framework.
    
    Supports both Motor (legacy) and PyMongo async (recommended) clients.
    When both libraries are available, PyMongo async is preferred.
    
    This storage backend provides persistence for agent sessions and user memory data.
    
    Example:
        ```python
        storage = AsyncMongoStorage(
            db_url="mongodb://localhost:27017",
            db_name="upsonic_db",
        )
        await storage._create_all_tables()
        
        # Upsert a session
        session = AgentSession(session_id="abc123", ...)
        result = await storage.aupsert_session(session)
        
        # Get a session
        session = await storage.aget_session(session_id="abc123")
        ```
    """

    # Client type constants (class-level access)
    CLIENT_TYPE_MOTOR = _CLIENT_TYPE_MOTOR
    CLIENT_TYPE_PYMONGO_ASYNC = _CLIENT_TYPE_PYMONGO_ASYNC
    CLIENT_TYPE_UNKNOWN = _CLIENT_TYPE_UNKNOWN

    def __init__(
        self,
        db_client: Optional[Union["AsyncIOMotorClient", "AsyncMongoClient"]] = None,
        db_name: Optional[str] = None,
        db_url: Optional[str] = None,
        session_collection: Optional[str] = None,
        user_memory_collection: Optional[str] = None,
        knowledge_collection: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize the async MongoDB storage.
        
        Args:
            db_client: The MongoDB async client to use. Can be either Motor's
                      AsyncIOMotorClient or PyMongo's AsyncMongoClient.
                      If not provided, a client will be created from db_url.
            db_name: The name of the database to use.
            db_url: The database URL to connect to.
            session_collection: Name of the collection to store sessions.
            user_memory_collection: Name of the collection to store user memories.
            knowledge_collection: Name of the collection to store knowledge entries.
            id: Unique identifier for this storage instance.
        
        Raises:
            ValueError: If neither db_url nor db_client is provided,
                       or if db_client type is unsupported.
            ImportError: If neither motor nor pymongo async is installed.
        """
        if not MOTOR_AVAILABLE and not PYMONGO_ASYNC_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="pymongo or motor",
                install_command='pip install "upsonic[mongo-storage]"',
                feature_name="MongoDB async storage provider"
            )
        
        super().__init__(
            session_table=session_collection,
            user_memory_table=user_memory_collection,
            knowledge_table=knowledge_collection,
            id=id,
        )

        # Detect client type if provided
        if db_client is not None:
            self._client_type = _detect_client_type(db_client)
            if self._client_type == self.CLIENT_TYPE_UNKNOWN:
                raise ValueError(
                    f"Unsupported MongoDB client type: {type(db_client).__name__}. "
                    "Only Motor (AsyncIOMotorClient) or PyMongo async (AsyncMongoClient) are supported."
                )
        else:
            # Auto-select preferred library when creating from URL
            # Prefer PyMongo async if available, fallback to Motor
            self._client_type = (
                self.CLIENT_TYPE_PYMONGO_ASYNC
                if PYMONGO_ASYNC_AVAILABLE
                else self.CLIENT_TYPE_MOTOR
            )

        # Store configuration for lazy initialization
        self._provided_client: Optional[AsyncMongoClientType] = db_client
        self.db_url: Optional[str] = db_url
        self.db_name: str = db_name if db_name is not None else "upsonic"

        if self._provided_client is None and self.db_url is None:
            raise ValueError("One of db_url or db_client must be provided")

        # Client and database will be lazily initialized per event loop
        self._client: Optional[AsyncMongoClientType] = None
        self._database: Optional[AsyncMongoDatabaseType] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Collection caches
        self._session_collection: Optional[AsyncMongoCollectionType] = None
        self._user_memory_collection: Optional[AsyncMongoCollectionType] = None
        self._cultural_knowledge_collection: Optional[AsyncMongoCollectionType] = None
        self._knowledge_collection: Optional[AsyncMongoCollectionType] = None
        self._sessions_initialized: bool = False
        self._user_memories_initialized: bool = False
        self._cultural_knowledge_initialized: bool = False
        self._knowledge_initialized: bool = False

    # ======================== Client Management ========================

    def _ensure_client(self) -> AsyncMongoClientType:
        """
        Ensure the MongoDB async client is valid for the current event loop.
        
        Both Motor's AsyncIOMotorClient and PyMongo's AsyncMongoClient are tied to
        the event loop they were created in. If we detect a new event loop, we need
        to refresh the client.
        
        Returns:
            A valid client for the current event loop.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, return existing client or create new one
            if self._client is None:
                if self._provided_client is not None:
                    self._client = self._provided_client
                elif self.db_url is not None:
                    if (
                        self._client_type == self.CLIENT_TYPE_PYMONGO_ASYNC
                        and PYMONGO_ASYNC_AVAILABLE
                    ):
                        self._client = AsyncMongoClient(self.db_url, driver=_ASYNC_DRIVER_INFO)  # type: ignore
                    elif self._client_type == self.CLIENT_TYPE_MOTOR and MOTOR_AVAILABLE:
                        self._client = AsyncIOMotorClient(self.db_url)  # type: ignore
                    else:
                        raise RuntimeError(
                            f"Client type '{self._client_type}' not available"
                        )
            return self._client  # type: ignore

        # Check if we're in a different event loop
        if self._event_loop is None or self._event_loop is not current_loop:
            # New event loop detected, create new client
            if self._provided_client is not None:
                client_type_name = (
                    "AsyncMongoClient"
                    if self._client_type == self.CLIENT_TYPE_PYMONGO_ASYNC
                    else "AsyncIOMotorClient"
                )
                _logger.debug(
                    f"New event loop detected. Using provided {client_type_name}, "
                    "which may cause issues if it was created in a different event loop."
                )
                self._client = self._provided_client
            elif self.db_url is not None:
                if (
                    self._client_type == self.CLIENT_TYPE_PYMONGO_ASYNC
                    and PYMONGO_ASYNC_AVAILABLE
                ):
                    self._client = AsyncMongoClient(self.db_url, driver=_ASYNC_DRIVER_INFO)  # type: ignore
                elif self._client_type == self.CLIENT_TYPE_MOTOR and MOTOR_AVAILABLE:
                    self._client = AsyncIOMotorClient(self.db_url)  # type: ignore
                else:
                    raise RuntimeError(
                        f"Client type '{self._client_type}' not available"
                    )

            self._event_loop = current_loop
            self._database = None
            # Clear collection caches and initialization flags
            self._session_collection = None
            self._user_memory_collection = None
            self._cultural_knowledge_collection = None
            self._sessions_initialized = False
            self._user_memories_initialized = False
            self._cultural_knowledge_initialized = False

        return self._client  # type: ignore

    @property
    def db_client(self) -> AsyncMongoClientType:
        """Get the MongoDB client, ensuring it's valid for the current event loop."""
        return self._ensure_client()

    @property
    def database(self) -> AsyncMongoDatabaseType:
        """Get the MongoDB database, ensuring it's valid for the current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
            if self._database is None or self._event_loop != current_loop:
                self._database = self.db_client[self.db_name]  # type: ignore
        except RuntimeError:
            if self._database is None:
                self._database = self.db_client[self.db_name]  # type: ignore
        return self._database

    async def close(self) -> None:
        """
        Close the MongoDB client connection.
        
        Should be called during application shutdown to properly release
        all database connections.
        """
        if self._client is not None:
            self._client.close()
            self._client = None
            self._database = None
            self._session_collection = None
            self._user_memory_collection = None
            self._cultural_knowledge_collection = None
            self._knowledge_collection = None

    # ======================== Table/Collection Management ========================

    async def table_exists(self, table_name: str) -> bool:
        """
        Check if a collection with the given name exists in the MongoDB database.
        
        Args:
            table_name: Name of the collection to check
        
        Returns:
            True if the collection exists in the database, False otherwise
        """
        collection_names = await self.database.list_collection_names()
        return table_name in collection_names

    async def _create_all_tables(self) -> None:
        """Create all configured MongoDB collections if they don't exist."""
        collections_to_create = [
            ("sessions", self.session_table_name),
            ("user_memories", self.user_memory_table_name),
            ("cultural_knowledge", self.cultural_knowledge_table_name),
            ("knowledge", self.knowledge_table_name),
        ]

        for collection_type, collection_name in collections_to_create:
            if collection_name and not await self.table_exists(collection_name):
                await self._get_collection(
                    collection_type, create_collection_if_not_found=True
                )

    def _should_reset_collection_cache(self) -> bool:
        """Check if collection cache should be reset due to event loop change."""
        try:
            current_loop = asyncio.get_running_loop()
            return self._event_loop is not current_loop
        except RuntimeError:
            return False

    async def _get_collection(
        self, collection_type: str, create_collection_if_not_found: bool = True
    ) -> Optional[AsyncMongoCollectionType]:
        """
        Get or create a collection based on collection type.
        
        Args:
            collection_type: The type of collection to get ('sessions' or 'user_memories')
            create_collection_if_not_found: Whether to create the collection if it doesn't exist
        
        Returns:
            The collection object
        
        Raises:
            ValueError: If collection type is unknown or collection name not provided
        """
        # Ensure client is valid for current event loop
        _ = self.db_client

        # Check if collections need to be reset due to event loop change
        reset_cache = self._should_reset_collection_cache()

        if collection_type == "sessions":
            if reset_cache or self._session_collection is None:
                if self.session_table_name is None:
                    raise ValueError(
                        "Session collection was not provided on initialization"
                    )
                self._session_collection = await self._get_or_create_collection(
                    collection_name=self.session_table_name,
                    collection_type="sessions",
                    create_collection_if_not_found=create_collection_if_not_found,
                )
            return self._session_collection

        if collection_type == "user_memories":
            if reset_cache or self._user_memory_collection is None:
                if self.user_memory_table_name is None:
                    raise ValueError(
                        "User memory collection was not provided on initialization"
                    )
                self._user_memory_collection = await self._get_or_create_collection(
                    collection_name=self.user_memory_table_name,
                    collection_type="user_memories",
                    create_collection_if_not_found=create_collection_if_not_found,
                )
            return self._user_memory_collection

        if collection_type == "cultural_knowledge":
            if reset_cache or self._cultural_knowledge_collection is None:
                if self.cultural_knowledge_table_name is None:
                    raise ValueError(
                        "Cultural knowledge collection was not provided on initialization"
                    )
                self._cultural_knowledge_collection = await self._get_or_create_collection(
                    collection_name=self.cultural_knowledge_table_name,
                    collection_type="cultural_knowledge",
                    create_collection_if_not_found=create_collection_if_not_found,
                )
            return self._cultural_knowledge_collection

        if collection_type == "knowledge":
            if reset_cache or self._knowledge_collection is None:
                if self.knowledge_table_name is None:
                    raise ValueError(
                        "Knowledge collection was not provided on initialization"
                    )
                self._knowledge_collection = await self._get_or_create_collection(
                    collection_name=self.knowledge_table_name,
                    collection_type="knowledge",
                    create_collection_if_not_found=create_collection_if_not_found,
                )
            return self._knowledge_collection

        raise ValueError(f"Unknown collection type: {collection_type}")

    async def _get_or_create_collection(
        self,
        collection_name: str,
        collection_type: str,
        create_collection_if_not_found: bool = True,
    ) -> Optional[AsyncMongoCollectionType]:
        """
        Get or create a collection with proper indexes.
        
        Args:
            collection_name: The name of the collection to get or create
            collection_type: The type of collection ('sessions' or 'user_memories')
            create_collection_if_not_found: Whether to create the collection if it doesn't exist
        
        Returns:
            The collection object
        """
        try:
            # Check if database is available (may be None after disconnect)
            if self.database is None:
                if not create_collection_if_not_found:
                    return None
                raise ValueError("Database connection is not available")
            
            collection = self.database[collection_name]

            # Check if already initialized
            init_flag = f"_{collection_type}_initialized"
            if not getattr(self, init_flag, False):
                if not create_collection_if_not_found:
                    return None
                # Create indexes asynchronously
                await create_collection_indexes_async(collection, collection_type)
                setattr(self, init_flag, True)
                _logger.debug(f"Initialized collection '{collection_name}'")
            else:
                _logger.debug(f"Collection '{collection_name}' already initialized")

            return collection

        except Exception as e:
            _logger.error(f"Error getting collection {collection_name}: {e}")
            raise


    def _get_session_type_value(self, session: "Session") -> str:
        """
        Extract session type value string from session object.
        
        Args:
            session: Session object
        
        Returns:
            Session type string value
        """
        session_type_value = "agent"
        if hasattr(session, "session_type"):
            st = session.session_type
            if hasattr(st, "value"):
                session_type_value = st.value
            elif isinstance(st, str):
                session_type_value = st
        return session_type_value

    # ======================== Session Methods ========================

    async def aupsert_session(
        self,
        session: "Session",
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Insert or update a session in the database (async).
        
        Args:
            session: The session to upsert. Must have session_id set.
            deserialize: If True, return deserialized Session object.
                        If False, return raw dictionary.
        
        Returns:
            The upserted session (deserialized or dict based on flag).
        
        Raises:
            ValueError: If session_id is not provided.
        """
        from upsonic.session.agent import AgentSession
        from upsonic.session.base import SessionType

        if not hasattr(session, "session_id") or session.session_id is None:
            raise ValueError("Session must have session_id set for upsert")

        try:
            collection = await self._get_collection(
                "sessions", create_collection_if_not_found=True
            )
            if collection is None:
                return None

            # Convert session to dict with serialize_flag=True
            session_dict = (
                session.to_dict(serialize_flag=True) if hasattr(session, "to_dict") else dict(session)
            )
            current_time = int(time.time())

            if isinstance(session, AgentSession):
                record = {
                    "session_id": session_dict.get("session_id"),
                    "session_type": SessionType.AGENT.value,
                    "agent_id": session_dict.get("agent_id"),
                    "team_id": session_dict.get("team_id"),
                    "workflow_id": session_dict.get("workflow_id"),
                    "user_id": session_dict.get("user_id"),
                    "session_data": session_dict.get("session_data"),
                    "agent_data": session_dict.get("agent_data"),
                    "metadata": session_dict.get("metadata"),
                    "runs": session_dict.get("runs"),
                    "messages": session_dict.get("messages"),
                    "summary": session_dict.get("summary"),
                    "usage": session_dict.get("usage"),
                    "created_at": session_dict.get("created_at") or current_time,
                    "updated_at": current_time,
                }
            else:
                # Fallback for other session types
                session_type_value = self._get_session_type_value(session)
                record = {
                    "session_id": session_dict.get("session_id"),
                    "session_type": session_type_value,
                    "agent_id": session_dict.get("agent_id"),
                    "team_id": session_dict.get("team_id"),
                    "workflow_id": session_dict.get("workflow_id"),
                    "user_id": session_dict.get("user_id"),
                    "session_data": session_dict.get("session_data"),
                    "agent_data": session_dict.get("agent_data"),
                    "team_data": session_dict.get("team_data"),
                    "workflow_data": session_dict.get("workflow_data"),
                    "metadata": session_dict.get("metadata"),
                    "runs": session_dict.get("runs"),
                    "messages": session_dict.get("messages"),
                    "summary": session_dict.get("summary"),
                    "usage": session_dict.get("usage"),
                    "created_at": session_dict.get("created_at") or current_time,
                    "updated_at": current_time,
                }

            result = await collection.find_one_and_replace(
                filter={"session_id": record["session_id"]},
                replacement=record,
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )

            if result is None:
                return None

            session_raw = deserialize_session_json_fields(result)

            if not deserialize:
                return session_raw

            if isinstance(session, AgentSession):
                return AgentSession.from_dict(session_raw, deserialize_flag=True)
            return deserialize_session(session_raw)

        except Exception as e:
            _logger.error(f"Error upserting session: {e}")
            raise

    async def aupsert_sessions(
        self,
        sessions: List["Session"],
        deserialize: bool = True,
    ) -> List[Union["Session", Dict[str, Any]]]:
        """
        Bulk insert or update multiple sessions (async).
        
        Args:
            sessions: List of sessions to upsert. Each must have session_id set.
            deserialize: If True, return deserialized Session objects.
                        If False, return raw dictionaries.
        
        Returns:
            List of upserted sessions.
        
        Raises:
            ValueError: If any session is missing session_id.
        """
        if not sessions:
            return []

        # Validate all sessions have session_id
        for session in sessions:
            if not hasattr(session, "session_id") or session.session_id is None:
                raise ValueError("All sessions must have session_id set for upsert")

        try:
            collection = await self._get_collection(
                "sessions", create_collection_if_not_found=True
            )
            if collection is None:
                _logger.info(
                    "Sessions collection not available, falling back to individual upserts"
                )
                return [
                    result
                    for session in sessions
                    if session is not None
                    for result in [await self.aupsert_session(session, deserialize=deserialize)]
                    if result is not None
                ]

            operations = []
            current_time = int(time.time())

            for session in sessions:
                if session is None:
                    continue

                # Convert session to dict with serialize_flag=True
                session_dict = (
                    session.to_dict(serialize_flag=True) if hasattr(session, "to_dict") else dict(session)
                )
                session_type_value = self._get_session_type_value(session)

                record = {
                    "session_id": session_dict.get("session_id"),
                    "session_type": session_type_value,
                    "agent_id": session_dict.get("agent_id"),
                    "team_id": session_dict.get("team_id"),
                    "workflow_id": session_dict.get("workflow_id"),
                    "user_id": session_dict.get("user_id"),
                    "session_data": session_dict.get("session_data"),
                    "agent_data": session_dict.get("agent_data"),
                    "team_data": session_dict.get("team_data"),
                    "workflow_data": session_dict.get("workflow_data"),
                    "metadata": session_dict.get("metadata"),
                    "runs": session_dict.get("runs"),
                    "messages": session_dict.get("messages"),
                    "summary": session_dict.get("summary"),
                    "usage": session_dict.get("usage"),
                    "created_at": session_dict.get("created_at") or current_time,
                    "updated_at": current_time,
                }

                operations.append(
                    ReplaceOne(
                        filter={"session_id": record["session_id"]},
                        replacement=record,
                        upsert=True,
                    )
                )

            results: List[Union["Session", Dict[str, Any]]] = []

            if operations:
                # Execute bulk write
                await collection.bulk_write(operations)

                # Fetch the results
                session_ids = [
                    session.session_id
                    for session in sessions
                    if session and session.session_id
                ]
                cursor = collection.find({"session_id": {"$in": session_ids}})

                async for doc in cursor:
                    session_raw = deserialize_session_json_fields(doc)

                    if deserialize:
                        results.append(deserialize_session(session_raw))
                    else:
                        results.append(session_raw)

            return results

        except Exception as e:
            _logger.error(
                f"Exception during bulk session upsert, falling back to individual upserts: {e}"
            )
            return [
                result
                for session in sessions
                if session is not None
                for result in [await self.aupsert_session(session, deserialize=deserialize)]
                if result is not None
            ]

    async def aget_session(
        self,
        session_id: Optional[str] = None,
        session_type: Optional["SessionType"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Get a session from the database (async).
        
        Args:
            session_id: ID of the session to retrieve. If None, returns latest session.
            session_type: Type of session (AGENT, TEAM, WORKFLOW).
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            deserialize: If True, return deserialized Session object.
                        If False, return raw dictionary.
        
        Returns:
            The session if found, None otherwise.
        """
        try:
            collection = await self._get_collection(
                "sessions", create_collection_if_not_found=True
            )
            if collection is None:
                return None
        except ValueError:
            return None

        try:
            query: Dict[str, Any] = {}

            if session_id is not None:
                query["session_id"] = session_id
            if session_type is not None:
                query["session_type"] = session_type.value
            if user_id is not None:
                query["user_id"] = user_id
            if agent_id is not None:
                query["agent_id"] = agent_id

            # If no session_id provided, get latest
            if session_id is None:
                cursor = collection.find(query).sort("created_at", -1).limit(1)
                result = await cursor.to_list(length=1)
                if not result:
                    return None
                result = result[0]
            else:
                result = await collection.find_one(query)

            if result is None:
                return None

            session_raw = deserialize_session_json_fields(result)

            if not deserialize:
                return session_raw

            return deserialize_session(session_raw, session_type)

        except Exception as e:
            _logger.error(f"Error getting session: {e}")
            raise

    async def aget_sessions(
        self,
        session_ids: Optional[List[str]] = None,
        session_type: Optional["SessionType"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        deserialize: bool = True,
    ) -> Union[List["Session"], Tuple[List[Dict[str, Any]], int]]:
        """
        Get multiple sessions from the database (async).
        
        Args:
            session_ids: List of session IDs to retrieve. If None, returns all sessions.
            session_type: Filter by session type.
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.
            sort_by: Field to sort by.
            sort_order: Sort order ('asc' or 'desc').
            deserialize: If True, return deserialized Session objects.
                        If False, return tuple of (raw dictionaries, total count).
        
        Returns:
            List of sessions or tuple of (list of dicts, total count).
        """
        try:
            collection = await self._get_collection(
                "sessions", create_collection_if_not_found=True
            )
            if collection is None:
                return [] if deserialize else ([], 0)
        except ValueError:
            return [] if deserialize else ([], 0)

        try:
            # Handle empty session_ids list - return empty result immediately
            if session_ids is not None and len(session_ids) == 0:
                return [] if deserialize else ([], 0)

            query: Dict[str, Any] = {}

            if session_ids is not None and len(session_ids) > 0:
                query["session_id"] = {"$in": session_ids}
            if session_type is not None:
                query["session_type"] = session_type.value
            if user_id is not None:
                query["user_id"] = user_id
            if agent_id is not None:
                query["agent_id"] = agent_id

            # Get total count
            total_count = await collection.count_documents(query)

            # Build cursor
            cursor = collection.find(query)

            # Apply sorting
            sort_criteria = apply_sorting({}, sort_by, sort_order)
            if sort_criteria:
                cursor = cursor.sort(sort_criteria)

            # Apply pagination
            cursor = apply_pagination(cursor, limit, offset)

            records = await cursor.to_list(length=None)

            if not records:
                return [] if deserialize else ([], 0)

            sessions_raw = [
                deserialize_session_json_fields(record) for record in records
            ]

            if not deserialize:
                return sessions_raw, total_count

            return [
                deserialize_session(s, session_type) for s in sessions_raw
            ]

        except Exception as e:
            _logger.error(f"Error getting sessions: {e}")
            raise

    async def adelete_session(self, session_id: str) -> bool:
        """
        Delete a session from the database (async).
        
        Args:
            session_id: ID of the session to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            ValueError: If session_id is not provided.
        """
        if not session_id:
            raise ValueError("session_id is required for delete")

        try:
            collection = await self._get_collection(
                "sessions", create_collection_if_not_found=False
            )
            if collection is None:
                return False
        except ValueError:
            return False

        try:
            result = await collection.delete_one({"session_id": session_id})
            deleted = result.deleted_count > 0
            if deleted:
                _logger.debug(f"Deleted session: {session_id}")
            else:
                _logger.debug(f"No session found to delete: {session_id}")
            return deleted

        except Exception as e:
            _logger.error(f"Error deleting session: {e}")
            raise

    async def adelete_sessions(self, session_ids: List[str]) -> int:
        """
        Delete multiple sessions from the database (async).
        
        Args:
            session_ids: List of session IDs to delete.
        
        Returns:
            Number of sessions deleted.
        
        Raises:
            ValueError: If session_ids is empty.
        """
        if not session_ids:
            raise ValueError("session_ids is required and cannot be empty")

        try:
            collection = await self._get_collection(
                "sessions", create_collection_if_not_found=False
            )
            if collection is None:
                return 0
        except ValueError:
            return 0

        try:
            result = await collection.delete_many({"session_id": {"$in": session_ids}})
            deleted_count = result.deleted_count
            _logger.debug(f"Deleted {deleted_count} sessions")
            return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting sessions: {e}")
            raise

    # ======================== User Memory Methods ========================

    async def aupsert_user_memory(
        self,
        user_memory: UserMemory,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """
        Insert or update user memory in the database (async).
        
        Args:
            user_memory: The UserMemory instance to store. Must have user_id set.
            deserialize: If True, return UserMemory object. If False, return raw dict.
        
        Returns:
            The upserted user memory record.
        
        Raises:
            ValueError: If user_id is not provided in user_memory.
        """
        if not user_memory.user_id:
            raise ValueError("user_memory must have user_id set for upsert")

        try:
            collection = await self._get_collection(
                "user_memories", create_collection_if_not_found=True
            )
            if collection is None:
                return None

            current_time = int(time.time())

            record = {
                "user_id": user_memory.user_id,
                "user_memory": user_memory.user_memory,
                "agent_id": user_memory.agent_id,
                "team_id": user_memory.team_id,
                "created_at": user_memory.created_at or current_time,
                "updated_at": current_time,
            }

            result = await collection.find_one_and_replace(
                filter={"user_id": user_memory.user_id},
                replacement=record,
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )

            if result is None:
                return None

            memory_dict = remove_mongo_id(result)
            if deserialize:
                return UserMemory.from_dict(memory_dict)
            return memory_dict

        except Exception as e:
            _logger.error(f"Error upserting user memory: {e}")
            raise

    async def aupsert_user_memories(
        self,
        user_memories: List[UserMemory],
        deserialize: bool = True,
    ) -> List[Union[UserMemory, Dict[str, Any]]]:
        """
        Bulk insert or update multiple user memories (async).
        
        Args:
            user_memories: List of UserMemory instances. Each must have user_id set.
            deserialize: If True, return UserMemory objects. If False, return raw dicts.
        
        Returns:
            List of upserted user memory records.
        
        Raises:
            ValueError: If any record is missing user_id.
        """
        if not user_memories:
            return []

        # Validate all have user_id
        for memory in user_memories:
            if not memory.user_id:
                raise ValueError("All user memories must have user_id set")

        try:
            collection = await self._get_collection(
                "user_memories", create_collection_if_not_found=True
            )
            if collection is None:
                _logger.info(
                    "User memories collection not available, falling back to individual upserts"
                )
                return [
                    result
                    for memory in user_memories
                    if memory is not None
                    for result in [
                        await self.aupsert_user_memory(
                            user_memory=memory,
                            deserialize=deserialize,
                        )
                    ]
                    if result is not None
                ]

            operations = []
            current_time = int(time.time())

            for memory in user_memories:
                if memory is None:
                    continue

                record = {
                    "user_id": memory.user_id,
                    "user_memory": memory.user_memory,
                    "agent_id": memory.agent_id,
                    "team_id": memory.team_id,
                    "created_at": memory.created_at or current_time,
                    "updated_at": current_time,
                }

                operations.append(
                    ReplaceOne(
                        filter={"user_id": memory.user_id},
                        replacement=record,
                        upsert=True,
                    )
                )

            results: List[Union[UserMemory, Dict[str, Any]]] = []

            if operations:
                # Execute bulk write
                await collection.bulk_write(operations)

                # Fetch the results
                user_ids = [
                    memory.user_id
                    for memory in user_memories
                    if memory and memory.user_id
                ]
                cursor = collection.find({"user_id": {"$in": user_ids}})

                async for doc in cursor:
                    memory_dict = remove_mongo_id(doc)
                    if deserialize:
                        results.append(UserMemory.from_dict(memory_dict))
                    else:
                        results.append(memory_dict)

            return results

        except Exception as e:
            _logger.error(
                f"Exception during bulk user memory upsert, falling back to individual upserts: {e}"
            )
            return [
                result
                for memory in user_memories
                if memory is not None
                for result in [
                    await self.aupsert_user_memory(
                        user_memory=memory,
                        deserialize=deserialize,
                    )
                ]
                if result is not None
            ]

    async def aget_user_memory(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """
        Get user memory from the database (async).
        
        Args:
            user_id: User ID to retrieve. If None, returns latest memory.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            deserialize: If True, return UserMemory object. If False, return raw dict.
        
        Returns:
            The user memory record if found, None otherwise.
        """
        try:
            collection = await self._get_collection(
                "user_memories", create_collection_if_not_found=True
            )
            if collection is None:
                return None
        except ValueError:
            return None

        try:
            query: Dict[str, Any] = {}

            if user_id is not None:
                query["user_id"] = user_id
            if agent_id is not None:
                query["agent_id"] = agent_id
            if team_id is not None:
                query["team_id"] = team_id

            # If no user_id provided, get latest
            if user_id is None:
                cursor = collection.find(query).sort("updated_at", -1).limit(1)
                result = await cursor.to_list(length=1)
                if not result:
                    return None
                result = result[0]
            else:
                result = await collection.find_one(query)

            if result is None:
                return None

            memory_dict = remove_mongo_id(result)
            if deserialize:
                return UserMemory.from_dict(memory_dict)
            return memory_dict

        except Exception as e:
            _logger.error(f"Error getting user memory: {e}")
            raise

    async def aget_user_memories(
        self,
        user_ids: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        deserialize: bool = True,
    ) -> Union[List[UserMemory], Tuple[List[Dict[str, Any]], int]]:
        """
        Get multiple user memories from the database (async).
        
        Args:
            user_ids: List of user IDs to retrieve. If None, returns all.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            limit: Maximum number of records to return.
            offset: Number of records to skip.
            deserialize: If True, return list of UserMemory objects. 
                         If False, return tuple of (list of dicts, total count).
        
        Returns:
            List of UserMemory objects if deserialize=True, otherwise tuple of (dicts, count).
        """
        try:
            collection = await self._get_collection(
                "user_memories", create_collection_if_not_found=False
            )
            if collection is None:
                if deserialize:
                    return []
                return [], 0
        except ValueError:
            if deserialize:
                return []
            return [], 0

        try:
            # Handle empty user_ids list - return empty result immediately
            if user_ids is not None and len(user_ids) == 0:
                if deserialize:
                    return []
                return [], 0

            query: Dict[str, Any] = {}

            if user_ids is not None and len(user_ids) > 0:
                query["user_id"] = {"$in": user_ids}
            if agent_id is not None:
                query["agent_id"] = agent_id
            if team_id is not None:
                query["team_id"] = team_id

            # Get total count
            total_count = await collection.count_documents(query)

            # Build cursor
            cursor = collection.find(query).sort("updated_at", -1)

            # Apply pagination
            cursor = apply_pagination(cursor, limit, offset)

            records = await cursor.to_list(length=None)

            if not records:
                if deserialize:
                    return []
                return [], 0

            if deserialize:
                return [
                    UserMemory.from_dict(remove_mongo_id(record)) for record in records
                ]
            return [
                remove_mongo_id(record) for record in records
            ], total_count

        except Exception as e:
            _logger.error(f"Error getting user memories: {e}")
            raise

    async def adelete_user_memory(self, user_id: str) -> bool:
        """
        Delete user memory from the database (async).
        
        Args:
            user_id: User ID of the memory to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            ValueError: If user_id is not provided.
        """
        if not user_id:
            raise ValueError("user_id is required for delete")

        try:
            collection = await self._get_collection(
                "user_memories", create_collection_if_not_found=False
            )
            if collection is None:
                return False
        except ValueError:
            return False

        try:
            result = await collection.delete_one({"user_id": user_id})
            deleted = result.deleted_count > 0
            if deleted:
                _logger.debug(f"Deleted user memory: {user_id}")
            else:
                _logger.debug(f"No user memory found to delete: {user_id}")
            return deleted

        except Exception as e:
            _logger.error(f"Error deleting user memory: {e}")
            raise

    async def adelete_user_memories(self, user_ids: List[str]) -> int:
        """
        Delete multiple user memories from the database (async).
        
        Args:
            user_ids: List of user IDs to delete.
        
        Returns:
            Number of records deleted.
        
        Raises:
            ValueError: If user_ids is empty.
        """
        if not user_ids:
            raise ValueError("user_ids is required and cannot be empty")

        try:
            collection = await self._get_collection(
                "user_memories", create_collection_if_not_found=False
            )
            if collection is None:
                return 0
        except ValueError:
            return 0

        try:
            result = await collection.delete_many({"user_id": {"$in": user_ids}})
            deleted_count = result.deleted_count
            _logger.debug(f"Deleted {deleted_count} user memories")
            return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting user memories: {e}")
            raise

    # ======================== Utility Methods ========================

    async def aclear_all(self) -> None:
        """
        Clear all data from all collections (async).
        
        This removes all sessions and user memories from the storage.
        Use with caution.
        """
        try:
            # Clear sessions
            try:
                collection = await self._get_collection(
                    "sessions", create_collection_if_not_found=False
                )
                if collection is not None:
                    await collection.delete_many({})
                    _logger.debug("Cleared all sessions")
            except ValueError:
                pass

            # Clear user memories
            try:
                collection = await self._get_collection(
                    "user_memories", create_collection_if_not_found=False
                )
                if collection is not None:
                    await collection.delete_many({})
                    _logger.debug("Cleared all user memories")
            except ValueError:
                pass

            # Clear knowledge
            try:
                collection = await self._get_collection(
                    "knowledge", create_collection_if_not_found=False
                )
                if collection is not None:
                    await collection.delete_many({})
                    _logger.debug("Cleared all knowledge entries")
            except ValueError:
                pass

            _logger.info("Cleared all data from storage")

        except Exception as e:
            _logger.error(f"Error clearing all data: {e}")
            raise

    # ======================== Generic Model Methods (Async) ========================

    async def _get_generic_model_collection(self, collection: str) -> "AsyncIOMotorCollection":
        """Get or create a collection for generic models."""
        collection_name = f"upsonic_models_{collection}"
        return self.db[collection_name]

    async def aupsert_model(
        self,
        key: str,
        model: Any,
        collection: str = "generic_models",
    ) -> None:
        """Insert or update a generic Pydantic model in storage (async)."""
        import time
        
        try:
            coll = await self._get_generic_model_collection(collection)
            current_time = int(time.time())
            
            # Serialize model
            if hasattr(model, 'model_dump'):
                model_data = model.model_dump(mode='json')
            elif hasattr(model, 'dict'):
                model_data = model.dict()
            else:
                model_data = dict(model)
            
            document = {
                "_id": key,
                "collection": collection,
                "model_data": model_data,
                "updated_at": current_time,
            }
            
            # Check if exists to preserve created_at
            existing = await coll.find_one({"_id": key})
            if existing:
                document["created_at"] = existing.get("created_at", current_time)
            else:
                document["created_at"] = current_time
            
            await coll.replace_one({"_id": key}, document, upsert=True)
            _logger.debug(f"Upserted model with key '{key}' in collection '{collection}'")
            
        except Exception as e:
            _logger.error(f"Error upserting model: {e}")
            raise

    async def aget_model(
        self,
        key: str,
        model_type: Any,
        collection: str = "generic_models",
    ) -> Optional[Any]:
        """Retrieve a generic Pydantic model from storage (async)."""
        try:
            coll = await self._get_generic_model_collection(collection)
            
            document = await coll.find_one({"_id": key})
            if document is None:
                return None
            
            model_data = document.get("model_data")
            if model_data is None:
                return None
            
            return model_type(**model_data)
            
        except Exception as e:
            _logger.error(f"Error getting model: {e}")
            return None

    async def adelete_model(
        self,
        key: str,
        collection: str = "generic_models",
    ) -> bool:
        """Delete a generic model from storage (async)."""
        try:
            coll = await self._get_generic_model_collection(collection)
            
            result = await coll.delete_one({"_id": key})
            
            if result.deleted_count > 0:
                _logger.debug(f"Deleted model with key '{key}' from collection '{collection}'")
                return True
            return False
            
        except Exception as e:
            _logger.error(f"Error deleting model: {e}")
            return False

    async def alist_models(
        self,
        model_type: Any,
        collection: str = "generic_models",
    ) -> List[Any]:
        """List all models of a type in a collection (async)."""
        try:
            coll = await self._get_generic_model_collection(collection)
            
            cursor = coll.find({"collection": collection})
            
            models = []
            async for document in cursor:
                try:
                    model_data = document.get("model_data")
                    if model_data:
                        models.append(model_type(**model_data))
                except Exception as e:
                    _logger.warning(f"Failed to deserialize model: {e}")
            
            return models
            
        except Exception as e:
            _logger.error(f"Error listing models: {e}")
            return []

    # ======================== Cultural Knowledge Methods ========================

    async def _get_cultural_knowledge_collection(
        self, create_collection_if_not_found: bool = False
    ) -> Optional[AsyncMongoCollectionType]:
        """Get the cultural knowledge collection, creating if needed."""
        return await self._get_collection(
            "cultural_knowledge",
            create_collection_if_not_found=create_collection_if_not_found,
        )

    async def adelete_cultural_knowledge(self, id: str) -> None:
        """Delete cultural knowledge from the database (async).
        
        Args:
            id: The ID of the cultural knowledge to delete.
        
        Raises:
            Exception: If an error occurs during deletion.
        """
        try:
            collection = await self._get_cultural_knowledge_collection(
                create_collection_if_not_found=False
            )
            if collection is None:
                return

            await collection.delete_one({"id": id})
            _logger.debug(f"Deleted cultural knowledge with ID: {id}")

        except Exception as e:
            _logger.error(f"Error deleting cultural knowledge: {e}")
            raise e

    async def aget_cultural_knowledge(
        self,
        id: str,
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """Get cultural knowledge from the database (async).
        
        Args:
            id: The ID of the cultural knowledge to get.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            CulturalKnowledge object or dict if found, None otherwise.
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        try:
            collection = await self._get_cultural_knowledge_collection(
                create_collection_if_not_found=False
            )
            if collection is None:
                return None

            result = await collection.find_one({"id": id})
            if result is None:
                return None

            # Remove MongoDB's _id field
            db_row = remove_mongo_id(result)
            if not deserialize:
                return db_row
            return CulturalKnowledge.from_dict(db_row)

        except Exception as e:
            _logger.error(f"Exception reading from cultural knowledge collection: {e}")
            raise e

    async def aget_all_cultural_knowledge(
        self,
        name: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Union[List["CulturalKnowledge"], Tuple[List[Dict[str, Any]], int]]:
        """Get all cultural knowledge entries from the database (async).
        
        Args:
            name: Filter by name.
            limit: Maximum number of records to return.
            page: Page number (1-indexed).
            sort_by: Column to sort by.
            sort_order: Sort order ('asc' or 'desc').
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            deserialize: If True, return list of CulturalKnowledge objects.
                        If False, return tuple of (list of dicts, total count).
        
        Returns:
            List of CulturalKnowledge objects or tuple of (list of dicts, total count).
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        try:
            collection = await self._get_cultural_knowledge_collection(
                create_collection_if_not_found=False
            )
            if collection is None:
                return [] if deserialize else ([], 0)

            # Build query
            query: Dict[str, Any] = {}
            if name is not None:
                query["name"] = name
            if agent_id is not None:
                query["agent_id"] = agent_id
            if team_id is not None:
                query["team_id"] = team_id

            # Get total count for pagination
            total_count = await collection.count_documents(query)

            # Apply sorting
            sort_criteria = apply_sorting({}, sort_by, sort_order)

            # Build cursor with sorting and pagination
            cursor = collection.find(query)
            if sort_criteria:
                cursor = cursor.sort(sort_criteria)
            
            # Apply pagination
            if limit is not None:
                if page is not None and page > 1:
                    offset = (page - 1) * limit
                    cursor = cursor.skip(offset)
                cursor = cursor.limit(limit)

            # Remove MongoDB's _id field from all results
            db_rows = [remove_mongo_id(item) async for item in cursor]

            if not deserialize:
                return db_rows, total_count
            return [CulturalKnowledge.from_dict(row) for row in db_rows]

        except Exception as e:
            _logger.error(f"Error reading from cultural knowledge collection: {e}")
            raise e

    async def aupsert_cultural_knowledge(
        self,
        cultural_knowledge: "CulturalKnowledge",
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """Upsert cultural knowledge into the database (async).
        
        Args:
            cultural_knowledge: The CulturalKnowledge instance to upsert.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            The upserted CulturalKnowledge object or dict, or None if error occurs.
        
        Raises:
            Exception: If an error occurs during upsert.
        """
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        import uuid
        
        try:
            collection = await self._get_cultural_knowledge_collection(
                create_collection_if_not_found=True
            )
            if collection is None:
                return None

            if cultural_knowledge.id is None:
                cultural_knowledge.id = str(uuid.uuid4())

            # Use to_dict for serialization
            data = cultural_knowledge.to_dict()
            
            # Convert RFC3339 timestamps to epoch seconds for storage
            if "created_at" in data and data["created_at"] is not None:
                if isinstance(data["created_at"], str):
                    from upsonic.utils.dttm import to_epoch_s
                    data["created_at"] = to_epoch_s(data["created_at"])
            if "updated_at" in data and data["updated_at"] is not None:
                if isinstance(data["updated_at"], str):
                    from upsonic.utils.dttm import to_epoch_s
                    data["updated_at"] = to_epoch_s(data["updated_at"])
            else:
                # Set updated_at to current time if not set
                data["updated_at"] = int(time.time())

            # Upsert using replace_one
            await collection.replace_one(
                {"id": cultural_knowledge.id},
                data,
                upsert=True,
            )

            if not deserialize:
                return data
            return CulturalKnowledge.from_dict(data)

        except Exception as e:
            _logger.error(f"Error upserting cultural knowledge: {e}")
            raise e

    # ======================== Knowledge Content Methods (Async) ========================

    async def _get_knowledge_collection(
        self, create_collection_if_not_found: bool = False
    ) -> Optional[AsyncMongoCollectionType]:
        """Get the knowledge collection, creating if needed."""
        return await self._get_collection(
            "knowledge",
            create_collection_if_not_found=create_collection_if_not_found,
        )

    async def aupsert_knowledge_content(
        self,
        knowledge_row: "KnowledgeRow",
    ) -> Optional["KnowledgeRow"]:
        """Insert or update a knowledge document registry entry (async)."""
        from upsonic.storage.schemas import KnowledgeRow

        try:
            collection = await self._get_knowledge_collection(
                create_collection_if_not_found=True
            )
            if collection is None:
                return None

            current_time = int(time.time())

            data = knowledge_row.to_dict()
            data.setdefault("created_at", current_time)
            data["updated_at"] = current_time

            result = await collection.find_one_and_replace(
                filter={"id": data["id"]},
                replacement=data,
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )

            if result is None:
                return None

            return KnowledgeRow.from_dict(remove_mongo_id(result))

        except Exception as e:
            _logger.error(f"Error upserting knowledge content: {e}")
            raise e

    async def aget_knowledge_content(
        self,
        id: str,
    ) -> Optional["KnowledgeRow"]:
        """Get a knowledge document registry entry by ID (async)."""
        from upsonic.storage.schemas import KnowledgeRow

        try:
            collection = await self._get_knowledge_collection(
                create_collection_if_not_found=False
            )
            if collection is None:
                return None

            result = await collection.find_one({"id": id})
            if result is None:
                return None

            return KnowledgeRow.from_dict(remove_mongo_id(result))

        except Exception as e:
            _logger.error(f"Error getting knowledge content: {e}")
            raise e

    async def aget_knowledge_contents(
        self,
        knowledge_base_id: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> Tuple[List["KnowledgeRow"], int]:
        """Get knowledge document registry entries with filtering and pagination (async)."""
        from upsonic.storage.schemas import KnowledgeRow

        try:
            collection = await self._get_knowledge_collection(
                create_collection_if_not_found=False
            )
            if collection is None:
                return [], 0

            query: Dict[str, Any] = {}
            if knowledge_base_id is not None:
                query["knowledge_base_id"] = knowledge_base_id

            total_count: int = await collection.count_documents(query)

            sort_criteria = apply_sorting({}, sort_by, sort_order)
            cursor = collection.find(query)
            if sort_criteria:
                cursor = cursor.sort(sort_criteria)

            if limit is not None:
                if page is not None and page > 1:
                    offset = (page - 1) * limit
                    cursor = cursor.skip(offset)
                cursor = cursor.limit(limit)

            rows = [KnowledgeRow.from_dict(remove_mongo_id(item)) async for item in cursor]
            return rows, total_count

        except Exception as e:
            _logger.error(f"Error getting knowledge contents: {e}")
            raise e

    async def adelete_knowledge_content(self, id: str) -> bool:
        """Delete a knowledge document registry entry by ID (async)."""
        try:
            collection = await self._get_knowledge_collection(
                create_collection_if_not_found=False
            )
            if collection is None:
                return False

            result = await collection.delete_one({"id": id})
            deleted: bool = result.deleted_count > 0
            if deleted:
                _logger.debug(f"Deleted knowledge content: {id}")
            return deleted

        except Exception as e:
            _logger.error(f"Error deleting knowledge content: {e}")
            raise e

    async def adelete_knowledge_contents(self, ids: List[str]) -> int:
        """Delete multiple knowledge document registry entries (async)."""
        if not ids:
            return 0

        try:
            collection = await self._get_knowledge_collection(
                create_collection_if_not_found=False
            )
            if collection is None:
                return 0

            result = await collection.delete_many({"id": {"$in": ids}})
            deleted_count: int = result.deleted_count
            _logger.debug(f"Deleted {deleted_count} knowledge entries")
            return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting knowledge contents: {e}")
            raise e