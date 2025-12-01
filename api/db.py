"""
MongoDB Database Layer - Singleton Pattern
Handles all database operations for PropelAI
"""

import os
from typing import Optional

try:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
    MOTOR_AVAILABLE = True
except ImportError as e:
    print(f"[DB] WARNING: motor not available: {e}")
    print("[DB] Please install: pip install motor>=3.3.0 dnspython>=2.3.0 pymongo>=4.3.3")
    MOTOR_AVAILABLE = False
    AsyncIOMotorClient = None
    AsyncIOMotorDatabase = None


class Database:
    """
    Singleton MongoDB connection manager using AsyncIO Motor.
    
    Provides access to collections:
    - rfps: RFP metadata, requirements, and state
    - chat_history: Message logs linked by rfp_id
    - company_library: Document metadata and content
    """
    
    _instance: Optional['Database'] = None
    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None
    
    def __new__(cls):
        """Singleton pattern - only one instance"""
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
        return cls._instance
    
    async def connect(self):
        """Initialize MongoDB connection"""
        if not MOTOR_AVAILABLE:
            print("[DB] ERROR: Motor not installed. MongoDB features disabled.")
            print("[DB] Install with: pip install motor>=3.3.0 dnspython>=2.3.0 pymongo>=4.3.3")
            return
        
        if self._client is None:
            mongo_url = os.getenv("MONGO_URL")
            if not mongo_url:
                print("[DB] WARNING: MONGO_URL not set. Using default: mongodb://localhost:27017/propelai")
                mongo_url = "mongodb://localhost:27017/propelai"
            
            try:
                self._client = AsyncIOMotorClient(mongo_url)
                # Get database name from URL or default to 'propelai'
                db_name = mongo_url.split('/')[-1].split('?')[0] if '/' in mongo_url else 'propelai'
                self._db = self._client[db_name]
                
                print(f"[DB] Connected to MongoDB: {db_name}")
            except Exception as e:
                print(f"[DB] ERROR: Failed to connect to MongoDB: {e}")
                print("[DB] Application will continue but database features may be limited")
    
    async def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            print("[DB] MongoDB connection closed")
    
    @property
    def rfps(self):
        """RFP collection - stores RFP metadata, requirements, state"""
        if self._db is None:
            # More helpful error message
            raise RuntimeError(
                "Database not connected. This usually means:\n"
                "1. FastAPI startup event hasn't completed yet\n"
                "2. MongoDB connection failed during startup\n"
                "Check logs for '[DB] Connected to MongoDB' message"
            )
        return self._db.rfps
    
    @property
    def chat_history(self):
        """Chat history collection - stores message logs by rfp_id"""
        if self._db is None:
            raise RuntimeError(
                "Database not connected. This usually means:\n"
                "1. FastAPI startup event hasn't completed yet\n"
                "2. MongoDB connection failed during startup\n"
                "Check logs for '[DB] Connected to MongoDB' message"
            )
        return self._db.chat_history
    
    @property
    def company_library(self):
        """Company library collection - stores document metadata"""
        if self._db is None:
            raise RuntimeError(
                "Database not connected. This usually means:\n"
                "1. FastAPI startup event hasn't completed yet\n"
                "2. MongoDB connection failed during startup\n"
                "Check logs for '[DB] Connected to MongoDB' message"
            )
        return self._db.company_library
    
    @staticmethod
    def serialize_doc(doc: dict) -> dict:
        """
        Convert MongoDB document to Pydantic-compatible format.
        Converts ObjectId '_id' to string 'id' and removes '_id'.
        
        Args:
            doc: MongoDB document dict
            
        Returns:
            Serialized dict with 'id' field
        """
        if doc is None:
            return None
        
        # Make a copy to avoid modifying original
        result = dict(doc)
        
        # Convert ObjectId to string if present
        if '_id' in result:
            if not result.get('id'):
                # Use _id as id if id doesn't exist
                result['id'] = str(result['_id'])
            del result['_id']
        
        return result
    
    @staticmethod
    def serialize_docs(docs: list) -> list:
        """
        Serialize a list of MongoDB documents.
        
        Args:
            docs: List of MongoDB documents
            
        Returns:
            List of serialized documents
        """
        return [Database.serialize_doc(doc) for doc in docs]


# Global instance
db = Database()
