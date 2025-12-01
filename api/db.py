"""
Local File-Based JSON Database
Stable, zero-dependency persistence for PropelAI
"""

import os
import json
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime


class JSONFileDB:
    """
    Simple file-based JSON storage with immediate persistence.
    
    Storage:
    - rfps.json: RFP metadata and requirements
    - chat_history.json: Chat messages by RFP ID
    - library.json: Company library metadata
    
    All data is loaded into memory on init and written to disk on every change.
    Thread-safe with simple locking.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize JSON file database with smart path resolution"""
        
        if data_dir is None:
            # Get project root (parent of 'api' directory)
            BASE_DIR = Path(__file__).parent.parent
            primary_path = BASE_DIR / "outputs" / "data"
            
            # Try to create primary path
            try:
                primary_path.mkdir(parents=True, exist_ok=True)
                # Test write permission
                test_file = primary_path / ".write_test"
                test_file.touch()
                test_file.unlink()
                self.data_dir = primary_path
                print(f"[DB] Using primary storage: {self.data_dir}")
            except (PermissionError, OSError) as e:
                # Fallback to /tmp which is always writable
                fallback_path = Path("/tmp/propelai_data")
                fallback_path.mkdir(parents=True, exist_ok=True)
                self.data_dir = fallback_path
                print(f"[DB] WARNING: Could not write to {primary_path}: {e}")
                print(f"[DB] Using fallback storage: {self.data_dir}")
                print(f"[DB] Data will NOT persist across container restarts!")
        else:
            # Use provided path (for testing)
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.rfps_file = self.data_dir / "rfps.json"
        self.chat_history_file = self.data_dir / "chat_history.json"
        self.library_file = self.data_dir / "library.json"
        
        # Thread locks for write safety
        self._rfps_lock = threading.Lock()
        self._chat_lock = threading.Lock()
        self._library_lock = threading.Lock()
        
        # Load data into memory
        self._rfps_data = self._load_json(self.rfps_file, {})
        self._chat_history_data = self._load_json(self.chat_history_file, {})
        self._library_data = self._load_json(self.library_file, {})
        
        print(f"[DB] JSONFileDB initialized at {self.data_dir}")
        print(f"[DB] Loaded {len(self._rfps_data)} RFPs, {len(self._chat_history_data)} chat histories")
    
    def _load_json(self, file_path: Path, default: Any) -> Any:
        """Load JSON file or return default if not exists"""
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[DB] Warning: Could not load {file_path.name}: {e}")
                return default
        return default
    
    def _save_json(self, file_path: Path, data: Any):
        """Save data to JSON file with atomic write"""
        temp_file = file_path.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            temp_file.replace(file_path)  # Atomic rename
        except Exception as e:
            print(f"[DB] Error saving {file_path.name}: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    # ============== RFPs Collection ==============
    
    def rfps_insert_one(self, document: Dict) -> Dict:
        """Insert RFP document"""
        with self._rfps_lock:
            rfp_id = document.get('id')
            if not rfp_id:
                raise ValueError("RFP document must have 'id' field")
            
            self._rfps_data[rfp_id] = document
            self._save_json(self.rfps_file, self._rfps_data)
            return document
    
    def rfps_find_one(self, query: Dict) -> Optional[Dict]:
        """Find one RFP by query"""
        rfp_id = query.get('id')
        if rfp_id:
            return self._rfps_data.get(rfp_id)
        return None
    
    def rfps_find(self, query: Dict = None) -> List[Dict]:
        """Find all RFPs matching query"""
        return list(self._rfps_data.values())
    
    def rfps_update_one(self, query: Dict, update: Dict) -> Dict:
        """Update one RFP"""
        with self._rfps_lock:
            rfp_id = query.get('id')
            if not rfp_id or rfp_id not in self._rfps_data:
                return {'matched_count': 0, 'modified_count': 0}
            
            # Apply $set updates
            if '$set' in update:
                self._rfps_data[rfp_id].update(update['$set'])
            else:
                self._rfps_data[rfp_id].update(update)
            
            self._save_json(self.rfps_file, self._rfps_data)
            return {'matched_count': 1, 'modified_count': 1}
    
    def rfps_delete_one(self, query: Dict) -> Dict:
        """Delete one RFP"""
        with self._rfps_lock:
            rfp_id = query.get('id')
            if rfp_id and rfp_id in self._rfps_data:
                del self._rfps_data[rfp_id]
                self._save_json(self.rfps_file, self._rfps_data)
                return {'deleted_count': 1}
            return {'deleted_count': 0}
    
    # ============== Chat History Collection ==============
    
    def chat_history_insert_one(self, document: Dict) -> Dict:
        """Insert chat message"""
        with self._chat_lock:
            rfp_id = document.get('rfp_id')
            if not rfp_id:
                raise ValueError("Chat document must have 'rfp_id' field")
            
            if rfp_id not in self._chat_history_data:
                self._chat_history_data[rfp_id] = []
            
            self._chat_history_data[rfp_id].append(document)
            self._save_json(self.chat_history_file, self._chat_history_data)
            return document
    
    def chat_history_find(self, query: Dict) -> List[Dict]:
        """Find chat messages for an RFP"""
        rfp_id = query.get('rfp_id')
        if rfp_id:
            return self._chat_history_data.get(rfp_id, [])
        return []
    
    # ============== Company Library Collection ==============
    
    def library_insert_one(self, document: Dict) -> Dict:
        """Insert library document"""
        with self._library_lock:
            doc_id = document.get('id')
            if not doc_id:
                raise ValueError("Library document must have 'id' field")
            
            self._library_data[doc_id] = document
            self._save_json(self.library_file, self._library_data)
            return document
    
    def library_find(self, query: Dict = None) -> List[Dict]:
        """Find all library documents"""
        return list(self._library_data.values())
    
    def library_delete_one(self, query: Dict) -> Dict:
        """Delete library document"""
        with self._library_lock:
            doc_id = query.get('id')
            if doc_id and doc_id in self._library_data:
                del self._library_data[doc_id]
                self._save_json(self.library_file, self._library_data)
                return {'deleted_count': 1}
            return {'deleted_count': 0}
    
    # ============== Utility Methods ==============
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'rfps_count': len(self._rfps_data),
            'chat_histories_count': len(self._chat_history_data),
            'library_docs_count': len(self._library_data),
            'storage_path': str(self.data_dir)
        }


# Global instance - available immediately at import time
db = JSONFileDB()
