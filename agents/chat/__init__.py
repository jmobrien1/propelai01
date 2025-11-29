"""
PropelAI: Chat with RFP Module

Enables natural language Q&A with RFP documents using RAG (Retrieval Augmented Generation).
"""

from .rfp_chat_agent import RFPChatAgent, DocumentChunk, ChatMessage

__all__ = ["RFPChatAgent", "DocumentChunk", "ChatMessage"]
