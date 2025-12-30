"""
PropelAI Library Ingestion Pipeline
Processes documents into searchable library with embeddings
"""

import asyncio
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from pathlib import Path
from enum import Enum

from rag.embeddings import EmbeddingService, get_embedding_service
from rag.chunker import TextChunker, Chunk, ChunkStrategy


logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Library entity types."""
    RESUME = "resume"
    PAST_PERFORMANCE = "past_performance"
    CAPABILITY = "capability"
    TEMPLATE = "template"
    BOILERPLATE = "boilerplate"
    POLICY = "policy"
    REFERENCE = "reference"


class IngestionStatus(str, Enum):
    """Document ingestion status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExtractedEntity:
    """Entity extracted from a document."""
    entity_type: EntityType
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    # Resume-specific
    skills: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    years_experience: Optional[int] = None
    education: List[str] = field(default_factory=list)

    # Past Performance-specific
    contract_value: Optional[float] = None
    contract_type: Optional[str] = None
    agency: Optional[str] = None
    period_of_performance: Optional[str] = None
    naics_codes: List[str] = field(default_factory=list)


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    document_id: str
    filename: str
    entity_type: EntityType
    entity_name: str
    chunk_count: int
    status: IngestionStatus
    processing_time_ms: int
    errors: List[str] = field(default_factory=list)
    extracted_entity: Optional[ExtractedEntity] = None


class LibraryIngestionPipeline:
    """
    Pipeline for ingesting documents into the Company Library.
    Handles parsing, entity extraction, chunking, and embedding.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        chunker: Optional[TextChunker] = None,
        db_pool: Optional[Any] = None,
        batch_size: int = 50,
    ):
        self.embedding_service = embedding_service or get_embedding_service()
        self.chunker = chunker or TextChunker(
            chunk_size=512,
            chunk_overlap=50,
            strategy=ChunkStrategy.SEMANTIC,
        )
        self.db_pool = db_pool
        self.batch_size = batch_size
        self._connection = None

    async def _get_connection(self):
        """Get database connection."""
        if self._connection is None and self.db_pool:
            self._connection = await self.db_pool.acquire()
        return self._connection

    async def ingest_document(
        self,
        file_path: str,
        tenant_id: str,
        entity_type: EntityType,
        entity_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionResult:
        """
        Ingest a single document into the library.

        Args:
            file_path: Path to the document
            tenant_id: Tenant ID
            entity_type: Type of entity (resume, past_performance, etc.)
            entity_name: Optional name override
            metadata: Additional metadata

        Returns:
            IngestionResult with status and details
        """
        start_time = datetime.utcnow()
        document_id = str(uuid.uuid4())
        filename = Path(file_path).name
        errors = []

        try:
            # Parse document
            from parsing.document_parser import DocumentParser
            parser = DocumentParser()
            parsed_doc = parser.parse(file_path)

            # Extract entity information
            extracted_entity = await self._extract_entity(
                parsed_doc.full_text,
                entity_type,
            )

            # Use extracted name if not provided
            if not entity_name:
                entity_name = extracted_entity.name or filename

            # Chunk the document
            chunks = self.chunker.chunk_with_pages(
                pages=[
                    {"page": page.page_number, "text": page.full_text}
                    for page in parsed_doc.pages
                ],
                metadata={
                    "document_id": document_id,
                    "entity_type": entity_type.value,
                    "entity_name": entity_name,
                    **(metadata or {}),
                },
            )

            if not chunks:
                return IngestionResult(
                    document_id=document_id,
                    filename=filename,
                    entity_type=entity_type,
                    entity_name=entity_name,
                    chunk_count=0,
                    status=IngestionStatus.FAILED,
                    processing_time_ms=self._elapsed_ms(start_time),
                    errors=["No content extracted from document"],
                )

            # Generate embeddings in batches
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self.embedding_service.embed_many(chunk_texts)

            # Store in database
            await self._store_document(
                document_id=document_id,
                tenant_id=tenant_id,
                filename=filename,
                entity_type=entity_type,
                entity_name=entity_name,
                chunks=chunks,
                embeddings=[e.embedding for e in embeddings],
                extracted_entity=extracted_entity,
                metadata=metadata,
            )

            return IngestionResult(
                document_id=document_id,
                filename=filename,
                entity_type=entity_type,
                entity_name=entity_name,
                chunk_count=len(chunks),
                status=IngestionStatus.COMPLETED,
                processing_time_ms=self._elapsed_ms(start_time),
                extracted_entity=extracted_entity,
            )

        except Exception as e:
            logger.exception(f"Failed to ingest document: {file_path}")
            return IngestionResult(
                document_id=document_id,
                filename=filename,
                entity_type=entity_type,
                entity_name=entity_name or filename,
                chunk_count=0,
                status=IngestionStatus.FAILED,
                processing_time_ms=self._elapsed_ms(start_time),
                errors=[str(e)],
            )

    async def ingest_batch(
        self,
        documents: List[Dict[str, Any]],
        tenant_id: str,
    ) -> AsyncGenerator[IngestionResult, None]:
        """
        Ingest multiple documents as a batch.

        Args:
            documents: List of {"path": str, "entity_type": str, "entity_name": str}
            tenant_id: Tenant ID

        Yields:
            IngestionResult for each document
        """
        for doc in documents:
            result = await self.ingest_document(
                file_path=doc["path"],
                tenant_id=tenant_id,
                entity_type=EntityType(doc.get("entity_type", "reference")),
                entity_name=doc.get("entity_name"),
                metadata=doc.get("metadata"),
            )
            yield result

    async def _extract_entity(
        self,
        text: str,
        entity_type: EntityType,
    ) -> ExtractedEntity:
        """Extract entity information from document text."""
        if entity_type == EntityType.RESUME:
            return await self._extract_resume(text)
        elif entity_type == EntityType.PAST_PERFORMANCE:
            return await self._extract_past_performance(text)
        else:
            return ExtractedEntity(
                entity_type=entity_type,
                name=self._extract_title(text),
            )

    async def _extract_resume(self, text: str) -> ExtractedEntity:
        """Extract resume information."""
        import re

        # Extract name (usually at the top)
        name = ""
        lines = text.strip().split("\n")
        for line in lines[:5]:
            line = line.strip()
            # Skip empty lines and common headers
            if not line or line.upper() in ["RESUME", "CURRICULUM VITAE", "CV"]:
                continue
            # Check if it looks like a name (2-4 words, title case or upper)
            words = line.split()
            if 2 <= len(words) <= 4 and all(
                w[0].isupper() for w in words if w.isalpha()
            ):
                name = line
                break

        # Extract skills
        skills = []
        skill_section = re.search(
            r"(?:SKILLS?|TECHNICAL\s+SKILLS?|COMPETENCIES)[:\s]*(.+?)(?=\n[A-Z]{2,}|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if skill_section:
            skill_text = skill_section.group(1)
            # Split by common delimiters
            raw_skills = re.split(r"[,;•|\n]", skill_text)
            skills = [s.strip() for s in raw_skills if s.strip() and len(s.strip()) < 50][:20]

        # Extract certifications
        certifications = []
        cert_patterns = [
            r"(?:PMP|CISSP|AWS|Azure|GCP|CISA|CISM|Six Sigma|ITIL|Scrum Master|CSM|PSM)",
            r"(?:Certified|Certificate|Certification)\s+[\w\s]+",
        ]
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend(matches)
        certifications = list(set(c.strip() for c in certifications))[:10]

        # Extract years of experience
        years_exp = None
        exp_match = re.search(
            r"(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)",
            text,
            re.IGNORECASE,
        )
        if exp_match:
            years_exp = int(exp_match.group(1))

        # Extract education
        education = []
        edu_patterns = [
            r"(?:Bachelor|Master|PhD|Doctorate|B\.S\.|M\.S\.|M\.B\.A\.|B\.A\.|M\.A\.)\s*(?:of|in)?\s*[\w\s,]+",
        ]
        for pattern in edu_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education.extend(m.strip() for m in matches if len(m.strip()) < 100)
        education = list(set(education))[:5]

        return ExtractedEntity(
            entity_type=EntityType.RESUME,
            name=name or "Unknown",
            skills=skills,
            certifications=certifications,
            years_experience=years_exp,
            education=education,
            confidence=0.8 if name else 0.5,
        )

    async def _extract_past_performance(self, text: str) -> ExtractedEntity:
        """Extract past performance information."""
        import re

        # Extract project/contract name
        name = ""
        title_patterns = [
            r"(?:Project|Contract|Task Order)\s*(?:Name|Title)?[:\s]+(.+?)(?:\n|$)",
            r"^([A-Z][\w\s-]{10,100})(?:\n|$)",
        ]
        for pattern in title_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                name = match.group(1).strip()
                break

        # Extract contract value
        contract_value = None
        value_match = re.search(
            r"\$\s*([\d,]+(?:\.\d{2})?)\s*(?:M|million|K|thousand)?",
            text,
            re.IGNORECASE,
        )
        if value_match:
            value_str = value_match.group(1).replace(",", "")
            try:
                contract_value = float(value_str)
                # Check for multipliers
                if "M" in value_match.group(0) or "million" in value_match.group(0).lower():
                    contract_value *= 1_000_000
                elif "K" in value_match.group(0) or "thousand" in value_match.group(0).lower():
                    contract_value *= 1_000
            except ValueError:
                pass

        # Extract agency
        agency = None
        agency_patterns = [
            r"(?:Agency|Client|Customer)[:\s]+(.+?)(?:\n|$)",
            r"((?:Department of|DoD|DOE|DOJ|DHS|HHS|VA|NASA|GSA|EPA|FAA|FEMA|FBI|CIA|NSA|NRO|USDA|USPTO|NOAA|NIH|CDC)[\w\s]*)",
        ]
        for pattern in agency_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                agency = match.group(1).strip()
                break

        # Extract NAICS codes
        naics_codes = re.findall(r"\b(\d{6})\b", text)
        naics_codes = list(set(naics_codes))[:5]

        # Extract contract type
        contract_type = None
        type_patterns = [
            r"(IDIQ|BPA|T&M|FFP|CPFF|CPIF|CPAF)",
            r"(Time and Materials|Firm Fixed Price|Cost Plus)",
        ]
        for pattern in type_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                contract_type = match.group(1).upper()
                break

        # Extract period of performance
        pop = None
        pop_match = re.search(
            r"(?:Period of Performance|POP|Duration)[:\s]*(.+?)(?:\n|$)",
            text,
            re.IGNORECASE,
        )
        if pop_match:
            pop = pop_match.group(1).strip()

        return ExtractedEntity(
            entity_type=EntityType.PAST_PERFORMANCE,
            name=name or "Unknown Contract",
            contract_value=contract_value,
            contract_type=contract_type,
            agency=agency,
            period_of_performance=pop,
            naics_codes=naics_codes,
            confidence=0.8 if name and agency else 0.5,
        )

    def _extract_title(self, text: str) -> str:
        """Extract document title from first few lines."""
        lines = text.strip().split("\n")
        for line in lines[:3]:
            line = line.strip()
            if 5 < len(line) < 100:
                return line
        return "Untitled Document"

    async def _store_document(
        self,
        document_id: str,
        tenant_id: str,
        filename: str,
        entity_type: EntityType,
        entity_name: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        extracted_entity: ExtractedEntity,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store document and embeddings in database."""
        conn = await self._get_connection()
        if conn is None:
            logger.warning("No database connection, skipping storage")
            return

        # Insert library document
        await conn.execute(
            """
            INSERT INTO library_documents (
                id, tenant_id, filename, entity_type, entity_name,
                extracted_data, metadata, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            """,
            uuid.UUID(document_id),
            uuid.UUID(tenant_id),
            filename,
            entity_type.value,
            entity_name,
            {
                "skills": extracted_entity.skills,
                "certifications": extracted_entity.certifications,
                "years_experience": extracted_entity.years_experience,
                "education": extracted_entity.education,
                "contract_value": extracted_entity.contract_value,
                "agency": extracted_entity.agency,
                "naics_codes": extracted_entity.naics_codes,
            },
            metadata or {},
        )

        # Insert embeddings in batches
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_embeddings = embeddings[i:i + self.batch_size]

            values = [
                (
                    uuid.uuid4(),
                    uuid.UUID(document_id),
                    chunk.text,
                    chunk.index,
                    chunk.source_page,
                    chunk.source_section,
                    chunk.to_dict().get("bbox"),
                    embedding,
                    chunk.metadata,
                )
                for chunk, embedding in zip(batch_chunks, batch_embeddings)
            ]

            await conn.executemany(
                """
                INSERT INTO library_embeddings (
                    id, document_id, chunk_text, chunk_index,
                    page_number, section, bbox, embedding, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector, $9)
                """,
                values,
            )

    def _elapsed_ms(self, start_time: datetime) -> int:
        """Calculate elapsed milliseconds."""
        return int((datetime.utcnow() - start_time).total_seconds() * 1000)

    async def update_embeddings(
        self,
        document_id: str,
        tenant_id: str,
    ) -> bool:
        """Re-generate embeddings for a document (e.g., after model upgrade)."""
        conn = await self._get_connection()
        if conn is None:
            return False

        # Get existing chunks
        rows = await conn.fetch(
            """
            SELECT id, chunk_text FROM library_embeddings
            WHERE document_id = $1
            ORDER BY chunk_index
            """,
            uuid.UUID(document_id),
        )

        if not rows:
            return False

        # Generate new embeddings
        texts = [row["chunk_text"] for row in rows]
        embeddings = await self.embedding_service.embed_many(texts)

        # Update embeddings
        for row, emb_result in zip(rows, embeddings):
            await conn.execute(
                """
                UPDATE library_embeddings
                SET embedding = $1::vector, updated_at = NOW()
                WHERE id = $2
                """,
                emb_result.embedding,
                row["id"],
            )

        return True

    async def delete_document(
        self,
        document_id: str,
        tenant_id: str,
    ) -> bool:
        """Delete a document from the library."""
        conn = await self._get_connection()
        if conn is None:
            return False

        # Delete embeddings first (cascade)
        await conn.execute(
            "DELETE FROM library_embeddings WHERE document_id = $1",
            uuid.UUID(document_id),
        )

        # Delete document
        result = await conn.execute(
            "DELETE FROM library_documents WHERE id = $1 AND tenant_id = $2",
            uuid.UUID(document_id),
            uuid.UUID(tenant_id),
        )

        return "DELETE 1" in result


# Convenience function
def get_ingestion_pipeline(
    embedding_service: Optional[EmbeddingService] = None,
    db_pool: Optional[Any] = None,
) -> LibraryIngestionPipeline:
    """Get a configured ingestion pipeline."""
    return LibraryIngestionPipeline(
        embedding_service=embedding_service,
        db_pool=db_pool,
    )
