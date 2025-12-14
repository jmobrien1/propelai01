"""
Gemini LLM Client - "The Librarian"

Implements Google's Gemini 1.5 Pro integration for PropelAI's massive
context ingestion capability. Key features:

- 1M token context window for full RFP ingestion
- Cross-document relationship detection
- Global attention for cross-reference discovery
- Compliance matrix generation from full context

Usage:
    client = GeminiClient()

    # Ingest full RFP package
    result = await client.ingest_rfp_package([
        "/path/to/solicitation.pdf",
        "/path/to/attachments.pdf",
        "/path/to/qa.pdf"
    ])

    # Generate compliance matrix
    matrix = await client.generate_compliance_matrix(result.content)
"""

import os
import time
import logging
import asyncio
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from .llm_clients import (
    BaseLLMClient,
    ModelRole,
    TaskType,
    LLMMessage,
    LLMResponse,
    GenerationConfig,
    TokenUsage,
    calculate_cost
)

logger = logging.getLogger(__name__)

# Check for google-generativeai package
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning(
        "google-generativeai not installed. Install with: "
        "pip install google-generativeai"
    )


@dataclass
class RFPIngestionResult:
    """Result of ingesting an RFP package"""
    success: bool
    content: str = ""
    document_count: int = 0
    total_pages: int = 0
    total_tokens: int = 0
    cross_references: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None

    # Extracted metadata
    solicitation_number: Optional[str] = None
    agency: Optional[str] = None
    due_date: Optional[str] = None
    contract_type: Optional[str] = None
    naics_code: Optional[str] = None
    set_aside: Optional[str] = None


@dataclass
class ComplianceMatrixEntry:
    """Single entry in the compliance matrix"""
    requirement_id: str
    requirement_text: str
    section_reference: str
    section_type: str  # C, L, M
    requirement_type: str
    binding_level: str  # MANDATORY, HIGHLY_DESIRABLE, etc.
    cross_references: List[str] = field(default_factory=list)
    evaluation_factor: Optional[str] = None


class GeminiClient(BaseLLMClient):
    """
    Gemini 1.5 Pro client for massive context operations.

    Role: "The Librarian" - handles full RFP ingestion and analysis
    with 1M token context window.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-pro",
        use_flash_for_counting: bool = True
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Google API key (or from GOOGLE_API_KEY env)
            model: Model to use (gemini-1.5-pro or gemini-1.5-flash)
            use_flash_for_counting: Use Flash for token counting (cheaper)
        """
        super().__init__(ModelRole.LIBRARIAN)

        self._model_name = model
        self._use_flash_for_counting = use_flash_for_counting

        # Get API key
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self._api_key:
            logger.warning("GOOGLE_API_KEY not set. Gemini features unavailable.")
            self._initialized = False
            return

        if not GEMINI_AVAILABLE:
            logger.error("google-generativeai package not installed")
            self._initialized = False
            return

        # Configure the API
        genai.configure(api_key=self._api_key)

        # Initialize the model
        self._model = genai.GenerativeModel(
            model_name=model,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        # Flash model for cheap operations
        self._flash_model = genai.GenerativeModel("gemini-1.5-flash")

        self._initialized = True
        logger.info(f"Gemini client initialized with model: {model}")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_context_tokens(self) -> int:
        # Gemini 1.5 Pro supports up to 2M tokens, but we use 1M for safety
        return 1_000_000

    @property
    def is_available(self) -> bool:
        return self._initialized

    async def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[GenerationConfig] = None,
        task_type: Optional[TaskType] = None
    ) -> LLMResponse:
        """
        Generate a completion using Gemini.

        Args:
            messages: Conversation messages
            config: Generation configuration
            task_type: Type of task for logging

        Returns:
            LLMResponse with generated content
        """
        if not self._initialized:
            return LLMResponse(
                content="Gemini client not initialized. Check API key.",
                model=self._model_name,
                role=self.role,
                finish_reason="error"
            )

        config = config or GenerationConfig()
        start_time = time.time()

        try:
            # Convert messages to Gemini format
            gemini_messages = self._convert_messages(messages)

            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                stop_sequences=config.stop_sequences or None
            )

            # Generate response
            response = await asyncio.to_thread(
                self._model.generate_content,
                gemini_messages,
                generation_config=generation_config
            )

            # Extract content
            content = ""
            if response.candidates:
                content = response.candidates[0].content.parts[0].text

            # Calculate token usage
            prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
            completion_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0

            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                estimated_cost_usd=calculate_cost(self._model_name, prompt_tokens, completion_tokens)
            )

            self._track_usage(usage)

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=content,
                model=self._model_name,
                role=self.role,
                task_type=task_type,
                token_usage=usage,
                finish_reason="stop",
                latency_ms=latency_ms,
                raw_response={"candidates": len(response.candidates) if response.candidates else 0}
            )

        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=self._model_name,
                role=self.role,
                task_type=task_type,
                finish_reason="error",
                latency_ms=(time.time() - start_time) * 1000
            )

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[GenerationConfig] = None,
        task_type: Optional[TaskType] = None
    ) -> AsyncIterator[str]:
        """
        Generate streaming completion using Gemini.

        Args:
            messages: Conversation messages
            config: Generation configuration
            task_type: Type of task for logging

        Yields:
            String chunks as they're generated
        """
        if not self._initialized:
            yield "Gemini client not initialized. Check API key."
            return

        config = config or GenerationConfig()

        try:
            gemini_messages = self._convert_messages(messages)

            generation_config = genai.types.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
            )

            response = self._model.generate_content(
                gemini_messages,
                generation_config=generation_config,
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            yield f"Error: {str(e)}"

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.

        Uses Flash model for cheaper counting.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        if not self._initialized:
            # Rough estimate: ~4 chars per token
            return len(text) // 4

        try:
            model = self._flash_model if self._use_flash_for_counting else self._model
            result = await asyncio.to_thread(model.count_tokens, text)
            return result.total_tokens
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Using estimate.")
            return len(text) // 4

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLMMessage format to Gemini format"""
        gemini_messages = []

        # Combine system message with first user message
        system_content = ""
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content + "\n\n"
            elif msg.role == "user":
                content = system_content + msg.content if system_content else msg.content
                gemini_messages.append({"role": "user", "parts": [content]})
                system_content = ""  # Reset after first user message
            elif msg.role == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg.content]})

        return gemini_messages

    # =========================================================================
    # RFP Ingestion Methods (The Librarian's Core Functions)
    # =========================================================================

    async def ingest_rfp_package(
        self,
        file_paths: List[str],
        extract_metadata: bool = True
    ) -> RFPIngestionResult:
        """
        Ingest a complete RFP package using massive context window.

        This is the core "Librarian" function that loads the entire RFP
        (solicitation + attachments + Q&A) into context for comprehensive
        analysis.

        Args:
            file_paths: Paths to RFP documents
            extract_metadata: Whether to extract solicitation metadata

        Returns:
            RFPIngestionResult with combined content and analysis
        """
        if not self._initialized:
            return RFPIngestionResult(
                success=False,
                error_message="Gemini client not initialized"
            )

        logger.info(f"Ingesting RFP package: {len(file_paths)} documents")

        # Combine all document content
        combined_content = []
        total_pages = 0

        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            # Read file content (assumes pre-extracted text)
            # In production, this would use the parser module
            try:
                content = path.read_text(encoding='utf-8', errors='ignore')
                combined_content.append(f"\n\n=== DOCUMENT: {path.name} ===\n\n{content}")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        full_content = "\n".join(combined_content)

        # Count tokens to ensure we're within limits
        token_count = await self.count_tokens(full_content)
        logger.info(f"RFP package: {token_count:,} tokens")

        if token_count > self.max_context_tokens:
            logger.warning(
                f"Content exceeds {self.max_context_tokens:,} tokens. "
                f"Consider chunking or summarization."
            )

        result = RFPIngestionResult(
            success=True,
            content=full_content,
            document_count=len(file_paths),
            total_tokens=token_count
        )

        # Extract metadata if requested
        if extract_metadata and token_count < self.max_context_tokens:
            metadata = await self._extract_rfp_metadata(full_content)
            result.solicitation_number = metadata.get("solicitation_number")
            result.agency = metadata.get("agency")
            result.due_date = metadata.get("due_date")
            result.contract_type = metadata.get("contract_type")
            result.naics_code = metadata.get("naics_code")
            result.set_aside = metadata.get("set_aside")

            # Detect cross-references
            result.cross_references = await self._detect_cross_references(full_content)

        return result

    async def _extract_rfp_metadata(self, content: str) -> Dict[str, Any]:
        """Extract solicitation metadata from RFP content"""
        prompt = """Analyze this RFP document and extract the following metadata.
Return ONLY a JSON object with these fields (use null if not found):

{
    "solicitation_number": "The solicitation/RFP number",
    "agency": "The contracting agency (e.g., NIH, Navy, GSA)",
    "due_date": "Proposal due date in ISO format (YYYY-MM-DD)",
    "contract_type": "Contract type (FFP, T&M, CPFF, IDIQ, etc.)",
    "naics_code": "NAICS code",
    "set_aside": "Set-aside type (8(a), HUBZone, WOSB, SDVOSB, Full and Open, etc.)",
    "place_of_performance": "Primary place of performance",
    "incumbent": "Incumbent contractor if mentioned"
}

RFP Content:
"""
        # Use first 50K tokens for metadata extraction
        truncated = content[:200000]  # ~50K tokens

        response = await self.generate(
            messages=[
                LLMMessage(role="user", content=prompt + truncated)
            ],
            config=GenerationConfig(temperature=0.1, max_output_tokens=1000),
            task_type=TaskType.RFP_INGESTION
        )

        # Parse JSON response
        try:
            import json
            # Find JSON in response
            text = response.content
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            logger.warning(f"Failed to parse metadata JSON: {e}")

        return {}

    async def _detect_cross_references(self, content: str) -> List[Dict[str, Any]]:
        """Detect cross-references between RFP sections using global attention"""
        prompt = """Analyze this RFP and identify ALL cross-references between sections.
Look for:
- Section C requirements that reference Section L instructions
- Section M evaluation criteria that reference Section C requirements
- Attachments that modify main solicitation requirements
- Q&A responses that clarify or modify requirements

Return a JSON array of cross-references:
[
    {
        "source_section": "C.3.1",
        "target_section": "Attachment J-4",
        "relationship": "refers_to",
        "description": "Staffing requirements reference Attachment J-4 labor categories"
    }
]

RFP Content:
"""
        # Use first 100K tokens for cross-reference detection
        truncated = content[:400000]  # ~100K tokens

        response = await self.generate(
            messages=[
                LLMMessage(role="user", content=prompt + truncated)
            ],
            config=GenerationConfig(temperature=0.1, max_output_tokens=4000),
            task_type=TaskType.CROSS_REFERENCE_DETECTION
        )

        try:
            import json
            text = response.content
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            logger.warning(f"Failed to parse cross-references: {e}")

        return []

    async def generate_compliance_matrix(
        self,
        content: str,
        include_l_requirements: bool = True,
        include_m_criteria: bool = True
    ) -> List[ComplianceMatrixEntry]:
        """
        Generate a comprehensive compliance matrix from the full RFP.

        This leverages Gemini's massive context to see the ENTIRE RFP
        at once, enabling detection of cross-document relationships
        that chunked approaches would miss.

        Args:
            content: Full RFP content
            include_l_requirements: Include Section L instructions
            include_m_criteria: Include Section M evaluation criteria

        Returns:
            List of ComplianceMatrixEntry objects
        """
        prompt = f"""You are an expert proposal compliance analyst. Analyze this complete RFP
and extract EVERY requirement into a compliance matrix.

For each requirement, identify:
1. requirement_id: Unique ID (e.g., C.3.1.2-001)
2. requirement_text: The exact requirement text
3. section_reference: Section number (e.g., C.3.1.2, L.4, M.2)
4. section_type: C (technical), L (instructions), or M (evaluation)
5. requirement_type: PERFORMANCE, DELIVERABLE, QUALIFICATION, FORMAT, etc.
6. binding_level: MANDATORY (shall/must), HIGHLY_DESIRABLE (should), OPTIONAL (may)
7. cross_references: Other sections this references
8. evaluation_factor: Section M factor this maps to (if applicable)

CRITICAL: Do NOT summarize. Extract EVERY "shall", "must", "will", "should" statement.
A missing requirement causes proposal disqualification.

Return as JSON array:
[
    {{
        "requirement_id": "C.3.1-001",
        "requirement_text": "The contractor shall provide...",
        "section_reference": "C.3.1",
        "section_type": "C",
        "requirement_type": "PERFORMANCE",
        "binding_level": "MANDATORY",
        "cross_references": ["L.4.2", "Attachment J-1"],
        "evaluation_factor": "Technical Approach"
    }}
]

{"Include Section L instructions." if include_l_requirements else "Skip Section L."}
{"Include Section M evaluation criteria." if include_m_criteria else "Skip Section M."}

RFP Content:
{content}
"""

        response = await self.generate(
            messages=[
                LLMMessage(role="user", content=prompt)
            ],
            config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=32000  # Large output for comprehensive matrix
            ),
            task_type=TaskType.COMPLIANCE_MATRIX_GENERATION
        )

        # Parse response into structured entries
        entries = []
        try:
            import json
            text = response.content
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                raw_entries = json.loads(text[start:end])
                for entry in raw_entries:
                    entries.append(ComplianceMatrixEntry(
                        requirement_id=entry.get("requirement_id", ""),
                        requirement_text=entry.get("requirement_text", ""),
                        section_reference=entry.get("section_reference", ""),
                        section_type=entry.get("section_type", ""),
                        requirement_type=entry.get("requirement_type", ""),
                        binding_level=entry.get("binding_level", ""),
                        cross_references=entry.get("cross_references", []),
                        evaluation_factor=entry.get("evaluation_factor")
                    ))
        except Exception as e:
            logger.error(f"Failed to parse compliance matrix: {e}")

        logger.info(f"Generated compliance matrix: {len(entries)} entries")
        return entries


# Factory function for easy instantiation
def create_gemini_client(
    model: str = "gemini-1.5-pro",
    api_key: Optional[str] = None
) -> GeminiClient:
    """
    Create a Gemini client instance.

    Args:
        model: Model to use (gemini-1.5-pro or gemini-1.5-flash)
        api_key: API key (or from environment)

    Returns:
        Configured GeminiClient
    """
    return GeminiClient(api_key=api_key, model=model)
