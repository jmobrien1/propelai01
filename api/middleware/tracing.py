"""
Request ID Tracing Middleware

Adds unique request IDs to all requests for distributed tracing and log correlation.
"""

import uuid
import contextvars
from starlette.middleware.base import BaseHTTPMiddleware

# Context variable to store request ID for the current request
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def get_request_id() -> str:
    """Get the current request ID from context."""
    return request_id_var.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Add unique request ID to all requests for tracing.

    - Preserves X-Request-ID header from upstream load balancers
    - Generates new UUID if no header present
    - Sets request_id context variable for logging
    - Returns X-Request-ID in response headers
    """

    async def dispatch(self, request, call_next):
        # Get existing request ID from header or generate new one
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Set context variable for logging
        request_id_var.set(request_id)

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response
