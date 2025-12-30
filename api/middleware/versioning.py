"""
API Versioning Middleware

Adds API version headers to all responses and handles version deprecation warnings.
"""

from starlette.middleware.base import BaseHTTPMiddleware

# API Version constants
API_VERSION = "4.1.0"
API_VERSION_MAJOR = 4
API_VERSION_MINOR = 1
API_VERSION_PATCH = 0


class APIVersionMiddleware(BaseHTTPMiddleware):
    """
    Add API version headers to all responses.

    Headers added:
    - X-API-Version: Current API version
    - X-API-Deprecated: "true" if client is using deprecated version
    - X-API-Upgrade-Message: Upgrade instructions if deprecated
    """

    async def dispatch(self, request, call_next):
        response = await call_next(request)

        # Add API version headers
        response.headers["X-API-Version"] = API_VERSION
        response.headers["X-API-Deprecated"] = "false"

        # Check for version in request header
        requested_version = request.headers.get("X-API-Version")
        if requested_version and requested_version != API_VERSION:
            # Log version mismatch (client might need to update)
            try:
                major = int(requested_version.split(".")[0])
                if major < API_VERSION_MAJOR:
                    response.headers["X-API-Deprecated"] = "true"
                    response.headers["X-API-Upgrade-Message"] = (
                        f"Please upgrade to API v{API_VERSION}"
                    )
            except (ValueError, IndexError):
                pass  # Invalid version format, ignore

        return response
