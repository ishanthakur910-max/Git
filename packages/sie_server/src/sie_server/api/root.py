"""Root endpoint router."""

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect the root path to the interactive OpenAPI docs."""
    return RedirectResponse(url="/docs")
