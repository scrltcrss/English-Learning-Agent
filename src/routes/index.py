from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index():
    with open("src/static/index.html", "r", encoding="utf-8") as file:
        html_content: str = file.read()
    return HTMLResponse(content=html_content)
