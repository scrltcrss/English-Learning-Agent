import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class MeasureTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time: float = time.perf_counter()
        response = await call_next(request)
        process_time: float = time.perf_counter() - start_time
        print(f"Endpoint {request.url.path} was executed in {process_time:.4f}s")
        return response
