import asyncio
import uuid
from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel
from typing import Optional, List, Dict
 
try:
    from my_asyc_agent import keBrain
except ImportError as e:
    raise RuntimeError(
        "无法导入 my_asyc_agent 模块。请确保 my_asyc_agent.py 在当前目录，"
        "且已修复语法错误。"
    ) from e
 
app = FastAPI()
_user_brains = {}
 
SESSION_COOKIE_NAME = "ke_session_id"
 
class ChatRequest(BaseModel):
    message: str
 
class ChatResponse(BaseModel):
    reply: str
    markers: Optional[List[Dict]] = None 
 
def get_or_create_session_id(request: Request) -> str:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id
 
@app.post("/chat", response_model=ChatResponse)
async def chat_with_ke(request: Request, chat_req: ChatRequest, response: Response):
    message = chat_req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")
 
    session_id = get_or_create_session_id(request)
 
    if SESSION_COOKIE_NAME not in request.cookies:
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            httponly=True,
            max_age=60 * 60 * 24 * 30,
            samesite="lax"
        )
 
    if session_id not in _user_brains:
        try:
            #keBrain 初始化现在接收 user_id
            _user_brains[session_id] = keBrain(user_id=session_id)
        except Exception as init_err:
            raise HTTPException(
                status_code=500,
                detail=f"初始化失败: {str(init_err)}"
            )
 
    brain = _user_brains[session_id]
 
    try:
        # think 方法现在返回字典，包含 reply 和 markers
        result = await asyncio.wait_for(brain.think(message), timeout=60.0)
        return ChatResponse(
            reply=result.get("reply", ""), 
            markers=result.get("markers")
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="")
    except Exception as e:
        print(f"[ERROR] Session {session_id} | Message: {message} | Error: {e}")
        raise HTTPException(
            status_code=500,
            detail="呜呜...小刻遇到问题了，请再试一次！"
        )
 
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "小刻在线，尾巴摇摇！"}
 
@app.get("/clear-session")
async def clear_session(response: Response):
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"message": "会话已清除，下次聊天将是全新的小刻！"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)