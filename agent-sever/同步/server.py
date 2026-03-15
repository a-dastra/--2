from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from agent.my_agents import  create_agent_for_mood, mood_check


app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(message: str):
    mood = mood_check(message)
    agent_executor = create_agent_for_mood(mood)
    response = agent_executor.invoke({"input": message})
    final_output = response.get("output", "呜...小刻与你的网络连接好像不太稳定")
    return {"response": final_output}



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")
        await websocket.close()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)