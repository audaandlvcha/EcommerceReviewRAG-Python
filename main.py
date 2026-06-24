"""入口 —— python main.py 启动 Web 服务"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("web:app", host="0.0.0.0", port=8000, reload=True)
