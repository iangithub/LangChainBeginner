from fastapi import FastAPI
from app.routes import router

app = FastAPI()

# 導入路由
app.include_router(router)

# 測試端點
@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
