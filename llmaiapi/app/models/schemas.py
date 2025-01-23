from pydantic import BaseModel

# 定義資料結構，例如請求與回應的格式。
class QuestionRequest(BaseModel):
    question: str