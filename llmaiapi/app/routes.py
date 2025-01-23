from fastapi import APIRouter, HTTPException, Request
from app.models.schemas import QuestionRequest
from app.services.llm_service import get_answer_from_llm
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

router = APIRouter()

# LINE Bot 設定
LINE_CHANNEL_ACCESS_TOKEN = 'your_channel_access_token'
LINE_CHANNEL_SECRET = 'your_channel_secret'

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 定義路由
@router.post("/copilot/qa")
async def ask_question(request: QuestionRequest):
    try:
        # 調用LLM服務處理問題
        answer = get_answer_from_llm(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# LINE Bot Webhook
@router.post("/webhook/line")
async def line_webhook(request: Request):
    # 獲取 X-Line-Signature header 值
    signature = request.headers.get('X-Line-Signature', '')
    
    # 獲取請求內容
    body = await request.body()
    body_decode = body.decode('utf-8')
    
    try:
        # 驗證簽章
        handler.handle(body_decode, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    return 'OK'

# LINE Bot 訊息處理器
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    try:
        # 取得使用者訊息
        user_message = event.message.text
        
         # 調用LLM服務處理回答
        answer = get_answer_from_llm(user_message)
        
        # 回覆訊息給使用者
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=answer)
        )
    except Exception as e:
        # 發生錯誤時發送錯誤訊息
        error_message = "抱歉，處理您的訊息時發生錯誤。請稍後再試。"
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=error_message)
        )