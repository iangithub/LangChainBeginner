from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import dotenv_values

# 讀取 .env 檔案
config = dotenv_values(".env")

# 初始化 OpenAI 模型
llm = ChatOpenAI(
    api_key=config.get("OPENAI_KEY"),
    temperature=0.8, 
    max_tokens=100
)

# 設定對話紀錄
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///./langchain.db")

# 初始化對話鏈
conversation = RunnableWithMessageHistory(
    runnable=llm,
    get_session_history=get_session_history
)

config = {"configurable": {"session_id": "1"}}

# 定義 system prompt
system_prompt = SystemMessage(content="""
你是一個有用的AI助手。請用繁體中文回答。
你的回答需要注意以下幾點：
1. 保持友善的語氣
2. 給予清晰的回應
""")

# 開始對話
response = conversation.invoke(
        [system_prompt,HumanMessage(content="你好！")],
        config=config
    )
print(response.content)

response = conversation.invoke(
        [system_prompt,HumanMessage(content="我叫Ian，你叫什麼名字？")],
        config=config
    )
print(response.content)

response = conversation.invoke(
        [system_prompt,HumanMessage(content="我叫什麼名字")],
        config=config
    )
print(response.content)

