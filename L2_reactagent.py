
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage,AIMessage
from datetime import datetime
from typing import List, Dict
from dotenv import dotenv_values

config = dotenv_values(".env")

# 初始化 OpenAI 模型
llm = ChatOpenAI(
    api_key=config.get("OPENAI_KEY"),
    temperature=0.2, 
    max_tokens=500
)

# 模擬房間的可用資料
rooms_availability: List[Dict] =  [
    {"roomno":"001","roomtype":"雙人房","available_date":"2024/9/1"},
    {"roomno":"001","roomtype":"雙人房","available_date":"2024/9/2"},
    {"roomno":"002","roomtype":"單人房","available_date":"2024/9/1"},
    {"roomno":"002","roomtype":"單人房","available_date":"2024/9/3"},
    {"roomno":"003","roomtype":"雙人房","available_date":"2024/9/1"},
    {"roomno":"003","roomtype":"雙人房","available_date":"2024/9/2"},
    {"roomno":"003","roomtype":"雙人房","available_date":"2024/9/3"},
    {"roomno":"003","roomtype":"雙人房","available_date":"2024/8/26"},
    {"roomno":"003","roomtype":"雙人房","available_date":"2024/8/27"}
]

# 取得當前日期function
@tool
def get_current_date() -> str:
    """
    取得今天日期。

    返回:
    str: 今天日期，格式為 YYYY/MM/DD
    """
    return datetime.now().strftime("%Y/%m/%d")


# 查詢指定日期的可用房間function
@tool
def check_room_availability(date: str) -> str:
    """
    查詢指定日期的可用房間。

    參數:
    date (str): 查詢日期，格式為 YYYY/MM/DD

    返回:
    str: 可用房間的資訊，如果沒有可用房間則返回無可預訂空房的訊息
    """
    try:
        # 驗證日期格式
        query_date = datetime.strptime(date, "%Y/%m/%d")
    except ValueError:
        return "日期格式不正確，請使用 YYYY/MM/DD 格式。"

    available_rooms = [
            room for room in rooms_availability 
            if datetime.strptime(room["available_date"], "%Y/%m/%d").date() == query_date.date()
        ]

    if not available_rooms:
        return f"抱歉，{date} 沒有可預訂的房間。"

    result = f"{date} 可預訂的房間如下：\n"
    for room in available_rooms:
        result += f"房間號碼：{room['roomno']}，類型：{room['roomtype']}\n"

    return result

# 輸出LLM回應過程(輸出整個推理過程)
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1] 
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# 設定工具
tools = [get_current_date,check_room_availability]

# # 建立 Agent 並掛載工具
agent = create_react_agent(llm, tools=tools,)

# Agent 啟動(輸出整個推理過程)
inputs = {"messages": [("user", "可以預約2024/9/3的住宿嗎")]}
print_stream(agent.stream(inputs, stream_mode="values"))

# Agent 啟動
# response = agent.invoke(
#     {"messages": [HumanMessage(content="可以預約2024/9/3的住宿嗎")]}
# )

# 輸出推理過程原始json message
# print(response["messages"]) 

# 僅輸出最後一個AI Message回應
# last_content = next((msg.content for msg in reversed(response["messages"]) 
#                     if isinstance(msg, AIMessage) and msg.content), None)
# print(last_content)