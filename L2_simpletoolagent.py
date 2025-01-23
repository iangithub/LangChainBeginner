from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from datetime import datetime
from dotenv import dotenv_values

# 讀取 .env 檔案
config = dotenv_values(".env")

# 初始化 OpenAI 模型
llm = ChatOpenAI(
    api_key=config.get("OPENAI_KEY"),
    temperature=0.2, 
    max_tokens=500
)

# 定義狀態 state 類別
class AgentState(TypedDict):
    user_input: str # 使用者輸入
    city: str   # 城市
    current_datetime: str # 現在日期時間
    outfit_suggestion: str # 建議穿搭

# 建立節點 node 函數, 用來處理使用者輸入, 並傳回 AgentState state 物件
def process_input(state: AgentState) -> AgentState:
    state['city'] = state.get('user_input', '').strip()
    return state

# 建立節點 node 函數, 用來取得日期, 並傳回 AgentState state 物件
def get_current_datetime(state: AgentState) -> AgentState:
    now = datetime.now()
    state['current_datetime'] = now.strftime("%Y-%m-%d %H:%M:%S")
    print("現在時間是(get_current_datetime)：", state['current_datetime'], "\n\n")  # 調試用
    return state

# 建立節點 node 函數, 用來取得穿搭建議, 並傳回 AgentState state 物件
def generate_outfit_suggestion(state: AgentState) -> AgentState:
    city = state.get('city', '')
    current_datetime = state.get('current_datetime', '')
    prompt = f"現在時間是{current_datetime},請提供這個季節到{city}旅行的穿搭建議?"
    outfit_suggestion = llm.invoke(prompt)
    state['outfit_suggestion'] = outfit_suggestion
    return state

# 初始化 StateGraph
graph_builder = StateGraph(AgentState)

# 加入 process_input, get_current_datetime, generate_outfit_suggestion 節點(node)
graph_builder.add_node("process_input", process_input)
graph_builder.add_node("get_current_datetime", get_current_datetime)
graph_builder.add_node("generate_outfit_suggestion", generate_outfit_suggestion)

# 設置節點(node)的連接順序
graph_builder.set_entry_point("process_input")
graph_builder.add_edge("process_input", "get_current_datetime")
graph_builder.add_edge("get_current_datetime", "generate_outfit_suggestion")
graph_builder.set_finish_point("generate_outfit_suggestion")

# 建立 graph
graph = graph_builder.compile()

# 執行 graph
initial_state = AgentState(user_input="台北", city="", current_datetime="", outfit_suggestion="")
result = graph.invoke(initial_state)
print(result['outfit_suggestion'].content)