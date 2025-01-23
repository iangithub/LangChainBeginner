from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import dotenv_values

# 讀取 .env 檔案
config = dotenv_values(".env")

# 初始化 OpenAI 模型
llm = ChatOpenAI(
    api_key=config.get("OPENAI_KEY"),
    temperature=0.8, 
    max_tokens=500
)

# 初始化 StrOutputParser
output_parser = StrOutputParser()

# 定義狀態 state 類別
class AgentState(TypedDict):
    user_input: str
    response: str

# 建立節點 node 函數, 用來處理使用者輸入, 並傳回 AgentState state 物件
def generate_response(state: AgentState) -> AgentState:
    user_input = state.get('user_input', '') # 取得使用者輸入
    llm_output = llm.invoke(user_input) # 使用 llm 語言模型產生回應
    state['response'] = llm_output # 將回應存入 state 物件
    return state # 傳回 state 物件

# 初始化 StateGraph
graph_builder = StateGraph(AgentState)

# 加入 process_input 節點(node)
graph_builder.add_node("generate_response", generate_response)


# 設定 graph 的 entry_point 和 finish_point
graph_builder.set_entry_point("generate_response")
graph_builder.set_finish_point("generate_response")

# 建立 graph
graph = graph_builder.compile()

# 執行 graph  
initial_state = AgentState(user_input="什麼是生成式AI,它與過去的AI類別有什麼差別", response="")  
result = graph.invoke(initial_state)

# print(result['response']) # json format
print(result['response'].content)

