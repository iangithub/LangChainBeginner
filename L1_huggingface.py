from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import dotenv_values

# 讀取 .env 檔案
config = dotenv_values(".env")

# 初始化語言模型
model = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.3, 
    do_sample=False,
    repetition_penalty=1.2,
    huggingfacehub_api_token = config.get("HF_API_TOKEN")
)

# 設定語言模型的 prompt
prompt = PromptTemplate.from_template("""
                                      請使用繁體中文,總結以下內容： 
                                      
                                      ## 原文內容
                                      {text}

                                      """)

# 建立可執行的 LLMChain
chain = prompt | model | StrOutputParser()

# 執行 LLMChain
result = chain.invoke({"text": 
"""
黃仁勳在演說中揭曉 RTX 50系列顯示卡，距離RTX40系列上市已間隔兩年多，50系列內部架構採用輝達最新的Blackwell架構，提供更高算力，並使用DLSS 4 畫格生成技術，透過AI來提升遊戲幀數，讓遊戲畫面更加流暢、解析度更高。
RTX 50系列顯卡包括四種型號，入門級的RTX 5070售價549美元（新台幣1.79萬元），價格僅為前一代RTX 4090頂級顯示卡的1/3，還有售價749美元的RTX 5070 Ti、999美元的RTX 5080，以及1,999美元的RTX 5090，預計1月起陸續上市。
輝達也指出，採用這些晶片的筆記型電腦電池續航力更長，而高性能桌上型電腦用戶也不必在遊戲時面臨追求反應速度或更真實的畫面之間的取捨。
"""
})

print(result) 
