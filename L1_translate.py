from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import dotenv_values

# 讀取 .env 檔案
config = dotenv_values(".env")

# 初始化語言模型
model = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    api_key=config.get("AZURE_OPENAI_KEY"),
)

# 設定語言模型的 prompt
prompt = PromptTemplate.from_template("Translate the following English text to zh-tw: {text}")

# 建立可執行的 Chain
chain = prompt | model | StrOutputParser()

# 執行 LLMChain
result = chain.invoke({"text": "Hello, how are you?"})

print(result) 
