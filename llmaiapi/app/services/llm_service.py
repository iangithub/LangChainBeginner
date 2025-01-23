from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings,ChatOpenAI,OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from dotenv import dotenv_values

# Load the document
config = dotenv_values(".env")

# 初始化語言模型
# generator_llm = AzureChatOpenAI(
#     azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
#     azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
#     openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
#     api_key=config.get("AZURE_OPENAI_KEY"),
# )

generator_llm = ChatOpenAI(api_key=config.get("OPENAI_KEY"),model="gpt-4o-2024-11-20")

# 初始化向量模型
# embedding_llm = AzureOpenAIEmbeddings(
#     azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
#     azure_deployment=config.get("AZURE_OPENAI_Embedding_DEPLOYMENT_NAME"),
#     api_key=config.get("AZURE_OPENAI_KEY"),
#     openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
# )

embedding_llm = OpenAIEmbeddings(
    api_key=config.get("OPENAI_KEY"),model="text-embedding-3-large"
)

# ----- 第一次要把知識文件加入Qdrant 向量資料庫時，執行以下程式碼 -----

# # Load PDF文件
# loader = PyPDFLoader("docs/勞動基準法.pdf")
# pages = loader.load_and_split()

# # 設定文本分割器, 每1000字元分割一次, 重疊400字元
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400) 
# # 分割文本
# splits = text_splitter.split_documents(pages) 

# # Qdrant向量資料庫
# qdrant = QdrantVectorStore.from_documents(
#     splits,
#     embedding=embedding_llm,
#     url=config.get("Qdrant_ENDPOINT"),  # Cloud Qdrant URL
#     api_key=config.get("Qdrant_API_KEY"),  # Cloud Qdrant API Key
#     collection_name="km_docs",
# )

#---------------------------------------------------------

# ------- 後續查詢時，已有向量資料，請執行以下程式碼 -------

# Qdrant client
client = QdrantClient(url=config.get("Qdrant_ENDPOINT"),api_key=config.get("Qdrant_API_KEY"))
collection_name = "km_docs"
qdrant = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_llm
    )

# -------------------------------------------------------

# 設置檢索器
retriever = qdrant.as_retriever(search_kwargs={"k": 3}) # 檢索前3個最相似的文檔

# 建立提示樣板
q_template = ChatPromptTemplate.from_template(
"""
你是一位精通台灣勞基法的專家。請根據以下參考資料回答問題：

###參考資料：
{context}

###問題：
{question}

"""
)

# 建立 QA Chain
qa_chain = (
    {
        "context": retriever ,
        "question": RunnablePassthrough(),
    }
    | q_template
    | generator_llm
    | StrOutputParser()
)

# 定義函數，接收問題，調用 QA Chain 返回答案。
def get_answer_from_llm(question: str) -> str:
    """
    接收問題，調用 QA Chain 返回答案。
    """
    try:
        response = qa_chain.invoke(question) # 進行查詢
        return response
    except Exception as e:
        raise RuntimeError(f"Error while processing the question: {str(e)}")
