import gradio as gr
import os

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from utils import ArgumentParser
os.environ["OPENAI_API_KEY"] = 'sk-k9jM0trGFbAQm1cu6bD951Cd503c4203A64d6aFf7aD8Ff09'
os.environ["OPENAI_BASE_URL"] = 'https://api.xiaoai.plus/v1'

def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # 从命令行参数中获取,如果不传参数则默认不开启大模型聊天
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()
    enable_chat = args.enable_chat if args.enable_chat else False
    

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导，请您稍等~"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="汽车销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化汽车销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
