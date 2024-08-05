# cd C:\chainlit_project
# netstat -ano | findstr :8000
# taskkill /PID <PID> /F
# tasklist /FI "IMAGENAME eq python.exe"
# taskkill /F /IM python.exe

# chainlit run creditcard_chatbotdb.py
# chainlit run creditcard_chatbotdb.py --port 8001


import os
import pandas as pd
from sqlalchemy import create_engine
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
import boto3
from langchain_experimental.comprehend_moderation import AmazonComprehendModerationChain
from langchain_community.llms.fake import FakeListLLM
from langchain_experimental.comprehend_moderation.base_moderation_exceptions import (
    ModerationPiiError,
)
from langchain_experimental.comprehend_moderation import (
    BaseModerationConfig,
    ModerationPiiConfig,
    ModerationPromptSafetyConfig,
    ModerationToxicityConfig,
)
from langchain_chroma import Chroma
import chainlit as cl

# Set up the Azure Chat OpenAI model
os.environ["AZURE_OPENAI_API_KEY"] = "####"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https:##########/"
OPENAI_API_KEY = "###########"
OPENAI_DEPLOYMENT_NAME = "gpt4"
MODEL_NAME = "gpt-4"
OPENAI_API_VERSION = "2024-03-01-preview"

# Set up the Amazon Comprehend Moderation
os.environ["AWS_ACCESS_KEY_ID"] = "####"
os.environ["AWS_SECRET_ACCESS_KEY"] = "######"


# Set up the Azure AI Search Retriever if using AZURE_COGNITIVE_SEARCH_SERVICE 
# os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"] = "https://######"
# os.environ["AZURE_AI_SEARCH_SERVICE_NAME"] = "##########"
# os.environ["AZURE_COGNITIVE_SEARCH_INDEX_NAME"] = "####"
# os.environ["AZURE_AI_SEARCH_INDEX_NAME"] = "CreditCards"
# os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"] = "########"
# os.environ["AZURE_AI_SEARCH_API_KEY"] = "######"


# Database connection
conn_str = 'postgresql://postgres:####'
engine = create_engine(conn_str)

# Function to fetch promotional credit card products
def fetch_promotional_cards():
    try:
        query = "SELECT * FROM credit_card_products WHERE annual_fee < 1"
        df_promotional_cards = pd.read_sql(query, engine)
        return df_promotional_cards
    except Exception as e:
        print(f"Error fetching promotional cards: {e}")
        return pd.DataFrame()

@cl.on_chat_start
async def on_chat_start():
    # Chatbot agent initialization
    chat_model = AzureChatOpenAI(
        openai_api_version=OPENAI_API_VERSION,
        azure_deployment=OPENAI_DEPLOYMENT_NAME,
        temperature=0
    )

    emb_model = AzureOpenAIEmbeddings(
        deployment='textembedding3large',
        model='text-embedding-3-large',
        openai_api_key=OPENAI_API_KEY,
        azure_endpoint="https://#####",
        openai_api_type="azure"
    )

    # Loads the vector database from the specified directory and returns a retriever for searching the embeddings.
    def get_retriever():
        loaded_vectordb = Chroma(
            persist_directory="../data/chroma_db",
            embedding_function=emb_model
        )
        retriever = loaded_vectordb.as_retriever()
        return retriever

    chat_retriever = get_retriever()
    
    chat_memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
        return_messages=True
    )

    # Amazon Comprehend Moderation config
    comprehend_client = boto3.client("comprehend", region_name="us-east-1")

    pii_labels = ["SSN", "DRIVER_ID", "ADDRESS", 'EMAIL', 'PHONE', 'CA_SOCIAL_INSURANCE_NUMBER']
    pii_config = ModerationPiiConfig(labels=pii_labels, redact=True, mask_character="X")
    toxicity_config = ModerationToxicityConfig(threshold=0.5)
    prompt_safety_config = ModerationPromptSafetyConfig(threshold=0.5)

    moderation_config = BaseModerationConfig(
        filters=[pii_config, toxicity_config, prompt_safety_config]
    )

    comp_moderation_with_config = AmazonComprehendModerationChain(
        moderation_config=moderation_config,  # specify the configuration
        client=comprehend_client,  # optionally pass the Boto3 Client
        verbose=True
    )

    # Define the system and human message template
    system_template = """
    You are a virtual assistant for Credit Card business.
    Only answer questions related to credit card business. 
    Include URL in your reply, return the full URL for your source document.
    Do not include any email.
    Use the answers from the retrieved document first.
    If you cannot find the answer from the pieces of context, just say that sorry you don't know nicely.
    Do not try to make up an answer.
    All the personal identifiable information will be redact with X. 
    Ignore the personal identifiable information and answer generally. 
    ---------------
    {context}
    """
    human_template = """Previous conversation: {chat_history}
    New human question: {question}
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]

    # Initialize the chain
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        chain_type='stuff',
        retriever=chat_retriever,
        memory=chat_memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    # Let the user know that the system is ready
    msg = cl.Message(
        content="Hello, this is AI helpdesk, feel free to ask me any questions!"
    )
    await msg.send()

    cl.user_session.set("chain", qa)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True
    )

    # Force final answer if necessary
    cb.answer_reached = True

    res = await chain.acall(message.content, callbacks=[cb])
 
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = [] # type: List[cl.Text]
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name, display="side")
            )
        source_names = [text_el.name for text_el in text_elements]
        answer += f"\n\nSources: {', '.join(source_names)}"
    else:
        answer += "\n\nNo sources found"

    # Fetch promotional credit card products
    df_promotional_cards = fetch_promotional_cards()
    promotional_cards_info = df_promotional_cards.to_dict(orient='records')
    
    # Append promotional cards information to the answer
    if promotional_cards_info:
        answer += "\n\nPromotional Credit Cards with No Annual Fee:\n"
        for card in promotional_cards_info:
            answer += f"- {card['card_name']} (Credit Limit: {card['credit_limit']}, Cashback: {card['cashback']}%, Sign-Up Bonus: {card['sign_up_bonus']} points)\n"

    await cl.Message(content=answer, elements=text_elements).send()
