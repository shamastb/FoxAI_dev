import os
import sys
import tempfile
import streamlit as st
import langchain
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationKGMemory, CombinedMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool, tool, DuckDuckGoSearchResults
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.docstore import InMemoryDocstore
from dotenv import load_dotenv

print("Python version:", sys.version)
print("LangChain version:", langchain.__version__)

st.set_page_config(page_title="FoxAI: Chat with Documents", page_icon="🦊")
st.title("🦊AI: Chat with Documents")

load_dotenv()

@st.cache_resource(ttl="1h")
def configure_qa_chain(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb_doc = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb_doc.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 4})

    # Setup LLM 
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True
    )

    # Setup memory for contextual conversation
    conv_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    #summary_memory = ConversationSummaryMemory(llm=llm, input_key="input")
    #graph_memory = ConversationKGMemory(llm=llm, memory_key="chat_history", return_messages=True)
    #memory = CombinedMemory(memories=[conv_memory, summary_memory])

    #QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=conv_memory, verbose=True
    )
    return qa_chain

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)

uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)

qa_chain_doc = configure_qa_chain(uploaded_files)

Search = st.sidebar.checkbox("Allow internet search inquery")
if Search:
    tools = [DuckDuckGoSearchResults(name="Search"),
        Tool(
            name = "Document Query",
            func = qa_chain_doc.run,
            description = "This is the primary tool. Useful to answer questions about the document or book or uploaded file. Answer questions like an analyst",
        ),
    ]
else:
    tools = [
        Tool(
            name = "Document Query",
            func = qa_chain_doc.run,
            description = "This is the primary tool. Useful to answer questions about the document or book or uploaded file. Answer questions like an analyst",
        ),
    ]

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything about what's been uploaded.")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True
    )

    SD_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = SD_agent.run(user_query, callbacks=[retrieval_handler, stream_handler])
        st.session_state.messages.append({"role": "assistant", "content": response})
