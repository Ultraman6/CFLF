import os
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from dotenv import load_dotenv
from llama_index.legacy import load_index_from_storage, ServiceContext, LLMPredictor, SimpleDirectoryReader, \
    GPTVectorStoreIndex, StorageContext
from nicegui import app


class Embedding:
    """
    This class is used to create and load embeddings, and to query them using langchain.
    """
    def __init__(self, openai_api_key, embedding_files, index_files):
        """
        Initializes the Embedding class with the necessary directories and services.
        """
        self.openai_api_key = openai_api_key
        self.openai_models = ["gpt-3.5-turbo", "gpt-4-1106-preview"]
        self.embedding_file_dir = embedding_files
        self.vector_dir = index_files
        self.service_context = (ServiceContext.from_defaults
                                (llm=ChatOpenAI(temperature=0, model_name=app.storage.user.
                                                get('last_model', self.openai_models[0]), request_timeout=120))) if self.openai_api_key != '' else None
    async def create_index(self):
        """
        Asynchronously creates an index from the documents in the embedding file directory.
        """
        self.documents = SimpleDirectoryReader(self.embedding_file_dir, recursive=True).load_data()
        self.index = GPTVectorStoreIndex.from_documents(
            self.documents, service_context=self.service_context
        )
        self.index.storage_context.persist(persist_dir=self.vector_dir)
       
    def load_index(self):
        
        """
        Loads an index from the vector directory.
        Returns:
            The loaded index.
        """
        storage_context = StorageContext.from_defaults(persist_dir=self.vector_dir)
        self.index = load_index_from_storage(storage_context)
        return self.index
    async def querylangchain(self, prompt):
        """
        Uses the embedding from llamaindex with langchain.
        Creates a langchain agent that uses the embedding. Asynchronously queries the langchain with a given prompt.
        
        Parameters:
            prompt (str): The prompt to query the langchain with.
        Returns:
            The response from the langchain.
        """
        llm = ChatOpenAI(temperature=0)
        index = self.load_index()
        memory = ConversationBufferMemory(memory_key="chat_history")

        self.tools = [
                Tool(
                    name="LlamaIndex",
                    func=lambda q: str(index.as_query_engine().query(q)),
                    description="useful for when you want to answer questions about the author. The input to this tool should be a complete english sentence.",
                    return_direct=True,
                ),
            ]      
        agent_executor = initialize_agent(
        self.tools, llm, agent="conversational-react-description", memory=memory)
        response = await agent_executor.arun(prompt)
        return response