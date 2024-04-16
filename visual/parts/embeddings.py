import os
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from dotenv import load_dotenv
from llama_index.legacy import load_index_from_storage, ServiceContext, LLMPredictor, SimpleDirectoryReader, \
    GPTVectorStoreIndex, StorageContext, OpenAIEmbedding
from nicegui import app


class Embedding:
    """
    This class is used to create and load embeddings, and to query them using langchain.
    """
    def __init__(self, config):
        """
        Initializes the Embedding class with the necessary directories and services.
        """
        self.api_key = config['api_key']
        self.api_base = config['api_base']
        self.last_model = config['last_model']
        self.embed_model = config['embed_model']
        self.embedding_file_dir = config['embedding_files']
        self.vector_dir = config['index_files']
        self.service_context = ServiceContext.from_defaults(embed_model=OpenAIEmbedding(model=self.embed_model),
                                                llm=ChatOpenAI(temperature=0, openai_api_key=self.api_key, openai_api_base=self.api_base,
                                                model_name=self.last_model, request_timeout=120)) if self.api_key != '' else None
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
        llm = ChatOpenAI(temperature=0, openai_api_key=self.api_key, openai_api_base=self.api_base, model_name=self.last_model, request_timeout=120)
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