import asyncio
import os
from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain_community.chat_models.openai import ChatOpenAI
# from langchain.chains.conversation.base import ConversationChain
from llama_index.legacy import ServiceContext, OpenAIEmbedding

# os.environ["OPENAI_API_KEY"] = 'sk-d5O7zsDceTuR0TIc592b29D61a31451e924e56Ab94FaA22d'
# os.environ["OPENAI_API_BASE"] = 'https://api.xty.app/v1'

# load_dotenv()#load environmental variables
# llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', base_url='https://api.xty.app/v1',
#                  api_key='sk-d5O7zsDceTuR0TIc592b29D61a31451e924e56Ab94FaA22d', temperature=0.1, context_length=4000)
# s = ServiceContext.from_defaults(embed_model=OpenAIEmbedding(model="text-embedding-3-large"), llm=llm)

llm = ConversationChain(
    llm=ChatOpenAI(model_name='gpt-3.5-turbo-0301',
                   openai_api_base='https://api.xty.app/v1',
                   openai_api_key='sk-d5O7zsDceTuR0TIc592b29D61a31451e924e56Ab94FaA22d'))
async def test():
    print(await llm.arun('你好！'))

asyncio.run(test())