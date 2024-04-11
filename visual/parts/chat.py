import json
import os
import shutil
from datetime import datetime
from typing import List, Dict
from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.memory.chat_memory import ChatMessageHistory
from llama_index.legacy import ServiceContext, OpenAIEmbedding
from nicegui import ui, app
from visual.parts.embeddings import Embedding


class ChatApp(Embedding):
    chat_name = None
    def __init__(self, config):
        super().__init__(config)  # Call the initializer of the parent class
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']
        self.max_retries = config['max_retries']
        self.memory = ConversationBufferMemory()
        self.embedding_switch = False
        self.uname = app.storage.user['user']['username']
        self.messages = []  # var that will contain an conversation
        self.thinking = False  #var for showing the spinner
        self.tokens_used = 0  # var for counting the tokens
        self.total_cost = 0  #var for cost in usd
        self.current_chat_name = ""  #name for the currently selected chat. will be filled when someone clicks on a chat in the aggrid
        self.llm = ConversationChain(
            llm=ChatOpenAI(model_name=config['last_model'],
                           openai_api_base=config['api_base'], max_tokens=self.max_tokens, max_retries=self.max_retries,
                           openai_api_key=config['api_key'], temperature=config['temperature']),
            memory=self.memory) if self.api_key != '' else None
        self.json_directory = config['chat_history']

    async def on_value_change(self, ename=None, etemp=None, embedding_switch=False, etok=None, eret=None, model_type='generation'):
        """
        Changes the value of the model and temperature for the ConversationChain.
        Parameters:
        ename (str): The name of the model.
        etemp (str): The temperature for the model.
        """
        #Open texts withe the models
        self.temperature = etemp if etemp is not None else self.temperature
        self.max_tokens = etok if etok is not None else self.max_tokens
        self.max_retries = eret if eret is not None else self.max_retries
        if model_type == 'generation':
            self.last_model = ename if ename is not None else self.last_model
            self.llm = ConversationChain(
                llm=ChatOpenAI(model_name=self.last_model,
                               openai_api_base=self.api_base, max_tokens=self.max_tokens,
                               max_retries=self.max_retries,
                               openai_api_key=self.api_key, temperature=self.temperature),
                memory=self.memory) if self.api_key != '' else None
        elif model_type == 'embedding':
            self.embed_model = ename if ename is not None else self.embed_model
            self.service_context = ServiceContext.from_defaults(embed_model=OpenAIEmbedding(model=self.embed_model),
                                                                llm=ChatOpenAI(temperature=0, openai_api_key=self.api_key,
                                                                               openai_api_base=self.api_base,
                                                                               model_name=self.last_model,
                                                                               request_timeout=120)) if self.api_key != '' else None
        self.embedding_switch = embedding_switch

    @ui.refreshable
    def chat_messages(self) -> None:
        """
        Displays the chat messages in the UI. Looks for the messages in the self.messages dict
        """

        async def copy_code(text):
            escaped_text = text.replace("\\", "\\\\").replace("`", "\\`")  #für saubere darstellung aus ui.markdown
            await ui.run_javascript(f'navigator.clipboard.writeText(`{escaped_text}`)')
            ui.notify("Text Copied!", type="positive")

        chatcolumn = ui.column().classes("w-full")
        for name, text in self.messages:
            with chatcolumn:
                if name == self.uname:
                    with ui.row().classes("overflow-auto no-wrap bg-cyan-400 rounded-lg text-white"):
                        ui.icon("person", size="40px").on("click",
                                                          lambda text=text, copy_code=copy_code: copy_code(text))
                        ui.markdown(text).classes("text-base pr-3")
                else:
                    with ui.row().classes("w-full no-wrap overflow-auto hover:bg-slate-100"):
                        ui.icon("smart_toy", size="40px")
                        ui.markdown(text).classes("w-full text-base pr-3")
                    with ui.row().classes("w-full justify-end -mt-9"):
                        with ui.icon('content_copy', size='xs', color="blue").classes(
                                'opacity-40 hover:opacity-80 cursor-pointer pb-5').on("click",
                                                                                      lambda text=text: copy_code(
                                                                                              text)):
                            ui.tooltip("Copy")
                        ui.icon('content_copy', size='xs').classes(
                            'opacity-20 hover:opacity-80 cursor-pointer pb-5').on("click",
                                                                                  lambda text=text: copy_code(text))
        if self.thinking:
            ui.spinner("comment", size='3rem')
            ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')

    @ui.refreshable
    def chat_history_grid(self):
        async def rename_chat(old_filename):
            old_file_path = os.path.join(self.json_directory, old_filename)
            new_file_path = os.path.join(self.json_directory, self.chat_name + ".json")

            if os.path.exists(old_file_path):
                shutil.move(old_file_path, new_file_path)
                ui.notify(f"Chat renamed from {old_filename} to {self.chat_name}.json", type="positive")
            else:
                ui.notify(f"No chat found with the name {old_filename}", type="negative")
            self.chat_history_grid.refresh()

        if not os.path.exists(self.json_directory):
            os.makedirs(self.json_directory)
        json_filenames = [f for f in os.listdir(self.json_directory) if f.endswith('.json')]

        # Create a list of tuples, each containing a filename and its corresponding timestamp
        timestamps_and_filenames = []
        for filename in json_filenames:
            with open(os.path.join(self.json_directory, filename), 'r') as f:
                data = json.load(f)
                timestamp = data['timestamp']
                timestamps_and_filenames.append((timestamp, filename))

        # Sort the list of tuples by the timestamp (in descending order)
        timestamps_and_filenames.sort(reverse=True)

        # Extract the sorted filenames
        sorted_filenames = [filename for timestamp, filename in timestamps_and_filenames]

        #Build the list of chats based on the sorted filenames
        with ui.column().classes("h-1/2 overflow-y-auto bg-white cursor-pointer").bind_visibility_from(self,
                                                                                                       "embedding_switch",
                                                                                                       value=False):
            with ui.element('q-list').props('bordered separator').classes("overflow-y-auto"):  #list element
                for filename in sorted_filenames:
                    with ui.element('q-item').classes("pt-2 hover:bg-slate-100"):  #item in the list
                        with ui.element('q-item-section').classes("overflow-hidden max-w-xs"):  #name of the chat
                            ui.label(filename).on("click",
                                                  lambda filename=filename: self.load_chat_history(filename)).classes(
                                "overflow-auto w-40")
                        with ui.element('q-item-section').props('side'):  #delete button and opening the dialog
                            with ui.dialog() as dialog, ui.card():
                                ui.label('Are you sure you want to delete the chat?')
                                with ui.row():
                                    ui.button('Yes',
                                              on_click=lambda filename=filename: self.delete_chat(filename)).classes(
                                        "bg-red")
                                    ui.button('No', on_click=dialog.close)
                            ui.icon('delete', color="red").on("click", dialog.open)
                        with ui.element('q-item-section').props('side'):  #edit name button
                            with ui.dialog() as edit_dialog, ui.card().classes("w-1/2"):
                                ui.label('Enter the new name')
                                name_input = ui.input(on_change=lambda e: setattr(self, 'chat_name', e.value)).classes(
                                    "w-full")
                                with ui.row():
                                    ui.button('Rename',
                                              on_click=lambda filename=filename: rename_chat(filename)).classes(
                                        "bg-black")
                                    ui.button('Close', on_click=edit_dialog.close)
                            ui.icon("edit").on("click", edit_dialog.open)

    async def send(self, text) -> None:
        """
        Sends a message to the chat. Appends the self.messages list with the new message, sends it to the llm using the self.llm.arun function
        also afte every sending the current chat is beeing updated in the json

        Parameters:
        text (str): The message to be sent. Text beeing given from the ui.input
        """
        self.thinking = True
        self.chat_messages.refresh()
        #message = text.value
        self.messages.append((self.uname, text))
        if self.embedding_switch is True:  ###if we are using embedding the chat history is not saved
            with get_openai_callback() as cb:
                response = await self.querylangchain(prompt=text)  ##using the langchain angent from the embeddings.py instead of a simple gpt call
                self.tokens_used = cb.total_tokens
                self.total_cost = round(cb.total_cost, 6)  # get the total tokens used
                self.messages.append(('GPT', response))
                self.thinking = False
                self.chat_messages.refresh()
        else:
            with get_openai_callback() as cb:  ##if we are not using embedding the chat history is saved
                try:
                    response = await self.llm.arun(text)
                except Exception as e:
                    response = f"An error occurred: {self.api_base, str(e)}"
                self.tokens_used = cb.total_tokens
                self.total_cost = cb.total_cost  # get the total tokens used
                self.messages.append(('GPT', response))
                await self.save_to_db(self.messages_to_dict(self.memory.chat_memory.messages))
                self.thinking = False
                self.chat_messages.refresh()

    async def clear(self):
        """
        Clears the chat memory and messages to "open" a new chat
        """
        self.llm.memory.clear()
        self.memory.clear()
        self.messages.clear()
        self.current_chat_name = ""
        self.tokens_used = "0"
        self.embedding_switch = False
        self.thinking = False
        self.chat_messages.refresh()

    def messages_to_dict(self, messages: List) -> List[Dict]:
        """
        Helper function for saving the messages to json. Converts the chat messages to a dictionary.

        Parameters:
        messages (List): The list of messages.

        Returns:
        List[Dict]: The dictionary representation of the messages.
        """
        return [{'type': type(m).__name__, 'content': m.content} for m in messages]

    async def save_to_db(self, data: List[Dict]) -> None:
        """
        Saves the chat history to the database. It checks if the current chat is already in the directory (thorugh self.current_chat_name) and if yes just updates the json file. if the chat is not in the directory
        a new json file is created. a call to the llm done before to sum the chat in 5 words and this becomde the filename fpr the json file.

        Parameters:
        data (List[Dict]): The chat history to be saved.
        """
        os.makedirs(self.json_directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_with_timestamp = {"timestamp": timestamp, "messages": data, "tokens_used": self.tokens_used}
        if self.current_chat_name:
            file_path = os.path.join(self.json_directory, f'{self.current_chat_name}')
            with open(file_path, 'w') as f:
                json.dump(data_with_timestamp, f)
        else:
            chat_history_text = '\n'.join(f'{"You" if m["type"] == "HumanMessage" else "GPT"}: {m["content"]}'
                                          for m in data)
            prompt_text = f"{chat_history_text}\n\nSummarize the above conversation with a descriptive name not longer than 5 words. Output only the name chosen."

            llm = ConversationChain(
                llm=ChatOpenAI(model_name=self.last_model, openai_api_key=self.api_key, openai_api_base=self.api_base, temperature=self.temperature))
            response = await llm.arun(prompt_text)
            #response = datetime.now()
            print(response)
            file_path = os.path.join(self.json_directory, f'{response}.json')
            with open(file_path, 'w') as f:
                json.dump(data_with_timestamp, f)
            self.current_chat_name = f'{response}.json'

    def load_and_convert_messages(self, filename: str) -> List:
        """
        Loads the chat history from the database and converts it to the appropriate message objects.

        Parameters:
        filename (str): The name of the file to be loaded.

        Returns:
        List: The list of message objects.
        """
        file_path = os.path.join(self.json_directory, f'{filename}')
        with open(file_path, 'r') as f:
            data_with_timestamp = json.load(f)
            self.tokens_used = data_with_timestamp["tokens_used"]
            messages_data = data_with_timestamp["messages"]

        # Convert the dictionary representation of messages to actual message objects
        messages = []
        for m in messages_data:
            if m['type'] == 'HumanMessage':
                messages.append(HumanMessage(content=m['content']))
            elif m['type'] == 'AIMessage':
                messages.append(AIMessage(content=m['content']))
        return messages

    async def delete_chat(self, filename):
        """
        Deletes a chat.

        Parameters:
            filename (str): The name of the file to be deleted.

        Returns:
            None
        """
        file_path = os.path.join(self.json_directory, filename)
        os.remove(file_path)
        await self.clear()
        self.chat_history_grid.refresh()

    async def load_chat_history(self, filename: str) -> None:
        """
        Loads the chat history. Gets the content of the selected json file and passes it as a langchain history object to the llm

        Parameters:
        filename (str): The name of the file to be loaded.
        """
        self.thinking = True
        self.current_chat_name = filename
        # Clear existing messages and memory before loading new history
        self.messages = []  # Reset the messages list to be empty
        self.memory.clear()  # Clear the chat memory

        # Load saved messages from JSON file
        retrieved_messages = self.load_and_convert_messages(filename)
        retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
        self.memory.chat_memory = retrieved_chat_history  # Update ConversationBufferMemory with loaded history

        # Directly set self.messages to the loaded messages only
        self.messages = [('You', m.content) if isinstance(m, HumanMessage) else ('GPT', m.content) for m in
                         retrieved_messages]
        self.thinking = False
        self.chat_messages.refresh()
