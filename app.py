import gradio as gr
from langchain.vectorstores import Chroma
from langchain.embeddings import CohereEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import (
    StuffDocumentsChain, LLMChain
)
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.manager import (
    trace_as_chain_group, 
)

### Set up our retriever

embeddings = CohereEmbeddings()
vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma")
retriever = vectorstore.as_retriever()


### Set up our chain that can answer questions based on documents

# This controls how each document will be formatted. Specifically,
# it will be passed to `format_document` - see that function for more
# details.
document_prompt = PromptTemplate(
    input_variables=["page_content"],
     template="{page_content}"
)
document_variable_name = "context"
llm = ChatOpenAI(temperature=0)
# The prompt here should take as an input variable the
# `document_variable_name`
prompt_template = """Use the following pieces of context to answer user questions. If you don't know the answer, just say that you don't know, don't try to make up an answer.

--------------

{context}"""
system_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
prompt = ChatPromptTemplate(
	messages=[
		system_prompt, 
		MessagesPlaceholder(variable_name="chat_history"), 
		HumanMessagePromptTemplate.from_template("{question}")
	]
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
combine_docs_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
    document_separator="---------"
)

### Set up a chain that controls how the search query for the vectorstore is generated

# This controls how the search query is generated.
# Should take `chat_history` and `question` as input variables.
template = """Combine the chat history and follow up question into a a search query.

Chat History:

{chat_history}

Follow up question: {question}
"""
prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(temperature=0)
question_generator_chain = LLMChain(llm=llm, prompt=prompt)


### Create our function to use

def qa_response(message, history):

	# Convert message history into format for the `question_generator_chain`.
	convo_string = "\n\n".join([f"Human: {h}\nAssistant: {a}" for h, a in history])

	# Convert message history into LangChain format for the final response chain.
	history_langchain_format = []
	for human, ai in history:
		history_langchain_format.append(HumanMessage(content=human))
		history_langchain_format.append(AIMessage(content=ai))

	# Wrap all actual calls to chains in a trace group.
	with trace_as_chain_group("qa_response") as group_manager:

		# Generate search query.
		search_query = question_generator_chain.run(
			question=message, 
			chat_history=convo_string, 
			callbacks=group_manager
		)

		# Retrieve relevant docs.
		docs = retriever.get_relevant_documents(search_query, callbacks=group_manager)

		# Answer question.
		return combine_docs_chain.run(
			input_documents=docs, 
			chat_history=history_langchain_format, 
			question=message, 
			callbacks=group_manager
		)

### Now we start the app!

gr.ChatInterface(qa_response).launch()
