from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = CohereEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="chroma")
