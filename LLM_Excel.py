# load required library
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import os
import pandas as pd


# set OpenAI key as the environmet variable
os.environ['OPENAI_API_KEY'] = ''

# Load the embedding and LLM model
embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", max_tokens = 200)


#VARIANT c группировкой по артикулу
# Путь к файлу
excel_file  = "Указать путь к файлу"

 # Загрузка данных из Excel файла
df = pd.read_excel(excel_file)

# Указать название колонки с артикулами!!!
articul_column = 'Артикул'

# Группировка по артикулу
grouped = df.groupby(articul_column)

# Создание списка документов
chunks = []

for articul, group in grouped:
    # Преобразование каждой группы в текст
    content = f"Артикул: {articul}\n"
    content += group.to_string(index=False)
    
    # Создание объекта Document
    chunks.append(Document(page_content=content, metadata={"Артикул": articul}))
    
    
    
# Store data into database
db = Chroma.from_documents(chunks, embedding = embeddings_model, persist_directory="test_index")
db.persist()



# Load the database
vectordb = Chroma(persist_directory="test_index", embedding_function = embeddings_model)

# Load the retriver
retriever = vectordb.as_retriever(search_kwargs = {"k" : 30})
chain = load_qa_chain(llm, chain_type="stuff")



# Функция для выполнения запроса
def ask(question):
    context = retriever.get_relevant_documents(question)
    answer = chain({"input_documents": context, "question": question}, return_only_outputs=True)['output_text']
    return answer


# Chat
while True:
    user_question = input("User: ")
    if user_question.lower() in ["exit", "quit", "stop"]:
        break
    answer = ask(user_question)
    print("Answer:", answer)