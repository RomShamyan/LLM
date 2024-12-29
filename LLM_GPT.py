# Setup libraries
!pip install langchain openai chromadb pypdf tiktoken
pip install rapidocr-onnxruntime
pip install boto3
pip install pdf2image
pip install pytesseract
pip install  tesseract
pip install PIL


# load required library
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os


# Load the embedding and LLM model
embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", max_tokens = 200)



# Load 1 PDF (text)
#pdf_link = "C:/Users/shamy/lesson_1/Job/Ecomseller/LLm + RAG/ZHI_ZHTmanual_outline.pdf"
#loader = PyPDFLoader(pdf_link, extract_images=False)
#pages = loader.load_and_split()

# Load several PDF files (text)
#pdf_link = "C:/Users/shamy/lesson_1/Job/Ecomseller/LLm + RAG"
#loader = PyPDFDirectoryLoader(pdf_link)
#pages = loader.load_and_split()




# Split data into chunks (PDF-text)
#text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
  #  chunk_size = 4000,
    #chunk_overlap  = 20,
    #length_function = len,
   # add_start_index = True,
#)
#chunks = text_splitter.split_documents(pages)



#Этот ниже преодразватель PDF в текст  лучше для файлов, в которых PDF файл полностью как картинки. Ниже для зарузки одного файла либо нескольктх сразу

# Load 1 PDF (image)
# Путь к исполняемому файлу Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Путь к файлу PDF
pdf_path = 'C:/Users/shamy/lesson_1/Job/Ecomseller/LLm_RAG/ZHI_ZHTmanual_outline.pdf'
# Путь к директории Poppler bin
poppler_path = r'C:\ProgramData\chocolatey\lib\poppler\tools\Library\bin'

  # Имя файла без расширения для сохранения текста
base_name = os.path.splitext(os.path.basename(pdf_path))[0]

# Конвертация PDF в список изображений с явным указанием poppler_path
images = convert_from_path(pdf_path, poppler_path=poppler_path)

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
chunks = []
# Проход по каждому изображению и распознавание текста
full_text = f"Document {doc_num}: {base_name}\n\n"  # Нумерация документа
for page_num, image in enumerate(images, start=1):
    # Распознавание текста с текущего изображения
    text = pytesseract.image_to_string(image, lang='rus')  # Укажите 'eng' для английского текста или 'rus' для русского
    full_text += f"Page {page_num}:\n{text}\n\n"  # Нумерация страницы

# Разделение текста на куски
all_chunks = text_splitter.split_text(full_text)

 # Преобразование кусков текста в объекты Document
for chunk in all_chunks:
    chunks.append(Document(page_content=chunk, metadata={"source": base_name}))
    
    
    
#------------    
    
# Load several PDF files (image)
# Указание пути к Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Путь к директории Poppler bin
poppler_path = r'C:\ProgramData\chocolatey\lib\poppler\tools\Library\bin'

# Путь к директории с PDF файлами
pdf_dir = 'C:/Users/shamy/lesson_1/Job/Ecomseller/LLm_RAG'  

# Путь к директории для сохранения текстовых файлов
output_dir = 'output_texts'
os.makedirs(output_dir, exist_ok=True)

# Получаем список всех PDF файлов в указанной директории
pdf_files = [os.path.join(pdf_dir, file) for file in os.listdir(pdf_dir) if file.endswith('.pdf')]

# Создаем экземпляр CharacterTextSplitter
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)

# Массив для хранения всех кусков текста
chunks = []

# Проход по каждому PDF файлу
for doc_num, pdf_file in enumerate(pdf_files, start=1):
    try:
        # Конвертация PDF в список изображений с явным указанием poppler_path
        images = convert_from_path(pdf_file, poppler_path=poppler_path)
        
        # Имя файла без расширения для сохранения текста
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        
        # Проход по каждому изображению и распознавание текста
        full_text = f"Document {doc_num}: {base_name}\n\n"  # Нумерация документа
        for page_num, image in enumerate(images, start=1):
            # Распознавание текста с текущего изображения
            text = pytesseract.image_to_string(image, lang='rus')  # Укажите 'eng' для английского текста или 'rus' для русского
            full_text += f"Page {page_num}:\n{text}\n\n"  # Нумерация страницы
        
        # Разделение текста на куски
        text_chunks = text_splitter.split_text(full_text)
        
        # Преобразование кусков текста в объекты Document
        for chunk in text_chunks:
            chunks.append(Document(page_content=chunk, metadata={"source": base_name}))
        
        print(f"Текст успешно извлечен и разделен на {len(text_chunks)} частей для файла {base_name}")
    
    except Exception as e:
        print(f"Ошибка при обработке файла {pdf_file}: {e}")
        
#--------



def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

calculate_chunk_ids(chunks)

#----

# Store data into database
db = Chroma.from_documents(chunks, embedding = embeddings_model, persist_directory="test_index")
db.persist()



# Load the database
vectordb = Chroma(persist_directory="test_index", embedding_function = embeddings_model)

# Load the retriver
retriever = vectordb.as_retriever(search_kwargs = {"k" : 3})
chain = load_qa_chain(llm, chain_type="stuff")



# A utility function for answer generation
def ask(question):
    context = retriever.get_relevant_documents(question)
    answer = (chain({"input_documents": context, "question": question}, return_only_outputs=True))['output_text']
    return answer



# Take the user input and call the function to generate output
user_question = input("User:")
answer = ask(user_question)
print("Answer:", answer)



