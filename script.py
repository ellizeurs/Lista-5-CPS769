import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Função de carregamento e processamento do PDF
def load_pdf(file_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Função para criar o vetor de conhecimento do PDF
def create_vectorstore_from_pdf(pdf_path):
    pdf_text = load_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(pdf_text)
    
    # Criar embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

# Inicializando o LLM e criando o vetor de conhecimento a partir de um PDF
load_dotenv()
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1,
    max_tokens=100,
    timeout=10,
    max_retries=3
)

# Substitua 'seu_pdf.pdf' pelo caminho do seu PDF
vectorstore = create_vectorstore_from_pdf("pdfs/524_icmlpaper.pdf")

# Criar a cadeia de recuperação de conversação
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Histórico de mensagens
messages = [
    SystemMessage(''),
]

# Loop de conversação
chat_history = []
while True:
    message = input('Você: ')
    if message == "exit()":
        break
    
    # Adicionar a entrada do usuário ao histórico
    chat_history.append(HumanMessage(message))
    
    # Executar a cadeia de QA com recuperação
    response = qa_chain({"question": message, "chat_history": chat_history})
    
    # Extrair a resposta e as fontes (se aplicável)
    answer = response["answer"]
    source_docs = response.get("source_documents", [])
    
    if source_docs:
        system_message_content = "Documentos relevantes:"
        for doc in source_docs:
            system_message_content += f"\n- {doc.page_content}"  # Exibe os primeiros 200 caracteres da página relevante
        #messages.append(SystemMessage(content=system_message_content))
        #answer = llm.invoke(messages)

    # Adicionar a resposta ao histórico
    print('ChatGPT: ', answer)
    chat_history.append(AIMessage(answer))
