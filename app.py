import streamlit as st
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

#Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://#/)
    - [LanchChain](https://#/)
    - [OpenAI](https://#/) LLM model
    ''')
    add_vertical_space(33)
    st.write('Made with Me [Prompt Engineer](https://#)')

load_dotenv()

def main():
    st.header("Chat with PDF file")
    
    #upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        store_name = pdf.name[:-4]

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)


        #embedding
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
                st.write("loaded")
        else:
            with open(f"{store_name}.pkl", 'wb') as f:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                pickle.dump(VectorStore, f)
                st.write("stored")    
        
        #Accept use question
        query = st.text_input("Ask question about your PDF file:")
        
        prompt_template = """Please answer the question.
        {context}
        question: {question}
        """


        if query:
            embedding_vector = OpenAIEmbeddings().embed_query(query)
            docs = VectorStore.similarity_search_by_vector(embedding_vector, k=3)

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=['context', 'question']
            )

            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type = "stuff", prompt=PROMPT)
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)    
            st.write(response)



if __name__ == '__main__':
    main()