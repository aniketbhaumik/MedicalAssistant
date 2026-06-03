from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA 
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    llm = ChatGroq(
            api_key = GROQ_API_KEY, 
            model="llama-3.3-70b-versatile", 
            temperature=0.7
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"], 
        template = """
            You are a helpful **Medical Assistant** that provides users with accurate and concise answers 
            to help them understand their medical documents and health-related questions.

            The user will ask you a question, and you will use the following retrieved context to answer the question.

            **Context:**
            {context}

            ** User Question:**
            {question}

            **Answer:**
            - Respond in a clam, factual and respectful tone.
            - Use simple explanations when needed.
            - If the context does not contain relevant information, say "I don't know. I could not find relevant facts in the document." 
            instead of making up an answer.
            - Do NOT provide any medical advice or recommendations.
            """
    )

    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
