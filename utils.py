import os
import re
import pandas as pd
from dotenv import load_dotenv
from prompts import *

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever

def load_api_key():
    load_dotenv('.env', override=True)
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
    return UPSTAGE_API_KEY

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)
    
def extract_answer(response):
    """
    extracts the answer from the response using a regular expression.
    expected format: "[ANSWER]: (A) convolutional networks"

    if there are any answers formatted like the format, it returns None.
    """
    pattern = r"\[ANSWER\]:\s*\((A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|W|Z)\)"
    matches = re.findall(pattern, response)
    
    if matches:
        return matches[-1]  # return last matching
    else:
        return extract_again(response)

def extract_again(response):
    pattern = r"\b[A-J]\b"
    matches = re.findall(pattern, response)
    if matches:
        return matches[-1]  
    else:
        return None
    
def read_data(data_path):
    data = pd.read_csv(data_path)
    prompts = data['prompts']
    answers = data['answers']
    # returns two lists: prompts and answers
    return prompts, answers

def read_data_for_final(data_path):
    data = pd.read_csv(data_path)
    prompts = data['question']
    answers = data['your_answer']
    # returns two lists: prompts and answers
    return prompts, answers

def route(llm, prompt):
    
    prompt_template = PromptTemplate.from_template(DOMAIN_PROMPT)
    chain = prompt_template | llm

    response = chain.invoke({"question": prompt})
    return response.content

def parse_question_and_choices(prompt):

    # extract question+options
    question_match = re.search(r'QUESTION\d+\)\s*(.*?)(?=\([A-Z]\))', prompt, re.DOTALL)

    if not question_match:
        raise ValueError("질문을 찾을 수 없습니다.")

    question = question_match.group(1).strip()

    # extract options
    choice_pattern = r'\(([A-Z])\)\s*(.*?)(?=\([A-Z]\)|$)'
    choices_matches = re.finditer(choice_pattern, prompt, re.DOTALL)

    choices = []
    for match in choices_matches:
        label = match.group(1)
        text = match.group(2).strip()
        choices.append({
            'label': label,
            'text': text
        })

    return question, choices

def ewha_context(query_embedding, search_type, k, lambda_mult, fetch_k, prompt):
    # load db
    db = FAISS.load_local("./faiss_vectorstore/ewha",
                            query_embedding,
                            allow_dangerous_deserialization=True)

    # retriever
    retriever = db.as_retriever(search_type=search_type,
                                search_kwargs={'k': k, 'lambda_mult': lambda_mult, 'fetch_k': fetch_k})
    
    # parse question & choice
    q, choice_list = parse_question_and_choices(prompt)

    context = ""
    # implement rag for each choice
    for _, choice in enumerate(choice_list):
        qa = f"{q}\n{choice}"
        docs = retriever.invoke(qa)
        for doc in docs:
            if doc.page_content not in context: # prevent duplication 
                context += f'\n\n{doc.page_content}'
    
    return context
    
def mmlu_context(routed_result, embedding_model, search_type, k, lambda_mult, fetch_k, prompt):
    # ---- Load FAISS DB ----
    db = FAISS.load_local(
        f"./faiss_vectorstore/{routed_result}",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    # ---- Dense Retriever (FAISS) ----
    dense_retriever = db.as_retriever(
        search_type=search_type,
        search_kwargs={'k': k, 'lambda_mult': lambda_mult, 'fetch_k': fetch_k}
    )

    # ---- BM25 Retriever ----
    documents = list(db.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    
    # ---- Retrieve Relevant Docs ----
    dense_docs = dense_retriever.invoke(prompt)
    sparse_docs = bm25_retriever.invoke(prompt)

    # ---- Merge (Reranking w/ Reciprocal Rank Fusion (RRF)) ----
    scores = {}
    rrf_constant = 60
    
    for rank, doc in enumerate(dense_docs, start=1):
        content = doc.page_content
        if content not in scores:
            scores[content] = {'doc': doc, 'score': 0}
        scores[content]['score'] += 1 / (rrf_constant + rank)
    
    for rank, doc in enumerate(sparse_docs, start=1):
        content = doc.page_content
        if content not in scores:
            scores[content] = {'doc': doc, 'score': 0}
        scores[content]['score'] += 1 / (rrf_constant + rank)
    
    # reranking based on scores -> top-k
    sorted_docs = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
    docs = [item['doc'] for item in sorted_docs[:k]]
    context = format_docs(docs=docs)
    
    return context    

def ewha_rag(prompt, context, llm):

    prompt_template = ChatPromptTemplate.from_messages([
    ("system",
    EWHA_SYSTEM_PROMPT),

    ("human",
    EWHA_HUMAN_PROMPT)
])

    # RAG chain
    rag_chain = prompt_template | llm

    # call RAG chain
    response = rag_chain.invoke({"question": prompt, "context": context})
    answer = response.content

    print(f"💬 answer: {answer}")

    return answer

def mmlu_law_rag(prompt, context, llm):

    # ---- Prompt Template ----
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
        MMLU_LAW_PROMPT),
        ("human",
        MMLU_HUMAN_PROMPT)
    ])

    rag_chain = prompt_template | llm

    # ---- Call RAG chain ----
    response = rag_chain.invoke({
        "question": prompt,
        "context": context
    })

    answer = response.content
    print(f"💬 answer: {answer}")

    return answer

def mmlu_psychology_rag(prompt, context, llm):

    # ---- Prompt Template ----
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
MMLU_PSYCHOLOGY_PROMPT),
        ("human",
MMLU_HUMAN_PROMPT)
    ])

    rag_chain = prompt_template | llm

    # ---- Call RAG chain ----
    response = rag_chain.invoke({
        "question": prompt,
        "context": context
    })

    answer = response.content
    print(f"💬 answer: {answer}")

    return answer

def mmlu_philosophy_rag(prompt, context, llm):

    # ---- Prompt Template ----
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
MMLU_PHILOSOPHY_PROMPT),
        ("human",
MMLU_HUMAN_PROMPT)
    ])

    rag_chain = prompt_template | llm

    # ---- Call RAG chain ----
    response = rag_chain.invoke({
        "question": prompt,
        "context": context
    })

    answer = response.content
    print(f"💬 answer: {answer}")

    return answer

def mmlu_history_rag(prompt, context, llm):

    # ---- Prompt Template ----
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
MMLU_HISTORY_PROMPT),
        ("human",
MMLU_HUMAN_PROMPT)
    ])

    rag_chain = prompt_template | llm

    # ---- Call RAG chain ----
    response = rag_chain.invoke({
        "question": prompt,
        "context": context
    })

    answer = response.content
    print(f"💬 answer: {answer}")

    return answer

def mmlu_business_rag(prompt, context, llm):

    # ---- Prompt Template ----
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
        MMLU_BUSINESS_PROMPT),
        ("human",
        MMLU_HUMAN_PROMPT)
    ])

    rag_chain = prompt_template | llm

    # ---- Call RAG chain ----
    response = rag_chain.invoke({
        "question": prompt,
        "context": context
    })

    answer = response.content
    print(f"💬 answer: {answer}")

    return answer

