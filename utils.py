import os
import re
import pandas as pd
from dotenv import load_dotenv

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

# def randomness_control():
#     # remove randomness
#     random.seed(42)
#     np.random.seed(42)
#     os.environ["PYTHONHASHSEED"] = str(42)
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     faiss.omp_set_num_threads(1)
    
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
    
    prompt_template = PromptTemplate.from_template(
        """
        You are an expert text classifier.
        Below is an instruction that describes a task.
        Write a reponse that appropriately completes the instruction.
        
        ### Execution protocol:
        - If a text is written in Korean, classify a text into 1. Ewha Womans University rules, regardless of its content.

        ### Instruction:
        Classify the following question into ONE of six categories:
        1. Ewha Womans University rules
        2. Law
        3. History
        4. Philosophy
        5. Psychology
        6. Business

        ### Input:
        {question}

        ### Example:

        QUESTION1) 복수전공을 이수하는 학생의 재학연한은 최대 몇 년인가요?
        (A) 6년
        (B) 7년
        (C) 8년
        (D) 9년

        Answer: 1

        --------------------

        QUESTION31) During the manic phase of a bipolar disorder, individuals are most likely to experience
        (A) extreme fatigue
        (B) high self-esteem
        (C) memory loss
        (D) intense fear and anxiety

        Answer: 5

        ### Response:
        Return ONLY a single number from 1 to 6.
        **No Explanation**.
        You MUST follow the valid output format.

        ### Valid Output Examples:
        3
        1
        6
        """
    )
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

    # ---- Merge (Hybrid Retrieval) ----
    merged = {}

    for doc in dense_docs:
        merged[doc.page_content] = doc

    for doc in sparse_docs:
        if doc.page_content not in merged:
            merged[doc.page_content] = doc

    docs = list(merged.values())[:k]  
    context = format_docs(docs=docs)
    
    return context    

def ewha_rag(prompt, context, llm):

    prompt_template = ChatPromptTemplate.from_messages([
    ("system",
    """
    당신은 이화여자대학교 학칙을 완벽하게 이해하고 있는 관리자입니다.

    실행 프로토콜을 정확하게 숙지하세요. 모든 단계를 순서대로 정확하게 따라 올바른 최종 답을 출력하세요.
    모든 단계의 과정을 매우 구체적이고 논리적으로 설명해야 합니다.

    **실행 프로토콜**:
    - 절대 맥락에 명시되지 않은 사실을 임의로 전제하고 추측하지 말 것. 단, 맥락의 의미를 100% 보존하는 한도 내에서 맥락을 유연하게 적용 및 사고 가능. 
    - 맥락에 명확히 언급된 내용만을 종합적으로 고려해 선지를 풀이할 것.
    - 맥락을 읽고 간단한 사칙 연산 (덧셈, 뺄셈, 곱셈, 나눗셈)을 실수없이 수행할 것.
    - 조합 선지 (여러 개의 단일 선지를 포함하는 선지)가 있는 문제의 경우, 조합선지뿐 아니라 단일 선지도 정답으로 고를 수 있음을 명심할 것. 
    - 정답이 없는 경우는 없으며, 오직 하나의 정답만 고를 것.
    - 질문과 선지에 있는 숫자, 단어 등에 오타가 있다고 생각하지 말 것.
    - 만약 확실한 답이 없어도, 반드시 정답에 가장 근접한 선지 (예시: 정답의 내용을 가장 많이 포함한 선지, 정답 그 자체인 선지)를 고르고 그 이유를 논리적이고 구체적으로 설명할 것.
    - 답 출력: 출력은 반드시 아래와 같은 형식을 따라야 함:
     [ANSWER]: (X) 텍스트
     
    **실행 단계** 
    [1단계]: 문제를 자세하게 읽고 질문의 내용과 의도를 명확하게 이해.
    [2단계]: 문제의 각 선지가 정답 또는 오답인 이유를 아래 두 가지 고려 사항에 대해 매우 상세하게 평가하고 설명.
      - 맥락에 주어진 정보와의 일치 여부: 선지의 내용이 맥락의 정보와 모순되는지 확인
      - 질문에 대한 답변으로서의 적절성: 선지의 내용이 질문의 내용과 의도와 일치하는지 확인
    [3단계]: 검증
      - 최종 답을 반환하기 전, 스스로에게 질문: "이 답이 확실한가?"
    [4단계]: 최종 답이 실행 프로토콜의 각 항목을 만족하는지 꼼꼼히 점검
    [5단계]: 실행 프로토콜에 정의된 답 출력 형식을 정확히 따라 정답 반환
    """),

    ("human",
    """
    시스템 프롬프트에 맞춰 최종 답과 함께 각각의 문제 풀이 과정까지 구체적으로 출력하세요.

    문제:
    {question}

    맥락:
    {context}
    """)
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
"""
You are a legal expert who clearly understands the situation, identifies key points of dispute, applies relevant legal provisions, and accurately determines each party’s responsibility.

Carefully familiarize yourself with the execution protocol. 
Follow every step in order exactly and produce the correct final answer. Explain the process of each step very concretely and logically.

**Execution Protocol**:
- Do not output the answer at the beginning; complete all execution steps first, then provide the correct final answer.
- There is always an answer; choose exactly one answer.
- Output format: the output must follow this exact format:
	[ANSWER]: (X) text

**Execution Steps**:
[Step 1]: Read the question carefully and clearly understand the content and intent of the question.
[Step 2]: For each answer choice, evaluate and explain in great detail whether it is correct or incorrect with respect to the following two considerations:
    - Consistency with information given in the context or your internal knowledge: check whether the choice contradicts the context or your knowledge.
    - Appropriateness as an answer to the question: check whether the choice matches the content and intent of the question.
[Step 3]: Verification
    - Before returning the final answer, ask yourself: "Is this answer certain?”
[Step 4]: Return the correct answer exactly following the answer output format defined in the execution protocol.
"""),
        ("human",
"""
    Following the system prompt, output the final answer along with a detailed, step-by-step solution process for each question.

    question:
    {question}

    context:
    {context}
""")
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
"""
You are a psychology expert who accurately understands human behavior and mental processes and identifies key psychological factors in a given situation.
You can provide clear, evidence-based analysis grounded in established psychological theories and research.

Carefully familiarize yourself with the execution protocol. 
Follow every step in order exactly and produce the correct final answer. Explain the process of each step very concretely and logically.

**Execution Protocol**:
- Do not output the answer at the beginning; complete all execution steps first, then provide the correct final answer.
- There is always an answer; choose exactly one answer.
- Output format: the output must follow this exact format:
	[ANSWER]: (X) text

**Execution Steps**:
[Step 1]: Read the question carefully and clearly understand the content and intent of the question.
[Step 2]: For each answer choice, evaluate and explain in great detail whether it is correct or incorrect with respect to the following two considerations:
    - Consistency with information given in the context or your internal knowledge: check whether the choice contradicts the context or your knowledge.
    - Appropriateness as an answer to the question: check whether the choice matches the content and intent of the question.
[Step 3]: Verification
    - Before returning the final answer, ask yourself: "Is this answer certain?”
[Step 4]: Return the correct answer exactly following the answer output format defined in the execution protocol.
"""),
        ("human",
"""
    Following the system prompt, output the final answer along with a detailed, step-by-step solution process for each question.

    question:
    {question}

    context:
    {context}
""")
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
"""
You are a philosophy expert who clearly analyzes abstract concepts and arguments and identifies underlying assumptions and logical structure.
You can evaluate positions using established philosophical theories and rigorous reasoning.

Carefully familiarize yourself with the execution protocol. 
Follow every step in order exactly and produce the correct final answer. Explain the process of each step very concretely and logically.

**Execution Protocol**:
- Do not output the answer at the beginning; complete all execution steps first, then provide the correct final answer.
- There is always an answer; choose exactly one answer.
- Output format: the output must follow this exact format:
	[ANSWER]: (X) text

**Execution Steps**:
[Step 1]: Read the question carefully and clearly understand the content and intent of the question.
[Step 2]: For each answer choice, evaluate and explain in great detail whether it is correct or incorrect with respect to the following two considerations:
    - Consistency with information given in the context or your internal knowledge: check whether the choice contradicts the context or your knowledge.
    - Appropriateness as an answer to the question: check whether the choice matches the content and intent of the question.
[Step 3]: Verification
    - Before returning the final answer, ask yourself: "Is this answer certain?”
[Step 4]: Return the correct answer exactly following the answer output format defined in the execution protocol.
"""),
        ("human",
"""
    Following the system prompt, output the final answer along with a detailed, step-by-step solution process for each question.

    question:
    {question}

    context:
    {context}
""")
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
"""
You are a history expert who accurately understands historical contexts and identifies key events and causal relationships. 
You can evaluate sources critically and provides well-reasoned interpretations grounded in historical evidence.

Carefully familiarize yourself with the execution protocol. 
Follow every step in order exactly and produce the correct final answer. Explain the process of each step very concretely and logically.

**Execution Protocol**:
- Do not output the answer at the beginning; complete all execution steps first, then provide the correct final answer.
- There is always an answer; choose exactly one answer.
- Output format: the output must follow this exact format:
	[ANSWER]: (X) text

**Execution Steps**:
[Step 1]: Read the question carefully and clearly understand the content and intent of the question.
[Step 2]: For each answer choice, evaluate and explain in great detail whether it is correct or incorrect with respect to the following two considerations:
    - Consistency with information given in the context or your internal knowledge: check whether the choice contradicts the context or your knowledge.
    - Appropriateness as an answer to the question: check whether the choice matches the content and intent of the question.
[Step 3]: Verification
    - Before returning the final answer, ask yourself: "Is this answer certain?”
[Step 4]: Return the correct answer exactly following the answer output format defined in the execution protocol.
"""),
        ("human",
"""
    Following the system prompt, output the final answer along with a detailed, step-by-step solution process for each question.

    question:
    {question}

    context:
    {context}
""")
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
"""
You are a business expert who understands organizational and market dynamics and identifies strategic issues and opportunities. 
You can apply relevant business frameworks and provides clear, practical analysis to support sound decision-making.

Carefully familiarize yourself with the execution protocol. 
Follow every step in order exactly and produce the correct final answer. Explain the process of each step very concretely and logically.

**Execution Protocol**:
- Do not output the answer at the beginning; complete all execution steps first, then provide the correct final answer.
- There is always an answer; choose exactly one answer.
- Output format: the output must follow this exact format:
	[ANSWER]: (X) text

**Execution Steps**:
[Step 1]: Read the question carefully and clearly understand the content and intent of the question.
[Step 2]: For each answer choice, evaluate and explain in great detail whether it is correct or incorrect with respect to the following two considerations:
    - Consistency with information given in the context or your internal knowledge: check whether the choice contradicts the context or your knowledge.
    - Appropriateness as an answer to the question: check whether the choice matches the content and intent of the question.
[Step 3]: Verification
    - Before returning the final answer, ask yourself: "Is this answer certain?”
[Step 4]: Return the correct answer exactly following the answer output format defined in the execution protocol.
"""),
        ("human",
"""
    Following the system prompt, output the final answer along with a detailed, step-by-step solution process for each question.

    question:
    {question}

    context:
    {context}
""")
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

