from utils import *
import argparse
from tqdm import tqdm
from collections import Counter

from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./datasets/testset.csv')
args = parser.parse_args()

def ewha(llm, query_embedding, prompt, n_votes):

    search_type = "mmr"
    k = 5 
    lambda_mult = 0.3 
    fetch_k = 20 
    
    context = ewha_context(query_embedding, search_type, k, lambda_mult, fetch_k, prompt)

    answers = []
    for n in range(n_votes):
        answer = ewha_rag(prompt=prompt,
                            llm=llm,
                            context=context)
        extracted_answer = extract_answer(answer)
        print(f"➡️ {n+1}-th case votes for {extracted_answer}")
        answers.append(extracted_answer)
    # majority voting
    vote_counts = Counter(answers)
    final_answer = vote_counts.most_common(1)[0][0]
    print(f"✅ Voted result: {final_answer}")

    return final_answer

def mmlu(llm, routed_result, query_embedding, prompt):
   
    ## hyperparameter
    search_type = "mmr"
    k = 8
    lambda_mult=0.3
    fetch_k = 20
    
    context = mmlu_context(routed_result, query_embedding, search_type, k, lambda_mult, fetch_k, prompt)  
    if routed_result == "law_2":
        answer = mmlu_law_rag(prompt, context, llm)
    elif routed_result == "psychology_3":
        answer = mmlu_psychology_rag(prompt, context, llm)
    elif routed_result == "philosophy_first":
        answer = mmlu_philosophy_rag(prompt, context, llm)
    elif routed_result == "history_first":
        answer = mmlu_history_rag(prompt, context, llm)
    elif routed_result == "business_3":
        answer = mmlu_business_rag(prompt, context, llm)
        
    extracted_answer = extract_answer(answer)
    print(f"✅ answer: {extracted_answer}")

    return extracted_answer


def main(prompts):

    api_key = load_api_key()
    
    DOMAIN = {1: "ewha", 2: "law_2", 3: "history_first",
              4: "philosophy_first", 5: "psychology_3", 6: "business_3"}
    
    llm_ewha = ChatUpstage(api_key=api_key,
                      model="solar-pro2",
                      temperature=0.5) # temperature setting for voting
    
    llm_mmlu = ChatUpstage(api_key=api_key,
                      model="solar-pro2",
                      temperature=0.5
                      )

    query_embedding = UpstageEmbeddings(
        model="solar-embedding-1-large-query",
        api_key=api_key
    )
    
    answer_list = []
    domain_list = []
    for i, prompt in tqdm(enumerate(prompts)):

        print(f"\n📌 Question {i+1}")
        print("="*100)
    
        # route domain 
        domain_result = route(llm=llm_mmlu, prompt=prompt)
        if not re.match(r'^[1-6]$', domain_result):
            domain = re.findall(r"\d", domain_result)[0]
            routed_result = DOMAIN[int(domain)]
        else:
            routed_result = DOMAIN[int(domain_result)]
        
        domain_list.append(routed_result)
        print(f"📚 domain: {routed_result}")
        
        if routed_result == "ewha":
            answer = ewha(llm=llm_ewha, 
                          query_embedding=query_embedding, 
                          prompt=prompt, 
                          n_votes=5)
        else:
            answer = mmlu(llm=llm_mmlu,
                          routed_result=routed_result,
                          query_embedding=query_embedding,
                          prompt=prompt)
        answer_list.append(answer)

    return answer_list, domain_list        

if __name__ == "__main__":

    prompts, answers = read_data(args.data_path)
    responses, domains = main(prompts=prompts)
    
    data = {
        'question': prompts,
        'your_answer': responses
    }

    df = pd.DataFrame(data)
    df.to_csv('./5_final.csv', index=False, encoding='utf-8-sig')