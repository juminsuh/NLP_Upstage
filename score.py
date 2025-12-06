from utils import *

def scoring(answers, responses):

    cnt = 0
    mistake = []
    for i, (answer, response) in enumerate(zip(answers, responses)):
        print("-"*10)
        generated_answer = response
        # check
        if generated_answer:
            print(f"{i+1}. generated answer: {generated_answer}, answer: {answer}")
        else:
            print(f"{i+1}. extraction fail")

        if generated_answer == None:
            continue
        if generated_answer in answer:
            cnt += 1
        else:
            mistake.append(i+1)

    print()
    num = len(answers)
    score = (cnt/num)*100
    return score

def main():
    _, answers = read_data("./datasets/testset.csv")
    _, responses = read_data("./5_final.csv")
    
    # ewha
    print("Ewha Scoring...")
    ewha_acc = scoring(answers=answers[:25], responses=responses[:25])
    # mmlu
    print("MMLU Scoring...")
    mmlu_acc = scoring(answers=answers[25:], responses=responses[25:])
    # total
    print("Total Scoring...")
    total_acc = scoring(answers=answers, responses=responses)
    
    print(f"✅ ewha acc: {ewha_acc}")
    print(f"✅ mmlu acc: {mmlu_acc}")
    print(f"✅ total acc: {total_acc}")
    
if __name__ == "__main__":
    main()

