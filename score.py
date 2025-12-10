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
    score = (cnt/len(answers))*100
    return score

def main():
    _, answers = read_data("./datasets/testset.csv")
    _, your_answer = read_data_for_final("./5_final.csv")
    print(your_answer)

    
    # # ewha
    # print("Ewha Scoring...")
    # ewha_acc = scoring(answers=answers[:25], responses=responses[:25])
    # # mmlu
    # print("MMLU Scoring...")
    # mmlu_acc = scoring(answers=answers[:70], responses=responses[:70])

    # total
    print("Total Scoring...")
    total_acc = scoring(answers=answers, responses=your_answer)
    
    # print(f"✅ ewha acc: {ewha_acc}")
    # print(f"✅ mmlu acc: {mmlu_acc}")
    print(f"✅ total acc: {total_acc}")
    
if __name__ == "__main__":
    main()

