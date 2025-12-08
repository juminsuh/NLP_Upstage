# Team5_NLP_Upstage
[25-2 NLP] Term Project. Enhancing a QA performance of LLM (Solar Pro 2) using RAG and Prompt engineering.

### ✨Performance for 5_final.csv
---
- EWHA: 100% (25/25)
- MMLU: 76% (19/25)
- Total: 88% (44/50)

### 🛠️ Setting
---
1. Clone this repository to your local.
   ```
   git clone https://github.com/juminsuh/Team5_NLP_Upstage.git
   ```
2. Please download ```faiss_vectorstore``` folder from [google drive](https://drive.google.com/drive/u/0/folders/1XI0D3OtXREUscMaqnuW2oKZk7bM7KV6U). (😂 It might take a few minutes due to it's large size (i.e., 1.1GB))
3. Unzip ```faiss_vectorstore```.
4. Make sure to set the ```faiss_vectorstore``` directory such as ➡️ ```./Team5_NLP_Upstage/faiss_vectorstore```
5. Make `.env` file and write ```UPSTAGE_API_KEY="your_upstage_api_key"``` to ```.env``` file. (The directory of your `.env` file is ```./Team5_NLP_Upstage/.env```)



### 🔗 Requirements
---
1. Create your virual enviroment and activate it
```
conda create -n team5 python=3.10 -y # create
conda activate team5 # activate
```
2. Install
```
pip install -r requirements.txt
```

**🥳 You are ready to run the code!**


### 🔥 Implement run.py
---
```
python run.py --data_path ./datasets/testset.csv
```
➡️ If you want to evaluate other testset, then just change the value of `--data_path`.
```
python run.py --data_path <your_testsets_directory>
```
✅ You can check total score by running the code below (You should modify the directory of score.py appropriately before you run the code):
```
python score.py
```

### 📚 Source
---
We utilized textbook and QA datasets from hugging face. 
- law
 
  https://huggingface.co/datasets/ymoslem/Law-StackExchange   
  https://huggingface.co/datasets/reglab/barexam_qa

- philosophy
  
  https://huggingface.co/datasets/burgerbee/philosophy_textbook
  https://huggingface.co/datasets/burgerbee/religion_textbook
  https://huggingface.co/datasets/sayhan/strix-philosophy-qa

- business

  https://huggingface.co/datasets/theoldmandthesea/17k_business_book
  https://huggingface.co/datasets/warrencain/Business_Knowledge_Dataset_Llama_3.1_Short_Token_Pairs

- history

  https://huggingface.co/datasets/nielsprovos/world-history-1500-qa
  https://huggingface.co/datasets/burgerbee/history_textbook

- psychology

  https://huggingface.co/datasets/BoltMonkey/psychology-question-answer
  https://huggingface.co/datasets/burgerbee/psychology_textbook/viewer/default/train?row=99&views%5B%5D=train
   



