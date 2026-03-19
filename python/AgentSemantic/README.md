
https://github.com/Zdong104/FNSPID_Financial_News_Dataset
wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip
wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv

ollama run qwen2.5:1.5b

python step1_collect_price_and_news.py
python step2_summarize_news.py --sentences 3 --show-lengths

python step3_agent_score_and_chain.py --limit 3

