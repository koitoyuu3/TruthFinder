
https://github.com/Zdong104/FNSPID_Financial_News_Dataset
wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip
wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv

ollama run qwen2.5:1.5b

python step1_collect_price_and_news.py
python step2_summarize_news.py --sentences 3 --show-lengths

python step3_agent_score_and_chain.py --limit 3

# Weather batch collection (US only, via api.weather.gov)
python step1_collect_weather_and_text.py --output -

python step1_collect_weather_and_text.py \
  --location "Seattle WA:47.6062,-122.3321" \
  --location "Boston MA:42.3601,-71.0589" \
  --daily-periods 4 \
  --hourly-periods 6

# CSV schema for --locations-csv:
# name,latitude,longitude
# Seattle WA,47.6062,-122.3321

# Weather news style CSV with the same 12 columns as nasdaq_external_news_300.csv
python step1_collect_weather_news_300.py
