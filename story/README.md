# Relative Fairness Testing

#### Data (included in the [data/](data/) folder)

 - [Story Point Estimation](https://github.com/awsm-research/gpt2sp?tab=readme-ov-file#about-the-datasets).
   + [jirasoftware_filtered.csv](jirasoftware_filtered.csv) filtered out duplicate items, separated data into two sensitive groups: internal user stories and external defect-fixing requests.

#### Outputs from two LLMs

 - [results/](results/)
   + [llama-3.3-70b-versatile-predictions_zero_shot.csv](results/llama-3.3-70b-versatile-predictions_zero_shot.csv) is the zero-shot output from the Llama-3.3 model.
   + [llama-3.3-70b-versatile-predictions_balanced.csv](results/llama-3.3-70b-versatile-predictions_balanced.csv) is the few-shot output from the Llama-3.3 model.
   + [moonshotai-kimi-k2-instruct-0905-predictions_zero_shot.csv](results/moonshotai-kimi-k2-instruct-0905-predictions_zero_shot.csv) is the zero-shot output from the kimi-k2 model.
   + [moonshotai-kimi-k2-instruct-0905-predictions_balanced.csv](results/moonshotai-kimi-k2-instruct-0905-predictions_balanced.csv) is the few-shot output from the kimi-k2 model.

#### Usage
0. Install dependencies:
```
pip install -r requirements.txt
```
1. Navigate to the source code:
```
cd story/src
```
2. Summarize results in [results/](results/)
```
python main.py
```
3. [evaluation.csv](results/evaluation.csv) will be generated after running main.py.
4. [evaluation_formatted.csv](results/evaluation_formatted.csv) is manually formatted to be shown in the paper.


