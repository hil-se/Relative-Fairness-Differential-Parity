from metrics import Metrics
import pandas as pd
from pdb import set_trace

def test_against_human(model, type):
    file = model+"-"+type
    data = pd.read_csv("../results/"+file)
    # train_examples = ["JSW-1271","JSW-2478","JSW-3107","JSW-1681","JSW-2881","JSW-4768","JSW-2912","JSW-3070","JSW-3106","JSW-4765"]
    train_examples = ["JSW-1271", "JSW-2478", "JSW-3107", "JSW-1681", "JSW-2881", "JSW-4768"]
    for example in train_examples:
        data = data.loc[data["issuekey"] != example]
    m = Metrics(data["actual_storypoint"], data["predicted_storypoint"])
    d = m.DPD(data["is_internal"])
    p = m.DPT(data["is_internal"])
    rho = m.pearsonr().statistic
    rs = m.spearmanr().statistic
    result = {"Model": model+"-GT", "Prompt": type, "Pearson": "%.2f" %rho, "Spearman": "%.2f" %rs, "DP": "(%.2f) %.2f" %(p,d)}
    return result

def test_between_llms(type):
    models = ["llama-3.3-70b-versatile", "moonshotai-kimi-k2-instruct-0905"]
    dfs = []
    for model in models:
        file = model+"-"+type
        data = pd.read_csv("../results/"+file)
        train_examples = ["JSW-1271", "JSW-2478", "JSW-3107", "JSW-1681", "JSW-2881", "JSW-4768"]
        for example in train_examples:
            data = data.loc[data["issuekey"] != example]
        dfs.append(data)
    m = Metrics(dfs[1]["predicted_storypoint"], dfs[0]["predicted_storypoint"])
    d = m.DPD(dfs[0]["is_internal"])
    p = m.DPT(dfs[0]["is_internal"])
    rho = m.pearsonr().statistic
    rs = m.spearmanr().statistic
    result = {"Model": "between", "Prompt": type, "Pearson": "%.2f" %rho, "Spearman": "%.2f" %rs, "DP": "(%.2f) %.2f" %(p,d)}
    return result

if __name__ == "__main__":
    models = ["llama-3.3-70b-versatile", "moonshotai-kimi-k2-instruct-0905"]
    # types = ["predictions_zero_shot.csv", "predictions_balanced.csv", "predictions_high.csv", "predictions_low.csv"]
    types = ["predictions_zero_shot.csv", "predictions_balanced.csv"]
    results = []
    for model in models:
        for type in types:
            results.append(test_against_human(model, type))
    for type in types:
        results.append(test_between_llms(type))
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("../results/evaluation.csv", index=False)
