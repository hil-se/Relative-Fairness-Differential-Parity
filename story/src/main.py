from metrics import Metrics
import pandas as pd
from pdb import set_trace

def test(file):
    data = pd.read_csv("../results/"+file)
    train_examples = ["JSW-1271","JSW-2478","JSW-3107","JSW-1681","JSW-2881","JSW-4768","JSW-2912","JSW-3070","JSW-3106","JSW-4765"]
    for example in train_examples:
        data = data.loc[data["issuekey"] != example]
    m = Metrics(data["actual_storypoint"], data["predicted_storypoint"])
    d = m.DPD(data["is_internal"])
    p = m.DPT(data["is_internal"])
    rho = m.pearsonr().statistic
    rs = m.spearmanr().statistic
    result = {"File": file, "Pearson": "%.2f" %rho, "Spearman": "%.2f" %rs, "DP": "(%.2f) %.2f" %(p,d)}
    return result

if __name__ == "__main__":
    files = ["predictions_zero_shot.csv", "predictions_balanced.csv", "predictions_high.csv", "predictions_low.csv"]
    results = []
    for file in files:
        results.append(test(file))
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("../results/evaluation.csv", index=False)
