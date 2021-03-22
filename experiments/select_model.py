import os
import sys
import pickle
import pandas as pd

sys.path.append('..')

from models.inference import Nowcaster

df = pd.read_csv(f"../data/cases-2020-12-30.csv")
all_ltlas = df.query("`Area type` == 'ltla'")['Area name'].unique()

for ltla in all_ltlas:
    save_name = ltla.replace(" ", "_")
    path = f"drift_model_scale_1/2020-12-14/{save_name}.pickle"
    model = pickle.load(open(path, 'rb'))
    evidence = sum(model.log_evidence)

    for i in range(2,10):
        path = f"drift_model_scale_{i}/2020-12-14/{save_name}.pickle"
        if os.path.exists(path):
            model_next = pickle.load(open(path, 'rb'))
            evidence_next = sum(model_next.log_evidence)
            if evidence_next > evidence:
                model = model_next
                evidence = evidence_next

    pickle.dump(model, open(f'results/2020-12-14/{save_name}.pickle', 'wb'))
