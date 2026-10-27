import argparse
from src.recommender import load_dataset,Recommender
from pathlib import Path
p=argparse.ArgumentParser(description='Group 6 Smartphone Recommender')
p.add_argument('--data',default='data/phones_2026_real_bdt.csv');p.add_argument('--budget',type=float,default=30000)
p.add_argument('--priority',choices=['balanced','gaming','camera','battery'],default='gaming')
p.add_argument('--method',choices=['content','utility','hybrid'],default='hybrid')
p.add_argument('--top',type=int,default=5)
a=p.parse_args();r=Recommender(load_dataset(Path(a.data) if Path(a.data).exists() else Path(__file__).resolve().parent / a.data));out=r.recommend(a.budget,a.priority,a.method,a.top)
print(out[['model','price_bdt','ram_gb','storage_gb','battery_mah','camera_mp','score']].to_string(index=False))
print('Evaluation:',r.evaluate(a.budget,a.priority,a.top))
