"""Smartphone recommendations: Content-Based, Utility-Based, and Hybrid.
The bundled CSV is a processed Kaggle-derived catalog. Original source rows are not included.
Hardware fields were parsed from text and should not be treated as independently verified.
All prices are BDT; no currency conversion is performed.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

ALIASES = {
 'brand':['brand','brand_name','smartphone_brand','company name','company','brand model'],
 'model':['model','model_name','model name','phone model','model name','device_name','name'],
 'price_bdt':['price_bdt','price (bdt)','price_bdt ','price'],
 'ram_gb':['ram_gb','ram','ram (gb)','ram_gb '],
 'storage_gb':['storage_gb','storage','internal storage','rom','memory_internal','internal memory'],
 'battery_mah':['battery_mah','battery capacity','battery','battery_capacity','battery capacity (mah)'],
 'camera_mp':['camera_mp','back camera','rear camera','main camera','main camera megapixels'],
 'screen_inches':['screen_inches','screen size','display size','screen size (inches)','display_size'],
 'rating':['rating','user rating','rating_score','ratings'],
 'processor':['processor','processor_name','chipset','platform_chipset'],
}
NUMERIC=['price_bdt','ram_gb','storage_gb','battery_mah','camera_mp','screen_inches','rating']
FEATURES=['ram_gb','storage_gb','battery_mah','camera_mp','screen_inches','rating']

def _number(value):
 if pd.isna(value): return np.nan
 if isinstance(value,(int,float,np.number)): return float(value)
 t=str(value).strip().replace(',','')
 if not t or t.lower() in ('nan','none','no','-'): return np.nan
 # RAM '8 GB', battery '5,000 mAh', cameras '50 MP + 8 MP'
 v=re.search(r'(?<!\w)\d+(?:\.\d+)?',t)
 return float(v.group()) if v else np.nan

def prepare(df:pd.DataFrame)->pd.DataFrame:
 keys={re.sub(r'[^a-z0-9]+',' ',str(c).lower()).strip():c for c in df.columns}
 rename={}
 for canonical,candidates in ALIASES.items():
  for name in [canonical]+candidates:
   key=re.sub(r'[^a-z0-9]+',' ',name.lower()).strip()
   if key in keys: rename[keys[key]]=canonical;break
 out=df.rename(columns=rename).copy()
 for col in ['brand','model']:
  if col not in out: raise ValueError(f'Missing required column: {col}. Check source CSV schema.')
 if 'price_bdt' not in out:
  raise ValueError('BDT price column missing. Use the provided 2025 CSV with Launched Price (USA), or add price_bdt. Do not treat other currencies as BDT.')
 for col in NUMERIC:
  if col not in out:out[col]=np.nan
  out[col]=out[col].map(_number)
 # ratings over 5 may be 0-100 scores; convert only when scale is consistent.
 if out['rating'].dropna().gt(5).any() and out['rating'].dropna().le(10).all():
  out['rating']=out['rating']/2.0
 out['brand']=out['brand'].astype(str).str.strip()
 out['model']=out['model'].astype(str).str.strip()
 out=out.replace([np.inf,-np.inf],np.nan)
 out=out.dropna(subset=['price_bdt'])
 out=out[out['price_bdt']>0].copy()
 out=out[(out['model']!='') & (out['model'].str.lower()!='nan')]
 # Domain screening: do not recommend feature phones or clearly corrupted specifications.
 # Retain the original source CSV unchanged for reproducibility.
 # Missing values can still be imputed; known implausible values are excluded.
 plausible = {
  'ram_gb': (2, 32), 'storage_gb': (16, 2048),
  'battery_mah': (2500, 20000), 'camera_mp': (2, 250),
  'screen_inches': (5.0, 9.0), 'rating': (0, 5),
 }
 for col,(minimum,maximum) in plausible.items():
  out=out[out[col].isna() | out[col].between(minimum,maximum)]
 out=out.drop_duplicates(subset=['brand','model'],keep='first')
 for col in FEATURES:
  median=out[col].median()
  if pd.isna(median):median=0
  out[col]=out[col].fillna(median)
  out[col]=out[col].clip(lower=0)
 if 'processor' not in out:out['processor']='Unknown'
 out['processor']=out['processor'].fillna('Unknown').astype(str)
 if 'source' not in out:out['source']='EXTERNAL_CSV'
 out=out.reset_index(drop=True)
 if len(out)<3:raise ValueError('Dataset must include at least three valid smartphones.')
 return out

class Recommender:
 def __init__(self, dataframe):
  self.df=prepare(dataframe)
  # Fit the same scaler once on the catalog so query and items are comparable.
  self.scaler=MinMaxScaler()
  self.feature_matrix=self.scaler.fit_transform(self.df[FEATURES])
  self.price_min=float(self.df.price_bdt.min());self.price_max=float(self.df.price_bdt.max())

 def recommend(self,budget_bdt=30000,priority='balanced',method='hybrid',top_n=5,reference=None,alpha=.5):
  if method not in ('content','utility','hybrid'):raise ValueError('method must be content, utility, or hybrid')
  if priority not in ('balanced','gaming','camera','battery'):raise ValueError('Invalid priority')
  if budget_bdt<=0:raise ValueError('Budget must be positive')
  if not 0<=alpha<=1:raise ValueError('alpha must be between 0 and 1')
  if not isinstance(top_n,int) or top_n<1:raise ValueError('top_n must be a positive integer')
  eligible=self.df[self.df.price_bdt<=budget_bdt].copy()
  if reference is not None:
   if reference not in set(self.df.model):raise ValueError('Reference model not found in dataset')
   eligible=eligible[eligible.model!=reference]
  if len(eligible)==0:return eligible.assign(content_score=[],utility_score=[],hybrid_score=[],score=[])
  weights={
    'balanced':[.22,.19,.19,.14,.10,.16],
    'gaming':[.39,.25,.12,.04,.08,.12],
    'camera':[.10,.10,.08,.45,.10,.17],
    'battery':[.12,.10,.48,.05,.10,.15],
  }[priority]
  profile={'balanced':[8,256,5000,50,6.5,4.0],'gaming':[12,256,5000,50,6.7,4.5],
           'camera':[8,256,4800,108,6.5,4.5],'battery':[8,128,6000,50,6.5,4.3]}[priority]
  if reference:
   matches=self.df.index[self.df.model==reference].tolist()
   if not matches:raise ValueError('Reference model not found in dataset')
   target=self.feature_matrix[matches[0]:matches[0]+1]
  else:
   target=self.scaler.transform(pd.DataFrame([dict(zip(FEATURES,profile))]))
  idx=eligible.index.to_numpy()
  similarities=cosine_similarity(self.feature_matrix[idx],target).ravel()
  # Educational proxy for gaming: RAM/storage only; real GPU/CPU benchmarks are absent.
  # A weighted utility score with normalized features plus a budget efficiency term.
  quality=np.dot(self.feature_matrix[idx],np.array(weights))
  affordability=np.clip(1-eligible.price_bdt.to_numpy()/budget_bdt,0,1)
  utility=.8*quality+.2*affordability
  # Normalize similarity within the eligible set to avoid near-all-ones cosine compression.
  if similarities.max()-similarities.min()>1e-10:
   sim_norm=(similarities-similarities.min())/(similarities.max()-similarities.min())
  else:sim_norm=np.ones_like(similarities)
  # utility already 0..1; no dependence on recommendation candidate rank.
  eligible['content_score']=sim_norm.round(5)
  eligible['utility_score']=utility.round(5)
  eligible['hybrid_score']=(alpha*sim_norm+(1-alpha)*utility).round(5)
  eligible['score']=eligible[{'content':'content_score','utility':'utility_score','hybrid':'hybrid_score'}[method]]
  return eligible.sort_values(['score','rating','model'],ascending=[False,False,True]).head(top_n).copy()

 def evaluate(self,budget_bdt=30000,priority='balanced',top_n=5):
  """Descriptive agreement and feature statistics; NOT ground-truth accuracy."""
  a=self.recommend(budget_bdt,priority,'content',top_n)
  b=self.recommend(budget_bdt,priority,'utility',top_n)
  c=self.recommend(budget_bdt,priority,'hybrid',top_n)
  ab=len(set(a.model)&set(b.model)); n=max(1,min(len(a),len(b)))
  return {'catalog_size':len(self.df),'eligible_size':int((self.df.price_bdt<=budget_bdt).sum()),
    'content_utility_overlap_at_k':ab/n,
    'hybrid_mean_rating':round(float(c.rating.mean()),3) if len(c) else None,
    'hybrid_mean_price_bdt':round(float(c.price_bdt.mean()),2) if len(c) else None,
    'metric_note':'Descriptive only; no user relevance ground truth / accuracy claim.'}

def load_dataset(path):return pd.read_csv(Path(path),encoding='utf-8-sig',low_memory=False)
