from pathlib import Path
import pandas as pd
import streamlit as st
from src.recommender import Recommender

st.set_page_config(page_title='Group 6 | Smartphone Recommender',page_icon='📱',layout='wide')
st.title('📱 Smartphone Recommendation System')
st.caption('Group 6 | Content-Based + Utility-Based + Hybrid')
st.info('Kaggle-derived catalog; implausible hardware and feature-phone records are screened at runtime. Prices are historical BDT listings, not live quotes or SAR.')
file=st.file_uploader('Upload real smartphone CSV (optional)',type=['csv'])
try:
 df=pd.read_csv(file) if file is not None else pd.read_csv(Path(__file__).resolve().parent/'data/phones_2026_real_bdt.csv')
 rec=Recommender(df)
 st.success(f'{len(rec.df):,} plausible smartphone records loaded from {len(df):,} source CSV rows. Prices in BDT, not SAR.')
 c1,c2,c3=st.columns(3)
 with c1:budget=st.number_input('Budget (BDT)',min_value=1.,value=30000.,step=1000.)
 with c2:priority=st.selectbox('Your main priority',['balanced','gaming','camera','battery'])
 with c3:method=st.selectbox('Recommendation method',['hybrid','content','utility'])
 reference=st.selectbox('Similar to (optional)', ['No reference']+rec.df.model.tolist())
 alpha=st.slider('Hybrid content weight',0.,1.,.5,.05)
 results=rec.recommend(budget,priority,method,top_n=5,reference=None if reference=='No reference' else reference,alpha=alpha)
 st.subheader('Top 5 recommendations')
 if results.empty:st.info('No smartphones match the budget. Increase the BDT budget.')
 else:st.dataframe(results[['brand','model','price_bdt','ram_gb','storage_gb','battery_mah','camera_mp','rating','score']],hide_index=True,use_container_width=True)
 with st.expander('Method comparison / descriptive evaluation'):
  st.json(rec.evaluate(budget,priority,5))
 st.caption('Scores are heuristic rankings, not measured accuracy. Gaming uses RAM/storage as proxies: CPU/GPU benchmark performance is NOT modeled. Historical BDT prices and parsed hardware fields require verification.')
except Exception as exc:
 st.error(f'Dataset could not be loaded: {exc}')
