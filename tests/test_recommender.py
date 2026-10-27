import pandas as pd
import pytest
from src.recommender import Recommender, prepare

def catalog():
 return pd.DataFrame({'brand':['A','B','C','D'],'model':['One','Two','Three','Four'],
  'price_bdt':[100,200,300,400],'ram_gb':[4,8,12,16],
  'storage_gb':[64,128,256,512],'battery_mah':[4000,5000,5500,6000],
  'camera_mp':[12,48,50,108],'screen_inches':[6,6.3,6.5,6.8],'rating':[4,4.2,4.6,4.8]})

@pytest.mark.parametrize('method',['content','utility','hybrid'])
def test_budget_and_order(method):
 x=Recommender(catalog()).recommend(250,method=method)
 assert len(x)==2 and (x.price_bdt<=250).all()
 assert x.score.is_monotonic_decreasing

def test_empty():assert len(Recommender(catalog()).recommend(30))==0

def test_reference_excluded():
 x=Recommender(catalog()).recommend(500,reference='One')
 assert 'One' not in set(x.model)

def test_missing_price_rejected():
 with pytest.raises(ValueError):prepare(catalog().drop(columns=['price_bdt']))


def test_outlier_and_feature_phone_screening():
 d=catalog()
 d.loc[0,'screen_inches']=1080
 d.loc[1,'ram_gb']=1
 d=pd.concat([d,pd.DataFrame([{**d.iloc[2].to_dict(),'model':'Five'}])],ignore_index=True)
 result=prepare(d)
 assert len(result)==3
 assert set(result.model)=={'Three','Four','Five'}

def test_missing_feature_median_imputation():
 d=catalog()
 d.loc[0,'camera_mp']=float('nan')
 assert prepare(d).camera_mp.notna().all()

@pytest.mark.parametrize('alpha',[-0.1,1.1])
def test_invalid_alpha(alpha):
 with pytest.raises(ValueError):Recommender(catalog()).recommend(500,alpha=alpha)

def test_unknown_reference_even_if_no_eligible():
 with pytest.raises(ValueError):Recommender(catalog()).recommend(1,reference='unknown')
