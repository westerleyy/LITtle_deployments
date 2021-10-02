# -*- coding: utf-8 -*-
"""
01 October 2021

This is a POC of the tomatoes project using the SBERT algorithm
"""
import pandas as pd
import streamlit as st
import altair as alt
from sentence_transformers import SentenceTransformer, util
import torch
import warnings
warnings.simplefilter('ignore', FutureWarning)

st.markdown("# Operation Tomatoes")
st.markdown('from [foodrazor](https://www.foodrazor.com/) with ❤️')

st.markdown('---')

st.markdown("## 📑 Data Upload")
st.warning('🏮⚠️ *Note:* Please only upload the CSV and model provided. If you have no idea what I am talking about, then you are in the wrong place anyway. ⚠️🏮')

product_name = st.file_uploader("📥 Sample Product List", type = 'csv')
model_list = st.file_uploader("📥 Model Product List", type = 'csv')
model = st.file_uploader("🚸 Model")
query = st.text_input('Product:')
query = str(query).upper()
top_k = st.number_input('Number of Similar Products', min_value = 1, max_value = 30, step = 1)

csv_file_reader = {"csv": pd.read_csv}
model_file_reader = {"file": torch.load}

st.markdown('---')

if product_name is not None and model is not None and query is not None and model_list is not None:
    product_name_df = pd.read_csv(product_name)
    model_list_df = pd.read_csv(model_list)
    model_list_df = list(model_list_df['product'])
    saved_embedding = torch.load(model)
    del product_name, model_list, model
    
    # sbert model encoding
    sbert_model = SentenceTransformer('stsb-mpnet-base-v2')
    
    # create the encoding for the queried phrase
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    
    # We use cosine-similarity and torch.topk to find the highest 20 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, saved_embedding)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    
    # using the scores, identify the corresponding items ordered
    similar_items = []
    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 20 most similar products ordered:")
    
    for idx in top_results[1]:
        print(model_list_df[idx])
        similar_items.append(model_list_df[idx])
    
    similar_product_names = product_name_df[product_name_df.productname_cleaned.isin(similar_items)].copy()
    
    # plot the chart
    chart = alt.Chart(similar_product_names).mark_circle().encode(
        x = alt.X('productname_cleaned', title = "Product"),
        y = alt.Y('amount', title = "Price"),
        color = 'productname_cleaned',
        tooltip = ['productname_cleaned']
        ).interactive()
    
    st.altair_chart(chart, use_container_width = True)
    