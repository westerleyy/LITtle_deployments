# -*- coding: utf-8 -*-
"""
24 October 2021

Expo
"""
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import re
import base64

st.markdown("# Expo 2020")
st.markdown('from [foodrazor](https://www.foodrazor.com/) with ❤️')

st.markdown('---')

st.markdown("## 📑 Data Upload")
st.warning('🏮⚠️ *Note:* Please only upload the CSVs and XLSX provided. ⚠️🏮')

pos_data = st.file_uploader("📥 POS Data", type = 'xlsx')
pos_sheet_name = st.text_input('POS Data Sheet Name:')
pos_sheet_name = str(pos_sheet_name)
recipe_data = st.file_uploader("📥 Recipe Data", type = 'xlsx')
recipe_sheet_name = st.text_input('Recipe Data Sheet Name:')
recipe_sheet_name = str(recipe_sheet_name)
stock_in_data = st.file_uploader("📥 Stock-In Data", type = 'csv')
category_exclusions = st.text_input('Categories to exclude from Stock-In (Comma-Separated):')
category_exclusions = str(category_exclusions) 
existing_inventory = st.file_uploader("📥 Starting Balance", type = 'csv') 
matched_ingredients_stock_in_amended = st.file_uploader("📥 Amended Ingredients to Stock-In Report", type = 'csv')
matched_recipe_pos_amended = st.file_uploader("📥 Amended Recipe items to POS Report", type = 'csv')


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')
    
    toth: [Chad_Mitchell](https://discuss.streamlit.io/t/heres-a-download-function-that-works-for-dataframes-and-txt/4052)

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode("UTF-8")).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

# initialize sbert
sbert_model = SentenceTransformer('stsb-mpnet-base-v2')

st.markdown('---')

if pos_data is not None and recipe_data is not None and stock_in_data is not None and pos_sheet_name is not None and recipe_sheet_name is not None:
    # import pos data
    pos_sheet_df = pd.read_excel(pos_data, skiprows = 6, sheet_name = pos_sheet_name)
    pos_sheet_df = pos_sheet_df.dropna(how = 'all')
    
    # remove all nonsensical values
    pos_sheet_cleaned = pos_sheet_df.loc[(pos_sheet_df['Article']!= 'Total')&(pos_sheet_df['Number of articles']>0),]
    pos_sheet_cleaned['Article'] = pos_sheet_cleaned.loc[:,'Article'].str.upper()
    all_pos_cleaned = pos_sheet_df.loc[(pos_sheet_df['Article']!= 'Total'),]
    all_pos_cleaned['Article'] = all_pos_cleaned.loc[:,'Article'].str.upper()
    
    # import recipe data
    recipe_sheet_df = pd.read_excel(recipe_data, skiprows = 5, sheet_name = recipe_sheet_name)
    recipe_sheet_df = recipe_sheet_df.dropna(how = 'all')
    
    # upper case all and clean the ingredient names
    recipe_sheet_df['Food Item (As per POS system)'] = recipe_sheet_df.loc[:,'Food Item (As per POS system)'].str.upper()
    recipe_sheet_df['Ingredient_Upper'] = recipe_sheet_df.loc[:,'Ingredient Ordered (if known)'].str.upper()
    recipe_sheet_df['Ingredient_Upper'] = recipe_sheet_df.Ingredient_Upper.apply(lambda x: re.sub(r'\([^)]*\)', '', str(x)))
    recipe_sheet_df['Ingredient_Upper'] = recipe_sheet_df.Ingredient_Upper.apply(lambda x: re.sub("[^a-zA-ZéÉíÍóÓúÚáÁ ]+", " ",x))
    
    ## drop single letters
    ## drop excessive whitespace
    recipe_sheet_df['Ingredient_Upper'] = recipe_sheet_df.Ingredient_Upper.apply(lambda x: ' '.join( [w for w in x.split() if len(w)>2] ))
    recipe_sheet_df['Ingredient_Upper'] = recipe_sheet_df.Ingredient_Upper.apply(lambda x: ' '.join(x.split()))
    recipe_sheet_df['Ingredient_Upper'] = recipe_sheet_df.Ingredient_Upper.apply(lambda x: x.upper())
    
    # get list of ingredients
    recipe_ingredients_list = recipe_sheet_df['Ingredient_Upper'].drop_duplicates().tolist()
    
    # detect similar menu items across recipe and pos sheets
    
    ## get list of food items in recipe sheet
    unique_recipe_items = recipe_sheet_df['Food Item (As per POS system)'].dropna().drop_duplicates().tolist()
    
    # encode recipe items
    recipe_item_embeddings = sbert_model.encode(unique_recipe_items, convert_to_tensor = True)
    
    # get unique menu items from POS
    pos_items = all_pos_cleaned['Article'].dropna().drop_duplicates().tolist()
    
    # get a list of most similar item on the menu from recipe and pos sheets
    most_similar = []
    for item in pos_items:
        query_embedding = sbert_model.encode(item, convert_to_tensor = True)
        cos_score = util.pytorch_cos_sim(query_embedding, recipe_item_embeddings)[0]
        best_match = torch.topk(cos_score, k = 1)
        for idx in best_match[1]:
            most_similar.append(unique_recipe_items[idx])
            
    # stacking into a df
    matched_recipe_pos_df = pd.DataFrame({
        'POS Items': pos_items,
        'Most_Similar': most_similar
        })
    
       
    # import stock-in data
    stock_in = pd.read_csv(stock_in_data, encoding = 'utf-8')
      
    
    # remove nas
    stock_in = stock_in.dropna()
    
    # remove those categories to be excluded
    if category_exclusions is not None:
        category_exclusions = category_exclusions.split(", ")
        exclusions = ~stock_in.Category.isin(category_exclusions)
        stock_in = stock_in[exclusions]
    
    # remove trailing white spaces and excessive whitespaces
    stock_in['Product Name'] = stock_in['Product Name'].apply(lambda x: x.rstrip())
    stock_in['productname_cleaned'] = stock_in.productname.apply(lambda x: re.sub(r'\([^)]*\)', '', x))
    stock_in['productname_cleaned'] = stock_in.productname_cleaned.apply(lambda x: re.sub("[^a-zA-ZéÉíÍóÓúÚáÁ ]+", " ", x))
    stock_in['productname_cleaned'] = stock_in.productname_cleaned.apply(lambda x: ' '.join(x.split()))
    stock_in['productname_cleaned'] = stock_in.productname_cleaned.apply(lambda x: ' '.join( [w for w in x.split() if len(w)>2] ))
    stock_in['productname_cleaned'] = stock_in.productname_cleaned.apply(lambda x: x.upper())

    # adjust unit price column
    def unit_price_adjustment(x):
        x = re.sub(r'AED ', '', x)
        x = re.sub(r',', '', x)
        x = float(x)
        return x

    stock_in['Unit Price'] = stock_in['Unit Price'].apply(lambda x: unit_price_adjustment(str(x)))



    # agg orders
    stock_in_agg = stock_in[['productname_cleaned', 'Qty', 'Unit Price', 'unitSize']].copy()
    stock_in_agg['total_quantity'] = stock_in_agg['Qty'] * stock_in_agg['unitSize']
    stock_in_agg['total_cost'] = stock_in_agg['Qty'] * stock_in_agg['Unit Price']
    stock_in_agg = stock_in_agg[['productname_cleaned', 'total_quantity', 'total_cost', 'Qty']]
    stock_in_agg = stock_in_agg.rename(columns = {'Qty': 'quantity'})
    stock_in_agg = stock_in_agg.groupby('productname_cleaned').sum()
    stock_in_agg = stock_in_agg.reset_index()
    
    
    # import existing inventory data
    if existing_inventory is not None:
        existing_inventory_df = pd.read_csv(existing_inventory, encoding = 'utf-8')
        existing_inventory_df = existing_inventory_df.fillna(0)
        existing_inventory_df['starting_balance'] = existing_inventory_df['Actual Balance'] + existing_inventory_df['Transfers']
        existing_inventory_df['Units Ordered'] = existing_inventory_df['starting_balance'] / (existing_inventory_df['Quantity Ordered']/existing_inventory_df['Units Ordered'])
        # existing_inventory_df = existing_inventory_df[['Product Ordered', 'Actual Balance', 'Transfers', 'Total Cost', 'Units Ordered']]
        
        
        ## append to stock_in_agg if applicable, then sum again
        
        existing_inventory_df_narrow = existing_inventory_df[['Product Ordered', 'starting_balance', 'Total Cost', 'Units Ordered']]
        existing_inventory_df_narrow = existing_inventory_df_narrow.rename(columns = {
            'Product Ordered': 'productname_cleaned',
            'starting_balance': 'total_quantity',
            'Total Cost': 'total_cost',
            'Units Ordered': 'quantity'
            })
        new_stock_in_agg = pd.concat([stock_in_agg, existing_inventory_df_narrow], ignore_index = True)
        new_stock_in_agg = new_stock_in_agg.groupby('productname_cleaned').sum()
        new_stock_in_agg = new_stock_in_agg.reset_index()
        stock_in_agg = new_stock_in_agg.copy()
        
    
    # get unit cost for profit-margin analysis
    stock_in_agg_margin = stock_in_agg.copy()
    stock_in_agg_margin['unit_cost'] = stock_in_agg['total_cost'] / stock_in_agg['total_quantity']
    
    
    ## convert to list
    unique_product_names = stock_in_agg['productname_cleaned'].drop_duplicates().tolist()
    
    # encode stock-in records
    stock_in_embeddings = sbert_model.encode(recipe_ingredients_list, convert_to_tensor = True)
    
    # get a list of most similar item stocked from recipe and stock-in sheets
    most_similar = []
    for item in unique_product_names:
        query_embedding = sbert_model.encode(item, convert_to_tensor = True)
        cos_score = util.pytorch_cos_sim(stock_in_embeddings, query_embedding)[0]
        best_match = torch.topk(cos_score, k = 1)
        for score, idx in zip(best_match[0], best_match[1]):
            most_similar.append(recipe_ingredients_list[idx])
            
    # stacking into a df
    matched_ingredients_stock_in_df = pd.DataFrame({
        'productname_cleaned': unique_product_names,
        'Ingredient_Upper': most_similar
        })
    
    if matched_ingredients_stock_in_amended is not None and matched_recipe_pos_amended is not None:
        matched_ingredients_stock_in_amended_df = pd.read_csv(matched_ingredients_stock_in_amended, encoding = '1252')
        matched_recipe_pos_amended_df = pd.read_csv(matched_recipe_pos_amended, encoding = '1252')
       
        # find out items ordered during the period 
        pos_sheet_cleaned_ordered = pos_sheet_cleaned[['Article', 'Number of articles']]
        pos_sheet_cleaned_ordered = pos_sheet_cleaned_ordered.merge(matched_recipe_pos_amended_df, left_on = 'Article', right_on = 'POS Items')
        pos_sheet_cleaned_ordered = pos_sheet_cleaned_ordered[['Article', 'Number of articles', 'Most_Similar']]
       
        # merge with recipes df to get all components
        recipe_ordered = recipe_sheet_df.merge(pos_sheet_cleaned_ordered,
                                               left_on = 'Food Item (As per POS system)',
                                               right_on = 'Most_Similar')
       
        # calculate quantity consumed by min serving size
        ## ceiling
        recipe_ordered['Quantity_Consumed'] = np.ceil(recipe_ordered['Number of articles']/recipe_ordered['Servings'])*recipe_ordered['Quantity']
       
        # focus on just ingredients
        # group by ingredients to consolidate total quantity consumed
        ingredient_consumption = recipe_ordered[['Ingredient_Upper', 'Quantity_Consumed', 'Unit of Measurement']]
        ingredient_consumption = ingredient_consumption.groupby(['Ingredient_Upper']).sum()
        ingredient_consumption = ingredient_consumption.reset_index()
       
        # ingredient stock in crosswalk
        ingredient_stockin_recipe_qc = matched_ingredients_stock_in_amended_df.merge(recipe_sheet_df[['Ingredient_Upper', 'Ingredient Ordered (if known)']].drop_duplicates(),
                                                                                     on = 'Ingredient_Upper')
       
        inventory_tracking = ingredient_consumption.merge(ingredient_stockin_recipe_qc)
        inventory_tracking = inventory_tracking.groupby(['productname_cleaned']).sum()
        inventory_tracking = inventory_tracking.reset_index()
        inventory_tracking = inventory_tracking.merge(stock_in_agg)
       
        ## get estimated balance
        inventory_tracking['estimated_balance'] = inventory_tracking['total_quantity'] - inventory_tracking['Quantity_Consumed']
        inventory_tracking['Cost of Quantity Consumed'] = inventory_tracking['Quantity_Consumed'] / inventory_tracking['total_quantity'] * inventory_tracking['total_cost']
        inventory_tracking.replace(np.inf, 0, inplace=True)
        inventory_tracking['Cost of Estimated Balance'] = inventory_tracking['total_cost'] - inventory_tracking['Cost of Quantity Consumed']
        
        inventory_tracking = inventory_tracking.rename(columns = {
            'productname_cleaned': 'Product Ordered',
            'quantity': 'Units Ordered',
            'Quantity_Consumed': 'Quantity Consumed',
            'total_quantity': 'Quantity Ordered',
            'estimated_balance': 'Estimated Balance',
            'total_cost': 'Total Cost'
            })
        
        
        
        # get items that cannot be matched to recipes
        unmatched = matched_ingredients_stock_in_amended_df.loc[matched_ingredients_stock_in_amended_df.Ingredient_Upper.isna(),]
        unmatched = unmatched.merge(stock_in_agg)
        unmatched = unmatched[['productname_cleaned','quantity', 'total_quantity', 'total_cost']]
        
        unmatched = unmatched.rename(columns = {
            'productname_cleaned': 'Product Ordered',
            'quantity': 'Units Ordered',
            'total_quantity': 'Quantity Ordered',
            'total_cost': 'Total Cost'
            })
        
        # get pos items that cannot be matched properly
        unmatched_pos = matched_recipe_pos_amended_df.loc[matched_recipe_pos_amended_df.Most_Similar.isna(),'POS Items']
        unmatched_pos = pd.DataFrame(unmatched_pos)
        unmatched_pos = unmatched_pos.rename(columns = {'POS Items': 'Articles'})
        
        # calculate margins
        # matched_ingredients_stock_in_amended_df = pd.read_csv(matched_ingredients_stock_in_amended, encoding = '1252')
        cost_calculation = recipe_ordered[['Food Item (As per POS system)', 'Ingredient_Upper', 'Quantity', 'Unit of Measurement']].copy()
        cost_calculation = cost_calculation.merge(matched_ingredients_stock_in_amended_df, on = 'Ingredient_Upper')
        cost_calculation = cost_calculation.merge(stock_in_agg_margin[['productname_cleaned', 'unit_cost']], on = 'productname_cleaned')
        cost_calculation.replace(np.inf, 0, inplace = True)
        
        # cost 
        cost_calculation = cost_calculation.assign(
            constituent_cost = lambda x: x['Quantity'] * x['unit_cost']
            )

        # averaging out the cost when there is more than one possible identical ingredient
        summarized_cost_calculation = cost_calculation.groupby(['Food Item (As per POS system)', 'Ingredient_Upper', 'Quantity']).agg(
            mean_constituent_cost=('constituent_cost', 'mean'))
        
        summarized_cost_calculation = summarized_cost_calculation.reset_index()
        summarized_cost_calculation = summarized_cost_calculation.drop_duplicates()
        
        # obtaining COGS
        cost_of_goods_sold = summarized_cost_calculation.groupby('Food Item (As per POS system)').sum()
        all_pos_cleaned = all_pos_cleaned.assign(
            total_revenue = lambda x: x['Net revenue'] + x['Fees']
            )
        
        
        ## crosswalk to connect to the POS system
        cost_of_goods_sold = cost_of_goods_sold.merge(matched_recipe_pos_amended_df, 
                                                      left_on = 'Food Item (As per POS system)', 
                                                      right_on = 'Most_Similar')
        
        cost_of_goods_sold = cost_of_goods_sold.merge(all_pos_cleaned[['Article', 'total_revenue']],
                                                      left_on = 'POS Items',
                                                      right_on = 'Article')
        
        ## slimming the DF down
        cost_of_goods_sold_narrow = cost_of_goods_sold[['Article', 'mean_constituent_cost', 'total_revenue']].copy()
        
        # obtaining margin
        cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.assign(
            margin = lambda x: 100*(1-x.mean_constituent_cost/x.total_revenue)
            )
        
        cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.dropna()
        cost_of_goods_sold_narrow.replace(-np.inf, 0, inplace = True)
        
        cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.rename(columns = {
            'mean_constituent_cost': 'Cost Per Sale',
            'total_revenue': 'Revenue Per Sale',
            'margin': 'Profit Margin Pct'
            })
        
        # calculate weighted ROI on ingredient
        cost_of_goods_sold = cost_of_goods_sold.assign(
            unit_revenue = lambda x: x.total_revenue/x.Quantity
            )
        
        summarized_cost_calculation = summarized_cost_calculation.merge(cost_of_goods_sold[['POS Items', 'unit_revenue']],
                                                                        left_on = 'Food Item (As per POS system)',
                                                                        right_on = 'POS Items'
                                                                        )
        
        summarized_cost_calculation = summarized_cost_calculation.assign(
            constituent_revenue = lambda x: x.unit_revenue * x.Quantity
            )
        summarized_cost_calculation = summarized_cost_calculation.merge(all_pos_cleaned[['Article', 'Number of articles']],
                                                                        left_on = 'Food Item (As per POS system)',
                                                                        right_on = 'Article'
                                                                        )
        summarized_cost_calculation = summarized_cost_calculation.drop_duplicates()
        summarized_cost_calculation = summarized_cost_calculation.assign(
            total_constituent_revenue = lambda x: x.constituent_revenue *  x['Number of articles']
            )
        
        ingredient_revenue = summarized_cost_calculation.groupby('Ingredient_Upper').sum()
        ingredient_revenue = ingredient_revenue.reset_index()
        ingredient_revenue = ingredient_revenue[['Ingredient_Upper', 'total_constituent_revenue']]
        ingredient_revenue = ingredient_revenue.merge(matched_ingredients_stock_in_amended_df)        
        ingredient_revenue = ingredient_revenue[['productname_cleaned', 'total_constituent_revenue']].copy()
        ingredient_revenue = ingredient_revenue.rename(columns = {
            'productname_cleaned': 'Product Ordered',
            'total_constituent_revenue': 'Attributable Revenue'
            })
        
        # adding weighted ROI per ingredient to inventory tracking 
        inventory_tracking = inventory_tracking.merge(ingredient_revenue, on = 'Product Ordered')
        inventory_tracking = inventory_tracking.assign(
            ProfitMargin = lambda x: 100 * (1- (x['Cost of Quantity Consumed']/x['Attributable Revenue']))
            )
        
        # preparing inventory tracking for final export
        inventory_tracking = inventory_tracking[['Product Ordered', 'Total Cost', 'Cost of Quantity Consumed', 'Cost of Estimated Balance', 'Attributable Revenue', 'ProfitMargin', 'Units Ordered', 'Quantity Ordered', 'Quantity Consumed', 'Estimated Balance']]
        new_cols_list = ['Actual Balance', 'Transfers']
        inventory_tracking_final = inventory_tracking.reindex(columns = [*inventory_tracking.columns.tolist(), *new_cols_list])


        
        if st.button('Download Inventory Reports as CSV'):
            tmp_download_link3 = download_link(inventory_tracking_final, 'estimated_inventory.csv', 'Click here to download your Inventory Report!')
            st.markdown(tmp_download_link3, unsafe_allow_html=True)
            
            tmp_download_link4 = download_link(unmatched, 'estimated_unused_orders.csv', 'Click here to download your Unused Inventory Report!')
            st.markdown(tmp_download_link4, unsafe_allow_html=True)
            
            #tmp_download_link7 = download_link(existing_inventory_df_narrow, 'existing_inventory_df_narrow.csv', 'Click here to download your Existing Inventory Report!')
            #st.markdown(tmp_download_link7, unsafe_allow_html=True)
            
            #tmp_download_link8 = download_link(new_stock_in_agg, 'new_stock_in_agg.csv', 'Click here to download your New Stock In Report!')
            #st.markdown(tmp_download_link8, unsafe_allow_html=True)
            
            
        
        if st.button('Download Unmatched POS Articles as CSV'):
            tmp_download_link5 = download_link(unmatched_pos, 'unmatched_pos_articles.csv', 'Click here to download your Unmatched POS Articles Report!')
            st.markdown(tmp_download_link5, unsafe_allow_html = True)
            
        if st.button('Download Margin Report as CSV'):
            tmp_download_link6 = download_link(cost_of_goods_sold_narrow, 'menu_items_margins.csv', 'Click here to download your Profit Margin (Menu Item) Report!')
            st.markdown(tmp_download_link6, unsafe_allow_html = True)
            
    
    if st.button('Download Crosswalks as CSV'):
        # ingredients_stock_in xwalk
        tmp_download_link = download_link(matched_ingredients_stock_in_df, 'ingredients_stockin.csv', 'Click here to download your file matching Ingredients to Stock-In Report!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
        # recipe_pos xwalk
        tmp_download_link2 = download_link(matched_recipe_pos_df, 'recipe_pos.csv', 'Click here to download your file matching Recipe items to POS Report!')
        st.markdown(tmp_download_link2, unsafe_allow_html=True)
        
        
       
    

                
    
  
    