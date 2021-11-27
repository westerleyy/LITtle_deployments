# -*- coding: utf-8 -*-
"""
24 October 2021
Last updated on Sat, Nov 27 2021


Expo
"""
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import re
import base64
from nltk import word_tokenize, pos_tag_sents
 

st.markdown("# Expo 2020")
st.markdown('from [foodrazor](https://www.foodrazor.com/) with ❤️')

st.markdown('---')

st.markdown("## 📑 Data Upload")
st.warning('🏮⚠️ *Note:* Please only upload the CSVs and XLSX provided. ⚠️🏮')




with st.form(key = 'upload_form'):
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
    corrected_stock_in_data = st.file_uploader("📥 Corrected Stock-In Data", type = 'csv')
    matched_ingredients_stock_in_amended = st.file_uploader("📥 Amended Ingredients to Stock-In Report", type = 'csv')
    matched_recipe_pos_amended = st.file_uploader("📥 Amended Recipe items to POS Report", type = 'csv')
    submit_button = st.form_submit_button(label='Submit')


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    
    toth: [Chad_Mitchell](https://discuss.streamlit.io/t/heres-a-download-function-that-works-for-dataframes-and-txt/4052)

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode("UTF-8")).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


st.markdown('---')

if pos_data is not None and recipe_data is not None and stock_in_data is not None and pos_sheet_name is not None and recipe_sheet_name is not None:
    
    # initialize sbert
    @st.cache(allow_output_mutation = True)
    def load_transformer():
        sbert_model = SentenceTransformer('stsb-mpnet-base-v2')
        return sbert_model
    
    sbert_model = load_transformer()
    
    # import pos data
    pos_sheet_df = pd.read_excel(pos_data, skiprows = 6, sheet_name = pos_sheet_name)
    pos_sheet_df = pos_sheet_df.dropna(how = 'all')
    del pos_data
    
    # remove all nonsensical values
    pos_sheet_cleaned = pos_sheet_df.loc[(pos_sheet_df['Article']!= 'Total')&(pos_sheet_df['Number of articles']>0),]
    pos_sheet_cleaned['Article'] = pos_sheet_cleaned.loc[:,'Article'].str.upper()
    all_pos_cleaned = pos_sheet_df.loc[(pos_sheet_df['Article']!= 'Total'),]
    all_pos_cleaned['Article'] = all_pos_cleaned.loc[:,'Article'].str.upper()
    
    # total revenue
    all_pos_cleaned = all_pos_cleaned.assign(
            Revenue = lambda x: x['Net revenue'] + x['Fees'] + x['Service charges']
            )
    all_revenue = all_pos_cleaned['Revenue'].sum()

    
    # import recipe data
    recipe_sheet_df = pd.read_excel(recipe_data, skiprows = 5, sheet_name = recipe_sheet_name)
    recipe_sheet_df = recipe_sheet_df.dropna(how = 'all')
    del recipe_data
    
    # adjust unit price column
    def names_cleaning(x):
        x = re.sub(r'\([^)]*\)', '', x)
        x = re.sub("[^a-zA-ZéÉíÍóÓúÚáÁ ]+", " ", x)
        x = ' '.join( [w for w in x.split() if len(w)>2] )
        x = ' '.join(x.split())
        x = x.upper()
        return x
    
    # upper case all and clean the ingredient names
    recipe_sheet_df['Food Item (As per POS system)'] = recipe_sheet_df.loc[:,'Food Item (As per POS system)'].str.upper()
    recipe_sheet_df['Ingredient'] = recipe_sheet_df.loc[:,'Ingredient Ordered (if known)'].str.upper()
    recipe_sheet_df['Ingredient'] = recipe_sheet_df.Ingredient.apply(lambda x: names_cleaning(str(x)))
    
    
    # get list of ingredients
    recipe_ingredients_list = recipe_sheet_df['Ingredient'].drop_duplicates().tolist()
    
    ## Crosswalk 1
    # detect similar menu items across recipe and pos sheets
    
    ## get list of food items in recipe sheet
    unique_recipe_items = recipe_sheet_df['Food Item (As per POS system)'].dropna().drop_duplicates().tolist()
    
    # encode recipe items
    
    @st.cache
    def recipe_item_embeddings_fn(unique_recipe_items = unique_recipe_items):
        recipe_item_embeddings = sbert_model.encode(unique_recipe_items, convert_to_tensor = True)
        return recipe_item_embeddings
    
    recipe_item_embeddings = recipe_item_embeddings_fn()
    
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
    
    del query_embedding
    
            
    # stacking into a df
    matched_recipe_pos_df = pd.DataFrame({
        'POS Items': pos_items,
        'Recipe Items': most_similar
        })
    
       
    # import stock-in data
    stock_in = pd.read_csv(stock_in_data, encoding = 'utf-8')
    del stock_in_data  
    
    # remove nas
    stock_in = stock_in.dropna()
    
    # remove those categories to be excluded
    if category_exclusions is not None:
        category_exclusion_split = category_exclusions.split(", ")
        exclusions = ~stock_in.Category.isin(category_exclusion_split)
        stock_in = stock_in[exclusions]
        del category_exclusions
    
    # remove trailing white spaces and excessive whitespaces
    stock_in['Product Name'] = stock_in['Product Name'].apply(lambda x: x.rstrip())
    
    def names_cleaning2(x):
        x = re.sub(r'\([^)]*\)', '', x)
        x = re.sub("[^a-zA-ZéÉíÍóÓúÚáÁ ]+", " ", x)
        x = ' '.join(x.split())
        x = ' '.join( [w for w in x.split() if len(w)>2] )
        x = x.upper()
        return x
    
    stock_in['productname_cleaned'] = stock_in['Product Name'].apply(lambda x: names_cleaning2(str(x))) 
    
    # creating a dictionary of product names
    product_name_dictionary = stock_in[['productname_cleaned', 'Product Name', 'Unit']].copy()
    product_name_dictionary = product_name_dictionary.drop_duplicates()
    product_name_dictionary = product_name_dictionary.rename(columns = {
        'productname_cleaned': 'Product Ordered Cleaned',
        'Unit': 'Unit of Measurement'
        })
    product_name_dictionary = product_name_dictionary.sort_values(by = ['Product Name'])
    #new_cols_list = ['Unit of Measurement']
    #product_name_dictionary = product_name_dictionary.reindex(columns = [*product_name_dictionary.columns.tolist(), *new_cols_list])
    
    
    # extracting uoms and unit sizes
    common_uoms = ['ML', 'KG', '[0-9]+GR', ' GR ', 'GMS', 'LTR', 'KGS', 'GM', '[0-9]+CL', 'LT', '[0-9]+L', '[0-9]+G', '[0-9]+ML', ' [0-9]+C', '[0-9]+ GR/', 'GRAMS', ' GR ', ' G ', '[0-9]+ GR', '[0-9]+ CL', '[0-9]+KS', '[0-9]+ CS']
    common_uoms_equivalent = ['ML', 'KG', 'GR', 'GR', 'GR', 'LTR', 'KG', 'GM', 'ML', 'LT', 'LTR', 'GR', 'ML', 'ML', 'GR', 'GR', 'GR', 'GR', 'GR', 'ML', 'KG', 'ML']
    
    for uom in range(len(common_uoms)):
        product_name_dictionary.loc[product_name_dictionary['Product Name'].str.contains(common_uoms[uom]), 'Unit of Measurement'] = common_uoms_equivalent[uom]
    
    # do pos tagging
    products_ordered_df = product_name_dictionary['Product Name'].copy().to_frame().reset_index()
    products_ordered = product_name_dictionary['Product Name'].tolist()
    tagged_products_ordered = pos_tag_sents(map(word_tokenize, products_ordered), tagset = 'universal')
    products_ordered_df['pos_tags'] = tagged_products_ordered
    
    # explode the column with pos tags
    tokens_exploded = products_ordered_df.explode('pos_tags')
    tokens_exploded[['tkn', 'tag']] = pd.DataFrame(tokens_exploded['pos_tags'].tolist(), index = tokens_exploded.index)
    
    # we just want those that contain numerals
    num_tokens = tokens_exploded[(tokens_exploded['tag'] == 'NUM')]    
    tokens = num_tokens['tkn'].tolist()    
    common_uoms = ['ML', 'KG', '[0-9]+GR', ' GR ', 'GMS', 'LTR', 'KGS', 'GM', '[0-9]+CL', 'LT', '[0-9]+L', '[0-9]+G', '[0-9]+ML', ' [0-9]+C', '[0-9]+ GR/', 'GRAMS', ' GR ', ' G ', '[0-9]+ GR', '[0-9]+ CL', '[0-9]+KS', '[0-9]+ CS']
    eligible_num_tokens_df = num_tokens.loc[num_tokens['tkn'].str.contains('|'.join(common_uoms)),]
    
    def multiplier_search(s):
        m = 1
        multiplier = re.search('[0-9.]+[xX]|[0-9.]+ [xX]', s)
        if multiplier:
            interim = multiplier.group()
            interim = re.sub('[a-zA-Z ]+', "", interim)
            m = int(interim)
        return m

    def unit_search(s):
        m = 1
        common_uoms = ['[0-9]+ML', '[0-9]+ML', ' [0-9]+C', '[0-9]+ CS',
                       '[0-9]+KG', '[0-9]+KGS', '[0-9]+KS',
                       '[0-9]+GR', '[0-9]+ GR ', '[0-9]+GMS', '[0-9]+GM', '[0-9]+G', '[0-9]+ GR/', '[0-9]+GRAMS', '[0-9+]+ G ',
                       '[0-9]+LTR', '[0-9]+LT','[0-9]+L',
                       '[0-9]+CL', '[0-9]+ CL']
        unit = re.search('|'.join(common_uoms), s)
        if unit:
            interim = unit.group()
            interim = re.sub('[a-zA-Z ]+', "", interim)
            uom = re.sub('[0-9. ]+', "", unit.group())
            if uom == 'CL':
                interim = float(interim) * 10
            m = float(interim)
        return m
    
    eligible_num_tokens_df['multiplier'] = eligible_num_tokens_df['tkn'].apply(lambda x: multiplier_search(x))
    eligible_num_tokens_df['unit_size'] = eligible_num_tokens_df['tkn'].apply(lambda x: unit_search(x))
    eligible_num_tokens_df = eligible_num_tokens_df.assign(
        unit_size = lambda x: x.unit_size * x.multiplier
    )
    eligible_num_tokens_df = eligible_num_tokens_df.rename(columns = {'unit_size':'Unit Size'})
    
    
    # merge in numerical tokens
    product_name_dictionary = product_name_dictionary.merge(eligible_num_tokens_df[['Product Name', 'Unit Size']], on = 'Product Name', how = 'left')
    
    cleaned_product_name_uom = product_name_dictionary[['Product Ordered Cleaned', 'Unit of Measurement', 'Unit Size']].copy().drop_duplicates()
    cleaned_product_name_uom = cleaned_product_name_uom['Unit Size'].fillna(1)
        
    # adjust unit price column
    def unit_price_adjustment(x):
        x = re.sub(r'AED ', '', x)
        x = re.sub(r',', '', x)
        x = float(x)
        return x
    
    stock_in['Unit Price'] = stock_in['Unit Price'].apply(lambda x: unit_price_adjustment(str(x)))
    
    # agg orders
    stock_in_agg = stock_in[['productname_cleaned', 'Qty', 'Unit Price']].copy()
    stock_in_agg['Est Total Cost'] = stock_in_agg['Qty'] * stock_in_agg['Unit Price']
    stock_in_agg = stock_in_agg[['productname_cleaned', 'Qty', 'Est Total Cost']]
    stock_in_agg = stock_in_agg.groupby('productname_cleaned').sum()
    stock_in_agg = stock_in_agg.reset_index()
    stock_in_agg = stock_in_agg.merge(cleaned_product_name_uom, left_on = 'productname_cleaned', right_on = 'Product Ordered Cleaned')
    
    # preparing stock-in report for export
    ## require cost managers to input the unit size etc    
    stock_in_agg = stock_in_agg.rename(columns = {
            'productname_cleaned': 'Product Ordered'
            })       
    
    # import existing inventory data
    if existing_inventory is not None:
        existing_inventory_df = pd.read_csv(existing_inventory, encoding = 'utf-8')
        existing_inventory_df = existing_inventory_df.fillna(0)
        existing_inventory_df = existing_inventory_df[['Product Ordered', 'Qty', 'Est Total Cost', 'Actual Balance']].copy()
        existing_inventory_df['Est Total Cost'] = existing_inventory_df['Est Total Cost']/existing_inventory_df['Qty'] * existing_inventory_df['Actual Balance']/existing_inventory_df['Unit Size']
        existing_inventory_df = existing_inventory_df.rename(columns = {
            'Est Total Cost': 'Existing Cost',
            'Actual Balance': 'Existing Actual Balance'
            })
        existing_inventory_df = existing_inventory_df[['Product Ordered', 'Existing Cost', 'Existing Actual Balance']].copy()
        stock_in_agg_final = stock_in_agg.merge(existing_inventory_df, on = 'Product Ordered', how = 'outer')
        stock_in_agg_final = stock_in_agg_final.fillna(0)
        stock_in_agg_final['Qty_Adj'] = round(stock_in_agg_final['Existing Actual Balance']/stock_in_agg_final['Unit Size'], 2)
        stock_in_agg_final = stock_in_agg_final.fillna(0)
        stock_in_agg_final['Qty_Adj'] = stock_in_agg_final['Qty_Adj'].replace(np.nan, 0) 
        stock_in_agg_final['Qty'] = stock_in_agg_final['Qty_Adj'] + stock_in_agg_final['Qty']
        stock_in_agg_final['Est Total Cost'] = round(stock_in_agg_final['Est Total Cost'] + stock_in_agg_final['Existing Cost'], 2)
        stock_in_agg_final = stock_in_agg_final[['Product Ordered', 'Qty', 'Est Total Cost', 'Unit Size', 'Unit of Measurement']]
        new_cols_list = ['Actual Balance','Transfers', 'Estimated Wastage']
        stock_in_agg_final = stock_in_agg_final.reindex(columns = [*stock_in_agg_final.columns.tolist(), *new_cols_list])
        
    if existing_inventory is None:
        new_cols_list = ['Unit Size', 'Actual Balance', 'Transfers', 'Estimated Wastage']
        stock_in_agg_final = stock_in_agg.reindex(columns = [*stock_in_agg.columns.tolist(), *new_cols_list])
        
    # remove rows with empty values
    stock_in_agg_final = stock_in_agg_final.loc[stock_in_agg_final['Product Ordered'] != '']       
               
    # allow for corrected stock in data to be uploaded
    if corrected_stock_in_data is not None:
        corrected_stock_in_data_df = pd.read_csv(corrected_stock_in_data, encoding = 'utf-8')
        corrected_stock_in_data_df = corrected_stock_in_data_df.fillna(0)
        del corrected_stock_in_data
    
        # get unit cost for profit-margin analysis
        stock_in_agg_margin = corrected_stock_in_data_df.copy()
    
    # in the event that corrected stock in data is not uploaded, the tool should be allowed to continue 
    if correct_stock_in_data is None:
        stock_in_agg_margin = stock_in_agg_final.copy()
        
    # continuation of above REGARDLESS of presence of corrected stock in data
    stock_in_agg_margin['unit_cost'] = stock_in_agg_margin['Est Total Cost'] / (stock_in_agg_margin['Unit Size'] * stock_in_agg_margin['Qty'])
        
    # total cost
    total_stock_in_cost = stock_in_agg_margin['Est Total Cost'].sum()
    
    ### Crosswalk 2: Recipe Ingredients to Stock-In Report
        
    ## convert to list
    unique_product_names = corrected_stock_in_data_df['Product Ordered'].drop_duplicates().tolist()
        
    # encode stock-in records
    @st.cache
    def stock_in_embeddings_fn(recipe_ingredients_list = recipe_ingredients_list):
        stock_in_embeddings = sbert_model.encode(recipe_ingredients_list, convert_to_tensor = True)
        return stock_in_embeddings
               
    stock_in_embeddings = stock_in_embeddings_fn()
        
        
    # get a list of most similar item stocked from recipe and stock-in sheets
    most_similar = []
    for item in unique_product_names:
        query_embedding = sbert_model.encode(item, convert_to_tensor = True)
        cos_score = util.pytorch_cos_sim(query_embedding, stock_in_embeddings)[0]
        best_match = torch.topk(cos_score, k = 1)
        for idx in best_match[1]:
            most_similar.append(recipe_ingredients_list[idx])
        
    del query_embedding
                
    # stacking into a df
    matched_ingredients_stock_in_df = pd.DataFrame({
        'Product Ordered': unique_product_names,
        'Ingredient': most_similar
        })
        
    # allow the tool to go on without amended crosswalks    
    if matched_ingredients_stock_in_amended is None and matched_recipe_pos_amended is None:
        
        # find out items ordered during the period 
        pos_sheet_cleaned_ordered = pos_sheet_cleaned[['Article', 'Number of articles', 'Total due amount']]
        pos_sheet_cleaned_ordered = pos_sheet_cleaned_ordered.merge(matched_recipe_pos_df, left_on = 'Article', right_on = 'POS Items')
        pos_sheet_cleaned_ordered = pos_sheet_cleaned_ordered[['Article', 'Number of articles', 'Recipe Items']]
        
        # ingredient stock in crosswalk
        ingredient_stockin_recipe_qc = matched_ingredients_stock_in_df.merge(recipe_sheet_df[['Ingredient', 'Ingredient Ordered (if known)']].drop_duplicates(),
                                                                                     on = 'Ingredient')
        
        # calculate margins
        cost_calculation = recipe_ordered[['Food Item (As per POS system)', 'Ingredient', 'Quantity', 'Unit of Measurement']].copy()
        cost_calculation = cost_calculation.merge(matched_ingredients_stock_in_df, on = 'Ingredient')
        cost_calculation = cost_calculation.merge(stock_in_agg_margin[['Product Ordered', 'unit_cost']], on = 'Product Ordered')
        cost_calculation.replace(np.inf, 0, inplace = True)
                
        # cost 
        cost_calculation = cost_calculation.assign(
            constituent_cost = lambda x: x['Quantity'] * x['unit_cost']
            )
        
        # averaging out the cost when there is more than one possible identical ingredient
        summarized_cost_calculation = cost_calculation.groupby(['Food Item (As per POS system)', 'Ingredient', 'Quantity']).agg(
            mean_constituent_cost=('constituent_cost', 'mean'))
        
        summarized_cost_calculation = summarized_cost_calculation.reset_index()
        summarized_cost_calculation = summarized_cost_calculation.drop_duplicates()
        
        # obtaining COGS
        cost_of_goods_sold = summarized_cost_calculation.groupby('Food Item (As per POS system)').sum()
        
        
        ## crosswalk to connect to the POS system
        cost_of_goods_sold = cost_of_goods_sold.merge(matched_recipe_pos_df, 
                                                      left_on = 'Food Item (As per POS system)', 
                                                      right_on = 'Recipe Items')
        
        cost_of_goods_sold = cost_of_goods_sold.merge(all_pos_cleaned[['Article', 'Revenue']],
                                                      left_on = 'POS Items',
                                                      right_on = 'Article')
    
    if matched_ingredients_stock_in_amended is not None and matched_recipe_pos_amended is not None:
        matched_ingredients_stock_in_amended_df = pd.read_csv(matched_ingredients_stock_in_amended, encoding = '1252')
        matched_recipe_pos_amended_df = pd.read_csv(matched_recipe_pos_amended, encoding = '1252')
        
       
        # find out items ordered during the period 
        pos_sheet_cleaned_ordered = pos_sheet_cleaned[['Article', 'Number of articles', 'Total due amount']]
        pos_sheet_cleaned_ordered = pos_sheet_cleaned_ordered.merge(matched_recipe_pos_amended_df, left_on = 'Article', right_on = 'POS Items')
        pos_sheet_cleaned_ordered = pos_sheet_cleaned_ordered[['Article', 'Number of articles', 'Recipe Items']]
        
        # ingredient stock in crosswalk
        ingredient_stockin_recipe_qc = matched_ingredients_stock_in_amended_df.merge(recipe_sheet_df[['Ingredient', 'Ingredient Ordered (if known)']].drop_duplicates(),
                                                                                     on = 'Ingredient')
        
        # get items that cannot be matched to recipes
        unmatched = matched_ingredients_stock_in_amended_df.loc[matched_ingredients_stock_in_amended_df.Ingredient.isna(),]
        unmatched = unmatched.merge(stock_in_agg_margin)
        unmatched = unmatched[['Product Ordered','Qty', 'Est Total Cost']]
    
        
        # get pos items that cannot be matched properly
        unmatched_pos = matched_recipe_pos_amended_df.loc[matched_recipe_pos_amended_df['Recipe Items'].isna(), 'POS Items']
        unmatched_pos = pd.DataFrame(unmatched_pos)
        unmatched_pos = unmatched_pos.rename(columns = {'POS Items': 'Article'})
        unmatched_pos = unmatched_pos.merge(all_pos_cleaned[['Article', 'Number of articles', 'Revenue']])
        
        # calculate margins
        cost_calculation = recipe_ordered[['Food Item (As per POS system)', 'Ingredient', 'Quantity', 'Unit of Measurement']].copy()
        cost_calculation = cost_calculation.merge(matched_ingredients_stock_in_amended_df, on = 'Ingredient')
        cost_calculation = cost_calculation.merge(stock_in_agg_margin[['Product Ordered', 'unit_cost']], on = 'Product Ordered')
        cost_calculation.replace(np.inf, 0, inplace = True)
                
        ## cost 
        cost_calculation = cost_calculation.assign(
            constituent_cost = lambda x: x['Quantity'] * x['unit_cost']
            )
        
        # averaging out the cost when there is more than one possible identical ingredient
        summarized_cost_calculation = cost_calculation.groupby(['Food Item (As per POS system)', 'Ingredient', 'Quantity']).agg(
            mean_constituent_cost=('constituent_cost', 'mean'))
        
        summarized_cost_calculation = summarized_cost_calculation.reset_index()
        summarized_cost_calculation = summarized_cost_calculation.drop_duplicates()
        
        # obtaining COGS
        cost_of_goods_sold = summarized_cost_calculation.groupby('Food Item (As per POS system)').sum()
                       
        ## crosswalk to connect to the POS system
        cost_of_goods_sold = cost_of_goods_sold.merge(matched_recipe_pos_amended_df, 
                                                      left_on = 'Food Item (As per POS system)', 
                                                      right_on = 'Recipe Items')
        
        cost_of_goods_sold = cost_of_goods_sold.merge(all_pos_cleaned[['Article', 'Revenue']],
                                                      left_on = 'POS Items',
                                                      right_on = 'Article')
        
        
        
       
    # merge with recipes df to get all components
    recipe_ordered = recipe_sheet_df.merge(pos_sheet_cleaned_ordered,
                                           left_on = 'Food Item (As per POS system)',
                                           right_on = 'Recipe Items')
       
    # calculate quantity consumed by min serving size
    ## ceiling
    recipe_ordered['Quantity Consumed'] = np.ceil(recipe_ordered['Number of articles']/recipe_ordered['Servings'])*recipe_ordered['Quantity']
       
    # focus on just ingredients
    # group by ingredients to consolidate total quantity consumed
    ingredient_consumption = recipe_ordered[['Ingredient', 'Quantity Consumed', 'Unit of Measurement']]
    ingredient_consumption = ingredient_consumption.groupby(['Ingredient']).sum()
    ingredient_consumption = ingredient_consumption.reset_index()
    
    # track inventory by looking at inventory stock in
    # independent of whether crosswalk was corrected or not
    
    inventory_tracking = ingredient_consumption.merge(ingredient_stockin_recipe_qc)
    inventory_tracking = inventory_tracking.groupby(['Product Ordered']).sum()
    inventory_tracking = inventory_tracking.reset_index()
    inventory_tracking = inventory_tracking.merge(stock_in_agg_margin, how = 'right')
       
    ## get estimated balance
    inventory_tracking['Estimated Balance'] = (inventory_tracking['Qty'] * inventory_tracking['Unit Size']) - inventory_tracking['Quantity Consumed']
    inventory_tracking['Estimated Actual Balance Difference'] = inventory_tracking['Estimated Balance'] - inventory_tracking['Actual Balance']
    inventory_tracking['Cost of Quantity Consumed'] = (inventory_tracking['Quantity Consumed'] / (inventory_tracking['Qty'] * inventory_tracking['Unit Size'])) * inventory_tracking['Est Total Cost']
    inventory_tracking.replace(np.inf, 0, inplace=True)
    inventory_tracking['Cost of Estimated Balance'] = inventory_tracking['Est Total Cost'] - inventory_tracking['Cost of Quantity Consumed']    
        
    ############# BREAK POINT 1015hrs ################
    ## continuing cost calculation 
        
    ## slimming the DF down
    cost_of_goods_sold_narrow = cost_of_goods_sold[['Article', 'mean_constituent_cost', 'Revenue']].copy()
    
    # obtaining margin
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.assign(
        cost_pct = lambda y: round(100*(y.mean_constituent_cost/y.Revenue),2),
        margin = lambda x: round(100*(1-x.mean_constituent_cost/x.Revenue),2)
        )
    
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.dropna()
    cost_of_goods_sold_narrow.replace(-np.inf, 0, inplace = True)
    
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.rename(columns = {
        'mean_constituent_cost': 'Cost Per Sale',
        'Revenue': 'Revenue Per Sale',
        'margin': 'Profit Margin Pct',
        'cost_pct': 'Cost of Production Pct'
        })
    
    # calculate weighted ROI on ingredient
    cost_of_goods_sold = cost_of_goods_sold.assign(
        unit_revenue = lambda x: x.Revenue/x.Quantity
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
        
    

    if matched_ingredients_stock_in_amended is not None and matched_recipe_pos_amended is not None:    
        ingredient_revenue = summarized_cost_calculation.groupby('Ingredient').sum()
        ingredient_revenue = ingredient_revenue.reset_index()
        ingredient_revenue = ingredient_revenue[['Ingredient', 'total_constituent_revenue']]
        ingredient_revenue = ingredient_revenue.merge(matched_ingredients_stock_in_amended_df)        
        ingredient_revenue = ingredient_revenue[['Product Ordered', 'total_constituent_revenue']].copy()
        ingredient_revenue = ingredient_revenue.rename(columns = {
            'Product Ordered': 'Product Ordered',
            'total_constituent_revenue': 'Attributable Revenue'
            })
    
    if matched_ingredients_stock_in_amended is None and matched_recipe_pos_amended is None:    
        ingredient_revenue = summarized_cost_calculation.groupby('Ingredient').sum()
        ingredient_revenue = ingredient_revenue.reset_index()
        ingredient_revenue = ingredient_revenue[['Ingredient', 'total_constituent_revenue']]
        ingredient_revenue = ingredient_revenue.merge(matched_ingredients_stock_in_df)        
        ingredient_revenue = ingredient_revenue[['Product Ordered', 'total_constituent_revenue']].copy()
        ingredient_revenue = ingredient_revenue.rename(columns = {
            'Product Ordered': 'Product Ordered',
            'total_constituent_revenue': 'Attributable Revenue'
            })

        
    # adding weighted ROI per ingredient to inventory tracking 
    inventory_tracking = inventory_tracking.merge(ingredient_revenue, on = 'Product Ordered', how = 'left')
    inventory_tracking = inventory_tracking.assign(
        ProfitMargin = lambda x: round(100 * (1- (x['Cost of Quantity Consumed']/x['Attributable Revenue'])), 2),
        CostMargin = lambda y: round(100 * (y['Cost of Quantity Consumed']/y['Attributable Revenue']), 2)
        )
    
    inventory_tracking = inventory_tracking[['Product Ordered', 'Qty', 'Unit Size', 'Quantity Consumed', 'Estimated Balance', 'Actual Balance', 'Estimated Actual Balance Difference', 'Transfers', 'Estimated Wastage', 'Est Total Cost', 'Cost of Quantity Consumed', 'Cost of Estimated Balance', 'Attributable Revenue', 'ProfitMargin', 'CostMargin']]
    total_cogs = round(inventory_tracking['Cost of Quantity Consumed'].sum(),2)
        
        
    # download buttons
    if st.button('Download Inventory Reports as CSV'):
        tmp_download_link2 = download_link(stock_in_agg_final, 'estimated_inventory.csv', 'Click here to download your Estimated Inventory Report!')
        st.markdown(tmp_download_link2, unsafe_allow_html=True)
        tmp_download_link2b = download_link(product_name_dictionary, 'product_name_dictionary.csv', 'Click here to download your Inventory Product Name Dictionary!')
        st.markdown(tmp_download_link2b, unsafe_allow_html=True)
        tmp_download_link3 = download_link(inventory_tracking, 'final_inventory.csv', 'Click here to download your Final Inventory Report!')
        st.markdown(tmp_download_link3, unsafe_allow_html=True)

           
    if matched_ingredients_stock_in_amended is not None and matched_recipe_pos_amended is not None:        
        if st.button('Download Unmatched POS Articles and Unused Inventory as CSV'):
            tmp_download_link4 = download_link(unmatched, 'estimated_unused_orders.csv', 'Click here to download your Unused Inventory Report!')
            st.markdown(tmp_download_link4, unsafe_allow_html=True)
            tmp_download_link5 = download_link(unmatched_pos, 'unmatched_pos_articles.csv', 'Click here to download your Unmatched POS Articles Report!')
            st.markdown(tmp_download_link5, unsafe_allow_html = True)
            
    del matched_ingredients_stock_in_amended, matched_recipe_pos_amended
        
    if st.button('Download Margin Report as CSV'):
        tmp_download_link6 = download_link(cost_of_goods_sold_narrow, 'menu_items_margins.csv', 'Click here to download your Profit Margin (Menu Item) Report!')
        st.markdown(tmp_download_link6, unsafe_allow_html = True)
        
    if st.button('Download Crosswalks as CSV'):
        # ingredients_stock_in xwalk
        tmp_download_link1a = download_link(matched_ingredients_stock_in_df, 'ingredients_stockin.csv', 'Click here to download your file matching Ingredients to Stock-In Report!')
        st.markdown(tmp_download_link1a, unsafe_allow_html=True)
        # recipe_pos xwalk
        tmp_download_link1b = download_link(matched_recipe_pos_df, 'recipe_pos.csv', 'Click here to download your file matching Recipe items to POS Report!')
        st.markdown(tmp_download_link1b, unsafe_allow_html=True)
        
    # show aggregated margins
    st.write("Total Revenue")
    st.write(all_revenue)
    st.write("Cost of Ingredients Sold")
    st.write(total_cogs)
    st.write('Costs as % of Revenue')
    aggregated_margin = round(100*total_cogs/all_revenue, 2)
    st.write(aggregated_margin)
            
        

    
    
   


            

  
