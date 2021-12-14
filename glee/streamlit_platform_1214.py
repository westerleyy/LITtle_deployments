# -*- coding: utf-8 -*-
"""
24 October 2021
Last updated on Sat, Nov 26, 2021


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



with st.form(key = 'upload_form'):
    pos_data = st.file_uploader("📥 POS ", type = 'xlsx')
    pos_sheet_name = st.text_input('POS Sheet Name:')
    pos_sheet_name = str(pos_sheet_name)
    recipe_data = st.file_uploader("📥 Recipe Data", type = 'xlsx')
    recipe_sheet_name = st.text_input('Recipe Data Sheet Name:')
    recipe_sheet_name = str(recipe_sheet_name)
    stock_in_data = st.file_uploader("📥 Invoice Details Report", type = 'csv')
    category_exclusions = st.text_input('Categories to exclude from Stock-In (Comma-Separated):')
    category_exclusions = str(category_exclusions) 
    matched_ingredients_stock_in_amended = st.file_uploader("📥 Amended Ingredients to Invoice Crosswalk", type = 'csv')
    matched_recipe_pos_amended = st.file_uploader("📥 Amended Recipe items to POS Crosswalk", type = 'csv')
    matched_invoice_unit_sizes = st.file_uploader("📥 Amended Ordered Items' Unit Sizes", type = 'csv')
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

def quantity_replacement(x):
    quantity = x
    try:
        quantity = float(quantity)
    except:
        quantity = 0
    return quantity

def names_cleaning(x):
    x = x.upper()
    x = re.sub(r'\([^)]*\)', '', x)
    sevenup_exception = re.search('7 UP|7UP', x)
    if sevenup_exception:
        x = x
    else:
        x = re.sub("[^a-zA-ZéÉíÍóÓúÚáÁ ]+", "", x)
        x = ' '.join( [w for w in x.split() if len(w)>2] )
    return x

# convert quantities to ml and gr
def quantity_conversion(x):
    thousand_multiplier = ['KG', 'K', 'KS', 'KGS', 'LTR', 'LT', 'LIT', 'LITRE', 'LITER']
    ounce_multiplier = ['OZ']
    alc_multiplier = ['CL']
    cup_multiplier = ['CUP', 'CUPS']
    spoon_multiplier = ['TBLS', 'TBSP']
    soda_multiplier = ['CAN', 'BTL']
    if x['Unit of Measurement'] in thousand_multiplier:
        m = float(x['Quantity']) * 1000
    elif x['Unit of Measurement'] in ounce_multiplier:
        m = float(x['Quantity']) * 30
    elif x['Unit of Measurement'] in alc_multiplier:
        m = float(x['Quantity']) * 10
    elif x['Unit of Measurement'] in cup_multiplier:
        m = float(x['Quantity']) * 237
    elif x['Unit of Measurement'] in spoon_multiplier:
        m = float(x['Quantity']) * 15
    elif x['Unit of Measurement'] in soda_multiplier:
        m = float(x['Quantity']) * 300
    else:
        m = x['Quantity']
    return m

def multiplier_search(s):
    m = 1
    multiplier = re.search('[0-9]+[xX]| [0-9]+ [xX] | [xX] [0-9]+ | [xX][0-9+] ', s)
    if multiplier:
        interim = multiplier.group()
        interim = re.sub('[^0-9]+', "", interim)
        m = float(interim)
    return m

def unit_search(s):
    m = 1000
    s = re.sub(',', '.', s)
    common_uoms = ['[0-9]+ML', ' [0-9]+ ML', 
                   ' [0-9]+ C', '[0-9]+C', '[0-9]+CS', '\-[0-9.]+ ML', ' [0-9]+ CS',
                   '[0-9.,]+KG', ' [0-9.,]+ KG', '\-[0-9,.]+KG', '[0-9.,]+KGS', ' [0-9.,]+ KGS', '\-[0-9.,]+KGS', '[0-9.,]+KS', ' [0-9.,]+ KS',
                   '[0-9]+GR', ' [0-9]+ GR ', '[0-9]+ GR//', ' [0-9]+ GR//', '[0-9]+GMS', ' [0-9.]+ GMS', '[0-9 ]+GM', ' [0-9 ]+ GM', '[0-9]+G', ' [0-9]+ G ', '[0-9\-]+GRAMS',  ' [0-9]+ GRAMS',                        
                   '[0-9]+LB', ' [0-9]+ LB',
                   '[0-9.]+CL', ' [0-9.]+ CL',
                   '[0-9.]+LTR', ' [0-9.]+ LTR', '[0-9.]+LT','[0-9.]+L', ' [0-9.]+ LT',' [0-9.]+ L',
                   '[0-9]+GAL', ' [0-9]+ GAL', '[0-9]+GL',
                   '[0-9.]+OZ', ' [0-9.]+ OZ'
                  ]
    thousand_multiplier = ['KG', 'KGS', 'KS', 'L', 'LT', 'LTR']
    gal_multiplier = ['GAL', 'GL']
    ounce_multiplier = ['OZ']
    lb_multiplier = ['LB']
    alcohol = ['CL']
    alcohol2 = ['C']
    unit = re.search('|'.join(common_uoms), s)
    if unit:
        interim = unit.group()
        interim = re.sub('[^0-9.]+', "", interim)
        uom = re.sub('[^a-zA-Z]+', "", unit.group())
        if interim != '.':
            if uom in alcohol:
                interim = float(interim) * 10
            elif uom in thousand_multiplier:
                interim = float(interim) * 1000
            elif uom in gal_multiplier:
                interim = float(interim) * 3780
            elif uom in ounce_multiplier:
                interim = float(interim) * 30
            elif uom in lb_multiplier:
                interim = float(interim) * 454
            elif uom in alcohol2:
                interim = float(interim) * 10
            m = float(interim)
            if m == 0:
                m = 1000
    return m

def soda_multiplier(t):
    sec_multiplier = 1
    soda_identifier = ' CS'
    soda_search = re.search(soda_identifier, t)
    if soda_search:
        sec_multiplier = 24
    return sec_multiplier  

def drinks_unit_price_multiplier(d):
    multiplier = d['Unit Price']
    if d['Upload Time'] < '2021-11-22' and (d['Category'] == 'Alcoholic Beverage' or d['Category'] == 'Beverage - Alcohol' or d['Category'] == 'Alcohol Beverage' or d['Category']== 'Alc Beverage'):
        multiplier = 1.3 * d['Unit Price']
    elif d['Upload Time'] < '2021-11-22' and (d['Category'] == 'Non Alcoholic Beverage' or d['Category'] == 'Beverage - Non Alcohol' or d['Category'] == 'Beverage - Soft' or d['Category'] == 'Beverage  - Soft' or d['Category'] == 'N-Alc Beverage'):
        multiplier_exclusion = re.search('JUICE|SYRUP', d['Product Name'])
        if multiplier_exclusion:
            multiplier = 1 * d['Unit Price']
        else:
            multiplier = 1.5 * d['Unit Price']
    return multiplier  

def unit_price_adjustment(x):
    x = re.sub(r'AED ', '', x)
    x = re.sub(r',', '', x)
    x = float(x)
    return x

def est_bal_adj(x):
    quantity_consumed = x['Quantity Consumed']
    if x['Estimated Balance'] < 0:
        quantity_consumed = x['Qty'] * x['Unit Size']
    return quantity_consumed

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
    all_pos_cleaned = all_pos_cleaned.rename(columns = {'Net revenue': 'Revenue'})
    all_revenue = all_pos_cleaned['Revenue'].sum()

    
    # import recipe data
    recipe_sheet_df = pd.read_excel(recipe_data, sheet_name = recipe_sheet_name)
    recipe_sheet_df = recipe_sheet_df.dropna(how = 'all')
    batch = True
    
    ### try to look for and import batched recipes
    try:
        recipe_sheet_batch_df = pd.read_excel(recipe_data, sheet_name = 'Batch')
        recipe_sheet_batch_df = recipe_sheet_batch_df.dropna(how = 'all')
        recipe_sheet_batch_df['Quantity'] = recipe_sheet_batch_df.Quantity.apply(lambda x: quantity_replacement(str(x)))
        recipe_sheet_batch_df = recipe_sheet_batch_df.assign(
            Servings = lambda x: x.Servings.astype(int),
            Quantity = lambda y: y.Quantity.astype(float)/y.Servings
            )
        recipe_sheet_batch_df.replace(np.inf, 0, inplace = True)
        recipe_sheet_batch_df['Food Item (As per POS system)'] = recipe_sheet_batch_df.loc[:,'Food Item (As per POS system)'].str.upper()
        recipe_sheet_batch_df['Unit of Measurement'] = recipe_sheet_batch_df.loc[:,'Unit of Measurement'].str.upper()    
        recipe_sheet_batch_df['Ingredient'] = recipe_sheet_batch_df['Ingredient Ordered (if known)'].apply(lambda x: names_cleaning(str(x)))
        
        # convert all units to ml and gr
        recipe_sheet_batch_df['NewQuantity'] = recipe_sheet_batch_df.apply(quantity_conversion, axis = 1)
        recipe_sheet_batch_df.drop('Quantity', axis = 1, inplace = True)
        recipe_sheet_batch_df.rename(columns = {'NewQuantity': 'Quantity'}, inplace = True)
        
        recipe_batch_ingredients_list = recipe_sheet_batch_df['Ingredient'].drop_duplicates().tolist()
        
        # calculate max theoretical weight of batch product
        batch_volume = recipe_sheet_batch_df.groupby('Food Item (As per POS system)').sum()
        batch_volume = batch_volume.reset_index()
        batch_volume = batch_volume[['Food Item (As per POS system)', 'Quantity']].drop_duplicates()
        batch = True
    except: 
        print ('No batched recipes detected.')
        batch = False
    
    del recipe_data
    
    # ensure servings column is an int, upper case all and clean the ingredient names
    # check and replace quantity column to make sure it is an integer
    recipe_sheet_df['Quantity'] = recipe_sheet_df.Quantity.apply(lambda x: quantity_replacement(str(x)))
    
    recipe_sheet_df = recipe_sheet_df.assign(
        Servings = lambda x: x.Servings.astype(int),
        Quantity = lambda y: y.Quantity.astype(float)/y.Servings
        )
    
    # upper case all and clean the ingredient names
    recipe_sheet_df.replace(np.inf, 0, inplace = True)
    recipe_sheet_df['Food Item (As per POS system)'] = recipe_sheet_df.loc[:,'Food Item (As per POS system)'].str.upper()
    recipe_sheet_df['Unit of Measurement'] = recipe_sheet_df.loc[:,'Unit of Measurement'].str.upper()        
    recipe_sheet_df['Ingredient'] = recipe_sheet_df['Ingredient Ordered (if known)'].apply(lambda x: names_cleaning(str(x))) 
    
    ## rename new quantity column accordingly
    recipe_sheet_df['NewQuantity'] = recipe_sheet_df.apply(quantity_conversion, axis = 1)
    recipe_sheet_df.drop('Quantity', axis = 1, inplace = True)
    recipe_sheet_df.rename(columns = {'NewQuantity': 'Quantity'}, inplace = True)
    
    # get list of ingredients
    recipe_ingredients_list = recipe_sheet_df['Ingredient'].drop_duplicates().tolist()
    
    try:
        recipe_ingredients_list.extend(recipe_batch_ingredients_list)
    except:
        print('No Batched Recipes detected.')
    
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
    
    # remove trailing white spaces
    stock_in['Category'] = stock_in['Category'].apply(lambda x: x.rstrip())
    
    # remove those categories to be excluded
    if category_exclusions is not None:
        category_exclusion_split = category_exclusions.split(", ")
        category_exclusion_split = [x.rstrip() for x in category_exclusion_split]
        exclusions = ~stock_in.Category.isin(category_exclusion_split)
        stock_in = stock_in[exclusions]
        del category_exclusions
    
    # remove trailing white spaces, excessive whitespaces and other stuff
    stock_in['Product Name'] = stock_in['Product Name'].apply(lambda x: x.rstrip())

    # extracting uoms and unit sizes
    # need to convert everything from kg to g, and ltr to ml
    common_uoms = ['GM', 'KG', '[0-9]+GR', ' GR ', 'GMS', 'KGS', '[0-9]+G', '[0-9.]+ GR//', 
                   'GRAMS', ' GR ', ' G ', '[0-9.]+ GR', '[0-9.]+KS',  '[0-9.]+ GMS', '[0-9]+ KGS', 
                   '[0-9]+LB', ' [0-9]+ LB', '[A-Z0-9/-]+KGS',
                   'LTR', 'ML','[0-9]+CL', 'LT', '[0-9]+L',  '[0-9]+ML', ' [0-9.]+C', '[0-9.]+ CL', 
                   '[0-9.]+ CS', '[0-9]+GAL', ' [0-9]+ GAL', '[0-9]+OZ', ' [0-9]+ OZ' ]
    
    common_uoms_equivalent = ['GR', 'GR', 'GR', 'GR', 'GR', 'GR', 'GR', 'GR', 'GR', 'GR', 'GR', 'GR', 
                              'GR', 'GR', 'GR', 'GR', 'GR', 'GR',
                              'ML', 'ML', 'ML', 'ML', 'ML', 'ML', 'ML', 'ML', 'ML', 'ML', 'ML', 'ML', 
                              'ML']
    
    for uom in range(len(common_uoms)):
        stock_in.loc[stock_in['Product Name'].str.contains(common_uoms[uom]), 'Unit of Measurement'] = common_uoms_equivalent[uom]
    
    stock_in['Unit'] = stock_in['Unit'].apply(lambda x: re.sub("KGS|KG", "GR", x))
        

    
    # adjust unit price column
    # look for multipliers, unit size, and apply
    stock_in['Unit Price'] = stock_in['Unit Price'].apply(lambda x: unit_price_adjustment(str(x)))    
    stock_in['Unit Price'] = stock_in.apply(drinks_unit_price_multiplier, axis = 1)
    stock_in['multiplier'] = stock_in['Product Name'].apply(lambda x: multiplier_search(x))
    stock_in['unit_size'] = stock_in['Product Name'].apply(lambda x: unit_search(x))
    stock_in['soda_multiplier'] = stock_in['Product Name'].apply(lambda x: soda_multiplier(x))
    stock_in = stock_in.assign(
       unit_size = lambda x: x.unit_size * x.multiplier * x.soda_multiplier
    )
    stock_in = stock_in.rename(columns = {'unit_size':'Unit Size'})
    stock_in.drop(['multiplier', 'soda_multiplier'], axis = 1, inplace = True)    
    
    # agg orders
    stock_in_agg = stock_in[['Product Name', 'Qty', 'Unit', 'Unit Size',  'Unit Price']].copy()
    stock_in_agg['Est Total Cost'] = stock_in_agg['Qty'] * stock_in_agg['Unit Price']
    stock_in_agg.drop('Unit Price', axis = 1, inplace = True)
    stock_in_agg = stock_in_agg.groupby(['Product Name', 'Unit', 'Unit Size']).sum()
    stock_in_agg = stock_in_agg.reset_index()
              
    # remove rows with empty values
    stock_in_agg = stock_in_agg.loc[stock_in_agg['Product Name'] != '']       
               
    # allow for corrected unit size dictionary to be uploaded
    if matched_invoice_unit_sizes is not None:
        matched_invoice_unit_sizes_df = pd.read_csv(matched_invoice_unit_sizes, encoding = 'utf-8')
        matched_invoice_unit_sizes_df = matched_invoice_unit_sizes_df.fillna(0)
    
        # replace unit size 
        matched_invoice_unit_sizes_df = matched_invoice_unit_sizes_df.rename(columns = {'Unit Size': 'Corrected Unit Size'})
        stock_in_agg = stock_in_agg.merge(matched_invoice_unit_sizes_df, on = 'Product Name')
        stock_in_agg.drop('Unit Size', axis = 1, inplace = True)
        stock_in_agg = stock_in_agg.rename(columns = {'Corrected Unit Size': 'Unit Size'})
        stock_in_agg_margin = stock_in_agg.copy()
        
   
    # in the event that corrected unit size dictionary is not uploaded, the tool should be allowed to continue 
    if matched_invoice_unit_sizes is None:
        stock_in_agg_margin = stock_in_agg.copy()

            
    
    # continuation of above REGARDLESS of presence of corrected stock in data
    stock_in_agg_margin['unit_cost'] = stock_in_agg_margin['Est Total Cost'] / (stock_in_agg_margin['Unit Size'] * stock_in_agg_margin['Qty'])
    
    stock_in_agg_margin['productname_cleaned'] = stock_in_agg_margin['Product Name'].apply(lambda x: names_cleaning(str(x))) 
    
    # creating a dictionary of product names
    product_name_dictionary = stock_in_agg_margin[['productname_cleaned', 'Product Name', 'Unit Size']].copy()
    product_name_dictionary = product_name_dictionary.drop_duplicates()
    unique_product_names = product_name_dictionary['productname_cleaned'].drop_duplicates().tolist()
    
    # product name and unit size dictionary to be exported
    product_name_dictionary2 = product_name_dictionary[['Product Name', 'Unit Size']]
        
    # total cost
    total_stock_in_cost = stock_in_agg_margin['Est Total Cost'].sum()
    
    ### Crosswalk 2: Recipe Ingredients to Stock-In Report
        
      
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
        'productname_cleaned': unique_product_names,
        'Ingredient': most_similar
        })
    
    matched_ingredients_stock_in_df = matched_ingredients_stock_in_df.merge(product_name_dictionary)
    matched_ingredients_stock_in_df.drop('productname_cleaned', axis = 1, inplace = True)
    matched_ingredients_stock_in_df = matched_ingredients_stock_in_df[['Product Name', 'Ingredient']]
        

    # allow the tool to go on without amended crosswalks    
    if matched_ingredients_stock_in_amended is None and matched_recipe_pos_amended is None:
        
        # find out items ordered during the period 
        pos_sheet_cleaned_ordered = pos_sheet_df[['Article', 'Number of articles']]
        pos_sheet_cleaned_ordered = pos_sheet_cleaned_ordered.merge(matched_recipe_pos_df, left_on = 'Article', right_on = 'POS Items')
        pos_sheet_cleaned_ordered = pos_sheet_cleaned_ordered[['Article', 'Number of articles', 'Recipe Items']]
        
        # ingredient stock in crosswalk
        ingredient_stockin_recipe_qc = matched_ingredients_stock_in_df.merge(recipe_sheet_df[['Ingredient', 'Ingredient Ordered (if known)']].drop_duplicates(),
                                                                                     on = 'Ingredient')
        
        # merge with recipes df to get all components
        recipe_ordered = recipe_sheet_df.merge(pos_sheet_cleaned_ordered,
                                               left_on = 'Food Item (As per POS system)',
                                               right_on = 'Recipe Items')
           
        # calculate quantity consumed by min serving size
        ## ceiling
        recipe_ordered['Quantity Consumed'] = recipe_ordered['Number of articles']*recipe_ordered['Quantity']
    
        # add the batched consumption back into the calculation
        if batch is True:
            batched_items = batch_volume['Food Item (As per POS system)'].drop_duplicates().tolist()
            batch_volume = batch_volume.rename(columns = {'Quantity': 'Servings Volume'})
            batch_consumption = batch_volume.merge(recipe_ordered[['Food Item (As per POS system)', 'Ingredient Ordered (if known)', 'Quantity Consumed', 'Article', 'Number of articles', 'Recipe Items', 'Quantity']],
                                                   left_on = 'Food Item (As per POS system)',
                                                   right_on = 'Ingredient Ordered (if known)')
            batch_consumption['Quantity'] = batch_consumption['Quantity']/batch_consumption['Servings Volume']
            batch_consumption = batch_consumption.rename(columns = {'Quantity': 'Proportion'})
            batch_consumption['Multiplier'] = batch_consumption['Quantity Consumed']/batch_consumption['Servings Volume']
            batch_consumption.drop(['Servings Volume', 'Food Item (As per POS system)_x'], axis = 1, inplace = True)
            batch_consumption = batch_consumption.merge(recipe_sheet_batch_df,
                                                        left_on = 'Ingredient Ordered (if known)',
                                                        right_on = 'Food Item (As per POS system)')
            batch_consumption['Quantity Consumed'] = batch_consumption['Multiplier'] * batch_consumption['Quantity']
            batch_consumption['Quantity'] = batch_consumption['Quantity']*batch_consumption['Proportion']
            batch_consumption = batch_consumption.rename(columns = {
                #'Food Item (As per POS system)_y': 'Food Item (As per POS system)',
                'Ingredient Ordered (if known)_y': 'Ingredient Ordered (if known)'
                })
            batch_consumption = batch_consumption[['Food Category (if applicable)', 'Food Item (As per POS system)_y',
                                                   'Ingredient as per invoices / supplier',
                                                   'Ingredient Ordered (if known)', 'Unit of Measurement', 'Servings',
                                                   'Ingredient', 'Quantity', 'Article', 'Number of articles',
                                                   'Recipe Items', 'Quantity Consumed']]
            batch_consumption = batch_consumption.rename(columns = {
                'Food Item (As per POS system)_y': 'Food Item (As per POS system)'
                })
            recipe_ordered = recipe_ordered.append(batch_consumption, ignore_index = True)
            recipe_ordered = recipe_ordered[recipe_ordered['Ingredient Ordered (if known)'].isin(batched_items) == False]
            recipe_ordered.reset_index(inplace = True)
    
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
        matched_ingredients_stock_in_amended_df = pd.read_csv(matched_ingredients_stock_in_amended, encoding = 'utf-8')
        matched_recipe_pos_amended_df = pd.read_csv(matched_recipe_pos_amended, encoding = 'utf-8')
        
       
        # find out items ordered during the period 
        pos_sheet_cleaned_ordered = pos_sheet_df[['Article', 'Number of articles']]
        pos_sheet_cleaned_ordered = pos_sheet_cleaned_ordered.merge(matched_recipe_pos_amended_df, left_on = 'Article', right_on = 'POS Items')
        pos_sheet_cleaned_ordered = pos_sheet_cleaned_ordered[['Article', 'Number of articles', 'Recipe Items']]
        
        # ingredient stock in crosswalk
        ingredient_stockin_recipe_qc = matched_ingredients_stock_in_amended_df.merge(recipe_sheet_df[['Ingredient', 'Ingredient Ordered (if known)']].drop_duplicates(),
                                                                                     on = 'Ingredient')
        
        # merge with recipes df to get all components
        recipe_ordered = recipe_sheet_df.merge(pos_sheet_cleaned_ordered,
                                               left_on = 'Food Item (As per POS system)',
                                               right_on = 'Recipe Items')
        st.write(recipe_ordered.tail())
        st.write(recipe_ordered.shape)
           
        # calculate quantity consumed by min serving size
        ## ceiling
        recipe_ordered['Quantity Consumed'] = recipe_ordered['Number of articles']*recipe_ordered['Quantity']
    
        
        # get items that cannot be matched to recipes
        unmatched = matched_ingredients_stock_in_amended_df.loc[matched_ingredients_stock_in_amended_df.Ingredient.isna(),]
        unmatched = unmatched.merge(stock_in_agg_margin)
        unmatched = unmatched[['Product Name','Qty', 'Est Total Cost']]
    
        
        # get pos items that cannot be matched properly
        unmatched_pos = matched_recipe_pos_amended_df.loc[matched_recipe_pos_amended_df['Recipe Items'].isna(), 'POS Items']
        unmatched_pos = pd.DataFrame(unmatched_pos)
        unmatched_pos = unmatched_pos.rename(columns = {'POS Items': 'Article'})
        unmatched_pos = unmatched_pos.merge(all_pos_cleaned[['Article', 'Number of articles', 'Revenue']])

        # add the batched consumption back into the calculation
        if batch is True:
            batched_items = batch_volume['Food Item (As per POS system)'].drop_duplicates().tolist()
            batch_volume = batch_volume.rename(columns = {'Quantity': 'Servings Volume'})
            batch_consumption = batch_volume.merge(recipe_ordered[['Food Item (As per POS system)', 'Ingredient Ordered (if known)', 'Quantity Consumed', 'Article', 'Number of articles', 'Recipe Items', 'Quantity']],
                                                   left_on = 'Food Item (As per POS system)',
                                                   right_on = 'Ingredient Ordered (if known)')
            batch_consumption['Quantity'] = batch_consumption['Quantity']/batch_consumption['Servings Volume']
            batch_consumption = batch_consumption.rename(columns = {'Quantity': 'Proportion'})
            batch_consumption['Multiplier'] = batch_consumption['Quantity Consumed']/batch_consumption['Servings Volume']
            batch_consumption.drop(['Servings Volume', 'Food Item (As per POS system)_x'], axis = 1, inplace = True)
            batch_consumption = batch_consumption.merge(recipe_sheet_batch_df,
                                                        left_on = 'Ingredient Ordered (if known)',
                                                        right_on = 'Food Item (As per POS system)')
            batch_consumption['Quantity Consumed'] = batch_consumption['Multiplier'] * batch_consumption['Quantity']
            batch_consumption['Quantity'] = batch_consumption['Quantity']*batch_consumption['Proportion']
            batch_consumption = batch_consumption.rename(columns = {
                #'Food Item (As per POS system)_y': 'Food Item (As per POS system)',
                'Ingredient Ordered (if known)_y': 'Ingredient Ordered (if known)'
                })
            batch_consumption = batch_consumption[['Food Category (if applicable)', 'Food Item (As per POS system)_y',
                                                   'Ingredient as per invoices / supplier',
                                                   'Ingredient Ordered (if known)', 'Unit of Measurement', 'Servings',
                                                   'Ingredient', 'Quantity', 'Article', 'Number of articles',
                                                   'Recipe Items', 'Quantity Consumed']]
            batch_consumption = batch_consumption.rename(columns = {
                'Food Item (As per POS system)_y': 'Food Item (As per POS system)'
                })
            recipe_ordered = recipe_ordered.append(batch_consumption, ignore_index = True)
            recipe_ordered = recipe_ordered[recipe_ordered['Ingredient Ordered (if known)'].isin(batched_items) == False]
            recipe_ordered.reset_index(inplace = True)
        
        # calculate margins
        cost_calculation = recipe_ordered[['Food Item (As per POS system)', 'Ingredient', 'Quantity']].copy()
        cost_calculation = cost_calculation.merge(matched_ingredients_stock_in_amended_df, on = 'Ingredient')
        cost_calculation = cost_calculation.merge(stock_in_agg_margin[['Product Name', 'unit_cost']], on = 'Product Name')
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
    recipe_ordered['Quantity Consumed'] = recipe_ordered['Number of articles']*recipe_ordered['Quantity']
       
    # focus on just ingredients
    # group by ingredients to consolidate total quantity consumed
    ingredient_consumption = recipe_ordered[['Ingredient', 'Quantity Consumed', 'Unit of Measurement']]
    ingredient_consumption = ingredient_consumption.groupby(['Ingredient']).sum()
    ingredient_consumption = ingredient_consumption.reset_index()
    
    # track inventory by looking at inventory stock in
    # independent of whether crosswalk was corrected or not
    
    inventory_tracking = ingredient_consumption.merge(ingredient_stockin_recipe_qc)
    inventory_tracking = inventory_tracking.groupby(['Product Name']).sum()
    inventory_tracking = inventory_tracking.reset_index()
    inventory_tracking = inventory_tracking.merge(stock_in_agg_margin, how = 'right')
    inventory_tracking['Quantity Consumed'] = inventory_tracking['Quantity Consumed'].fillna(0)
       
    # # get estimated balance
    # inventory_tracking['Estimated Balance'] = (inventory_tracking['Qty'] * inventory_tracking['Unit Size']) - inventory_tracking['Quantity Consumed']
    # inventory_tracking['Quantity Consumed'] = inventory_tracking.apply(est_bal_adj, axis = 1)
    # inventory_tracking['Cost of Quantity Consumed'] = (inventory_tracking['Quantity Consumed'] / (inventory_tracking['Qty'] * inventory_tracking['Unit Size'])) * inventory_tracking['Est Total Cost']
    # inventory_tracking.replace(np.inf, 0, inplace=True)
    # inventory_tracking = inventory_tracking.fillna(0)    
        
    ## continuing cost calculation 
        
    ## slimming the DF down
    cost_of_goods_sold_narrow = cost_of_goods_sold[['Article', 'mean_constituent_cost', 'Revenue']].copy()
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.merge(pos_sheet_cleaned_ordered[['Article', 'Number of articles']])
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.rename(columns = {
        'Number of articles': 'Quantity'
        })
    
    # obtaining margin
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.assign(
        cost_pct = lambda y: round(100*(y.mean_constituent_cost/(y.Revenue/y.Quantity)),2),
        margin = lambda x: round(100*(1-x.mean_constituent_cost/(x.Revenue/x.Quantity)),2),
        tfc = lambda z: round(z.mean_constituent_cost*z.Quantity, 2)
        )
    
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.dropna()
    cost_of_goods_sold_narrow.replace(-np.inf, 0, inplace = True)
    
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.rename(columns = {
        'mean_constituent_cost': 'Food Cost per Unit',
        'Revenue': 'Total Revenue',
        'margin': 'Profit Margin Pct',
        'cost_pct': 'Food Cost Pct',
        'Quantity': 'Quantity Sold',
        'tfc': 'Theoretical Food Cost'
        })
    
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow[['Article', 'Quantity Sold', 'Total Revenue', 'Theoretical Food Cost', 'Food Cost per Unit', 'Food Cost Pct', 'Profit Margin Pct']]
    
    cost_of_goods_sold_narrow.replace(np.inf, 0, inplace = True)
    cost_of_goods_sold_narrow.replace(-np.inf, 0, inplace = True)
    
# =============================================================================
#     # calculate weighted ROI on ingredient
#     cost_of_goods_sold = cost_of_goods_sold.assign(
#         unit_revenue = lambda x: x.Revenue/x.Quantity
#         )
#     
#     summarized_cost_calculation = summarized_cost_calculation.merge(cost_of_goods_sold[['POS Items', 'unit_revenue']],
#                                                                     left_on = 'Food Item (As per POS system)',
#                                                                     right_on = 'POS Items'
#                                                                     )
#     
#     summarized_cost_calculation = summarized_cost_calculation.assign(
#         constituent_revenue = lambda x: x.unit_revenue * x.Quantity
#         )
#     summarized_cost_calculation = summarized_cost_calculation.merge(all_pos_cleaned[['Article', 'Number of articles']],
#                                                                     left_on = 'Food Item (As per POS system)',
#                                                                     right_on = 'Article'
#                                                                     )
#     summarized_cost_calculation = summarized_cost_calculation.drop_duplicates()
#     summarized_cost_calculation = summarized_cost_calculation.assign(
#         total_constituent_revenue = lambda x: x.constituent_revenue *  x['Number of articles']
#         )
#         
#     ingredient_revenue = summarized_cost_calculation.groupby('Ingredient').sum()
#     ingredient_revenue = ingredient_revenue.reset_index()
#     ingredient_revenue = ingredient_revenue[['Ingredient', 'total_constituent_revenue']]
# 
#     if matched_ingredients_stock_in_amended is not None and matched_recipe_pos_amended is not None:    
#         ingredient_revenue = ingredient_revenue.merge(matched_ingredients_stock_in_amended_df)        
#     
#     if matched_ingredients_stock_in_amended is None and matched_recipe_pos_amended is None:    
#         ingredient_revenue = ingredient_revenue.merge(matched_ingredients_stock_in_df)
#         
#     ingredient_revenue = ingredient_revenue[['Product Ordered', 'total_constituent_revenue']].copy()
#     ingredient_revenue = ingredient_revenue.rename(columns = {
#         'Product Ordered': 'Product Ordered',
#         'total_constituent_revenue': 'Attributable Revenue'
#         })
# 
#         
#     # adding weighted ROI per ingredient to inventory tracking 
#     inventory_tracking = inventory_tracking.merge(ingredient_revenue, on = 'Product Name', how = 'left')
#     inventory_tracking = inventory_tracking.assign(
#         ProfitMargin = lambda x: round(100 * (1- (x['Cost of Quantity Consumed']/x['Attributable Revenue'])), 2),
#         CostMargin = lambda y: round(100 * (y['Cost of Quantity Consumed']/y['Attributable Revenue']), 2),
#         QuantityStockedIn = lambda z: z['Unit Size'] * z['Qty']
#         )
#     inventory_tracking = inventory_tracking.rename(columns = {
#         'QuantityStockedIn': 'Quantity Stocked In',
#         'ProfitMargin': 'Profit Margin',
#         'CostMargin': 'Cost Margin'
#         })
# =============================================================================
    
#    inventory_tracking = inventory_tracking[['Product Name', 'Qty', 'Unit Size', 'Quantity Stocked In', 'Unit of Measurement', 'Quantity Consumed', 'Estimated Balance', 'Est Total Cost', 'Cost of Quantity Consumed', 'Attributable Revenue', 'Profit Margin', 'Cost Margin']]
    inventory_tracking = inventory_tracking.fillna(0)
    inventory_tracking.replace(np.inf, 0, inplace = True)
    inventory_tracking.replace(-np.inf, 0, inplace = True)
    inventory_tracking = inventory_tracking.drop_duplicates()
       
    
    total_cogs = round(cost_of_goods_sold_narrow['Theoretical Food Cost'].sum(),2)
    total_stockin_cost = round(inventory_tracking['Est Total Cost'].sum(),2)  
    total_cogs_pct = round(100*total_cogs/all_revenue, 2)
    total_stockin_cost = round(inventory_tracking['Est Total Cost'].sum(),2)
    total_stockin_pct = round(100*total_stockin_cost/all_revenue, 2)
    
    food_cost_revenue_summary = [all_revenue, total_cogs, total_cogs_pct, total_stockin_cost, total_stockin_pct]
    food_cost_revenue = ['Total Revenue', 'Theoretical Food Cost', 'Theoretical Food Cost Percentage', 'Total Stocked In', 'Total Stocked In Percentage']
    food_cost_revenue_summary_df = pd.DataFrame({
        'Cost-Revenue Summary': food_cost_revenue,
        'Cost-Revenue Values': food_cost_revenue_summary
        })
        
    # download buttons
    if st.button('Download Crosswalks as CSV'):
        # ingredients_stock_in xwalk
        tmp_download_link1a = download_link(matched_ingredients_stock_in_df, 'ingredients_stockin.csv', 'Click here to download your Ingredients to Stock-In Crosswalk!')
        st.markdown(tmp_download_link1a, unsafe_allow_html=True)
        # recipe_pos xwalk
        tmp_download_link1b = download_link(matched_recipe_pos_df, 'recipe_pos.csv', 'Click here to download your Recipe items to POS Crosswalk!')
        st.markdown(tmp_download_link1b, unsafe_allow_html=True)
        # stock in unit size
        tmp_download_link1c = download_link(product_name_dictionary2, 'invoice_details_unit_size.csv', 'Click here to download your Unit Size of Items Ordered Report!')
        st.markdown(tmp_download_link1c, unsafe_allow_html=True)
        
# =============================================================================
#     if st.button('Download Inventory Report as CSV'):
#         tmp_download_link3 = download_link(inventory_tracking, 'final_inventory.csv', 'Click here to download your Inventory Report!')
#         st.markdown(tmp_download_link3, unsafe_allow_html=True)
# =============================================================================

           
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
        tmp_download_link7 = download_link(summarized_cost_calculation, 'recipe_with_cost_of_ingredients.csv', 'Click here to download your Recipe Report with cost of ingredients!')
        st.markdown(tmp_download_link7, unsafe_allow_html = True)
        
            
# =============================================================================
#     if corrected_stock_in_data is not None:
#         
#         corrected_stock_in_data_df_transformed = corrected_stock_in_data_df.copy()
#         corrected_stock_in_data_df_transformed = corrected_stock_in_data_df_transformed.rename(columns = {
#             'Quantity Stocked In': 'QuantityStockedIn',
#             'Quantity Consumed': 'QuantityConsumed',
#             'Cost of Quantity Consumed': 'CostofQuantityConsumed',
#             'Cost of Estimated Balance': 'CostofEstimatedBalance'
#             })
#         
#         
#         corrected_stock_in_data_df_transformed = corrected_stock_in_data_df_transformed.assign(
#             QuantityStockedIn = lambda w: w['Qty'] * w['Unit Size'],
#             QuantityConsumed = lambda x: x['QuantityStockedIn'] - x['Estimated Balance'],
#             CostofQuantityConsumed = lambda y: round(y['Est Total Cost'] * y['QuantityConsumed']/y['QuantityStockedIn'], 2),
#             CostofEstimatedBalance = lambda z: z['Est Total Cost'] - z['CostofQuantityConsumed']
#             )
#         
#         corrected_stock_in_data_df_transformed = corrected_stock_in_data_df_transformed.rename(columns = {
#             'QuantityStockedIn': 'Quantity Stocked In',
#             'QuantityConsumed': 'Quantity Consumed',
#             'CostofQuantityConsumed': 'Cost of Quantity Consumed',
#             'CostofEstimatedBalance': 'Cost of Estimated Balance'
#             })
#         
#         if st.button('Download Corrected Inventory Reports as CSV'):
#             tmp_download_link7 = download_link(corrected_stock_in_data_df_transformed, 'corrected_final_inventory.csv', 'Click here to download your Final Inventory Report!')
#             st.markdown(tmp_download_link7, unsafe_allow_html=True)
#         
#         # rewrite revenue printout
#         total_cogs = sum(corrected_stock_in_data_df_transformed['Cost of Quantity Consumed'])
#         st.write("Total Revenue")
#         st.write(all_revenue)
#         st.write("Cost of Ingredients Sold")
#         st.write(total_cogs)
#         st.write('Costs as % of Revenue')
#         aggregated_margin = round(100*total_cogs/all_revenue, 2)
#         st.write(aggregated_margin)
# =============================================================================
    
    # show aggregated margins
    st.write("Total Revenue")
    st.write(all_revenue)
    st.write("Cost of Ingredients Sold")
    if total_cogs > total_stock_in_cost:
        st.write(total_stock_in_cost)
        st.write('Costs as % of Revenue')
        aggregated_margin = round(100*total_stock_in_cost/all_revenue, 2)
        st.write(aggregated_margin)
    else:
        st.write(total_cogs)
        st.write('Costs as % of Revenue')
        aggregated_margin = round(100*total_cogs/all_revenue, 2)
        st.write(aggregated_margin)
        
        

    
    
   


            

  
