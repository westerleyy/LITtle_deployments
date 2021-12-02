# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 22:04:18 2021
The idea behind this script is to process the Glee reports as fast as we can without going through Streamlit.
This script will follow the original preferred workflow of getting staff to check the crosswalks and unit sizes first before moving to calculate balances.
@author: wesch
"""
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

path = r'C:\Users\wesch\OneDrive\Documents\FoodRazor\expo'


# data import
## POS
pos_sheet_df = pd.read_excel(path + "\data\POS\Jubilee_Adrift Sales.xlsx", skiprows = 6, sheet_name = "Scarpetta")
pos_sheet_df = pos_sheet_df.dropna(how = 'all')

## RECIPE
recipe_sheet_df = pd.read_excel(path + "\data\Recipe\Scarpetta_Recipe.xlsx", skiprows = 5, sheet_name = 'Food & Bev menu_Wes')
recipe_sheet_df = recipe_sheet_df.dropna(how = 'all')
batch = False
### try to look for and import batched recipes
try:
    recipe_sheet_batch_df = pd.read_excel(path, sheet_name = 'Batch')
    recipe_sheet_batch_df = recipe_sheet_batch_df.dropna()
    batch = True
except: 
    print ('No batched recipes detected.')

## STOCK-IN
stock_in = pd.read_csv(path + "\data\Stock_In\FoodRazor_App_StockIn_Scarpetta.csv", encoding = 'utf-8')
stock_in = stock_in.dropna()

# initialize sbert
def load_transformer():
    sbert_model = SentenceTransformer('stsb-mpnet-base-v2')
    return sbert_model

sbert_model = load_transformer()

## POS DATA 
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

## RECIPE DATA
# function to clean ingredient names and stock-in names
def names_cleaning(x):
    x = x.upper()
    x = re.sub(r'\([^)]*\)', '', x)
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
        m = float(x['Quantity']) * 330
    else:
        m = x['Quantity']
    return m

# ensure servings column is an int, upper case all and clean the ingredient names
recipe_sheet_df = recipe_sheet_df.assign(
    Servings = lambda x: x.Servings.astype(int)
    )
recipe_sheet_df['Food Item (As per POS system)'] = recipe_sheet_df.loc[:,'Food Item (As per POS system)'].str.upper()
recipe_sheet_df['Unit of Measurement'] = recipe_sheet_df.loc[:,'Unit of Measurement'].str.upper()    
recipe_sheet_df['Ingredient'] = recipe_sheet_df['Ingredient Ordered (if known)'].apply(lambda x: names_cleaning(str(x)))
 
# convert all units to ml and gr
recipe_sheet_df['NewQuantity'] = recipe_sheet_df.apply(quantity_conversion, axis = 1)
recipe_sheet_df.drop('Quantity', axis = 1, inplace = True)
recipe_sheet_df.rename(columns = {'NewQuantity': 'Quantity'}, inplace = True)

# get list of ingredients
recipe_ingredients_list = recipe_sheet_df['Ingredient'].drop_duplicates().tolist()

# handling of batched ingredients
if batch is True:
    # ensure servings column is an int, upper case all and clean the ingredient names
    recipe_sheet_batch_df = recipe_sheet_batch_df.assign(
        Servings = lambda x: x.Servings.astype(int)
        )
    recipe_sheet_batch_df['Food Item (As per POS system)'] = recipe_sheet_batch_df.loc[:,'Food Item (As per POS system)'].str.upper()
    recipe_sheet_batch_df['Unit of Measurement'] = recipe_sheet_batch_df.loc[:,'Unit of Measurement'].str.upper()    
    recipe_sheet_batch_df['Ingredient'] = recipe_sheet_batch_df['Ingredient Ordered (if known)'].apply(lambda x: names_cleaning(str(x)))
     
    # convert all units to ml and gr
    recipe_sheet_batch_df['NewQuantity'] = recipe_sheet_batch_df.apply(quantity_conversion, axis = 1)
    recipe_sheet_batch_df.drop('Quantity', axis = 1, inplace = True)
    recipe_sheet_batch_df.rename(columns = {'NewQuantity': 'Quantity'}, inplace = True)
    
    # get list of ingredients
    recipe_batch_ingredients_list = recipe_sheet_batch_df['Ingredients'].drop_duplicates().tolist()
    
    # append to recipe ingredients list
    recipe_ingredients_list.extend(recipe_batch_ingredients_list)

## Crosswalk 1
# detect similar menu items across recipe and pos sheets

## get list of food items in recipe sheet
unique_recipe_items = recipe_sheet_df['Food Item (As per POS system)'].dropna().drop_duplicates().tolist()

# encode recipe items    
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

## STOCK-IN DATA
# remove those categories to be excluded
category_exclusions = ["Printing & Stationary", "Printing and Stationery Supplies",  "Tax Adjustment", "CAPEX", "Other",
                        "Cleaning", "Cleaning Supplies", "Kitchen Supplies", "Discount", "Guest Supplies", "Rounding",
                        "General Supplies", "Packaging", "Bar Expenses", "Operating Supplies General",
                        "Cleaning & Chemical", "Utilities", "Music & entertainment", "Payroll & Related Expenses",
                        "PR & Marketing", "Payroll Provision (Guest Chefs)", "Transport", "Accommodation & Air Tickets",
                        "OS&E - Kitchen", "OS&E - FOH", "Supplies Kitchen", "Supplies Others", "Supplies Cleaning",
                        "Pre-Opening Printing and Stationery Supplies",
                        "Pre-Opening - Kitchenware",
                        "Pre-Opening - Payroll / HR Related", 
                        "Pre-Opening - Accommodation & Air Tickets",
                        "Pre-Opening - Linen & Uniform", 
                        "Pre-Opening - Legal / Licenses",
                        "Pre-Opening-PR & Marketing",
                        "Pre-Opening - IT & Technology",
                        "Pre-Opening - China/Glass/Silver",
                        "Pre-Opening - Staff Meal",
                        "Pre-Opening Expenses",
                        "Pre-Opening Operating Supplies",
                        "Pre-Opening Training",
                        "Pre-Opening-PR & Marketing",
                        "Pre-Opening Music and Entertain. Expenses"
                           ]
exclusions = ~stock_in.Category.isin(category_exclusions)
stock_in = stock_in[exclusions]

# remove trailing white spaces
# clean the product names for easier matching later
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
    
    
def multiplier_search(s):
    m = 1
    multiplier = re.search('[0-9.]+[xX]| [0-9.]+ [xX] | [xX] [0-9]+ | [xX][0-9+] ', s)
    if multiplier:
        interim = multiplier.group()
        interim = re.sub('[^0-9.]+', "", interim)
        m = float(interim)
    return m

def unit_search(s):
    m = 1000
    s = re.sub(',', '.', s)
    common_uoms = ['[0-9]+ML', ' [0-9]+ ML', 
                   '[0-9.]+CL', ' [0-9.]+ CL',
                   ' [0-9]+ C', '[0-9]+C', '[0-9]+CS', '\-[0-9.]+ ML', ' [0-9]+ CS',
                   '[0-9.,]+KG', ' [0-9.,]+ KG', '\-[0-9,.]+KG', '[0-9.,]+KGS', ' [0-9.,]+ KGS', '\-[0-9.,]+KGS', '[0-9.,]+KS', ' [0-9.,]+ KS',
                   '[0-9]+GR', ' [0-9]+ GR ', '[0-9]+ GR//', ' [0-9]+ GR//', '[0-9]+GMS', ' [0-9.]+ GMS', '[0-9 ]+GM', ' [0-9 ]+ GM', '[0-9]+G', ' [0-9]+ G ', '[0-9\-]+GRAMS',  ' [0-9]+ GRAMS',                        
                   '[0-9.]+LTR', ' [0-9.]+ LTR', '[0-9.]+LT','[0-9.]+L', ' [0-9.]+ LT',' [0-9.]+ L',
                   '[0-9]+GAL', ' [0-9]+ GAL',
                   '[0-9.]+OZ', ' [0-9.]+ OZ',
                   '[0-9]+LB', ' [0-9]+ LB'
                  ]
    thousand_multiplier = ['KG', 'KGS', 'KS', 'L', 'LT', 'LTR']
    gal_multiplier = ['GAL']
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
    if d['Upload Time'] < '2021-11-22' and d['Category'] == 'Alcoholic Beverage':
        multiplier = 1.3 * d['Unit Price']
    elif d['Upload Time'] < '2021-11-22' and d['Category'] == 'Non Alcoholic Beverage':
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

# adjust unit price column
# hunt down multiplier, unit size, and multiply
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
stock_in_agg = stock_in[['Product Name', 'Qty', 'Unit', 'Unit Size', 'Unit Price']].copy()
stock_in_agg['Est Total Cost'] = stock_in_agg['Qty'] * stock_in_agg['Unit Price']
stock_in_agg.drop('Unit Price', axis = 1, inplace = True)
stock_in_agg = stock_in_agg.groupby(['Product Name', 'Unit', 'Unit Size']).sum()
stock_in_agg = stock_in_agg.reset_index()
stock_in_agg = stock_in_agg.assign(
    UnitPrice = lambda x: round(x['Est Total Cost']/x['Qty'], 2),
    UnitCost = lambda y: round(y['Est Total Cost']/(y['Qty'] * y['Unit Size']), 2)
    )

stock_in_agg['Cleaned Product Name'] = stock_in_agg['Product Name'].apply(lambda x: names_cleaning(str(x)))

## cleaned product name and product name dictionary
## product name and unit size dictionary
product_name_dictionary = stock_in_agg[['Product Name', 'Cleaned Product Name']].copy()
stock_in_unit_product_dict = stock_in_agg[['Product Name', 'Unit Size']].copy()

# get product names list
unique_product_names = stock_in_agg['Cleaned Product Name'].drop_duplicates().tolist() 

### Crosswalk 2: Recipe Ingredients to Stock-In Report    
  
# encode stock-in records
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
    'Cleaned Product Name': unique_product_names,
    'Ingredient': most_similar
    })

matched_ingredients_stock_in_df = matched_ingredients_stock_in_df.merge(product_name_dictionary)
matched_ingredients_stock_in_df.drop('Cleaned Product Name', axis = 1, inplace = True)
matched_ingredients_stock_in_df = matched_ingredients_stock_in_df[['Product Name', 'Ingredient']]

 
# EXPORT
matched_ingredients_stock_in_df.to_csv(r'C:\Users\wesch\OneDrive\Documents\FoodRazor\expo\output\nov21_reporting\sample\stock_in_ingredients_xwalk.csv', index = False)
pd.DataFrame(recipe_ingredients_list).to_csv(r'C:\Users\wesch\OneDrive\Documents\FoodRazor\expo\output\nov21_reporting\sample\recipe_ingredients_list.csv', index = False)
matched_recipe_pos_df.to_csv(r'C:\Users\wesch\OneDrive\Documents\FoodRazor\expo\output\nov21_reporting\sample\recipe_pos_xwalk.csv', index = False)
stock_in_unit_product_dict.to_csv(r'C:\Users\wesch\OneDrive\Documents\FoodRazor\expo\output\nov21_reporting\sample\stock_in_unit_qc.csv', index = False)
pd.DataFrame(unique_recipe_items).to_csv(r'C:\Users\wesch\OneDrive\Documents\FoodRazor\expo\output\nov21_reporting\sample\recipe_items_list.csv', index = False)
