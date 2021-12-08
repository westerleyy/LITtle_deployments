# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 23:06:12 2021
The idea behind this script is to process the Glee reports as fast as we can without going through Streamlit.
This script will follow the original preferred workflow of calculating balances with corrected crosswalks.
@author: wesch
"""
import pandas as pd
import re
import numpy as np
import os 

path = r'C:\Users\wesch\OneDrive\Documents\FoodRazor\expo\output\nov21_reporting'
os.chdir(path)

done_weird = ['\Kojaki']
outlet = [r'\Adrift', r'\Mina Seyahi', r'\National', r'\Scarpetta',r'\Kutir', r'\Baron' ]

batch_outlet = [r'\Baron']

for outlet in outlet:
    stock_in_agg = pd.read_csv(path + outlet + '\stock_in_agg.csv')
    stock_in_unit_qc = pd.read_csv(path + outlet + '\stock_in_unit_qc_amended.csv')
    stock_in_ingredients_xwalk = pd.read_csv(path + outlet + '\stock_in_ingredients_xwalk_amended.csv')
    recipe_pos_xwalk = pd.read_csv(path + outlet + r'\recipe_pos_amended.csv') 
    
    ## Stock In Data Handling
    stock_in_agg = stock_in_agg[['Product Name', 'Qty', 'Est Total Cost', 'Unit']]
    stock_in_agg = stock_in_agg.merge(stock_in_unit_qc)
    stock_in_agg = stock_in_agg.assign(
        UnitPrice = lambda x: round(x['Est Total Cost']/x['Qty'], 2),
        UnitCost = lambda y: y['Est Total Cost']/(y['Qty'] * y['Unit Size'])
        )
    
    
    ## POS
    pos_sheet_df = pd.read_excel(path + outlet + "\POS.xlsx", skiprows = 6, sheet_name = "Revenue per article")
    pos_sheet_df = pos_sheet_df.dropna(how = 'all')
    
    # remove all nonsensical values
    pos_sheet_cleaned = pos_sheet_df.loc[(pos_sheet_df['Article']!= 'Total')&(pos_sheet_df['Number of articles']>0),]
    pos_sheet_cleaned['Article'] = pos_sheet_cleaned.loc[:,'Article'].str.upper()
    all_pos_cleaned = pos_sheet_cleaned.loc[(pos_sheet_cleaned['Article']!= 'Total'),]
    all_pos_cleaned.dropna(inplace = True)
    
    # total revenue
    all_revenue = all_pos_cleaned['Net revenue w/o discount'].sum()
    
    
    ## RECIPE
    recipe_sheet_df = pd.read_excel(path + outlet + "\Recipe.xlsx", sheet_name = 'Reformatted')
    recipe_sheet_df = recipe_sheet_df.dropna(how = 'all')
    batch = False
    
    ### try to look for and import batched recipes
    try:
        recipe_sheet_batch_df = pd.read_excel(path + outlet + "\Recipe.xlsx", sheet_name = 'Batch')
        recipe_sheet_batch_df = recipe_sheet_batch_df.dropna(how = 'all')
        batch = True
    except: 
        print ('No batched recipes detected.')
    
    ## RECIPE DATA
    # function to clean ingredient names and stock-in names
    def names_cleaning(x):
        x = x.upper()
        x = re.sub(r'\([^)]*\)', '', x)
        sevenup_exception = "UP "
        if sevenup_exception in x:
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
            m = float(x['Quantity']) * 330
        else:
            m = x['Quantity']
        return m
    
    # ensure servings column is an int, upper case all and clean the ingredient names
    # check and replace quantity column to make sure it is an integer
    
    def quantity_replacement(x):
        quantity = x
        try:
            quantity = float(quantity)
        except:
            quantity = 0
        return quantity
        
    recipe_sheet_df['Quantity'] = recipe_sheet_df.Quantity.apply(lambda x: quantity_replacement(str(x)))
    
    recipe_sheet_df = recipe_sheet_df.assign(
        Servings = lambda x: x.Servings.astype(int),
        Quantity = lambda y: y.Quantity.astype(float)/y.Servings
        )
    recipe_sheet_df.replace(np.inf, 0, inplace = True)
    recipe_sheet_df['Food Item (As per POS system)'] = recipe_sheet_df.loc[:,'Food Item (As per POS system)'].str.upper()
    recipe_sheet_df['Unit of Measurement'] = recipe_sheet_df.loc[:,'Unit of Measurement'].str.upper()    
    recipe_sheet_df['Ingredient'] = recipe_sheet_df['Ingredient Ordered (if known)'].apply(lambda x: names_cleaning(str(x)))
     
    # convert all units to ml and gr
    recipe_sheet_df['NewQuantity'] = recipe_sheet_df.apply(quantity_conversion, axis = 1)
    recipe_sheet_df.drop('Quantity', axis = 1, inplace = True)
    recipe_sheet_df.rename(columns = {'NewQuantity': 'Quantity'}, inplace = True)
    
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
        
        # calculate max theoretical weight of batch product
        recipe_sheet_batch_df['Quantity'] = recipe_sheet_batch_df.Quantity.apply(lambda x: quantity_replacement(str(x)))
        recipe_sheet_batch_df = recipe_sheet_batch_df.assign(
            Servings = lambda x: x.Servings.astype(int),
            Quantity = lambda y: y.Quantity.astype(float)/y.Servings
            )
        recipe_sheet_batch_df.replace(np.inf, 0, inplace = True)
        
        batch_volume = recipe_sheet_batch_df.groupby('Food Item (As per POS system)').sum()
        batch_volume = batch_volume.reset_index()
        batch_volume = batch_volume[['Food Item (As per POS system)', 'Quantity']].drop_duplicates()
        
        
    # find out items ordered during the period 
    pos_sheet_cleaned_ordered = pos_sheet_cleaned[['Article', 'Number of articles', 'Net revenue w/o discount']]
    pos_sheet_cleaned_ordered = pos_sheet_cleaned_ordered.merge(recipe_pos_xwalk, left_on = 'Article', right_on = 'POS Items')
    pos_sheet_cleaned_ordered = pos_sheet_cleaned_ordered[['Article', 'Number of articles', 'Recipe Items']]
             
            
    # merge with recipes df to get all components
    recipe_ordered = recipe_sheet_df.merge(pos_sheet_cleaned_ordered,
                                           left_on = 'Food Item (As per POS system)',
                                           right_on = 'Recipe Items')
               
    # calculate quantity consumed by serving size of 1
    recipe_ordered['Quantity Consumed'] = recipe_ordered['Number of articles']*recipe_ordered['Quantity']
    
          
    # get items that cannot be matched to recipes
    unmatched = stock_in_ingredients_xwalk.loc[stock_in_ingredients_xwalk.Ingredient.isna(),]
    unmatched = unmatched.merge(stock_in_agg)
    unmatched = unmatched[['Product Name','Qty', 'Est Total Cost']]
        
            
    # get pos items that cannot be matched properly
    unmatched_pos = recipe_pos_xwalk.loc[recipe_pos_xwalk['Recipe Items'].isna(), 'POS Items']
    unmatched_pos = pd.DataFrame(unmatched_pos)
    unmatched_pos = unmatched_pos.rename(columns = {'POS Items': 'Article'})
    unmatched_pos = unmatched_pos.merge(all_pos_cleaned[['Article', 'Number of articles', 'Net revenue w/o discount']])
               
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
    cost_calculation = cost_calculation.merge(stock_in_ingredients_xwalk, on = 'Ingredient')
    cost_calculation = cost_calculation.merge(stock_in_agg[['Product Name', 'UnitCost']], on = 'Product Name')
    cost_calculation.replace(np.inf, 0, inplace = True)
                    
    ## cost 
    cost_calculation = cost_calculation.assign(
        constituent_cost = lambda x: x['Quantity'] * x['UnitCost']
        )
            
    # averaging out the cost when there is more than one possible identical ingredient
    summarized_cost_calculation = cost_calculation.groupby(['Food Item (As per POS system)', 'Ingredient', 'Quantity']).agg(
        mean_constituent_cost=('constituent_cost', 'mean'))        
    summarized_cost_calculation = summarized_cost_calculation.reset_index()
    summarized_cost_calculation = summarized_cost_calculation.drop_duplicates()
            
    # obtaining COGS
    cost_of_goods_sold = summarized_cost_calculation.groupby('Food Item (As per POS system)').sum()
                           
    ## crosswalk to connect to the POS system
    cost_of_goods_sold = cost_of_goods_sold.merge(recipe_pos_xwalk, 
                                                  left_on = 'Food Item (As per POS system)', 
                                                  right_on = 'Recipe Items')
    
    cost_of_goods_sold = cost_of_goods_sold.merge(all_pos_cleaned[['Article', 'Net revenue w/o discount']],
                                                  left_on = 'POS Items',
                                                  right_on = 'Article')
    
    cost_of_goods_sold = cost_of_goods_sold.rename(columns = {
        'Net revenue w/o discount': 'Revenue'
        })
            
    ## slimming the DF down
    cost_of_goods_sold_narrow = cost_of_goods_sold[['Article', 'mean_constituent_cost', 'Revenue']].copy()
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.merge(pos_sheet_cleaned_ordered[['Article', 'Number of articles']])
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.rename(columns = {
        'Number of articles': 'Quantity'
        })
    
    
    # obtaining margin
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.assign(
        Revenue = lambda w: round(w.Revenue/w.Quantity,2),
        cost_pct = lambda y: round(100*(y.mean_constituent_cost/y.Revenue),2),
        margin = lambda x: round(100*(1-x.mean_constituent_cost/x.Revenue),2)
        )
    
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.dropna()
    cost_of_goods_sold_narrow.replace(-np.inf, 0, inplace = True)
    cost_of_goods_sold_narrow.replace(np.inf, 0, inplace = True)
        
    cost_of_goods_sold_narrow = cost_of_goods_sold_narrow.rename(columns = {
        'mean_constituent_cost': 'Cost Per Sale',
        'Revenue': 'Revenue Per Sale',
        'margin': 'Profit Margin Pct',
        'cost_pct': 'Cost of Production Pct'
        })        
            
           
    
    
    
    # focus on just ingredients
    # group by ingredients to consolidate total quantity consumed
    ingredient_consumption = recipe_ordered[['Ingredient', 'Quantity Consumed']]
    ingredient_consumption = ingredient_consumption.groupby(['Ingredient']).sum()
    ingredient_consumption = ingredient_consumption.reset_index()
    
    # track inventory by looking at inventory stock in
    # ingredient stock in crosswalk
    ingredient_stockin_recipe_qc = stock_in_ingredients_xwalk.merge(recipe_sheet_df[['Ingredient', 'Ingredient Ordered (if known)']].drop_duplicates(), 
                                                                    on = 'Ingredient')
        
    inventory_tracking = ingredient_consumption.merge(ingredient_stockin_recipe_qc)
    inventory_tracking = inventory_tracking.groupby(['Product Name']).sum()
    inventory_tracking = inventory_tracking.reset_index()
    inventory_tracking = inventory_tracking.merge(stock_in_agg, how = 'right')
    inventory_tracking['Quantity Consumed'] = inventory_tracking['Quantity Consumed'].fillna(0)
           
    
    def est_bal_adj(x):
        quantity_consumed = x['Quantity Consumed']
        if x['Estimated Balance'] < 0:
            quantity_consumed = x['Qty'] * x['Unit Size']
        return quantity_consumed
    
    ## get estimated balance
    inventory_tracking['Estimated Balance'] = (inventory_tracking['Qty'] * inventory_tracking['Unit Size']) - inventory_tracking['Quantity Consumed']
    inventory_tracking['Quantity Consumed'] = inventory_tracking.apply(est_bal_adj, axis = 1)
    inventory_tracking['Cost of Quantity Consumed'] = (inventory_tracking['Quantity Consumed'] / (inventory_tracking['Qty'] * inventory_tracking['Unit Size'])) * inventory_tracking['Est Total Cost']
    inventory_tracking.replace(np.inf, 0, inplace=True)
    inventory_tracking = inventory_tracking.fillna(0)    
                              
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

            
        
    
    ingredient_revenue = summarized_cost_calculation.groupby('Ingredient').sum()
    ingredient_revenue = ingredient_revenue.reset_index()
    ingredient_revenue = ingredient_revenue[['Ingredient', 'constituent_revenue']]
    ingredient_revenue = ingredient_revenue.merge(stock_in_ingredients_xwalk)        
    ingredient_revenue = ingredient_revenue[['Product Name', 'constituent_revenue']].copy()
    ingredient_revenue = ingredient_revenue.groupby('Product Name').sum()
    ingredient_revenue = ingredient_revenue.reset_index()
    ingredient_revenue = ingredient_revenue.rename(columns = {
        'constituent_revenue': 'Attributable Revenue'
        })
            
    # adding weighted ROI per ingredient to inventory tracking 
    inventory_tracking = inventory_tracking.merge(ingredient_revenue, on = 'Product Name', how = 'left')
    inventory_tracking = inventory_tracking.assign(
        ProfitMargin = lambda x: round(100 * (1- (x['Cost of Quantity Consumed']/x['Attributable Revenue'])), 2),
        CostMargin = lambda y: round(100 * (y['Cost of Quantity Consumed']/y['Attributable Revenue']), 2),
        QuantityStockedIn = lambda z: z['Unit Size'] * z['Qty']
        )
    inventory_tracking = inventory_tracking.rename(columns = {
        'QuantityStockedIn': 'Quantity Stocked In',
        'ProfitMargin': 'Profit Margin',
        'CostMargin': 'Cost Margin'
        })
        
    inventory_tracking = inventory_tracking[['Product Name', 'Qty', 'Unit Size', 'Quantity Stocked In', 'Quantity Consumed', 'Estimated Balance', 'Est Total Cost', 'Cost of Quantity Consumed',  'Attributable Revenue', 'Profit Margin', 'Cost Margin']]
    inventory_tracking = inventory_tracking.fillna(0)
    inventory_tracking.replace(np.inf, 0, inplace = True)
    inventory_tracking.replace(-np.inf, 0, inplace = True)
    inventory_tracking = inventory_tracking.drop_duplicates()
        
        # # uom conversion
        # common_uoms = ['KG', 'GMS', 'LTR', 'KGS', 'GM', 'CL', 'LT', 'L', 'C', 'GRAMS', 'G', 'KS', 'CS', 'GMS', 'KGS']
        # common_uoms_equivalent = ['GR', 'GR', 'ML', 'GR', 'GR', 'ML', 'ML', 'ML', 'ML', 'GR', 'GR', 'GR', 'ML', 'GR', 'GR']
        
        # for uom in range(len(common_uoms)):
        #     inventory_tracking.loc[inventory_tracking['Unit of Measurement'].str.contains(common_uoms[uom]), 'Unit of Measurement'] = common_uoms_equivalent[uom]
        
        
        
    total_cogs = round(inventory_tracking['Cost of Quantity Consumed'].sum(),2)
    total_cogs_pct = round(100*total_cogs/all_revenue, 2)
    total_stockin_cost = round(inventory_tracking['Est Total Cost'].sum(),2)
    total_stockin_pct = round(100*total_stockin_cost/all_revenue, 2)
    
    food_cost_revenue_summary = [all_revenue, total_cogs, total_cogs_pct, total_stockin_cost, total_stockin_pct]
    food_cost_revenue = ['Total Revenue', 'Theoretical Food Cost', 'Theoretical Food Cost Percentage', 'Total Stocked In', 'Total Stocked In Percentage']
    food_cost_revenue_summary_df = pd.DataFrame({
        'Cost-Revenue Summary': food_cost_revenue,
        'Cost-Revenue Values': food_cost_revenue_summary
        })
    
    inventory_tracking.to_csv(path + outlet + r'\final_inventory.csv', index = False)
    cost_of_goods_sold_narrow.to_csv(path + outlet + r'\menu_margin.csv', index = False)
    unmatched.to_csv(path + outlet + r'\unmatched_inventory.csv', index = False)
    unmatched_pos.to_csv(path + outlet + r'\unmatched_pos_articles.csv', index = False)
    food_cost_revenue_summary_df.to_csv(path + outlet + r'\overall_summary.csv', index = False)
    
    print('Done')
    print(all_revenue)
    print(total_cogs)
    print(total_stockin_cost)