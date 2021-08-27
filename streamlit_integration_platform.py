# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 07:39:07 2021

Integrated Inventory Management Platform: 
Integrating stock-in records with recipes and sales
"""

import streamlit as st
import numpy as np
import pandas as pd
import base64

st.markdown("# Integrated Inventory Management")
st.markdown('from [foodrazor](https://www.foodrazor.com/) with ❤️')

st.markdown('---')

st.markdown("## 📑 Data Upload")
st.warning('⚠️ *Note:* Please only upload CSVs in the same format as the sample provided')
st.warning('🏮 **Is the intention for end-users to report internal transfers on their own or within the FR app?** 🏮')

stock_in = st.file_uploader("📥 Stock-In Report", type = 'csv')
start_of_period_inventory = st.file_uploader("📮 Start of Period Inventory", type = 'csv')
recipes = st.file_uploader("🍴 Recipes", type = 'csv')
sales = st.file_uploader("💰 Sales", type = 'csv')
crosswalk = st.file_uploader("🚸 Ingredients, Orders, POS Crosswalk", type = ['csv'])

file_readers = {"csv": pd.read_csv}

st.markdown('---')



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

# def stock_in_tf(stock_in, xwalk):
#     stock_in = stock_in.dropna()
#     stock_in['Product Name'] = stock_in['Product Name'].str.rstrip()
#     sample_orders['Total_float'] = sample_orders.apply(
#         lambda x: float(str(x.Total)[2:].replace(",", "")), axis = 1
#     )
#     stock_in_summary = stock_in.groupby(['Location', 'Supplier Name', 'Product Name']).agg(
#         Qty = ('Qty', 'sum'),
#         Total = ('Total_float', 'sum')
#         )
#     stock_in_summary = stock_in_summary.assign(
#         Unit_Price = lambda x: round(x.Total/x.Qty, 2)
#         )
#     stock_in_summary = stock_in_summary.reset_index()
       
#     stock_in2 = stock_in[['Location', 'Supplier Name', 'Product Name', 'Invoice Date', 'Qty']]   
#     stock_in2 = stock_in2.groupby(['Location', 'Supplier Name', 'Product Name', 'Invoice Date']).agg(sum).reset_index()
#     stock_in_wide = pd.pivot(stock_in, columns = 'Invoice Date', values = 'Qty', index = ['Location', 'Supplier Name', 'Product Name'])
#     stock_in_wide = stock_in_wide.fillna(0).reset_index()
    
#     stock_in_joined = stock_in_wide.merge(stock_in_summary, on = ['Location', 'Supplier Name', 'Product Name'])
    
#     return stock_in_joined

# def consumption(sales_report, recipes, xwalk):
#     # adjusting pos crosswalk
#     label_pos_xwalk = xwalk.loc[:,['Label', 'pos_recipe', 'size']].drop_duplicates()
    
#     # calculating consumption from cocktails
#     cocktail_sales = sales_report[(sales_report['Sub Category'] == 'Cocktails')].copy()
#     cocktail_sales = cocktail_sales[['Item', 'Number Sold']]
#     cocktail_consumption = cocktail_sales.merge(recipes, left_on = 'Item', right_on = 'cocktail')
#     cocktail_consumption = cocktail_consumption.assign(
#         qty_sold = lambda x: x['qty'] * x['Number Sold'],
#         btl_sold = lambda x: x['qty_sold'] / x['btl_size']
#         )
#     cocktail_spirits_consumption = cocktail_consumption.groupby(['ingredient']).agg(
#         btl_sold = ('btl_sold', 'sum')    
#         )
#     cocktail_spirits_consumption.reset_index(inplace = True)
#     cocktail_spirits_consumption = cocktail_spirits_consumption.merge(label_pos_xwalk, left_on = 'ingredient', right_on = 'pos_recipe' )

#     # calculating consumption from shots
#     shots_sales = sales_report[(sales_report['Portion'] == "SHOT")]
#     shots_sales = sales_report[["Item", "Number Sold"]]
#     shots_sales = shots_sales.merge(label_pos_xwalk, left_on = "Item", right_on = "pos_recipe")
#     shots_sales = shots_sales.assign(
#         qty_sold = lambda x: 30 * x['Number Sold'],
#         btl_sold = lambda x: x['qty_sold']/x['size']
#         )
    
#     # calculating soda consumption 
#     non_alc_sales = sales_report[(sales_report['Sub Category'] == 'Sodas')|(sales_report['Sub Category'] == 'Water')].copy()
#     non_alc_sales = non_alc_sales[['Item', 'Number Sold']]
#     non_alc_sales = non_alc_sales.rename(columns = {'Number Sold': 'btl_sold'})
#     non_alc_sales = non_alc_sales.merge(label_pos_xwalk, left_on = "Item", right_on = "pos_recipe")
    
#     # amalgamating to get all spirits
#     spirits_sales = cocktail_spirits_consumption[['Label', 'pos_recipe', 'btl_sold']].append(shots_sales[['Label', 'pos_recipe', 'btl_sold']], ignore_index =  True)
#     spirits_sales = spirits_sales.groupby('Label').agg(
#         btl_sold = ('btl_sold', 'sum')
#     )
#     spirits_sales.reset_index(inplace = True)
#     spirits_sales = spirits_sales.append(non_alc_sales[['Label', 'btl_sold']], ignore_index = True)
    
#     return spirits_sales 

# def inventory(inventory_df):
#     sales = consumption(sales_df, recipes_df, crosswalk_df)
#     bar_inventory = inventory_df.drop_duplicates()
#     bar_inventory = bar_inventory.merge(sales, 'left')
#     bar_inventory = bar_inventory.fillna(0)

#     ## crosswalk with stock-in
#     label_ordering_xwalk = xwalk.loc[:,['Label', 'orders_item']].drop_duplicates()

#     ## adjust orders
#     orders_summary = stock_in_tf(stock_in_df, crosswalk_df)
#     orders_summary_xwalk = orders_summary.merge(label_ordering_xwalk, left_on = 'name', right_on = 'orders_item', how = 'left')
#     orders_summary_xwalk = orders_summary_xwalk.groupby('Label').agg(
#         total_ordered = ('total_ordered', 'sum')
#     )
#     orders_summary_xwalk.reset_index(inplace = True)

#     ## integrate with sales
#     bar_inventory_orders = bar_inventory.merge(orders_summary_xwalk, on = 'Label', how = "left")
#     bar_inventory_orders['total_ordered'] = bar_inventory_orders['total_ordered'].fillna(0)

#     # calculate estimated inventory
#     bar_inventory_orders = bar_inventory_orders.assign(
#         Estimated_Balance = lambda x: x.Opening + x.Net_Transfers + x.total_ordered - x.btl_sold
#     )
#     bar_inventory_orders = bar_inventory_orders.rename(columns = {'btl_sold':'Btl_Sold', 'total_ordered':'Total_Ordered'})
    
#     return bar_inventory_orders


            

st.markdown("## ⏬ Data Download")

if crosswalk is not None and recipes is not None and sales is not None and stock_in is not None and start_of_period_inventory is not None:
    for filetype, reader in file_readers.items():
        if crosswalk.name.endswith(filetype) and recipes.name.endswith(filetype) and sales.name.endswith(filetype) and stock_in.name.endswith(filetype) and start_of_period_inventory.name.endswith(filetype):
            crosswalk_df = reader(crosswalk)
            recipes_df = reader(recipes)
            sales_df = reader(sales)
            stock_in_df = reader(stock_in)
            inventory_df = reader(start_of_period_inventory)
            del crosswalk, sales, stock_in, recipes, start_of_period_inventory
            
            def stock_in_tf(stock_in, xwalk):
                stock_in = stock_in.dropna()
                stock_in['Product Name'] = stock_in['Product Name'].str.rstrip()
                stock_in['Total_float'] = stock_in.apply(
                    lambda x: float(str(x.Total)[2:].replace(",", "")), axis = 1
                )
                stock_in_summary = stock_in.groupby(['Location', 'Supplier Name', 'Product Name']).agg(
                    Qty = ('Qty', 'sum'),
                    Total = ('Total_float', 'sum')
                    )
                stock_in_summary = stock_in_summary.assign(
                    Unit_Price = lambda x: round(x.Total/x.Qty, 2)
                    )
                stock_in_summary = stock_in_summary.reset_index()
                   
                stock_in2 = stock_in[['Location', 'Supplier Name', 'Product Name', 'Invoice Date', 'Qty']]   
                stock_in2 = stock_in2.groupby(['Location', 'Supplier Name', 'Product Name', 'Invoice Date']).agg(sum).reset_index()
                stock_in_wide = pd.pivot(stock_in2, columns = 'Invoice Date', values = 'Qty', index = ['Location', 'Supplier Name', 'Product Name'])
                stock_in_wide = stock_in_wide.fillna(0).reset_index()
                
                stock_in_joined = stock_in_wide.merge(stock_in_summary, on = ['Location', 'Supplier Name', 'Product Name'])
                
                return stock_in_joined
            
            def consumption(sales_report, recipes, xwalk):
                
                
                # calculating consumption from cocktails
                cocktail_sales = sales_report[(sales_report['Sub Category'] == 'Cocktails')].copy()
                cocktail_sales = cocktail_sales[['Item', 'Number Sold']]
                cocktail_consumption = cocktail_sales.merge(recipes, left_on = 'Item', right_on = 'cocktail')
                cocktail_consumption = cocktail_consumption.assign(
                    qty_sold = lambda x: x['qty'] * x['Number Sold'],
                    btl_sold = lambda x: x['qty_sold'] / x['btl_size']
                    )
                cocktail_spirits_consumption = cocktail_consumption.groupby(['ingredient']).agg(
                    btl_sold = ('btl_sold', 'sum')    
                    )
                cocktail_spirits_consumption.reset_index(inplace = True)
                
                # adjusting pos crosswalk
                label_pos_xwalk = xwalk[['Label', 'pos_recipe', 'size']].drop_duplicates()
                
                cocktail_spirits_consumption = cocktail_spirits_consumption.merge(label_pos_xwalk, left_on = 'ingredient', right_on = 'pos_recipe' )
            
                # calculating consumption from shots
                shots_sales = sales_report[(sales_report['Portion'] == "SHOT")]
                shots_sales = sales_report[["Item", "Number Sold"]]
                shots_sales = shots_sales.merge(label_pos_xwalk, left_on = "Item", right_on = "pos_recipe")
                shots_sales = shots_sales.assign(
                    qty_sold = lambda x: 30 * x['Number Sold'],
                    btl_sold = lambda x: x['qty_sold']/x['size']
                    )
                
                # calculating soda consumption 
                non_alc_sales = sales_report[(sales_report['Sub Category'] == 'Sodas')|(sales_report['Sub Category'] == 'Water')].copy()
                non_alc_sales = non_alc_sales[['Item', 'Number Sold']]
                non_alc_sales = non_alc_sales.rename(columns = {'Number Sold': 'btl_sold'})
                non_alc_sales = non_alc_sales.merge(label_pos_xwalk, left_on = "Item", right_on = "pos_recipe")
                
                # amalgamating to get all spirits
                spirits_sales = cocktail_spirits_consumption[['Label', 'pos_recipe', 'btl_sold']].append(shots_sales[['Label', 'pos_recipe', 'btl_sold']], ignore_index =  True)
                spirits_sales = spirits_sales.groupby('Label').agg(
                    btl_sold = ('btl_sold', 'sum')
                )
                spirits_sales.reset_index(inplace = True)
                spirits_sales = spirits_sales.append(non_alc_sales[['Label', 'btl_sold']], ignore_index = True)
                
                return spirits_sales 
            
            def inventory(inventory_df, xwalk):
                sales = consumption(sales_report = sales_df, recipes = recipes_df, xwalk = crosswalk_df)
                bar_inventory = inventory_df.drop_duplicates()
                bar_inventory = bar_inventory.merge(sales, 'left')
                bar_inventory = bar_inventory.fillna(0)
            
                ## crosswalk with stock-in
                label_ordering_xwalk = xwalk.loc[:,['Label', 'orders_item']].drop_duplicates()
            
                ## adjust orders
                orders_summary = stock_in_tf(stock_in_df, crosswalk_df)
                orders_summary_xwalk = orders_summary.merge(label_ordering_xwalk, left_on = 'Product Name', right_on = 'orders_item', how = 'left')
                orders_summary_xwalk = orders_summary_xwalk.groupby('Label').agg(
                    Qty = ('Qty', 'sum'),
                    Total = ('Total', 'sum')
                )
                orders_summary_xwalk['Qty'] = orders_summary_xwalk['Qty'].fillna(0)
                orders_summary_xwalk['Total'] = orders_summary_xwalk['Total'].fillna(0)
                orders_summary_xwalk = orders_summary_xwalk.assign(
                    Unit_Price = lambda x: round(x.Total/x.Qty, 2)
                    )
                orders_summary_xwalk.reset_index(inplace = True)
            
                ## integrate with sales
                bar_inventory_orders = bar_inventory.merge(orders_summary_xwalk, on = 'Label', how = "left")
                bar_inventory_orders = bar_inventory_orders.fillna(0)
            
                # calculate estimated inventory
                bar_inventory_orders = bar_inventory_orders.assign(
                    Estimated_Balance = lambda x: x.Opening + x.Net_Transfers + x.Qty - x.btl_sold
                )
                bar_inventory_orders = bar_inventory_orders.rename(columns = {'btl_sold':'Btl_Sold', 
                                                                              'Qty':'Total_Ordered',
                                                                              'Total': 'Total_Cost'})
                bar_inventory_orders = bar_inventory_orders[['Label', 'Opening', 'Net_Transfers', 'Total_Ordered', 'Total_Cost', 'Unit_Price', 'Btl_Sold', 'Estimated_Balance']]
                
                
                return bar_inventory_orders, orders_summary
            
            final_inventory_report_df, orders_summary = inventory(inventory_df, crosswalk_df)
            
            
            if st.button('Download Sales + Inventory Report as CSV'):
                tmp_download_link = download_link(final_inventory_report_df, 'final_inventory.csv', 'Click here to download your Integrated Sales + Inventory Data!')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                tmp_download_link2 = download_link(orders_summary, 'final_stock_in.csv', 'Click here to download your Stock-In data!')
                st.markdown(tmp_download_link2, unsafe_allow_html=True)
            
                
                
        else: 
            print ("Wrong file type.")
            