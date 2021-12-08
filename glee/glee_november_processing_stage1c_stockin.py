# -*- coding: utf-8 -*-
"""
Created on Thurs Dec 02 2021
The idea behind this script is to process the Glee reports as fast as we can without going through Streamlit.
This script will check the stock-in reports and recipes first in the absence of POS
@author: wesch
"""
import pandas as pd
import re
import numpy as np
import os

path = r'C:\Users\wesch\OneDrive\Documents\FoodRazor\expo\output\nov21_reporting'
os.chdir(path)
outlet = ["\Baron"]

for outlet in outlet:
   
      
    ## STOCK-IN
    stock_in = pd.read_csv(path + outlet + "\StockIn.csv", encoding = 'utf-8')
    stock_in = stock_in.dropna()
    
    
    ## STOCK-IN DATA
    
    # remove trailing white spaces
    # clean the product names for easier matching later
    stock_in['Product Name'] = stock_in['Product Name'].apply(lambda x: x.rstrip())
    stock_in['Category'] = stock_in['Category'].apply(lambda x: x.rstrip())
    
    # remove those categories to be excluded
    category_exclusions = ["Printing & Stationary", "Printing and Stationery Supplies",  "Tax Adjustment", "CAPEX", "Other",
                           "Cleaning", "Cleaning Supplies", "Kitchen Supplies", "Discount", "Guest Supplies", "Rounding",
                           "General Supplies", "Packaging", "Bar Expenses", "Operating Supplies General", "HR", "Payroll",
                           "Cleaning & Chemical", "Utilities", "Music & entertainment", "Payroll & Related Expenses",
                           "PR & Marketing", "Payroll Provision (Guest Chefs)", "Transport", "Accommodation & Air Tickets",
                           "OS&E - Kitchen", "OS&E - FOH", "Supplies Kitchen", "Supplies Others", "Supplies Cleaning",
                           "Provision", "Accommodation", "Crockery", "Small Equipment", "Departmental Supplies",
                           "Cleaning ,Disposables and Chemicals", "Payroll and HR related", "Staff Training",
                           "Paper Supplies", "Uniforms", "Passage", "Travel Other", "Miscellaneous Expenses", 
                           "VisaVisa MedicalsMedical Lev", "Linen", "Equipment Hire", 'Fuel', 'Meals', 
                           'Marketing Expense', "Managment Fee", "Legal / Licenses",
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
                           "Pre-Opening Music and Entertain. Expenses",
                           "OE-China", "OE-Uniform", "OE-Others", "OE-Security/ Cleaning", "OE-Kitchen supplies",
                           "OE-Music & entertainment", "OE-Provision", "FC-Bank Charges", "OE - Admin - Supplies",
                           "OE-Glasswares", "OE-Provision", "OE-Laundry", "OE-Supply cleaning", "OE-Packaging", 
                           "OE-Guest supplies", "OE-Printing & stationary", "OE - ADMIN - Printing", "OE-Others",
                           "FC-PR & Marketing", "OE - Admin - Transport", "FC-IT & Technology", 'OE-Bar Expenses',
                           "OE-Admin - Meal Allocation", "Task force"
                           ]
    exclusions = ~stock_in.Category.isin(category_exclusions)
    stock_in = stock_in[exclusions]
    
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
        
    # function to clean ingredient names and stock-in names
    def names_cleaning(x):
        x = x.upper()
        x = re.sub(r'\([^)]*\)', '', x)
        x = re.sub("[^a-zA-ZéÉíÍóÓúÚáÁ ]+", "", x)
        x = ' '.join( [w for w in x.split() if len(w)>2] )
        return x    
    
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
    stock_in_unit_product_dict = stock_in_agg[['Product Name', 'Unit Size']].copy()
    
 
    
   
    stock_in_unit_product_dict.to_csv(path + outlet + '\stock_in_unit_qc.csv', index = False)
    stock_in_agg.to_csv(path + outlet + '\stock_in_agg.csv', index = False)
    
