import flask
from flask import Flask, jsonify, request, render_template, redirect
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from formencode import variabledecode


# create random data only first time
# dates = pd.date_range('2017-07-23',periods=50, freq='W')
# data = pd.DataFrame(np.random.randint(100, size=(len(dates), 10)))
# data.set_index(dates, inplace=True)
# new_cols = []
# for i in data.columns:
#     new_cols.append(str(i))
# data.columns = new_cols
# data.to_csv('data.csv')

data = pd.read_csv('data.csv',index_col=0,parse_dates=True)

#create random categories
cats = {'cat1':['0','1','2','3'],'cat2':['4','5','6'],'cat3':['7','8','9']}

#create dummy product info only first 5 products
prod_info = {
        "Product Information":
        {
        "0":{
        "SKU":"0",
        "Product Description":"Lined Paper, 50 pages",
        "Price":"7.00",
        "Promotion":"6.50",
        "Product Category":"paper_goods"
        },
        "1":{
        "SKU":"1",
        "Product Description":"Legal Paper, 50 pages",
        "Price":"15.00",
        "Promotion":"13.00",
        "Product Category":"paper_goods"
        },
        "2":{
        "SKU":"2",
        "Product Description":"Red Stapler",
        "Price":"4.50",
        "Promotion":"nan",
        "Product Category":"fasteners"
        },
        "3":{
        "SKU":"3",
        "Product Description":"Heavy Duty Stapler",
        "Price":"5.99",
        "Promotion":"nan",
        "Product Category":"fasteners"
        },
        "4":{
        "SKU":"4",
        "Product Description":"Paper Clips (1000ct)",
        "Price":"9.99",
        "Promotion":"nan",
        "Product Category":"fasteners"
        }
        }
        }

#begin Flask
app = Flask(__name__)

@app.route('/api/v1.0/skus/<id>', methods=["GET"])
def info(id):
    return jsonify({str(id):prod_info['Product Information'][id]})

@app.route('/api/v1.0/skus/forecasts/table', methods=["GET","POST"])
def table():

    skus = request.args.get('skus')
    wk_range = request.args.get('wk_range')
    cat = request.args.get('cat')
    wk_start = (datetime.today() - timedelta(days=28)).date()
    wk_end = (datetime.today() + timedelta(days=63)).date()

    if request.method == "POST":
        postvars = variabledecode.variable_decode(request.form, dict_char='_')
        overrides = {"skus":postvars}
        for sku in overrides['skus'].keys():
            for date,val in overrides['skus'][sku].items():
                data.loc[date][sku] = val
        data.to_csv('data.csv')

        #return jsonify(overrides)
        return redirect(request.referrer)

    if cat:
        skus = cats[cat]

        filter_df = data.loc[wk_start:wk_end][skus]
        wk_totals = data.loc[wk_start:wk_end][skus].sum(axis=1)
        filter_df = pd.concat([filter_df,wk_totals.rename('Category')],axis=1)
        filter_df = filter_df.append(filter_df.sum(numeric_only=True).rename('Q1'))

        output = filter_df.T.to_dict(orient='split')
        return render_template('table4.html', result = output)

    if skus:
        skus = skus.split(',')

        filter_df = data.loc[wk_start:wk_end][skus]
        wk_totals = data.loc[wk_start:wk_end][skus].sum(axis=1)
        filter_df = pd.concat([filter_df,wk_totals.rename('Category')],axis=1)
        filter_df = filter_df.append(filter_df.sum(numeric_only=True).rename('Q1'))

        output = filter_df.T.to_dict(orient='split')
        return render_template('table4.html', result = output)

@app.route('/api/v1.0/skus/forecasts', methods=["GET","POST"])
def forecasts():
    '''Return json formatted data based on filter input.

    GET
    filter skus: skus=1 or skus=1,3,5
    filter categories: cat=cat1 or cat=cat2 or cat=cat3
    filter date range of data: wk_range=2017-07-23 or wk_range=2017-07-23,2017-07-30
    filter totals by skus, dates, or grand total: total=sku or total=wk or total=grand

    POST
    json format: {"skus":{"0":{"2017-07-23": "0","2017-07-30": "0"}}} or
        {"skus":{"0":{"2015-01-04": "11","2015-01-11": "25"},"1":{"2015-01-04": "2","2015-01-11": "4"}}}
    '''

    skus = request.args.get('skus')
    wk_range = request.args.get('wk_range')
    cat = request.args.get('cat')
    total = request.args.get('total')
    price = request.args.get('price')

    if request.method == 'POST':
        print(request.data)
        overrides = {"skus":request.json['skus']}
        for sku in overrides['skus'].keys():
            for date,val in overrides['skus'][sku].items():
                data.loc[date][sku] = val
        data.to_csv('data.csv')
        return '\npost successful! \n\n'

    if cat and wk_range and price:
        skus = cats[cat]
        prices = []
        for sku in skus:
            prices.append(float(prod_info['Product Information'][sku]['Price']))
        if ',' in wk_range:
            wk_start, wk_end = wk_range.split(',')
        else:
            wk_start = wk_range
            wk_end = wk_range
        return (data.loc[wk_start:wk_end][skus]*prices).to_json(date_format='iso')

    if cat and wk_range and total:
        skus = cats[cat]
        if ',' in wk_range:
            wk_start, wk_end = wk_range.split(',')
        else:
            wk_start = wk_range
            wk_end = wk_range
        if 'sku' in total:
            return data.loc[wk_start:wk_end][skus].sum().to_json(date_format='iso')
        if 'wk' in total:
            return data.loc[wk_start:wk_end][skus].sum(axis=1).to_json(date_format='iso')
        if 'grand' in total:
            return jsonify({"grand total":str(data.loc[wk_start:wk_end][skus].sum().sum())})

    if cat and wk_range:
        skus = cats[cat]
        if ',' in wk_range:
            wk_start, wk_end = wk_range.split(',')
        else:
            wk_start = wk_range
            wk_end = wk_range
        return data.loc[wk_start:wk_end][skus].to_json(date_format='iso')

    if cat:
        skus = cats[cat]
        return data[skus].to_json(date_format='iso')

    # skus filters
    if skus and wk_range and price:
        skus = skus.split(',')
        prices = []
        for sku in skus:
            prices.append(float(prod_info['Product Information'][sku]['Price']))
        if ',' in wk_range:
            wk_start, wk_end = wk_range.split(',')
        else:
            wk_start = wk_range
            wk_end = wk_range
        return (data.loc[wk_start:wk_end][skus]*prices).to_json(date_format='iso')

    if skus and wk_range and total:
        skus = skus.split(',')
        if ',' in wk_range:
            wk_start, wk_end = wk_range.split(',')
        else:
            wk_start = wk_range
            wk_end = wk_range
        if 'sku' in total:
            return data.loc[wk_start:wk_end][skus].sum().to_json(date_format='iso')
        if 'wk' in total:
            return data.loc[wk_start:wk_end][skus].sum(axis=1).to_json(date_format='iso')
        if 'grand' in total:
            return jsonify({"grand total":str(data.loc[wk_start:wk_end][skus].sum().sum())})

    if skus and wk_range:
        skus = skus.split(',')
        if ',' in wk_range:
            wk_start, wk_end = wk_range.split(',')
        else:
            wk_start = wk_range
            wk_end = wk_range
        return data.loc[wk_start:wk_end][skus].to_json(date_format='iso')

    if skus:
        skus = skus.split(',')
        return data[skus].to_json(date_format='iso')

    # wk_range filter
    if wk_range:
        if ',' in wk_range:
            wk_start, wk_end = wk_range.split(',')
        else:
            wk_start = wk_range
            wk_end = wk_range
        return data.loc[wk_start:wk_end].to_json(date_format='iso')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
