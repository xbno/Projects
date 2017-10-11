# API assignment for an interview

To run the app, type 'sudo python v1_api.py' into terminal when in the local directory of the file. Then to access the api type 'http://0.0.0.0/' followed by the resource you want to use.

For example, to see the forecast of all products in category2 and category3 type:
  - http://0.0.0.0/api/v1.0/skus/forecasts?cat=cat2,cat3

To filter based on a select week or range of weeks with certain sku(s) enter those filters:
  - http://0.0.0.0/api/v1.0/skus/forecasts?wk_range=2017-08-01,2017-10-01&skus=1

The results will look like:

  - {"1":{"2017-08-06T00:00:00.000Z":22,"2017-08-13T00:00:00.000Z":51,"2017-08-20T00:00:00.000Z":101,"2017-08-27T00:00:00.000Z":9,"2017-09-03T00:00:00.000Z":50,"2017-09-10T00:00:00.000Z":21,"2017-09-17T00:00:00.000Z":67,"2017-09-24T00:00:00.000Z":79,"2017-10-01T00:00:00.000Z":48}}

To use the HTML version type:
  - http://0.0.0.0/api/v1.0/skus/forecasts/table?cat=cat1
  - Update a forecast by modifying a value in an editable cell and then press 'Update Values'

![](table.png)
