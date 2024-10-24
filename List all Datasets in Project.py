# FOR LOCAL
project_id = "advancedbusinessanalytics"

myclient = bigquery.Client(project = project_id)

for dataset in myclient.list_datasets(include_all=False, max_results=None, page_token=None):
  print(dataset.dataset_id)
  
  
  #----------------
# FOR PUBLIC  

public_project_id = 'bigquery-public-data'

public_client = bigquery.Client(project = public_project_id)

for dataset in public_client.list_datasets(max_results = 205):
  print(dataset.dataset_id)
  
# CHECK
  
for dataset in myclient.list_datasets():
  print(dataset.dataset_id)