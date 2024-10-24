from google.cloud import bigquery

# https://cloud.google.com/resource-manager/docs/creating-managing-projects

project_id = 'advancedbusinessanalytics'

client = bigquery.Client(project=project_id)

for dataset in client.list_datasets():
  print(dataset.dataset_id)