from google.cloud import bigquery

project_id = "advancedbusinessanalytics"

myclient = bigquery.Client(project = project_id)

ml_data_id = "bigquery-public-data.ml_datasets"

ml_data = myclient.get_dataset(ml_data_id)

ml_tables = myclient.list_tables(ml_data)

for table in ml_tables:
  print(table.table_id)