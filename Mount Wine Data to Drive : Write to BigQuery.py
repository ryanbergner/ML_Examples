from google.colab import drive
drive.mount('/content/drive/')

myfilepath = "/content/drive/MyDrive/Advanced Business Analytics/data/wine.csv"

import pandas as pd
winedf = pd.read_csv(myfilepath, index_col = 0)

#--------------
# Writing to BigQuery

wine_id  = f"{project_id}.mydataset.winedata"

job_config = bigquery.LoadJobConfig(autodetect = True,
                                    write_disposition = "WRITE_TRUNCATE" ) # overwrite

load_job = myclient.load_table_from_dataframe(dataframe = winedf,
                                              destination = wine_id,
                                              job_config = job_config
load_job.result()

wine_table = myclient.get_table(wine_id)

print(f"Loaded {wine_table.num_rows} rows")



# CHECK

dataset = myclient.get_dataset(f"{project_id}.mydataset" )

tables = list(myclient.list_tables(dataset))

for table in tables:
  print(table.table_id)