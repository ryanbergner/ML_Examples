mascots_id = f"{project_id}.mydataset.mascots"
job_config = bigquery.QueryJobConfig(destination = mascots_id,
                                     write_disposition = "WRITE_TRUNCATE" )

sql_query_1 = '''
SELECT
  *
FROM
  bigquery-public-data.ncaa_basketball.mascots
'''

query_job = myclient.query(sql_query_1, job_config = job_config)
query_job.result()

mascots_table = myclient.get_table(mascots_id)
print(f"Query results loaded to the table {mascots_id}")
print (f"Loaded {mascots_table.num_rows} rows and {len(mascots_table.schema)} columns")
