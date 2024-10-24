winetable = myclient.get_table(wine_id)

print(f"Wine rows: {winetable.num_rows}, Wine columns: {len(winetable.schema)}")

for column in winetable.schema:
  print(column)