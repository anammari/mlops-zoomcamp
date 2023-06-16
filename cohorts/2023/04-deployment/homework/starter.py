#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import os

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def get_parquet_file_size_in_mb(parquet_file_path):
  """Returns the size of the parquet file in megabytes.

  Args:
    parquet_file_path: The path to the parquet file.

  Returns:
    The size of the parquet file in megabytes.
  """

  file_size_bytes = os.stat(parquet_file_path).st_size
  file_size_mb = file_size_bytes / (1024 * 1024)
  return file_size_mb

if __name__ == "__main__":

  df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-02.parquet')

  dicts = df[categorical].to_dict(orient='records')
  X_val = dv.transform(dicts)
  y_pred = model.predict(X_val)

  df['y_pred'] = y_pred
  df['y_pred'].std()

  year = 2022
  month = 2
  output_file = 'output.parquet'
  df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
  df_result = df [['ride_id', 'y_pred']]
  df_result.to_parquet(
      output_file,
      engine='pyarrow',
      compression=None,
      index=False
  )

  parquet_file_path = output_file
  file_size_mb = get_parquet_file_size_in_mb(parquet_file_path)
  print(f"The size of the parquet file is {file_size_mb} MB.")