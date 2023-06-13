import os

from time import sleep
from prefect_aws import S3Bucket, AwsCredentials


def create_aws_creds_block():
    # Before, set your keys in the terminal with `export AWS_ACCESS_KEY_ID=XXXXXXXXXXXXXXXXX`.
    # See https://docs.wandb.ai/guides/track/environment-variables
    # API secret keys should never be put in a client-side code or should be hidden.
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id=AWS_ACCESS_KEY_ID, 
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name="my-first-bucket-ahm-amm", credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="s3-bucket-example", overwrite=True)


if __name__ == "__main__":
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block()