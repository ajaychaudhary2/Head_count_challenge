import boto3
import zipfile
import os

# Function to initialize the S3 client
def create_s3_client(aws_access_key_id, aws_secret_access_key, region_name):
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

# Function to download the zip file from S3
def download_file_from_s3(s3_client, bucket_name, zip_file_key, local_file_path):
    try:
        print(f"Downloading {zip_file_key} from S3...")
        s3_client.download_file(bucket_name, zip_file_key, local_file_path)
        print(f"File {zip_file_key} downloaded successfully.")
    except Exception as e:
        print(f"Error downloading file: {str(e)}")

# Function to extract the zip file
def extract_zip_file(zip_file_path, extract_dir):
    try:
        print(f"Extracting {zip_file_path} to {extract_dir}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Files extracted to {extract_dir}")
    except Exception as e:
        print(f"Error extracting files: {str(e)}")

# # Main function to download and extract the zip file
# def main():
#     # Define your IAM credentials and S3 details
#     aws_access_key_id = ''  
#     aws_secret_access_key = ''  
#     region_name = 'ap-south-1' 
#     bucket_name = 'hackdataimgs'
#     zip_file_key = 'Train_data.zip'
#     local_file_path = 'Train_data.zip'  
#     extract_dir = 'train_data'  

    
    os.makedirs(extract_dir, exist_ok=True)

    # Create the S3 client
    s3_client = create_s3_client(aws_access_key_id, aws_secret_access_key, region_name)

    # Download the zip file from S3
    download_file_from_s3(s3_client, bucket_name, zip_file_key, local_file_path)

    # Extract the zip file
    extract_zip_file(local_file_path, extract_dir)

# Run the main function
if __name__ == "__main__":
    main()
