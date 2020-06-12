import util
import os
import shutil
from DatasetCreator import LivedoorDatasetCreator
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.model import PyTorchModel
import datetime
import argparse
from decorator import Timer

# create decorator
self_timer = Timer()

def setup_pretrained_model(base_dir:str):
    # get the pretrained bert model if needed
    model_dir = os.path.join(base_dir, "pretrained_model")
    if os.path.exists(model_dir) is False:
        download_url  = "http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JapaneseBertPretrainedModel/Japanese_L-12_H-768_A-12_E-30_BPE_transformers.zip&name=Japanese_L-12_H-768_A-12_E-30_BPE_transformers.zip"
        download_dir  = os.path.join(base_dir, "tmp")
        download_path = os.path.join(download_dir, "Japanese_L-12_H-768_A-12_E-30_BPE_transformers.zip")

        os.makedirs(download_dir, exist_ok=True)
        util.download_and_extract_if_needed(download_url, download_path, model_dir)
        shutil.rmtree(download_dir)
    return model_dir


def setup_livedoor_dataset(base_dir:str):
    # get dataset of livedoor news if needed
    origin_dir = os.path.join(base_dir, "dataset", "origin")
    if os.path.exists(origin_dir) is False:
        dataset_url   = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
        download_dir  = os.path.join(base_dir, "tmp")
        download_path = os.path.join(download_dir, "ldcc-20140209.tar.gz")

        os.makedirs(download_dir, exist_ok=True)
        util.download_and_extract_if_needed(dataset_url, download_path, origin_dir)
        shutil.rmtree(download_dir)
    
    # convert livedoor dataset if needed
    dataset_dir = os.path.join(base_dir, "dataset", "target")
    if os.path.exists(dataset_dir) is False:
        os.makedirs(dataset_dir, exist_ok=True)
        creator = LivedoorDatasetCreator()
        if creator.exists(dataset_dir=dataset_dir) is False:
            creator.load(origin_dir)
            creator.create(dataset_dir)
    return dataset_dir


def create_s3_bucket(s3, bucket_name:str, region_name:str):
    # create S3 bucket if target bucket is not exist
    try:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": region_name}
        )
    except s3.exceptions.BucketAlreadyOwnedByYou as e:
        print(e)


def upload_dataset_into_S3(s3, session, bucket_name:str, key_prefix:str, dataset_dir:str):
    # upload dataset into S3 if needed
    result = s3.list_objects(Bucket=bucket_name, Prefix=key_prefix)
    if "Contents" in result:
        s3_train_data = "s3://" + bucket_name + "/" + key_prefix
        print("training data url is {} . this url is already exists".format(s3_train_data))
    else:
        s3_train_data = session.upload_data(path=dataset_dir, bucket=bucket_name, key_prefix=key_prefix)
        print("training data url is {}".format(s3_train_data))

    # create data map
    data_channels = {"train": s3_train_data}
    return data_channels


@self_timer.time_deco
def train_in_sagemaker(role, data_channels:dict, server_source_dir:str, aws_account_id:str, aws_region:str, device:str, debug:bool, hyperparameters:dict):
    instance_type, image_version = __get_instance_info(device=device, debug=debug, mode="training")

    # create estimator
    image_url_training = "{}.dkr.ecr.{}.amazonaws.com/youyakuman:{}".format(aws_account_id, aws_region, image_version)
    print("image_url : {}".format(image_url_training))
    estimator = PyTorch(entry_point="youyakuman_train_and_deploy.py",
                        source_dir=server_source_dir,
                        role=role,
                        framework_version='1.5.0',
                        train_instance_count=1,
                        train_instance_type=instance_type,
                        hyperparameters=hyperparameters,
                        image_name=image_url_training)

    # start to train
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = "youyakuman-{}-{}".format(device, date_str)
    print("job_name is {}".format(job_name))
    estimator.fit(data_channels, job_name=job_name)
    
    return estimator, job_name


@self_timer.time_deco
def deploy_model_into_sagemaker(estimator, role, 
                                server_source_dir:str, 
                                endpoint_name:str, 
                                aws_account_id:str, 
                                aws_region:str, 
                                device:str, 
                                debug:bool):
    instance_type, image_version = __get_instance_info(device=device, debug=debug, mode="inference")
    image_url_inference = "{}.dkr.ecr.{}.amazonaws.com/youyakuman:{}".format(aws_account_id, aws_region, image_version)
    p_model = PyTorchModel(
        model_data=estimator.model_data,
        image=image_url_inference,
        role=role,
        framework_version=estimator.framework_version,
        entry_point=estimator.entry_point,
        source_dir=server_source_dir
    )
    predictor = p_model.deploy(initial_instance_count=1, 
                               instance_type=instance_type, 
                               endpoint_name=endpoint_name)
    return predictor


def get_account_id():
    sts = boto3.client("sts")
    id_info = sts.get_caller_identity()
    return id_info["Account"]


# mode is "inference" of "training"
def __get_instance_info(device:str, debug:bool, mode:str):
    if debug:
        image_version = "cpu_{}_0.9".format(mode)
        instance_type = "local"
    elif device == "cuda":
        image_version = "gpu_{}_1.0".format(mode)
        instance_type = "ml.p2.xlarge" # for GPU
    else:
        image_version = "cpu_{}_0.9".format(mode)
        instance_type = "ml.m4.xlarge" # for CPU
    return instance_type, image_version


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook_instance_type", type=str, default="local", help="notebook instance type which create training instance and inference instance to prepare the model. you can select the following types -> 'local', 'cloud'")
    parser.add_argument("--role", type=str, help="execution role of AWS. you need to set this parameter if notebook instance type is 'local'")
    parser.add_argument("--server_source_dir", type=str, default="../server")
    parser.add_argument("--bucket_name", type=str, default="youyakuman-dataset")
    parser.add_argument("--region_name", type=str, default="us-east-2")
    parser.add_argument("--endpoint_name", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    print("start to setup...")
    base_dir = os.path.dirname(__file__)
    session = boto3.session.Session()
    aws_account_id = get_account_id()
    
    args = get_argument()
    notebook_instance_type = args.notebook_instance_type
    if notebook_instance_type == "local":
        # TODO : check whether set role
        role = args.role
    else:
        role = sagemaker.get_execution_role()
    bucket_name = args.bucket_name
    region_name = args.region_name
    server_source_dir = args.server_source_dir
    endpoint_name = args.endpoint_name
    
    # TODO : 以下のパラメーターを外から設定できるようにする
    train_device = "cuda"
    infer_device = "cpu"
    debug = False
    hyperparameters = {
        'batch_size'            : 5,
        'report_every'          : 10,
        'train_steps'           : 100,
        'save_checkpoint_steps' : 50,
        'device'                : train_device,
        'pretrained_model_path' : "lib/pretrained_model/Japanese_L-12_H-768_A-12_E-30_BPE_transformers",
        'config_path'           : "config.ini"
    }

    os.environ["AWS_DEFAULT_REGION"]=region_name
    
    print("setup pretrained model...")
    model_dir = setup_pretrained_model(base_dir)
    
    print("setup dataset...")
    dataset_dir = setup_livedoor_dataset(base_dir)

    print("upload dataset into S3...")
    s3 = boto3.client("s3")
    key_prefix  = "data"
    create_s3_bucket(s3, bucket_name=bucket_name, region_name=region_name)
    data_channels = upload_dataset_into_S3(s3, session, bucket_name=bucket_name, key_prefix=key_prefix, dataset_dir=dataset_dir)
    
    print("train model...")
    estimator, job_name = train_in_sagemaker(
        role=role, 
        data_channels=data_channels, 
        server_source_dir=server_source_dir, 
        aws_account_id=aws_account_id,
        aws_region=region_name,
        device=train_device,
        debug=debug,
        hyperparameters=hyperparameters
    )
    
    print("deploy model...")
    if endpoint_name is None:
        endpoint_name = job_name
    predictor = deploy_model_into_sagemaker(
        estimator=estimator, 
        role=role, 
        server_source_dir=server_source_dir, 
        endpoint_name=endpoint_name, 
        aws_account_id=aws_account_id, 
        aws_region=region_name, 
        device=infer_device, 
        debug=debug
    )
    
    print("process time is {}".format(self_timer.p_time_dict))
    
    # TODO : test to infer sample data
    print("deploy is completed!!")
