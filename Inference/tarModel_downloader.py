import os
import mlflow

mlflow.set_tracking_uri("http://192.168.0.56:5000")
remote_server_uri = "http://192.168.0.56:5000" # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.56:9090"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
mlflow.set_experiment("Transfer-CvT_OneLabel")



# mlflow.artifacts.download_artifacts('s3://mlflow/13/f98accf721cd462da6a8207d4be1b62c/artifacts/ep17_model_best.pth.tar', dst_path='/home/hwi/weights/red/normal')
# mlflow.artifacts.download_artifacts('s3://mlflow/13/f98accf721cd462da6a8207d4be1b62c/artifacts/ep18_model_best.pth.tar', dst_path='/home/hwi/weights/red/normal')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/2c76a5b068b44e819343b8ac6887708c/artifacts/ep3_model_best.pth.tar', dst_path='/home/hwi/weights/red/86')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/e57f1dab726d4aa4a1d0641b586cc9b9/artifacts/ep9_model_best.pth.tar', dst_path='/home/hwi/weights/belly/normal')
# mlflow.artifacts.download_artifacts('s3://mlflow/13/e57f1dab726d4aa4a1d0641b586cc9b9/artifacts/ep2_model_best.pth.tar', dst_path='/home/hwi/weights/belly/normal')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/f5747aa3c4a74b7f8e3088afb6c22552/artifacts/ep8_model_best.pth.tar', dst_path='/home/hwi/weights/belly/86')
# mlflow.artifacts.download_artifacts('s3://mlflow/13/f5747aa3c4a74b7f8e3088afb6c22552/artifacts/ep4_model_best.pth.tar', dst_path='/home/hwi/weights/belly/86')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/6f2468f5ddd3492c921ade464962ea16/artifacts/ep1_model_best.pth.tar', dst_path='/home/hwi/weights/F/normal')
# mlflow.artifacts.download_artifacts('s3://mlflow/13/6f2468f5ddd3492c921ade464962ea16/artifacts/ep2_model_best.pth.tar', dst_path='/home/hwi/weights/F/normal')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/0db4e16193cb430da432859e46577589/artifacts/ep9_model_best.pth.tar', dst_path='/home/hwi/weights/F/86')
# mlflow.artifacts.download_artifacts('s3://mlflow/13/0db4e16193cb430da432859e46577589/artifacts/ep15_model_best.pth.tar', dst_path='/home/hwi/weights/F/86')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/023905e76a8143228fcd602682359ff6/artifacts/ep9_model_best_Acc.pth.tar', dst_path='/home/hwi/weights/C/normal')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/c4f4098d745b49ae96da3b90bf33e768/artifacts/ep1_model_best.pth.tar', dst_path='/home/hwi/weights/C/86')
# mlflow.artifacts.download_artifacts('s3://mlflow/13/c4f4098d745b49ae96da3b90bf33e768/artifacts/ep2_model_best.pth.tar', dst_path='/home/hwi/weights/C/86')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/3d5d5c519d24475f959901809515dd35/artifacts/ep3_model_best.pth.tar', dst_path='/home/hwi/weights/S/normal')
# mlflow.artifacts.download_artifacts('s3://mlflow/13/3d5d5c519d24475f959901809515dd35/artifacts/ep4_model_best.pth.tar', dst_path='/home/hwi/weights/S/normal')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/e36a985eeed3430987b65d95b218be5c/artifacts/ep3_model_best.pth.tar', dst_path='/home/hwi/weights/S/86')
# mlflow.artifacts.download_artifacts('s3://mlflow/13/e36a985eeed3430987b65d95b218be5c/artifacts/ep7_model_best.pth.tar', dst_path='/home/hwi/weights/S/86')

# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/chest/86')
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/chest/86')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/dbbdcd3d81d649ad80109b0d7ebd8cd3/artifacts/ep3_model_best.pth.tar', dst_path='/home/hwi/weights/frac/normal')
# mlflow.artifacts.download_artifacts('s3://mlflow/13/dbbdcd3d81d649ad80109b0d7ebd8cd3/artifacts/ep4_model_best.pth.tar', dst_path='/home/hwi/weights/frac/normal')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/16464a563bb14dc4956088d712793dc8/artifacts/ep6_model_best.pth.tar', dst_path='/home/hwi/weights/frac/86')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/e9a160396c9c4143bc74e8ef37a402a3/artifacts/ep6_model_best.pth.tar', dst_path='/home/hwi/weights/chest/normal')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/95b4179481f749c89d50aec7171b2382/artifacts/ep6_model_best.pth.tar', dst_path='/home/hwi/weights/frac/86')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/45d09cb145104383bb487f2ebc9169dc/artifacts/ep4_model_best_mAP.pth.tar', dst_path='/home/hwi/weights/wing/normal')
# mlflow.artifacts.download_artifacts('s3://mlflow/13/45d09cb145104383bb487f2ebc9169dc/artifacts/ep6_model_best_Acc.pth.tar', dst_path='/home/hwi/weights/wing/normal')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/d46bdcc5d1934c4ca44286c22ef0e658/artifacts/ep2_model_best_mAP.pth.tar', dst_path='/home/hwi/weights/wing/86')
# mlflow.artifacts.download_artifacts('s3://mlflow/13/d46bdcc5d1934c4ca44286c22ef0e658/artifacts/ep5_model_best_mAP.pth.tar', dst_path='/home/hwi/weights/wing/86')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/91945e9a6d4d46d38754841eb34aa250/artifacts/ep4_model_best.pth.tar', dst_path='/home/hwi/weights/leg/normal')

# mlflow.artifacts.download_artifacts('s3://mlflow/13/9eb4183333c44342a8841fc44b4aa637/artifacts/ep4_model_best.pth.tar', dst_path='/home/hwi/weights/leg/86')
# mlflow.artifacts.download_artifacts('s3://mlflow/13/9eb4183333c44342a8841fc44b4aa637/artifacts/ep6_model_best.pth.tar', dst_path='/home/hwi/weights/leg/86')

# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/red/86') # same with acc
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/red/normal') # same with acc
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/belly/86')  #없어요
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/belly/normal') # same with acc
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/F/86') # same with acc
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/F/normal') # same with acc
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/C/86')
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/C/normal')
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/F/86')
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/F/normal')
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/chest/86')
mlflow.artifacts.download_artifacts('s3://mlflow/13/e9a160396c9c4143bc74e8ef37a402a3/artifacts/ep3_model_best.pth.tar', dst_path='/home/hwi/weights/chest/normal')
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/chest/86')
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/chest/normal')
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/chest/86')
# mlflow.artifacts.download_artifacts('path', dst_path='/home/hwi/weights/chest/normal')



print('finfish')