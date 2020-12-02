# outlier-detection

## Quick start


``` sh
#prepare environment
pip install -r requirements.txt -i https://opentuna.cn/pypi/web/simple

#模型训练
cd source
python model.py
```

```sh
#使用已经训练生成的模型进行推理
cd source
python forecast.py
```

``` sh
#模型部署
sh build_and_push.sh
docker run -v -d -p 8080:8080 outlier
```



# result
On Training Data:
XGBOD ROC:0.9992, precision @ rank n:0.9375

On Test Data:
XGBOD ROC:0.8058, precision @ rank n:0.0833