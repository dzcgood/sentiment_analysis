# 后端运行相关说明

## 模型相关

+ 预训练模型`bert-base-chinese`相关网址：`https://huggingface.co/google-bert/bert-base-chinese`
+ 训练好的模型大小为`1.40GB`，故未上传至Github

## 爬虫功能相关

由于涉及到对微博数据的爬取，所以需要模拟用户的登陆，使用相关功能前需要获取Cookie并保存至后端工作目录的`cookie.txt`文件中。

## 图片回显相关

图片回显使用到了腾讯云的对象存储服务，需要在后端工作目录建立`tx_bucket_key.ini`文件并指定`secret_id`, `secret_key`和`region`
