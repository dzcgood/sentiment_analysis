# 情感分析系统

## 作者信息

+ 学号：2023244069
+ 姓名：邓智超
+ 学院：智能与计算学部

## 组织架构

+ `back-end`：模型构建、文本爬取、网页后端实现
+ `webapps`：网页前端代码

## 运行

### 前端

```shell
npm install
npm run serve
```

### 后端

创建虚拟环境

```shell
conda create --name sentiment_analysis python=3.7
activate sentiment_analysis
```

安装依赖

```shell
cd back-end
pip install -r requirements.txt
```

运行后端程序

```shell
python app.py
```

