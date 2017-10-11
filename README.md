# Side-Learning
This is the repo where I will be saving my projects and ongoing learning. Some have been inspired by Kaggle others are inspired by courses or tutorials I've found online.

I am excited to continue learning/using classic ML algorithms for data science projects and have also been learning keras and exploring deep learning in computer vision and natural language processing. In doing so, built my own gpu server at home which has made my neural net learning rate so much quicker.

I've also created an [overview of machine learning methods](https://docs.google.com/spreadsheets/d/1lOBXArptpihQ3WFSC6C6D1X9e4KPZjYbLoU6v3XtbTw/edit?usp=sharing) for my benefit. A work in progress, I add everything I learn about new algos, terminology, and logically useful things about dealing with datasets.

---
- Models_Scratch
  - Things Learned:
    - OOP, inheritance, data manipulation in numpy, better understanding of Random Forests,
- Cats_Dogs_Redux - Notebooks to explore and produce top 22% score on Kaggle
  - Things learned:
    - Data augmentation, the value of a sample set, finetuning, the effects of different initializers and learning rates, convolutional architectures (VGG, ResNet, Inception), batch training and loading from directory, bottleneck training, working with data larger than working memory (both gpu and ram), ensembling, predicting and evaluating model performance, saving/loading model weights, how to build a linux gpu box and how to configure ssh, port forwarding, ddns for public ip,
  - [Kaggle Page](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/discussion/27613)
- Flask API
  - Things learned:
    - Curl, host a webapp, GET and POST requests, structure a RESTful API, create a html form, send POST requests via html logic, port forwarding
  - GET requests to pull forecast and product data
    - Filters based on sku(s), wk_range(s), product category(s)
  - POST requests to update underlying forecast data
    - HTML table/form GUI to update future forecast values
- Mnist - Multiple notebooks learning NNs with Mnist and Fashion Mnist datasets
  - Things learned:
    - The affect of size of data as it relates to ability of model to learn, how convolutions work, how to construct autoencoders, input shapes and reshaping data, batchnorm
- NLP Notes
  - Notes based on Stanford's NLP overview course from coursera in 2012
  - [Course Videos](https://www.youtube.com/playlist?list=PLqNqLI7n_fDbisqKkkAzrFpWQOg8E6KEf)
- NLTK Exploration
  - Notes, materials, and code for Stanford's Deep Learning NLP course CS224d
  - [Course Syllabus](http://cs224d.stanford.edu/syllabus.html)
- POS tagger
  - Tutorial of how to build a POS tagger from scratch.
  - Load treebank pretagged dataset from NLTK
  - Create word features to differentiate POS
  - Compare preformance to NLTK built-in POS tagger
