
<!-- Add banner here -->

<p align="center">
  <img src="https://github.com/DSPOWER93/Data/blob/main/GitHub_nlp_question_banner.png?raw=true" alt="Sublime's custom image"/>
</p>


![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)
![GitHub last commit](https://img.shields.io/badge/last%20commit-Dec--2021-blue)
![GitHub issues](https://img.shields.io/github/issues-raw/navendu-pottekkat/awesome-readme)
![GitHub](https://img.shields.io/github/license/navendu-pottekkat/awesome-readme)

# Inappropriate Question Classifier-API

The Inappropriate question API aims to identify questions that are asked with intent of having insincere perpespective in form of scores. Classifier identifies questions that are having toxic and divisive content. API helps to keep online trolls at bay by keeping web content adheres policy of "Be Nice & Respectful"

### Sample business Use Cases

API can flag inappropriate content in: 
- **Public forum Q&A sessions**
- **E-commerce websites product Q&A**
- **Customer support tickets**

### Business benefits

- **Reduces trolls infiltration to platform**
- **Faster resolution time on reported tickets**


## Model Deployment Structure

<p align="center">
  <img src="https://github.com/DSPOWER93/Data/blob/main/deployment%20flow.png" alt="Sublime's custom image"/>
</p>

### **Flow**: 
➤➤Client Sends data using /POST command to AWS HTTP Gateway rest API.

➤➤ AWS HTTP gateway Rest API triggers AWS Lambda to spin up the instance.
       
➤➤AWS Lambda Instance runs the python handler. Handler access AWS EFS to get Python Packages & model Artifacts back to handler.

➤➤Handler sends reponse to post request with results, through HTTP gateway back to client.

## Model Visuals

<p align="center">
  <img src="https://github.com/DSPOWER93/artifacts/blob/main/Model_vis.png" alt="Sublime's custom image"/>
</p>

## Shortlisted Models

<p align="center">
  <img src="https://github.com/DSPOWER93/artifacts/blob/main/model_comparison.png" alt="Sublime's custom image"/>
</p>

Out of the two NLP models, Hybrid Bi-LSTM model was finalized as it met following criteria: <br/>

- **Computationally in-expensive:** As Lambda Handler Instance has been set up with lower capacity, a light model would be a better fit compared to transformers. To Set up a transformer model rendering, a powerful infrastructure needs to be chosen.
- **Execution Speed:** Though transformer models are designed for parallel processing. A transformer model have high set of parameters, it makes it bit slower to render results faster from low capacity server.<br/>
 
*The AUC score of LSTM model is almost at par with transformer model post multiple hypertuning. It seem to be optimum fit for business requirement. 

Training files can be referred from following links:<br/>
- **Hybrid LSTM:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/DSPOWER93/quora-insincere/blob/main/Bi_LSTM_insincere_question_Classifier.ipynb)<br/>
- **Distilbert:**      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/DSPOWER93/quora-insincere/blob/main/Distilbert_Classifier_Insincere_Questions.ipynb)

## Data Source

The Data for the project has been taken from Kaggle competition 
[Quora Insincere Question Classification](https://www.kaggle.com/c/quora-insincere-questions-classification)

## API Inference

#### API Inference through Python, also can be tested through [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/DSPOWER93/Insincere-Questions/blob/main/Lambda_API_Deployment_File.ipynb)<br/>

Invoke url = https://bl5i4vptna.execute-api.us-east-1.amazonaws.com/dev/api

```
/POST Method

# Python Packages
import requests, json

>>url = "https://bl5i4vptna.execute-api.us-east-1.amazonaws.com/dev/api"
```

#### Low Insincerity Content
```python
>>data = json.dumps({"queryStringParameters":{
                        "text":"what time does sun rise in Japan"
      		        }})
>>r = requests.post(url, data)
>>print(r.json())

{'statusCode': 200, 'output': '"0.004349351"'}
```

#### High Insincerity Content
```python
>>data = json.dumps({"queryStringParameters":{
                    "text":"does USA hate gay"
      		        }})
>>r = requests.post(url, data)
>>print(r.json())

{'statusCode': 200, 'output': '"0.8821814"'}
```

## Localhost Deployment

To reproduce API on Localhost using rest API Flask, following code to be ran on bash: 

**Creating Virtual Environment**
```bash
$virtualenv virtualenv_name
$virtualenv -p /usr/bin/python3.8 virtualenv_name
$source virtualenv_name/bin/activate
```
**Model Deployment**
```bash
(virtualenv_name)$git clone https://github.com/DSPOWER93/quora-insincere.git
(virtualenv_name)$pip install -r requirements.txt 
(virtualenv_name)$python -m spacy download en_core_web_sm
(virtualenv_name)$python3 app.py
```
Model Inferencing would result same at localhost, just Invoke Url would local host link generated by Flask Rest API.

## FAQs

- **Why Hybrid Model was selected compared to normal Bi-LSTM Model?**<br />

The conclusion to reach on Hy-brid model was after many multiple attempts and hyper tuning to nlp models. Though Simple Bi-LSTM had resulted good outputs but to hypertuning with additional layers has improvised AUC better, I had used conv1D in addition to following reasons:<br />
➤ As Conv1D helps in retaining sequential information between the words better, conv1D comes as a good choice for 1D array.<br />
➤ Adding multiple conv1D channels helps model understanding data from multiple perspective, these channels are working as multiple variables to model without having collinearity. The enlightenment to use this technique was taken from following article kaggle first poistion by Psi [Kaggle Article](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568).

- **Why is AWS Lambda selected to deploy API instead of other AWS infrastructure?**<br />
 
The primary reason behind deploying to AWS Lambda is the request load, as the API doesn't recieve high or frequent request load so AWS Lambda turns out to be best choice. If request load reaches to certain threshold, then would move it to infrastructure like EC2 or ECS.

- **Upgrade Modules of API?**<br />

There are two part in upgrading API:<br /> 
➤**Model Weights**: If model weights are upgraded over time then model Artifacts to be updated in AWS EFS.<br />
➤**Model Architecture**: If Model Architecture is changed then Lambda handler & AWS EFS needs to be updated.<br />

- **How is model designed to deal with New/unseen data?**<br />

Pre-trained word Embedding would come to rescue here. Model has built dependency pre-trained embedding **Glove**, embeddings would play major role in classifying unseen data. If performance of API drops, then model finetuning would be required.

## API infrastructure Limitations:

- **Computation power:** The API is deployed on on Severless Lambda Instance which is designed to take low size request. If traffic to API Increases exponentially, then present API infrastructure would face challenges.<br />
- **Rendering speed:** Lambda Instance is designed to spin instance of max 2GB RAM, which is not sufficient to deal with request coming in at extreme rate for e.g. 10000+ request at single instance.


## About me

I am Mohammed working as Sr. Business Analyst @ Affine.ai. I am presently working gaming Industry, delivering end to end Machine Learning projects based on clients requirement. 
Github profile:
[![Mohammed](https://img.shields.io/badge/Github-white?style=flat&logo=github&labelColor=black)](https://github.com/DSPOWER93/)


#### 👀 We can connect on <br/>
[![Mohammed](https://img.shields.io/badge/Linkedin-blue?style=flat&logo=Linkedin&labelColor=blue)](https://www.linkedin.com/in/mohammed-taher-13934a51/)
[![Mohammed](https://img.shields.io/badge/Gmail-white?style=flat&logo=gmail&labelColor=white)](mailto:md786.52@gmail.com)
[![Mohammed](https://img.shields.io/badge/Instagram-white?style=flat&logo=Instagram&labelColor=white)](https://www.instagram.com/mdboy93/)

