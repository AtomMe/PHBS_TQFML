# The Prediction of Credit User's Overdue event Based on Machine Learning Method

----------

## Team members
Tang Jieqiang (Atom)
## Idea
This project aims to implenment Machine Learning method to predict whether or not a loan will be overdue. This can help banks decide who can get finance and on what terms and can make or break investment decisions. The model was trained on multi-demension input data. 

## Dataset Description
### Output Variables
* **y_prediction**
	* The user overdue performance prediction of the validation set. 0 is not overuse and 1 is overdue.
### Input Variables
There are totally 199 input variables which may account for the overdue event.On the whole, They are divided into six categories.

* **Identity information and property status** 
	* gender, age, house,heath and etc.
* **Bank card holding information**
	* bank level, bank location,bank hoding numbers and etc.
* **Transaction information**
	* The last six months' balance,The last six months'trading amount and etc.
* **Lending information**
	* max/min amount of money in 30/90/180 days,the last amount of money and etc.
* **Lending information**
	* max/min amount of money in 30/90/180 days,the last amount of money and etc.
* **Repayment information**
	* times of sucessed/failed repayment in 30/90/180 days, max/min repayment in 30/90/180 days and etc.
* **Application for loan information**
	* numbers for loan in 30/90/180 days, numbers for sucessed/failed loan in 30/90/180 days and etc.
### Dataset source

[A competition dataset.](https://open.chinaums.com/#/intro)

## Model
* LR
* Deep Neural Networks
* SVM
* Desicion Tree



