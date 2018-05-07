## The Prediction of Credit User's Overdue event Based on Machine Learning Method 


### Project description
* Project 	Source
	* This is a competition project. You can check some details from the official website of this [competition](https://open.chinaums.com/#/intro). Also, you can see my [proposal](./Proposal.md). 
* Project Valuation
	* The competition will select candidate finalists by evaluating **F1 score** on test dataset. If you become one of the candidates, you should submit a document which expains your model. Here is my [document](./document)(about 30 pages).

* Project Requirement(python packages)
	* **missingno** (The missingno library provides a few further useful expressions for a deeper look into this subject. There are some useful method to **visualize the data**)
	* **Impyute** (Impyute is a library of missing data imputation algorithms. Data imputations library to preprocess datasets with missing data. More infomation you can check the website [http://impyute.readthedocs.io/](http://impyute.readthedocs.io/))
	* **imbalanced-learn** (This package aims to process the imbalaned data. There are many methods,including oversapmling and under-sampling)
	* **keras and tensorflow-gpu** (using Deep Neural Network method with GPU)
	* **skicit-learn**
### Data
There are totally 199 input variables which may account for the overdue event.On the whole, They are divided into six categories.

<table>
  <tr>
    <th width=10%, bgcolor=yellow >参数</th>
    <th width=40%, bgcolor=yellow>详细解释</th>
    <th width="50%", bgcolor=yellow>备注</th>
  </tr>
  <tr>
    <td bgcolor=#eeeeee> -l </td>
    <td> use a long listing format  </td>
    <td> 以长列表方式显示（显示出文件/文件夹详细信息）  </td>
  </tr>
  <tr>
    <td bgcolor=#00FF00>-t </td>
    <td> sort by modification time </td>
    <td> 按照修改时间排序（默认最近被修改的文件/文件夹排在最前面） </td>
  <tr>
    <td bgcolor=rgb(0,10,0)>-r </td>
    <td> reverse order while sorting </td>
    <td>  逆序排列 </td>
  </tr>
</table>

* **Identity information and property status** 
	* gender, age, house,heath and etc.
* **Bank card holding information**
	* bank level, bank location,bank hoding numbers and etc.
* **Transaction information**
	* The last six months' balance,The last six months'trading amount and etc.
* **Lending information**
	* max/min amount of money in 30/90/180 days,the last amount of money and etc.
* **Repayment information**
	* times of sucessed/failed repayment in 30/90/180 days, max/min repayment in 30/90/180 days and etc.
* **Application for loan information**
	* numbers for loan in 30/90/180 days, numbers for sucessed/failed loan in 30/90/180 days and etc.
### Features 



### Methods





### Implementation


### Conclusion
* **Results and Findings**

	* The models seemed to perform better without the missing values being imputed than when imputing the missing values
	* Stacking and Voting and the combination of the two models generally tend to have very high predictive power compared to plain Ensemble models
	* Feature Engineering improved the AUC score for single models (Naive Bayes and Logistic Regression) from ~0.7 to ~0.85 but did not have much impact on the Tree based methods
	* The incremental increase in the predictive accuracy (AUC) is of the order of 0.0001 as we move towards the top of the Kaggle leaderboard (top 2%) and the tuning gets a lot harder
* **Lessons Learned and Insights**
	* Hyperparameter tuning is a very time consuming process and it is better to have the team split this effort and work in parallel
	* Cross Validation is very critical and it is worth spending time testing the impact of various folds on the model accuracy
	* The model needs to be tuned at a much more granular level as the dataset gets smaller in size (both in terms of number of features and observations)
	* Following an Agile parallel process has continued to be a proven factor for maximizing success
* **Future Work**
	* Tune the parameters for Deep Learning using Theano / Keras and compare the predictive accuracy and performance against Stacking / Voting models
	* Explore the possibility of adding new polynomial and transformed features, and evaluate the predictive accuracy.
### References
[1]	Lang, W. (2009). Consumer Credit Risk Modeling and the Financial Crisis. International Atlantic Economic Conference.

[2]	Marlin, B. M. (2008). Missing data problems in machine learning. University of Toronto.

[3]	Kitchin, R. (2015). The opportunities, challenges and risks of big data for official statistics. Statistical Journal of the Iaos, 31(3), 471-481.

[4]	孙存一, & 王彩霞. (2015). 机器学习法在信贷风险预测识别中的应用. 中国物价(12), 45-47.

[5]	Lacković, I. D., Kovšca, V., & Vincek, Z. L. (2016). Framework for big data usage in risk management process in banking institutions. Central European Conference on Information and Intelligent Systems, International Conference 2016 / Hunjak, Tihomir ; Kirinić, Valentina ; Konecki, Mario.


