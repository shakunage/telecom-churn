# Installing required packages
install.packages('caret')
install.packages('recipes')
install.packages('ggcorrplot')

# Importing the required library
library(caret)
library(recipes)
library(ggcorrplot)
library(tidyr)

# Loading the training dataset into a variable
telecom_train <- read.csv("./telecom_train.csv", sep=";")

# Inspecting the dataset
str(telecom_train)

# Output: 

# 'data.frame':	3516 obs. of  18 variables:
#   $ female           : int  0 0 1 1 0 0 0 1 1 0 ...
# $ SeniorCitizen    : int  0 0 0 0 0 0 0 0 0 0 ...
# $ partner          : int  0 0 0 1 0 0 0 1 0 0 ...
# $ dependents       : int  0 0 0 0 1 0 0 1 0 1 ...
# $ tenure           : int  2 45 10 28 62 16 49 69 52 71 ...
# $ phone.service    : int  1 0 0 1 1 1 1 1 1 1 ...
# $ multiple.lines   : int  0 0 0 1 0 0 1 1 0 1 ...
# $ online.security  : int  1 1 1 0 1 0 0 1 0 1 ...
# $ online.backup    : int  1 0 0 0 1 0 1 1 0 0 ...
# $ device.protection: int  0 1 0 1 0 0 1 1 0 1 ...
# $ tech.support     : int  0 1 0 1 0 0 0 1 0 0 ...
# $ streaming.tv     : int  0 0 0 1 0 0 1 1 0 1 ...
# $ streaming.movies : int  0 0 0 1 0 0 1 1 0 1 ...
# $ contract.one.year: int  0 1 0 0 1 0 0 0 1 0 ...
# $ contract.two.year: int  0 0 0 0 0 1 0 1 0 1 ...
# $ paperless.billing: int  1 0 0 1 0 0 1 0 0 0 ...
# $ monthly.charges  : chr  "53,85" "42,3" "29,75" "104,8" ...
# $ churn            : int  1 0 0 1 0 0 1 0 0 0 ...



# We can see that the dataset contains 18 variables, 16 of which are categorical, while the remaining two being continous.
# The target variable is called 'churn', 1 meaning that the customer churned (cut the contract) and 0 indicating no churn. 

telecom_train_churns <- telecom_train[,18]==1　
sum(telecom_train_churns)
sum(telecom_train_churns)/length(telecom_train_churns)
# Output: 
# 933
# 0.2653584

# Around 26.5% of the entries in the dataset are churns. 

# We can run t-tests on the two continuous varibles to see if they predict 'churn'. 
# First, let's turn the character string variable into a numeric one. 

charges_vector <- unlist(telecom_train[,17])
charges_vector <- scan(text=charges_vector, dec=",", sep=".")
telecom_train[,17] <- c(charges_vector)

# Now that the values in monthly.charges are in numeric format, let's do the t-tests. 

ttest_tenure <- t.test(tenure~churn,data=telecom_train)
ttest_tenure

ttest_monthlycharges <- t.test(monthly.charges~churn,data=telecom_train)
ttest_monthlycharges

# Both t-tests return results with low p-values (2.2e-16), and thus, are likely to be associated with churn. 

# Next, let us generate a correlation matrix for between the variables.

corr<-round(cor(telecom_train),1)
ggcorrplot(corr, lab = TRUE)

# Looking at the correlation matrix, the most highly correlated independent variable seems to be ‘tenure’, which has a negative 0.4 correlation to ‘churn’. 
# This means that the longer the customer has been subscribed, the lower the chance is that they are going to cancel their contract. 
# Other than that, there are some independent variables that seem to be correlated with each other. 
# Subscription services (‘streaming.tv’ & ‘streaming.movies’) are significantly correlated with ‘monthly.charges’. Naturally, subscription to additional services seems to increase the total monthly cost to the customer. 
# Some variables in the data have overlapping definitions. 
# The variable ’tenure’ is obviously highly correlated to ‘contract.two.year’ as they essentially measure the same thing, contract length in a different manner. 

# Now, let's move on to the next stage. Here, we will build a model based on the training set.
# Then, we'll continue on to validate that model with the test set.  

churn_regression <- glm(churn ~ female + SeniorCitizen + partner + dependents + tenure + phone.service + multiple.lines + online.security + online.backup + device.protection + tech.support + streaming.tv + streaming.movies + contract.one.year + contract.two.year + paperless.billing + monthly.charges, family=binomial, data=telecom_train)
summary(churn_regression)

exp(churn_regression$coefficients)

# Output: 

# (Intercept)       -1.079732   0.181657  -5.944 2.78e-09 ***
# female             0.032552   0.091327   0.356  0.72152    
# SeniorCitizen      0.115150   0.118424   0.972  0.33087    
# partner           -0.051888   0.109326  -0.475  0.63506    
# dependents        -0.163628   0.125021  -1.309  0.19060    
# tenure            -0.035086   0.003315 -10.583  < 2e-16 ***
# phone.service     -1.245055   0.188513  -6.605 3.99e-11 ***
# multiple.lines    -0.064116   0.114852  -0.558  0.57667    
# online.security   -0.559968   0.117672  -4.759 1.95e-06 ***
# online.backup     -0.237403   0.109222  -2.174  0.02974 *  
# device.protection -0.091455   0.112696  -0.812  0.41707    
# tech.support      -0.582976   0.118111  -4.936 7.98e-07 ***
# streaming.tv      -0.130957   0.120892  -1.083  0.27870    
# streaming.movies  -0.076039   0.120731  -0.630  0.52881    
# contract.one.year -0.649845   0.149170  -4.356 1.32e-05 ***
# contract.two.year -1.530186   0.257859  -5.934 2.95e-09 ***
# paperless.billing  0.272464   0.103742   2.626  0.00863 ** 
# monthly.charges    0.040115   0.003418  11.738  < 2e-16 ***

# (Intercept)            female     SeniorCitizen           partner        dependents            tenure     phone.service    multiple.lines   online.security 
# 0.3396864         1.0330877         1.1220418         0.9494349         0.8490577         0.9655225         0.2879251         0.9378962         0.5712275 
# online.backup device.protection      tech.support      streaming.tv  streaming.movies contract.one.year contract.two.year paperless.billing   monthly.charges 
# 0.7886736         0.9126021         0.5582347         0.8772554         0.9267802         0.5221265         0.2164954         1.3131965         1.0409310

# Out of the statistically significant variables, only ‘monthly.charges’ and ‘paperless.billing’ have a positive relationship with churning. 
# Higher monthly bill seems to have a relationship of cancelling contract, while customer who choose paperless billing seem to be more prone to cancel as well.
# The variable contract.two.year has the smallest coefficient, so it has the largest effect to the target variable.

# In the final phase, we will test our model with the test dataset, and analyze the results.
# Let's start by reading the test data into a variable, and fixing the data a bit. 


telecom_test <- read.csv("telecom_test.csv", sep=";")
charges_vector <- unlist(telecom_test[,17])
charges_vector <- scan(text=charges_vector, dec=",", sep=".")
telecom_test[,17] <- c(charges_vector)

# Now, we can validate our model and generate the confusion matrix to illustrate accuracy and calculate performance indicators easily.

predicted_values <- predict(churn_regression, newdata=telecom_test, type="response")

predicted_binary <- ifelse(predicted_values > 0.5, 1, 0)

y <- as.numeric(unlist(telecom_test[18]))

data <- data.frame(Actual = y, Prediction = predicted_binary)
table(data$Prediction, data$Actual)

# Output: 

#   0    1
# 0 2329  430
# 1  251  506

# The columns represent the actual values, while the rows represent the guesses our model made on the training data. 
# From this confusion matrix, we can calculate the following, rounded to the nearest percentage point:
# Accuracy: 81%
# Precision: 67%
# Recall 54%

# Overall, the performance seems pretty good at a first glance - it got 81% of the it's predictions right in the test dataset.
# However, predicting the minority class was difficult, with only 67% of churn predictions hitting the mark.
# Out of all churning customers, the model was able to detect 54% of them. 
# If the cost of missing a churning customer is high and falsely identifying a non-churning customer as churning is low, we could consider tuning the model more aggressive, making it predict more customers as churning.
# This however is something that needs to be looked into independently. 

# In conclusion, to increase customer retention, the company should consider locking customers into 2-year contracts.
# This what the data suggests, but in practice, the regulatory environment does not allow that in many regions (eg. the EU and Japan).
# A more realistic change to increase customer retention could be to offer a discount, either for a fixed period or indefinitely, on existing plans for customers who apply for cancellation.
# Another way to implement a similar scheme would be to offer “win-back” offers for customers who have just churned, sending them an offer that could potentially undercut competition. 
# One marketing trigger to increase retention could be to offer online backup and security services at a discounted price for customers who might be about to churn, or just otherwise direct marketing to them that would persuade them to use the services.
