---
title: "Exploring the Application of Naive Bayes Algorithm Classifier: Theory, Implementation, and Evaluation in Solving the Airline Prediction Problem"
author: "Adekunle Joseph Damilare"
date: "`r Sys.Date()`"
output: 
  html_document:
    toc: true
    toc_float: true
    number_sections: true
    theme: united
    code_folding: hide
    keep_tex: true
    pdf_document: default
---
```{r sd}
set.seed(20)
```
# Introduction

Naive Bayes is a popular and powerful classification algorithm used in machine learning and data mining tasks. It's based on Bayes' theorem, which describes the probability of an event occurring based on prior knowledge of conditions that might be related to the event.

In Naive Bayes classification, the algorithm assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. This assumption is known as "naive" because it simplifies the calculation of probabilities. Despite its simplicity, Naive Bayes often performs surprisingly well in practice, especially in text classification and spam filtering tasks.

One of the key advantages of Naive Bayes is its computational efficiency and ability to handle large datasets with high-dimensional features. It's also relatively easy to implement and requires minimal training data compared to other classification algorithms.

Naive Bayes is a versatile and efficient algorithm that's widely used in various real-world applications due to its simplicity, speed, and effectiveness, especially in scenarios where the independence assumption holds reasonably well.

In this manual, we'll do a short demonstration of this algorithm, and also implement it in R.

To start with, consider the table below, which is the training data. From this data we're to train a model that is capable of predicting Airline type based on some given features. After training this model, new traveller on the test dataset who want to travel, should be able to decide on which of the Airline (A, B, and C) should they book based on those features.In this problem, the target variable is Airline, while the features are Aircraft_Type,	Demand and	Weather_Conditions.	Number_of_Stops
```{r code, include=FALSE}

library(gt)
library(tidyverse)
train <- read.csv("C:/Users/DELL/Desktop/2024_Projects/Healthcare prediction/Full linear regression/train.csv")
test <- read.csv("C:/Users/DELL/Desktop/2024_Projects/Healthcare prediction/Full linear regression/test.csv")
```

```{r dp, cars-table, echo=FALSE, fig.cap="Test dataset"}
train_subset <- sample_n(train, 9) %>% select(Aircraft_Type, Demand, Weather_Conditions, Number_of_Stops, Airline )
gt(train_subset)
```



# Computational Implementation
$$ P(C_k | x) = \frac{P(x | C_k) \cdot P(C_k)}{P(x)}$$
In this formula:

- $P(C_k|x)$ is the posterior probability of class $C_k$ given predictor x.
- $P(x|C_k)$ is the likelihood, the probability of predictor x given class $C_k$.
- $P(C_k)$ is the prior probability of class $C_k$.
- $P(x)$ is the total probability of predictor x, also known as evidence. It's usually computed as the sum of $P(x|C_k) \times P(C_k)$ over all classes.



## Process of Training
We start training the model by calculating the condition probability of each instances. 
\begin{equation}
\text{Probability of Airline A} = P(\text{Airline} = \text{Airline A}) = \frac{3}{9} = \frac{1}{3}
\end{equation}

\begin{equation}
\text{Probability of Airline B} = P(\text{Airline} = \text{Airline B}) = \frac{5}{9}
\end{equation}

\begin{equation}
\text{Probability of Airline C} = P(\text{Airline} = \text{Airline C}) = \frac{1}{9}
\end{equation}

\begin{equation}
n(\text{Airline} = \text{A}) = 3, \quad n(\text{Airline} = \text{B}) = 5, \quad n(\text{Airline} = \text{C}) = 1
\end{equation}

### then calculate condition probability of each attribute

#### Conditional probability of Airbus type

```{r carble, echo=FALSE, fig.cap="Conditional probability of Airbus type"}
library(gt)

# Sample data frame
df <- data.frame(
  Aircraft_Type = c("Airbus A320", "Airbus A380", "Airbus 777","Airbus 737", "Airbus 787"),
  `Airline A` = c(2/3, 1/3, 0/3,0/3,0/3 ),
  `Airline B` = c(0/5, 2/5, 1/5, 1/5,1/5),
  `Airline C` = c(0/1, 0/1, 1/1, 0/1,0/1)
)


# Create a gt table
gt(df)
```

#### Conditional probability of Airbus type

```{r cartable, echo=FALSE, fig.cap="Conditional probability of Airbus type"}
library(gt)

# Sample data frame
f <- data.frame(
  Demand = c("Low", "Medium", "High"),
  `Airline A` = c(3/3, 0/3,0/3),
  `Airline B` = c(4/5,1/5,0/5),
  `Airline C` = c(0/1, 1/1,0/1)
)


# Create a gt table
gt(f)
```

#### Conditional probability of Weather Condition

```{r cars-tabl, echo=FALSE, fig.cap="Conditional probability of Weather Condition"}
library(gt)

# Sample data frame
fg <- data.frame(
  `Weather Condition` = c("Cloudy", "Rain", "Snow", "Clear"),
  `Airline A` = c(1/3,2/3,0/3,0/3),
  `Airline B` = c(1/5,2/5,2/5,0/5),
  `Airline C` = c(0/1,0/1,0/1,1/1)
)


# Create a gt table
gt(fg)
```

#### Conditional probability of Number of stops

```{r cars-tab, echo=FALSE, fig.cap="Conditional probability of Number of stops"}
library(gt)

# Sample data frame
fgp <- data.frame(
  `Number of stops` = c("0", "1", "2", "3"),
  `Airline A` = c(3/3,0/3,0/3,0/3),
  `Airline B` = c(3/5,1/5,0/5,1/5),
  `Airline C` = c(0/1,1/1,0/1,0/1)
)


# Create a gt table
gt(fgp)
```



## Process of Prediction
We're to predict and classifier new instances in the table below. You can assume yourself as a traveler, planning on embarking on a journey, but so confused about the best airline suitable for such journey. Thank God, you have some information regarding the journey. And here we're to help you decide. The machine we've trained will do this job.

```{r dop, cars-table, echo=FALSE, fig.cap="Test dataset"}
test_subset <- sample_n(test, 10) %>% select(Aircraft_Type, Demand, Weather_Conditions, Number_of_Stops)
Airline <- data.frame(Airline = replicate( 10, "?"))
test_subset <- cbind(test_subset, Airline)
gt(test_subset)
```

### Prediction formula

$$
U_{NB} = \text{argMax}_{V_j} \left[ \sum_{n}^{J} P(a_1 | U_j) \right] 
$$

$$
 = argMaxP(U_j)
 $$

Where:


- $U_{NB}$ is the predicted class using Naive Bayes.
- $V_j$ is a class label. 
- $P(a_1 | U_j)$ is the conditional probability of the first attribute given class  $U_j$ .
- $U_j$ is a class label in the set of all possible classes  J .


Following this, we make prediction as follows:


To predict the values for the "Airline" column based on the given features using a Naive Bayes classifier, we would follow these steps:

- Calculate the posterior probability for each class label (airline) given the predictor variables.
- Select the class label with the highest posterior probability as the predicted class for each instance.
- The prediction formula for a Naive Bayes classifier can be represented as:
\begin{equation}
\text{Predicted Class} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} P(\text{Aircraft_Type} = a_1 | \text{Airline} = U_j) \right] \times P(\text{Airline} = U_j)
\end{equation}

Where:

- Predicted Class: Predicted Class is the predicted value for the "Airline" column.
- $V_j$ represents each possible class label (airline).
- $P(Aircraft Type} = a_1 |$ Airline = $U_j)$ is the conditional probability of the "Aircraft_Type" given each class label $U_j$
- $P(Airline = U_j)$ is the prior probability of each class label 
- $J$ represents the set of all possible class labels.


### Predition for the first row

To predict the airline for the first row using the given features (Aircraft_Type, Demand, Weather_Conditions, Number_of_Stops), we can use the Naive Bayes prediction formula:
\begin{equation}
\text{Predicted Class} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} P(\text{Aircraft_Type} = \text{Boeing 777} | \text{Airline} = U_j) \\ \times P(\text{Demand} = \text{Low} | \text{Airline} = U_j) \times P(\text{Weather_Conditions} = \text{Rain} | \text{Airline} = U_j) \\ \times P(\text{Number_of_Stops} = 3 | \text{Airline} = U_j) \right] \times P(\text{Airline} = U_j)
\end{equation}
Here, we substitute the specific values from the condition probability tables:
\begin{equation}
\text{Airline A} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0 \times 1 \times 2/3 \times 0 \right] \times 1/3 = 0
\end{equation}

\begin{equation}
\text{Airline B} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 1/5 \times 4/5 \times 2/5 \times 1/5 \right] \times 5/9 = 0.0071
\end{equation}

\begin{equation}
\text{Airline C} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 1 \times 0 \times 0 \times 0 \right] \times 1/9 = 0
\end{equation}

**Decision:** Since $P(\text{Airline B}) > P(\text{Airline A})$ and $P(\text{Airline B}) > P(\text{Airline C})$, we classify the new instance as Airline B.


### Predition for the second row

To predict the airline for the first row using the given features (Aircraft_Type = Airbus A320, Demand = Low, Weather_Conditions = Snow, Number_of_Stops = 0), we can use the Naive Bayes prediction formula:
\begin{equation}
\text{Predicted Class} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} P(\text{Aircraft_Type} = \text{Boeing 777} | \text{Airline} = U_j) \times \\ P(\text{Demand} = \text{Low} | \text{Airline} = U_j) \times P(\text{Weather_Conditions} = \text{Snow} | \text{Airline} = U_j) \\ \times P(\text{Number_of_Stops} = 0 | \text{Airline} = U_j) \right] \times P(\text{Airline} = U_j)
\end{equation}
Here, we substitute the specific values from the condition probability tables:
\begin{equation}
\text{Airline A} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 2/3 \times 1 \times 0 \times 1/3 \right] \times 1/3 = 0
\end{equation}

\begin{equation}
\text{Airline B} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0 \times 4/5 \times 2/5 \times 5/9 \right] \times 5/9 = 0
\end{equation}

\begin{equation}
\text{Airline C} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0 \times 0 \times 0 \times 0 \right] \times 1/9 = 0
\end{equation}

**Decision:** Since $P(\text{Airline B}) = P(\text{Airline A})$ = P(\text{Airline C})$. This become inconclusive. Hence, we use the Laplace smoothing formula

$$P_{lap_k}(x,y) = \frac{c(x,y)+K}{c(y) + K(X)}$$
Where K is the smoothing parameter and it set to 2.

### Laplace Smoothing Prediction

#### Probability Calculation:

- **Airline A:**
\[
\begin{align*}
&= \left( \frac{2 + 2}{3+2(9)} \times \frac{3 + 2}{3+2(9)}\times \frac{0 + 2}{3+2(9)} \times \frac{0+2}{3+2(9)} \right) \times \frac{1}{3} \\
&= \left( \frac{4}{21} \times \frac{3}{21} \times \frac{2}{21} \times \frac{2}{21} \right) \times \frac{1}{3} \\
&= \left( \frac{48}{194,481} \right) \times \frac{1}{3} \\
&= \frac{48}{583,443} \\
&= 0.0000083
\end{align*}
\]

- **Airline B:**
\[
\begin{align*}
&= \left( \frac{0+2}{5+2(9)} \times \frac{4+2}{5+2(9)}\times \frac{2+2}{5+2(9)} \times \frac{3+2}{5+2(9)}\right) \times \frac{5}{9} \\
&= \left( \frac{2}{23} \times \frac{6}{23} \times \frac{4}{23} \times \frac{5}{23} \right) \times \frac{5}{9} \\
&= \left( \frac{240}{279,841} \right) \times \frac{5}{9} \\
&= \frac{1,200}{2,518,569} \\
&= 0.0005
\end{align*}
\]

- **Airline C:**
\[
\begin{align*}
&= \left( \frac{(0 + 2)}{1+2(9)} \times \frac{0 + 2}{1 + 2(9)} \times \frac{0+2}{1+2(9)} \times \frac{0+2}{1+2(9)} \right) \times \frac{1}{9} \\
&= \left( \frac{2}{19} \times \frac{2}{19} \times \frac{2}{19} \times \frac{2}{19} \right) \times \frac{1}{9} \\
&= \left( \frac{16}{130,321} \right) \times \frac{1}{9} \\
&= \frac{16}{1,172,889} \\
&= 0.0000014
\end{align*}
\]

Now, we can compare these probabilities and make a decision. From the results above, Airline B > Airline A and Airline B > Airline C. Therefore, we predict this instance to be Airline B.


### Predition for new instance in third row

To predict the airline for the first row using the given features (Aircraft_Type = Airbus A380, Demand = High, Weather_Conditions = Clear, Number_of_Stops = 0).
\begin{equation}
\text{Predicted Class} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} P(\text{Aircraft_Type} = \text{Airbus A380} | \text{Airline} = U_j) \times \\ P(\text{Demand} = \text{High} | \text{Airline} = U_j) \times P(\text{Weather_Conditions} = \text{Clear} | \text{Airline} = U_j) \times \\ P(\text{Number_of_Stops} = 0 | \text{Airline} = U_j) \right] \times P(\text{Airline} = U_j)
\end{equation}
Here, we substitute the specific values from the condition probability tables:
\begin{equation}
\text{Airline A} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 1/3 \times 0 \times 0 \times 0 \right] \times 1/3 = 0
\end{equation}

\begin{equation}
\text{Airline B} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 2/5 \times 4/5 \times 0/5 \times 3/5 \right] \times 5/9 = 0
\end{equation}

\begin{equation}
\text{Airline C} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0 \times 0 \times 1 \times 0 \right] \times 1/9 = 0
\end{equation}

**Decision:** Since $P(\text{Airline B}) = P(\text{Airline A})$ = $P(\text{Airline C})$. This become inconclusive. Hence, we use the Laplace smoothing formula


### Laplace Smoothing Prediction

- **Airline A:**
\[
\begin{align*}
&= \left( \frac{1 + 2}{3+2(9)} \times \frac{0 + 2}{3+2(9)}\times \frac{0 + 2}{3+2(9)} \times \frac{0+2}{3+2(9)} \right) \times \frac{1}{3} \\
&= \left( \frac{3}{21} \times \frac{2}{21} \times \frac{2}{21} \times \frac{2}{21} \right) \times \frac{1}{3} \\
&= \left( \frac{24}{194,481} \right) \times \frac{1}{3} \\
&= \frac{24}{583,443} \\
&= 0.00004
\end{align*}
\]

- **Airline B:**
\[
\begin{align*}
&= \left( \frac{2+2}{5+2(9)} \times \frac{4+2}{5+2(9)}\times \frac{0+2}{5+2(9)} \times \frac{3+2}{5+2(9)}\right) \times \frac{5}{9} \\
&= \left( \frac{4}{23} \times \frac{6}{23} \times \frac{2}{23} \times \frac{5}{23} \right) \times \frac{5}{9} \\
&= \left( \frac{240}{279,841} \right) \times \frac{5}{9} \\
&= \frac{240}{2,518,569} \\
&= 0.000095
\end{align*}
\]

- **Airline C:**
\[
\begin{align*}
&= \left( \frac{(0 + 2)}{1+2(9)} \times \frac{0 + 2}{1 + 2(9)} \times \frac{1+2}{1+2(9)} \times \frac{0+2}{1+2(9)} \right) \times \frac{1}{9} \\
&= \left( \frac{2}{19} \times \frac{2}{19} \times \frac{3}{19} \times \frac{2}{19} \right) \times \frac{1}{9} \\
&= \left( \frac{24}{130,321} \right) \times \frac{1}{9} \\
&= \frac{24}{1,172,889} \\
&= 0.00002
\end{align*}
\]

Now, we can compare these probabilities and make a decision. From the results above, Airline B > Airline A and Airline B > Airline C. Therefore, we predict this instance to be Airline B.



### Predition for new instance in fourth row

To predict the airline for the fourth row using the given features (Aircraft_Type = Airbus A320, Demand = Low, Weather_Conditions = Rain, Number_of_Stops = 0).
\begin{equation}
\text{Predicted Class} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} P(\text{Aircraft_Type} = \text{Airbus A320} | \text{Airline} = U_j) \times \\ P(\text{Demand} = \text{Low} | \text{Airline} = U_j) \times P(\text{Weather_Conditions} = \text{Rain} | \text{Airline} = U_j) \times \\ P(\text{Number_of_Stops} = 0 | \text{Airline} = U_j) \right] \times P(\text{Airline} = U_j)
\end{equation}
Here, we substitute the specific values from the condition probability tables:
\begin{equation}
\text{Airline A} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 2/3 \times 3/3 \times 2/3 \times 3/3 \right] \times 1/3 \\ = 36/243 = 0.15
\end{equation}

\begin{equation}
\text{Airline B} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0 \times 4/5 \times 2/5 \times 3/5 \right] \times 5/9 = 0
\end{equation}

\begin{equation}
\text{Airline C} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0 \times 0 \times 1 \times 0 \right] \times 1/9 = 0
\end{equation}

**Decision:** Since $P(\text{Airline A}) > P(\text{Airline B})$ = $P(\text{Airline C})$. We conclude that the new instance is Airline A.


### Predition for new instance in fift row : **Note:** This instance is the same to that of third row. Hence, we proceed to predicting for sixth instance.


### Predition for new instance in sixth row

To predict the airline for the first row using the given features (Aircraft_Type = Boeing 737, Demand = Low, Weather_Conditions = Snow, Number_of_Stops = 1).
\begin{equation}
\text{Predicted Class} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} P(\text{Aircraft_Type} = \text{Boeing 737} | \text{Airline} = U_j) \times \\ P(\text{Demand} = \text{Low} | \text{Airline} = U_j) \times P(\text{Weather_Conditions} = \text{Snow} | \text{Airline} = U_j) \times \\ P(\text{Number_of_Stops} = 1 | \text{Airline} = U_j) \right] \times P(\text{Airline} = U_j)
\end{equation}
Here, we substitute the specific values from the condition probability tables:
\begin{equation}
\text{Airline A} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0/3 \times 3/3 \times 0/3 \times 0/3 \right] \times 1/3 = 0
\end{equation}

\begin{equation}
\text{Airline B} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 1/5 \times 4/5 \times 2/5 \times 1/5 \right] \times 5/9 = 8/625 = 0.01
\end{equation}

\begin{equation}
\text{Airline C} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0/1 \times 0/1 \times 0/1 \times 2/5 \right] \times 1/9 = 0
\end{equation}


**Decision:** Since $P(\text{Airline B}) > P(\text{Airline A})$ = $P(\text{Airline C})$. We conclude that the new instance is Airline B.



### Predition for new instance in seventh row

To predict the airline for the first row using the given features (Aircraft_Type = Boeing 777, Demand = Low, Weather_Conditions = Snow, Number_of_Stops = 1).
\begin{equation}
\text{Predicted Class} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} P(\text{Aircraft_Type} = \text{Boeing 777} | \text{Airline} = U_j) \times \\ P(\text{Demand} = \text{Low} | \text{Airline} = U_j) \times P(\text{Weather_Conditions} = \text{Snow} | \text{Airline} = U_j) \times \\ P(\text{Number_of_Stops} = 1 | \text{Airline} = U_j) \right] \times P(\text{Airline} = U_j)
\end{equation}
Here, we substitute the specific values from the condition probability tables:
\begin{equation}
\text{Airline A} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0/3 \times 3/3 \times 0/3 \times 0/3 \right] \times 1/3 = 0
\end{equation}

\begin{equation}
\text{Airline B} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 1/5 \times 4/5 \times 2/5 \times 1/5 \right] \times 5/9 = 8/625 = 0.01
\end{equation}

\begin{equation}
\text{Airline C} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 1/1 \times 0/1 \times 0/1 \times 1/1 \right] \times 1/9 = 0
\end{equation}


**Decision:** Since $P(\text{Airline B}) > P(\text{Airline A})$ = $P(\text{Airline C})$. We conclude that the new instance is Airline B.


### Predition for new instance in eight row

To predict the airline for the first row using the given features (Aircraft_Type = Boeing 777, Demand = Low, Weather_Conditions = Cloudy, Number_of_Stops = 1). 
\begin{equation}
\text{Predicted Class} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} P(\text{Aircraft_Type} = \text{Boeing 777} | \text{Airline} = U_j) \times \\ P(\text{Demand} = \text{Low} | \text{Airline} = U_j) \times P(\text{Weather_Conditions} = \text{Cloudy} | \text{Airline} = U_j) \times \\ P(\text{Number_of_Stops} = 1 | \text{Airline} = U_j) \right] \times P(\text{Airline} = U_j)
\end{equation}
Here, we substitute the specific values from the condition probability tables:
\begin{equation}
\text{Airline A} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0/3 \times 3/3 \times 1/3 \times 0/3 \right] \times 1/3 = 4/243 = 0.016
\end{equation}

\begin{equation}
\text{Airline B} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 1/5 \times 4/5 \times 1/5 \times 1/5 \right] \times 5/9 = 8/625 = 0.01
\end{equation}

\begin{equation}
\text{Airline C} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 1/1 \times 0/1 \times 0/1 \times 1/1 \right] \times 1/9 = 0
\end{equation}


**Decision:** Since $P(\text{Airline B}) > P(\text{Airline A})$ = $P(\text{Airline C})$. We conclude that the new instance is Airline B.



### Predition for new instance in ninth row

To predict the airline for the first row using the given features (Aircraft_Type = Airbus A380, Demand = Low, Weather_Conditions = Rain, Number_of_Stops = 1):
\begin{equation}
\text{Predicted Class} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} P(\text{Aircraft_Type} = \text{Airline A320} | \text{Airline} = U_j) \\ \times P(\text{Demand} = \text{Medium} | \text{Airline} = U_j) \\ \times P(\text{Weather_Conditions} = \text{Clear} | \text{Airline} = U_j) \times P(\text{Number_of_Stops} = 1 | \text{Airline} = U_j) \right] \\ \times P(\text{Airline} = U_j)
\end{equation}
Here, we substitute the specific values from the condition probability tables:
\begin{equation}
\text{Airline A} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 2/3 \times 0/3 \times 0/3 \times 3/3 \right] \times 1/3 = 0
\end{equation}

\begin{equation}
\text{Airline B} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0/5 \times 1/5 \times 0/5 \times 1/5 \right] \times 5/9 = 0
\end{equation}

\begin{equation}
\text{Airline C} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0/1 \times 1/1 \times 0/1 \times 1/1 \right] \times 1/9 = 0
\end{equation}

**Decision:** Since $P(\text{Airline B}) = P(\text{Airline A})$ = P(\text{Airline C})$. This become inconclusive. Hence, we use the Laplace smoothing formula


### Laplace Smoothing Prediction

- **Airline A:**
\[
\begin{align*}
&= \left( \frac{2 + 2}{3+2(9)} \times \frac{0 + 2}{3+2(9)}\times \frac{0 + 2}{3+2(9)} \times \frac{3+2}{3+2(9)} \right) \times \frac{1}{3} \\
&= \left( \frac{4}{21} \times \frac{2}{21} \times \frac{2}{21} \times \frac{5}{21} \right) \times \frac{1}{3} \\
&= \left( \frac{80}{194,481} \right) \times \frac{1}{3} \\
&= \frac{80}{583,443} \\
&= 0.00014
\end{align*}
\]

- **Airline B:**
\[
\begin{align*}
&= \left( \frac{0+2}{5+2(9)} \times \frac{1+2}{5+2(9)}\times \frac{0+2}{5+2(9)} \times \frac{1+2}{5+2(9)}\right) \times \frac{5}{9} \\
&= \left( \frac{2}{23} \times \frac{3}{23} \times \frac{2}{23} \times \frac{3}{23} \right) \times \frac{5}{9} \\
&= \left( \frac{36}{279,841} \right) \times \frac{5}{9} \\
&= \frac{180}{2,518,569} \\
&= 0.000072
\end{align*}
\]

- **Airline C:**
\[
\begin{align*}
&= \left( \frac{(0 + 2)}{1+2(9)} \times \frac{1 + 2}{1 + 2(9)} \times \frac{0+2}{1+2(9)} \times \frac{1+2}{1+2(9)} \right) \times \frac{1}{9} \\
&= \left( \frac{2}{19} \times \frac{3}{19} \times \frac{2}{19} \times \frac{3}{19} \right) \times \frac{1}{9} \\
&= \left( \frac{36}{130,321} \right) \times \frac{1}{9} \\
&= \frac{36}{1,172,889} \\
&= 0.0000031
\end{align*}
\]

Now, we can compare these probabilities and make a decision. From the results above, Airline A > Airline B and Airline A > Airline C. Therefore, we predict this instance to be Airline A.

### Predition for new instance in tenth row

To predict the airline for the first row using the given features (Aircraft_Type = Boeing 777, Demand = Low, Weather_Conditions = Snow, Number_of_Stops = 1). 
\begin{equation}
\text{Predicted Class} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} P(\text{Aircraft_Type} = \text{Boeing 777} | \text{Airline} = U_j) \times \\ P(\text{Demand} = \text{Low} | \text{Airline} = U_j) \times P(\text{Weather_Conditions} = \text{Snow} | \text{Airline} = U_j) \times \\ P(\text{Number_of_Stops} = 1 | \text{Airline} = U_j) \right] \times P(\text{Airline} = U_j)
\end{equation}
Here, we substitute the specific values from the condition probability tables:
\begin{equation}
\text{Airline A} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 1/3 \times 3/3 \times 2/3 \times 3/3 \right] \times 1/3 = 2/27 = 0.075
\end{equation}

\begin{equation}
\text{Airline B} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 2/5 \times 4/5 \times 2/5 \times 1/5 \right] \times 5/9 = 80/5625 = 0.014
\end{equation}

\begin{equation}
\text{Airline C} = \underset{V_j}{\text{argmax}} \left[ \sum_{n}^{J} 0/1 \times 0/1 \times 0/1 \times 1/1 \right] \times 1/9 = 0
\end{equation}


**Decision:** Since $P(\text{Airline A}) > P(\text{Airline B})$, and $P(\text{Airline A}) > P(\text{Airline C})$. We conclude that the new instance is Airline A.




```{r dopFG, cars-table, echo=FALSE, fig.cap="Test dataset"}
test_subset <- sample_n(test, 10) %>% select(Aircraft_Type, Demand, Weather_Conditions, Number_of_Stops)
Airline <- data.frame(Airline = c( "Airline B", "Airline B", " Airline B","Airline A","Airline B", "Airline B"," Airline B"
                                   ,"Airline B", "Airline A", "Airline A"))
test_subset <- cbind(test_subset, Airline)
gt(test_subset)
```


# Implementation in R

## Unpacking neccessary packages
```{r  dopjjf}
library(dlookr)
library(tidyverse)
library(caret)
library(caTools)
library(e1071)
```
## Data cleaning and Data Wrangling
```{r cleang}
train <- read.csv("C:/Users/DELL/Desktop/2024_Projects/Healthcare prediction/Full linear regression/train.csv")
test <- read.csv("C:/Users/DELL/Desktop/2024_Projects/Healthcare prediction/Full linear regression/test.csv")
train_subset <- sample_n(train, 10) %>% select(Aircraft_Type, Demand, Weather_Conditions, Number_of_Stops, Airline )
head(test, 5)
head(train, 5)
```

```{r gmvg}
emp <-train%>%filter(Airline=="" ) %>% count() %>% summarise(Empty = (n/length(train$Airline))*100)
nempt <-train%>%filter(Airline!="" )%>% count() %>% summarise(Not_Empty = (n/length(train$Airline))*100)
cbind(emp, nempt) %>% pivot_longer(cols = everything()) 

```
## Checking for Missing values
```{r par}
plot_na_pareto(test)
plot_na_pareto(train)
```

## Checking the percentage of missing rows

```{r fig.width=8, fig.height=6}

emp_c <-test%>%filter(Month_of_Travel=="" ) %>% count() %>% summarise(Empty = (n/length(test$Month_of_Travel))*100)
nempt_c <-test%>%filter(Month_of_Travel!="" )%>% count() %>% summarise(Not_Empty = (n/length(test$Month_of_Travel))*100)
cbind(emp_c, nempt_c) %>% pivot_longer(cols = everything()) 



empp <-test%>%filter(Day_of_Week=="" ) %>% count() %>% summarise(Empty = (n/length(test$Day_of_Week))*100)
nemptp <-test%>%filter(Day_of_Week!="" )%>% count() %>% summarise(Not_Empty = (n/length(test$Day_of_Week))*100)
cbind(empp, nemptp) %>% pivot_longer(cols = everything()) 


emp_de <-test%>%filter(Demand=="" ) %>% count() %>% summarise(Empty = (n/length(test$Demand))*100)
nempt_de <-test%>%filter(Demand!="" )%>% count() %>% summarise(Not_Empty = (n/length(test$Demand))*100)
cbind(emp_de, nempt_de) %>% pivot_longer(cols = everything()) 


emp_hs <-test%>%filter(Holiday_Season=="" ) %>% count() %>% summarise(Empty = (n/length(test$Holiday_Season))*100)
nempt_hs <-test%>%filter(Holiday_Season!="" )%>% count() %>% summarise(Not_Empty = (n/length(test$Holiday_Season))*100)
cbind(emp_hs, nempt_hs) %>% pivot_longer(cols = everything()) 


emp_h <-test%>%filter(Weather_Conditions=="" ) %>% count() %>% summarise(Empty = (n/length(test$Weather_Conditions))*100)
nempt_h <-test%>%filter(Weather_Conditions!="" )%>% count() %>% summarise(Not_Empty = (n/length(test$Weather_Conditions))*100)
cbind(emp_h, nempt_h) %>% pivot_longer(cols = everything()) 


emp_hsf <-test%>%filter(Promotion_Type=="" ) %>% count() %>% summarise(Empty = (n/length(test$Promotion_Type))*100)
nempt_hsf <-test%>%filter(Promotion_Type!="" )%>% count() %>% summarise(Not_Empty = (n/length(test$Promotion_Type))*100)
cbind(emp_hsf, nempt_hsf) %>% pivot_longer(cols = everything()) 


emp_hsp <-test%>%filter(Number_of_Stops=="" ) %>% count() %>% summarise(Empty = (n/length(test$Number_of_Stops))*100)
nempt_hsp <-test%>%filter(Number_of_Stops!="" )%>% count() %>% summarise(Not_Empty = (n/length(test$Number_of_Stops))*100)
 cbind(emp_hsp, nempt_hsp) %>% pivot_longer(cols = everything()) 

emp_hspq <-test%>%filter(Aircraft_Type=="" ) %>% count() %>% summarise(Empty = (n/length(test$Aircraft_Type))*100)
nempt_hspq <-test%>%filter(Aircraft_Type!="" )%>% count() %>% summarise(Not_Empty = (n/length(test$Aircraft_Type))*100)
cbind(emp_hspq, nempt_hspq) %>% pivot_longer(cols = everything()) 

```

## Cleaning the dataset : Removing all empty values
```{r cleaning}
test_nempty <-test%>%filter(Airline!="" )
test_nempty <-test_nempty%>%filter(Day_of_Week!="" )
test_nempty <-test_nempty%>%filter(Departure_City!="" )
test_nempty <-test_nempty%>%filter(Arrival_City!="" )
test_nempty <-test_nempty%>%filter(Aircraft_Type!="" )
test_nempty <-test_nempty%>%filter(Promotion_Type!="" )
test_nempty <-test_nempty%>%filter(Number_of_Stops!="" )
test_nempty <-test_nempty%>%filter(Weather_Conditions!="" )
test_nempty <-test_nempty%>%filter(Demand!="" )
test_nempty <-test_nempty%>%filter(Holiday_Season!="" )
test_nempty <-test_nempty%>%filter(Month_of_Travel!="" )
head(test_nempty,5)
test_nempty <-test_nempty%>%drop_na()
```
# Data Exploration: Descriptve Analysis and Feature 

```{r lscfvfv}

Airline <- test_nempty %>% select(Airline, Flight_ID)%>%group_by(Airline)%>%count() %>%summarise(percentage = (n / length(test_nempty$Flight_ID)) * 100)

# Plot the bar plot with data labels as percentages
ggplot(Airline, aes(x = Airline, y = percentage)) +
  geom_bar(stat = "identity", fill = "red") + geom_text(aes(label = paste0(round(percentage, 1), "%")), position = position_dodge(width = 0.9), vjust=-0.5, hjust=0.5, size=7.5) +  # Use geom_text_repel instead
  labs(y = " ",x =" ", title = "Distribution of Airlines:This result shows data balancing") +
  theme_minimal()+ 
  theme(plot.title = element_text(size=25), axis.text.y = element_blank(), axis.text.x = element_text(face="bold", color='black', size = 18, angle = 45, hjust=1))


# Calculate percentage of each category
Day_of_Week <- test_nempty %>%
  group_by(Day_of_Week, Airline) %>%
  count() %>%
  summarise(percentage = (n / nrow(test_nempty)) * 100)

# Plot the bar plot with data labels as percentages
ggplot(Day_of_Week, aes(x = Day_of_Week, y = percentage, fill = Airline)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), position = position_dodge(width = 0.9), vjust=-0.5, hjust=0.5, angle =90, size=7.5) +  # Use geom_text_repel instead
  labs(y = " ",x =" ", title = "Distribution of Airlines by Day of the Week") +
  theme_minimal()+ 
  theme(plot.title = element_text(size=25), axis.text.y = element_blank(), axis.text.x = element_text(face="bold", color='black', size = 18, angle = 45, hjust=1), legend.text = element_text(size = 18, face = "bold"))


# Calculate percentage of each category
Aircraft_Type <- test_nempty %>%
  group_by(Aircraft_Type, Airline) %>%
  count() %>%
  summarise(percentage = (n / nrow(test_nempty)) * 100)

# Plot the bar plot with data labels as percentages
ggplot(Aircraft_Type, aes(x = Aircraft_Type, y = percentage, fill = Airline)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), position = position_dodge(width = 0.9), vjust=-0.5, hjust=0.5, angle =90, size=7.5) +  # Use geom_text_repel instead
labs(y = " ",x =" ", title = "Distribution of Airlines by Aircraft_Type")+
  theme_minimal()+ 
 theme(plot.title = element_text(size=25), axis.text.y = element_blank(), axis.text.x = element_text(face="bold", color='black', size = 18, angle = 45, hjust=1), legend.text = element_text(size = 18, face = "bold"))


# Calculate percentage of each category
Promotion_Type <- test_nempty %>%
  group_by(Promotion_Type, Airline) %>%
  count() %>%
  summarise(percentage = (n / nrow(test_nempty)) * 100)

# Plot the bar plot with data labels as percentages
ggplot(Promotion_Type, aes(x = Promotion_Type, y = percentage, fill = Airline)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), position = position_dodge(width = 0.9), vjust=-0.5, hjust=0.5, angle =90, size=7.5) +  # Use geom_text_repel instead
  labs(y = " ",x =" ", title = "Distribution of Airlines by Promotion_Type") +
  theme_minimal()+ 
 theme(plot.title = element_text(size=25), axis.text.y = element_blank(), axis.text.x = element_text(face="bold", color='black', size = 18, angle = 45, hjust=1), legend.text = element_text(size = 18, face = "bold"))


# Calculate percentage of each category
Number_of_Stops <- test_nempty %>%
  group_by(Number_of_Stops, Airline) %>%
  count() %>%
  summarise(percentage = (n / nrow(test_nempty)) * 100)

# Plot the bar plot with data labels as percentages
ggplot(Number_of_Stops, aes(x = Number_of_Stops, y = percentage, fill = Airline)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), position = position_dodge(width = 0.9), vjust=-0.5, hjust=0.5, angle =90, size=7.5) +  # Use geom_text_repel instead
    labs(y = " ",x =" ", title = "Distribution of Airlines by Number_of_Stops")+
  theme_minimal()+ 
 theme(plot.title = element_text(size=25), axis.text.y = element_blank(), axis.text.x = element_text(face="bold", color='black', size = 18, angle = 45, hjust=1), legend.text = element_text(size = 18, face = "bold"))



# Calculate percentage of each category
Weather_Conditions <- test_nempty %>%
  group_by(Weather_Conditions, Airline) %>%
  count() %>%
  summarise(percentage = (n / nrow(test_nempty)) * 100)

# Plot the bar plot with data labels as percentages
ggplot(Weather_Conditions, aes(x = Weather_Conditions, y = percentage, fill = Airline)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), position = position_dodge(width = 0.9), vjust=-0.5, hjust=0.5, angle =90, size=7.5) +  # Use geom_text_repel instead
   labs(y = " ",x =" ", title = "Distribution of Airlines by Weather_Conditions")+
  theme_minimal()+ 
  theme(plot.title = element_text(size=25), axis.text.y = element_blank(), axis.text.x = element_text(face="bold", color='black', size = 18, angle = 45, hjust=1), legend.text = element_text(size = 18, face = "bold"))



# Calculate percentage of each category
Demand <- test_nempty %>%
  group_by(Demand, Airline) %>%
  count() %>%
  summarise(percentage = (n / nrow(test_nempty)) * 100)

# Plot the bar plot with data labels as percentages
ggplot(Demand, aes(x = Demand, y = percentage, fill = Airline)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), position = position_dodge(width = 0.9), vjust=-0.5, hjust=0.5, angle =90, size=7.5) +  # Use geom_text_repel instead
    labs(y = " ",x =" ", title = "Distribution of Airlines by Demand") +
  theme_minimal()+ 
 theme(plot.title = element_text(size=25), axis.text.y = element_blank(), axis.text.x = element_text(face="bold", color='black', size = 18, angle = 45, hjust=1), legend.text = element_text(size = 18, face = "bold"))


# Calculate percentage of each category
Holiday_Season <- test_nempty %>%
  group_by(Holiday_Season, Airline) %>%
  count() %>%
  summarise(percentage = (n / nrow(test_nempty)) * 100)

# Plot the bar plot with data labels as percentages
ggplot(Holiday_Season, aes(x = Holiday_Season, y = percentage, fill = Airline)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), position = position_dodge(width = 0.9), vjust=-0.5, hjust=0.5, angle =90, size=7.5) +  # Use geom_text_repel instead
    labs(y = " ",x =" ", title = "Distribution of Airlines by Holiday_Season") +
  theme_minimal()+ 
  theme(plot.title = element_text(size=25), axis.text.y = element_blank(), axis.text.x = element_text(face="bold", color='black', size = 18, angle = 45, hjust=1), legend.text = element_text(size = 18, face = "bold"))






# Calculate percentage of each category
Month_of_Travel <- test_nempty %>%
  group_by(Month_of_Travel, Airline) %>%
  count() %>%
  summarise(percentage = (n / nrow(test_nempty)) * 100)

# Plot the bar plot with data labels as percentages
ggplot(Month_of_Travel, aes(x = Month_of_Travel, y = percentage, fill = Airline)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), position = position_dodge(width = 0.9), vjust=-0.5, hjust=0.5, angle =90, size=7.5) +  # Use geom_text_repel instead
  labs(y = " ",x =" ", title = "Distribution of Airlines by Month of Travel") +
  theme_minimal()+ 
  theme(plot.title = element_text(size=25), axis.text.y = element_blank(), axis.text.x = element_text(face="bold", color='black', size = 18, angle = 45, hjust=1), legend.text = element_text(size = 18, face = "bold"))
```




# Model training


```{r model}
test_nempty$Airline <- as.factor(test_nempty$Airline)
test_nempty$Number_of_Stops <- as.factor(test_nempty$Number_of_Stops)

sample <- sample.split(test_nempty$Flight_ID, SplitRatio = 0.8)
test_data <- subset(test_nempty, sample==T)
train_data <- subset(test_nempty, sample==F)
```

## Training the model

```{r train}
NB_AT <-naiveBayes(Airline~ Aircraft_Type, train_data)
NB_NS <-naiveBayes(Airline~ Number_of_Stops, train_data)
NB_DW <-naiveBayes(Airline~ Day_of_Week, train_data)
NB_MT <-naiveBayes(Airline~ Month_of_Travel, train_data)
NB_HS <-naiveBayes(Airline~ Holiday_Season, train_data)
NB_DE <-naiveBayes(Airline~ Demand, train_data)
NB_WC <-naiveBayes(Airline~ Weather_Conditions, train_data)
NB_PT <-naiveBayes(Airline~ Promotion_Type, train_data)
#Let take a look at the results

NB_AT
NB_NS
NB_DW
NB_MT
NB_HS
NB_DE
NB_WC
NB_PT

# let summarise the results
```

```{r sumenefj, results='hide'}
summary(NB_AT)
summary(NB_NS)
summary(NB_DW)
summary(NB_MT)
summary(NB_HS)
summary(NB_DE)
summary(NB_WC)
summary(NB_PT)

```

## Predictions

```{r predic}

pre_NB_AT <- predict(NB_AT, newdata = test_data, response ="class")
pre_NB_NS <- predict(NB_NS, newdata = test_data, response ="class")
pre_NB_DW <- predict(NB_DW, newdata = test_data, response ="class")
pre_NB_MT <- predict(NB_MT, newdata = test_data, response ="class")
pre_NB_HS <- predict(NB_HS, newdata = test_data, response ="class")
pre_NB_DE <- predict(NB_DE, newdata = test_data, response ="class")
pre_NB_WC <- predict(NB_WC, newdata = test_data, response ="class")
pre_NB_PT <- predict(NB_PT, newdata = test_data, response ="class")
```

## Model Evaluation: Confusion Matrix
```{r eva}
eva_NB_AT <- confusionMatrix(as.factor(test_data$Airline), pre_NB_AT)
eva_NB_NS <- confusionMatrix(as.factor(test_data$Airline), pre_NB_NS)
eva_NB_DW <- confusionMatrix(as.factor(test_data$Airline), pre_NB_DW)
eva_NB_MT <- confusionMatrix(as.factor(test_data$Airline), pre_NB_MT)
eva_NB_HS <- confusionMatrix(as.factor(test_data$Airline), pre_NB_HS)
eva_NB_DE <- confusionMatrix(as.factor(test_data$Airline), pre_NB_DE)
eva_NB_WC <- confusionMatrix(as.factor(test_data$Airline), pre_NB_WC)
eva_NB_PT <- confusionMatrix(as.factor(test_data$Airline), pre_NB_PT)
### Results

ca<-eva_NB_AT$byClass
da<-eva_NB_NS$byClass
ea<-eva_NB_DW$byClass
fa<-eva_NB_MT$byClass
ga<-eva_NB_HS$byClass
ha<-eva_NB_DE$byClass
ia<-eva_NB_WC$byClass
ja<-eva_NB_PT$byClass
da <-rbind(ca,da,ea,fa,ga,ha,ia,ja)
da <- as.data.frame(da)


install.packages("gt")
library(gt)
##############


c<-eva_NB_AT$overall
d<-eva_NB_NS$overall
e<-eva_NB_DW$overall
f<-eva_NB_MT$overall
g<-eva_NB_HS$overall
h<-eva_NB_DE$overall
i<-eva_NB_WC$overall
j<-eva_NB_PT$overall
d <-rbind(c,d,e,f,g,h,i,j)
 nam <- c("NB_AT","NB_NS", "NB_DW","NB_MT","NB_HS","NB_DE","NB_WC","NB_PT")
rownames(d)<-nam
head(d)
d <- as.data.frame(d)
d$Model <- nam
d_long <- pivot_longer(d, -Model, names_to = "Metric", values_to = "Value")


ggplot(d_long , aes(x = Metric, y= Value, fill=Model))+geom_bar(stat = "identity", position = "dodge", col="black") +
labs(y = " ",x =" ", title = "Performance metric of the eight models with \n difference features to predict type of Airline") +
  theme_minimal()+ 
  theme(plot.title = element_text(size=25), axis.text.y = element_blank(), axis.text.x = element_text(face="bold", color='black', size = 18, angle = 45, hjust=1), legend.text = element_text(size = 19, face = "bold"))

# Classifications based on multiple conditions
NB_DC_T <-naiveBayes(Airline~Number_of_Stops+Day_of_Week+Month_of_Travel+
                     Holiday_Season+Demand+Weather_Conditions+Promotion_Type , train_data)


pre_NB_DC_T <- predict(NB_DC_T, newdata = test_data, type = "class")
confusionMatrix(pre_NB_DC_T, test_data$Airline)

```

# Conclusion


The performance of the model in this study was observed to be lower than expected. However, it's essential to contextualize this outcome within the broader scope of the study's objectives. The primary aim of this research endeavor was not solely focused on achieving optimal predictive accuracy but rather to demonstrate the application of the Naive Bayes algorithm in the context of airline prediction.

It's crucial to recognize that various machine learning algorithms exist, each with its unique strengths and weaknesses. While Naive Bayes is a valuable tool in certain scenarios, its performance may not always meet expectations, especially in complex prediction tasks like airline classification.

Nonetheless, the study's findings offer valuable insights. One noteworthy aspect highlighted by this research is the implementation of Laplace smoothing to address instances where decision outcomes are inconclusive. Laplace smoothing, also known as additive smoothing, is a technique used to handle cases where certain combinations of feature values have zero probabilities in the training data. By introducing a small amount of pseudo-counts to all observed feature-value combinations, Laplace smoothing prevents zero probabilities and ensures more robust model predictions.

In summary, while the observed performance of the Naive Bayes model may be modest, the study provides valuable methodological insights and highlights the importance of considering alternative approaches, such as Laplace smoothing, to enhance the robustness and reliability of predictive models in real-world applications.
