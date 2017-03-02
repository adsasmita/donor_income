# donor_income

A machine learning project using [Scikit-learn](http://scikit-learn.org/stable/) Python library

## Files

To open the main code, simply open [`donor_income.ipynb`](https://github.com/adsasmita/donor_income/blob/master/donor_income.ipynb) on any desktop browser, or you can download and run the cells in a Python 2 environment. The code is presented in a [Jupyter Notebook](https://github.com/jupyter/notebook) / iPython notebook for readability purposes.

Visualization codes are contained in [`visuals.py`](https://github.com/adsasmita/donor_income/blob/master/visuals.py)

## Overview

Suppose that we were tasked to identify donors for a charity. The organization decided that potential donors are individuals that have yearly income of more than $50,000. We agreed to approach this task by employing several supervised algorithms to accurately model individuals' income using data collected from the 1994 U.S. Census. We will then choose the best candidate algorithm from preliminary results and further optimize it to best model the data.

Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with. While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features.

## Data

The dataset used in this project is included as `census.csv`. The dataset originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper [*"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",*](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf) by Ron Kohavi.

**Features**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Masters, Doctorate, Assoc-voc,etc)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status
- `occupation`: Work Occupation
- `relationship`: Relationship Status
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country

**Target Variable**
- `income`: Income Class (<=50K, >50K)

## Dependencies

This project requires **Python 2.7** and the following Python libraries installed:

* [Scikit-learn](http://scikit-learn.org/stable/)
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [Bokeh](http://bokeh.pydata.org/en/latest/) and [matplotlib](http://matplotlib.org/) for data visualization



