# Models from Scratch

I spoke with a Data Scientist at a meetup recently that mentioned he had found building models from scratch very rewarding. He said it gave him great intuition on the underlying calculations and helped him learn the different styles of models; Lazy, Greedy, etc.

With some inspiration from [Machine Learning Mastery](https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/) I decided to build a few and compare my performance to Sklearn. I modified the approach taken in the tutorials to use numpy and created classes to emulate the Sklearn API. Then used Sklearn and mine implementation to compare performance and timing.

This was a 2 day project which I plan to continue with other algos like K-means and heirarchical clustering. It was enlightening from a machine learning perspective and from a performance perspective. My algorithms have a long way to go from performing optimally, but at this point they are functionally correct and are on pace with Sklearn's accuracy.

---

### Models:
- K-Nearest-Neighbors - OOP implementation of KNN for Classification
  - Learned:
    - Stronger knowledge of numpy, manipulations of array shapes,
  - Logic:
    - On majority vote ties, tie goes to the highest frequency class in the training data
  - Metrics:
    - Euclidean distance - the straight line distance between points
    - Manhattan distance - the path between points which only travels along the axes to get from one to another

- Decision Trees - OOP implementation of DTs for Classification
  - Learned:
    - How to use recursion within a class, stack/concatenate in numpy,
  - Metrics:
    - Gini
    - Would like to add Entropy
  - Logic:
    - min_num_samples for a split
    - max_depth to grow tree

- Random Forest - OOP implementation of Random Forest inheriting the DT class
  - Learned:
    - How to build a class on top of an inherited class. This is my first attempt so this may not be the most appropriate way to go about it, but it does work successfully.
    - That at each level of the trees within the forest, the algo splits on a subset of different features. This was left at the 'recommended amount' sqrt(input_features)
    - Which variables should be saved as instance variables and which need to be local variables for recursion.
  - Logic:
    - num_trees to build the forest from
    - sample_ratio to determine amount of sampling with replacement each tree should be based on from the original dataset

---

### Compare with Sklearn
- Models from scratch
  - Shows working implementations of my models and compares performance between them and the Sklearn versions
- scratch.py
  - Allows the import of all my models
  - Includes an accuracy function to compare between Sklearn
