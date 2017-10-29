# 30 Day Readmissions

This dataset is a small sample of a geriatric population in a hospital setting with a number of features. The goal of this code challenge was to first create the target column and then build a model to predict it.

---

Description:

30-Day All-Cause Hospital Readmissions is a quality measure that many healthcare organizations use to track their performance. Lower readmission rates indicate better patient outcomes, while higher ones tend to indicate system problems that are negatively impacting patients. The goal of this exercise is to analyze a dataset that simulates hospitalizations for a geriatric patient population in 2015 and 2016 to predict if a patient is likely to have a readmission based on the information available at the time of their initial admission.

Data Dictionary:
- admissions.csv
    - Patient - a unique patient identifier string.
    - AdmitDate - start date of hospital admission (yyyy-MM-dd formatted).
    - LOS - length of hospital stay for the admission in days.
- claims.csv
    - Patient - a unique patient identifier string.
    - AdmitDate - start date of hospital admission (yyyy-MM-dd formatted).
    - Age - patient's current age at time of admission.
    - Gender - single character gender value for the patient (limited to 2 values for simplicity).
    - PrimaryDx - the primary diagnosis code for the patient.
    - Dx2 - the secondary diagnosis code for the patient (nullable).
    - Dx3 - the tertiary diagnosis code for the patient (nullable).
    - PastPCPVisits - the count of primary care physician visits the patient had in the 12 months prior to admission.

---

I completed it in three stages: EDA, Gridsearch Models, and Stacking Models. This was my first attempt at stacking models and I'm happy I was able to achieve an accuracy 2% better with stacking.

In addition, what I found was that when including all original features in the dataset stacking wasn't able perform better than individual models. Same was true for using only the stacked model outputs as the features. However, when I used the 20 top features based on feature importance via Logistic Regression's coefs and Random Forest's feature importances the stacking model performed best.

I followed this structure for [stacking](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)

---
