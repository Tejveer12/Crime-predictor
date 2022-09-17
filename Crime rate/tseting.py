{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a5dd2d35",
   "metadata": {},
   "outputs": [],
   "source": [
    "from joblib import load\n",
    "from sklearn.metrics import mean_squared_error,accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "436e62ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "model=load(\"ViolentCrimePerPop.joblib\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4556d8d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "%store -r test_set\n",
    "%store -r my_pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c5e402b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_features=test_set.drop(\"ViolentCrimesPerPop\",axis=1)\n",
    "test_label=test_set[\"ViolentCrimesPerPop\"].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "420e173a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(399, 123)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_features.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a3e947e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(399,)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_label.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "0f266c9c",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_features_tr=my_pipeline.fit_transform(test_features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "17f0a70f",
   "metadata": {},
   "outputs": [],
   "source": [
    "predicted=model.predict(test_features_tr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "fc44eb6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "error=mean_squared_error(test_label,predicted)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3d7c8587",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.002621051027568923"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "error"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
