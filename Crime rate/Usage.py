{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b3fc7094",
   "metadata": {},
   "outputs": [],
   "source": [
    "from joblib import load\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5aa2f26a",
   "metadata": {},
   "outputs": [],
   "source": [
    "model=load(\"ViolentCrimePerPop.joblib\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "25eea4d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "input=np.array([[1.  , 0.19, 0.33, 0.02, 0.9 , 0.12, 0.17, 0.34, 0.47, 0.29, 0.32,\n",
    "       0.2 , 1.  , 0.37, 0.72, 0.34, 0.6 , 0.29, 0.15, 0.43, 0.39, 0.4 ,\n",
    "       0.39, 0.32, 0.27, 0.27, 0.36, 0.41, 0.08, 0.19, 0.1 , 0.18, 0.48,\n",
    "       0.27, 0.68, 0.23, 0.41, 0.25, 0.52, 0.68, 0.4 , 0.75, 0.75, 0.35,\n",
    "       0.55, 0.59, 0.61, 0.56, 0.74, 0.76, 0.04, 0.14, 0.03, 0.24, 0.27,\n",
    "       0.37, 0.39, 0.07, 0.07, 0.08, 0.08, 0.89, 0.06, 0.14, 0.13, 0.33,\n",
    "       0.39, 0.28, 0.55, 0.09, 0.51, 0.5 , 0.21, 0.71, 0.52, 0.05, 0.26,\n",
    "       0.65, 0.14, 0.06, 0.22, 0.19, 0.18, 0.36, 0.35, 0.38, 0.34, 0.38,\n",
    "       0.46, 0.25, 0.04, 0.  , 0.12, 0.42, 0.5 , 0.51, 0.64, 0.03, 0.13,\n",
    "       0.96, 0.17, 0.06, 0.18, 0.44, 0.13, 0.94, 0.93, 0.03, 0.07, 0.1 ,\n",
    "       0.07, 0.02, 0.57, 0.29, 0.12, 0.26, 0.2 , 0.06, 0.04, 0.9 , 0.5 ,\n",
    "       0.32, 0.14]]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e8bbc85a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.195])"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict(input)"
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
