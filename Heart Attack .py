{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 98.54%\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    " \n",
    "# Load the data from the CSV file\n",
    "data = pd.read_csv(\"/Users/adithyau/Downloads/heart.csv\")\n",
    " \n",
    "# Split the data into features and target\n",
    "X = data.drop(\"target\", axis=1) # Features are all columns except target\n",
    "y = data[\"target\"] # Target is the column named target\n",
    " \n",
    "# Split the data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    " \n",
    "# Create and fit a decision tree classifier\n",
    "clf = DecisionTreeClassifier()\n",
    "clf.fit(X_train, y_train)\n",
    " \n",
    "# Make predictions on the test set\n",
    "y_pred = clf.predict(X_test)\n",
    " \n",
    "# Evaluate the accuracy of the model\n",
    "acc = accuracy_score(y_test, y_pred)\n",
    "print(f\"Accuracy: {acc*100:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 98.54%\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    " \n",
    "# Load the data from the CSV file\n",
    "data = pd.read_csv(\"/Users/adithyau/Downloads/heart.csv\")\n",
    " \n",
    "# Split the data into features and target\n",
    "X = data.drop(\"target\", axis=1) # Features are all columns except target\n",
    "y = data[\"target\"] # Target is the column named target\n",
    " \n",
    "# Split the data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    " \n",
    "# Create and fit a random forest classifier\n",
    "clf = RandomForestClassifier()\n",
    "clf.fit(X_train, y_train)\n",
    " \n",
    "# Make predictions on the test set\n",
    "y_pred = clf.predict(X_test)\n",
    " \n",
    "# Evaluate the accuracy of the model\n",
    "acc = accuracy_score(y_test, y_pred)\n",
    "print(f\"Accuracy: {acc*100:.2f}%\")\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 0 1 0 1 0 0 1 0 1 0 1 1 0 1 0 1 1 0 1 0 0 0 1 1 1 1 0 1 1 1 0 1 1 1 1\n",
      " 0 1 1 0 0 1 0 0 0 0 0 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 1 0 0 1 1 0 1 1 0 0 1\n",
      " 1 1 0 1 1 1 0 0 0 0 1 0 1 0 0 1 1 0 1 1 1 1 1 0 0 0 0 0 1 1 0 1 0 1 0 1 1\n",
      " 1 1 0 1 1 1 1 1 0 0 1 0 0 0 0 1 1 1 1 1 0 1 0 0 1 0 1 1 1 0 1 1 0 1 1 0 1\n",
      " 1 0 0 0 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 0 0 1 0 1 1 0 1 1 1 0 1 1 1 0 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 0 1 0 1 0 1 0\n",
      " 0 1 1 1 1 1 0 0 1 0 1 1 0 0 0 0 1 1 0 1 1 0 0 1 0 0 1 0 1 0 0 0 0 1 0 1 0\n",
      " 1 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 0 1 1 0 1\n",
      " 1 0 0 1 1 0 1 1 1 0 0 1 1 0 1 1 1 1 0 0 0 1 0 1 0 1 1 1 0 0 0 0 0 1 1 1 1\n",
      " 0 0 0 1 0 1 1 0 0 1 1 1 1 0 0 1 0 1 1 1 0 0 1 0 0 1 1 1 0 1 1 0 1 1 1 1 0\n",
      " 0 1 0 1 1 0 1 0 0 1 1 1 0 1 1 1 0 0 1 0 1 0 1 1 1 1 1 1 0 1 0 1]\n",
      "Accuracy: 81.59203980099502\n"
     ]
    }
   ],
   "source": [
    "#svm\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "df = pd.read_csv(\"/Users/adithyau/Downloads/heart.csv\")\n",
    "x = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.392, random_state=42)\n",
    "model = SVC(kernel='linear', random_state=42)\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 80.00%\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "data = pd.read_csv(\"/Users/adithyau/Downloads/heart.csv\")\n",
    "\n",
    "X = data.drop(\"target\", axis=1)\n",
    "y = data[\"target\"]\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "clf = GaussianNB()\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "y_pred = clf.predict(X_test)\n",
    "\n",
    "acc = accuracy_score(y_test, y_pred)\n",
    "print(f\"Accuracy: {acc*100:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 0 1 0 1 0 0 1 0 1 1 1 1 0 1 0 1 1 0 0]\n",
      "Accuracy: 85.71428571428571\n"
     ]
    }
   ],
   "source": [
    "#knn\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "df = pd.read_csv(\"/Users/adithyau/Downloads/heart.csv\")\n",
    "x = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.02, random_state=42)\n",
    "model = KNeighborsClassifier(n_neighbors=5)\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 0 1 0 1 0 0 1 0 1 0 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1\n",
      " 0 1 1 0 0 1 1 0 0 0 0 1 0 1 0 1 0 1 1 0 0 1 1 1 0 0 0 0 0 1 1 0 1 1 0 0 1\n",
      " 1 1 0 1 1 1 0 0 0 0 1 0 1 0 0 1 0 0 1 1 1 1 1 0 0 0 1 0 1 1 0 1 0 1 0 1 1\n",
      " 1 1]\n",
      "Accuracy: 81.41592920353983\n"
     ]
    }
   ],
   "source": [
    "#LogisticRegression\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "df = pd.read_csv(\"/Users/adithyau/Downloads/heart.csv\")\n",
    "x = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.11, random_state=42)\n",
    "model = LogisticRegression(max_iter=1000, random_state=42)\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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1a8vffxxOMxta3z9LruE4XD8+MIsIbBtGzfmrf4CAAAgX7AYS7xz49e9I9viHjwayYhH1F8AAIBIEPTinVce3F4vna9nhULEI+ovAABAJAh6ibLJtKWmKr1XL21atkzpPXtu7gXx9Sy8gkSqv154hfoLAABQYAh68c49HV9/HS5Y8eWXQUN5oRvKvXtLX3wRXv7VV9K6dVGXFMh9/b3hBmn6dOovAABAASHoxTP3crzxRrhlwpdfSvXqaW1qqv7+++/gZ7BQiy93796bb9IrgsSpvxUqhJfNmBFuq0D9BQAAyFesuhnPvNn5WWdJZ56Z87LzmbdhSEkJb+/VC4F4r7+urz5Q0bSp9PTT0sEHU38BAADyET168cwhrlSpcJuFbfH1vh0rFyKR6u+xx0r77isdfbQ0eTL1FwAAIB8R9OKZG8i5bfz6dtsLhEC81d/+/cO99tq1k5YuLaySAQAAJD2CHoDouDfv5JPDsHf77VGXBgAAIGkQ9ABEy716Nnq0NH9+1KUBAABICgQ9ANFq0kQ6/fRw1c0BA6IuDQAAQFIg6AGIn169556T5s2LujQAAAAJj6AHIHqNG0tt2kjp6ZtDHwAAAHYYQQ9AfOjXL/w5bpw0Z07UpQEAAEhokW+YvmrVKt1999165513tH79eh177LHq1auXKleurOOPP16//PJLjvcbM2aMjjjiiByvO/HEE/XTTz9luaxNmza66667CuQ5AMgHDRtK550njR8f9upNmBB1iQAAABJW5EHvuuuu08KFCzVo0CDtvffeuv/++3XxxRfrpZde0oQJE7TRy67/PwfBjh07qmrVqjr00ENzfLzVq1fr559/1siRI1W/fv2My0uXLl0ozwfATujbNwx4EydKs2dLhxwSdYkAAAASUqRDN+fNm6fp06drwIABatGihQ444AANGTJEv//+uyZPnqzddttNVapUyTi5F+/vv//Wfffdp+LFc86oCxYs0KZNm4IgmPm+FSpUKPTnByCPfHDmggs2hz4AAAAkXtBbtGhR8PPwww/PuKxcuXKqXr26Pv300y0C3OjRo9WzZ88gAG7Nd999p913310VK1YswJIDKDAOeMWKSa+8Is2aFXVpAAAAElKkQzf32GOP4OeSJUu0//77B+c9VHPp0qXBHL3Mhg8frgMPPFBnnXXWNh/TQa9s2bL697//rS+++EK7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Nzx/993S339HXSIAAFCEEfQAIL+0ayfVqSMtXy4NGxZ1aQAAQBFG0AOA/JKaKvXrF54fOlRasSLqEgEAgCKKoAcA+en886X69aW//pLuuy/q0gAAgCKKoAcA+alYMal///C8g56HcQIAABQygh4A5Lc2baRGjaSVK8MhnAAAAIWMoAcABdmr50VZ/vgj6hIBAIAiJi6C3qpVq9S3b181b95cTZo0Ubdu3bRs2bKM6ydOnKgzzjhDhxxyiE488UQ9+uij2rhx41Yf77ffflOdOnW2OL344ouF9IwAFHlnnik1biz980+43QIAAEAhKq44cN1112nhwoUaNGiQ9t57b91///26+OKL9dJLL2nq1KlBCLz11lvVtGlTffPNN8H59evX69prr83x8b799luVKlVKb7/9tlJSUjIur1ChQiE+KwBFmj97BgyQTj893ED9ppukPfeMulQAAKCIiDzozZs3T9OnT9djjz2mY489NrhsyJAhOu644zR58mSNGzdOZ599tv71r38F11WrVk0//vijxo8fv9WgN3/+fNWoUUN77LFHoT4XAMji1FOlI4+UPvlEGjxYuvfeqEsEAACKiMiHbi5atCj4efjhh2dcVq5cOVWvXl2ffvppMIzz8ssvz3KfYsWK6S8vXb4V3333nfbff/8CLDUA5KFXzx5+WPr116hLBAAAiojIe/RivW5LlizJCGeef7d06VJVrlxZhx12WJbbr1y5Us8//7yOOeaYrT6me/R23XVXtW/fPuj9c2js0qVLRo/hjkhPT9fq1asVtTVr1mT5CSSSIll/mzVTqaZNlTpjhtJuv11p99wTdYmwg4pk/UXSoP4ikcVb/U1PT88yPSxepaS7pBHyXLuzzjorCHVDhw5VxYoVNXz4cD399NM68sgj9eSTT2bc9p9//lHnzp2DIOcFWvbdd98tHm/Dhg3Boi21a9dWz549Vb58+WAI6KhRo4KT5/nl1Zw5c4JyAsCOqPDZZzqwSxdtKlFC37z0ktKqVo26SAAAYCeULFlSDRo0UDyLPOiZF2Lp3r17sNBKiRIlghU23XPnIZoOffbHH3/oqquu0uLFi/XEE09s84V1IExNTVXp0qUzLrviiiuCn48//vgOBT2/TA6PUfORDA939RzEMmXKRF0cIE+KbP1NT1epU05R6ocfKu2KK5TmLReQcIps/UVSoP4ikcVb/V2wYEHQoxfvQS/yoZvmIZvuoVuxYoWKFy8e9MK1bdtWRx11VEYQdFDbtGmTnn32WR1wwAHbfDzP8cvO9/GiLzvK/8yyZcsqXriSx1N5gLwokvX39tulFi1U4umnVaJPH6lGjahLhB1UJOsvkgb1F4ksXupvSgIM24yLxVi8h16HDh2CLREqVaoUhDz32s2dO1fNmjXTzz//rEsuuST4x77wwgvbDXnff/+9GjdurE+8yl0m7i2Mhx45AEWU5wi3bi2lpUmDBkVdGgAAkOQiD3oOdh4W6T30HNI8TNILp7g3z/PpevfuHcyPu/fee4PePg/hjJ1ili9fHgz1jPUO1qpVSwMGDNCsWbOC3sA777xTs2fPDh4XACLTv3/4c9QoD1WIujQAACCJxcXQTYe4gQMHql27dsHExhNPPFE333yzfvvtt2CLBfOCLTlto2Ae5tmkSRPdddddwby+Rx55JFjY5frrr9fff/+tevXqBQuxHHjggYX+3AAgw9FHSyefLE2dGg7ldOADAABI1qC35557asSIETnOtYuFuW159913s/y+++67B714ABCXvXoOeqNHS716SRyAAgAAyTh0EwCKlCZNpNNPlzZt2ryZOgAAQD4j6AFAVHP1nntOmjcv6tIAAIAkRNADgMLWuLHUpk2wv15G6AMAAMhHBD0AiEK/fuHPceOkOXOiLg0AAEgyBD0AiELDhtJ559GrBwAACgRBDwCi0revlJIiTZwozZ4ddWkAAEASIegBQFTq15cuuGBz6AMAAMgnBD0AiJIDXrFi0iuvSLNmRV0aAACQJAh6ABClOnWkDh3C8/TqAQCAfELQA4Co3XqrlJoqTZkizZwZdWkAAEASIOgBQNRq15YuuSQ8T68eAACIIuitW7cuP/4uACCzW26RiheX3nxTmj496tIAAICiFvSaNWumvn376uuvvy6YEgFAUVSzptSxY3j+ttuiLg0AAChqQa9jx46aOXOm/vWvf+nUU0/V448/rj/++KNgSgcARUmfPlLJktK0aeEJAACgsILe1VdfrTfeeEPPPvusDjvsMI0cOVItW7bUlVdeGVyelpa2o2UBgKKtWjWpU6fNc/XS06MuEQAAKGqLsTRu3FgDBw7URx99pGHDhmnNmjW6/vrr1bx5cw0ePFi//PJL/pYUAIqCXr2kUqWkDz+U3nkn6tIAAICiuOrmkiVL9OSTT2r48OH67LPPVKNGDZ1zzjn64IMPgmGdU7xUOAAg9/bZR+rcefNcPXr1AABAYQS9VatWaeLEibrooovUqlUrPfroo6pfv76ee+45vf766+rRo4cmT56so446SnfccceOlAkAiraePaUyZaQZM6SpU6MuDQAASEDFd2TVTW+xcMghh2jAgAFBz13ZsmW3uF2DBg00d+7c/ConABQdVat6QrQ0dGjYq3fyyVJKStSlAgAAydyj1759+2BI5gsvvKC2bdvmGPLssssu03vvvZcfZQSAoqd7d6lcOWnWLOm116IuDQAASPag1717d/3555968MEHMy5zz911112nb775JuOycuXKKTU1Nf9KCgBFyR57SF27hueZqwcAAAo66L3//vu65JJLNH369IzLUlJStGjRIl144YWa5aPPAICd162bVL68NHu2NGlS1KUBAADJHPQeeOABnXbaacHiKzF169bVyy+/rFNOOUX33ntvfpcRAIqmypWl66/fvK/epk1RlwgAACRr0Fu4cKHOPvvsoBcvO1/+7bff5lfZAAA33ijtsos0Z440YULUpQEAAMka9CpUqKAff/wxx+t+/vnnrS7OAgDYAbvuGoY969dP2rgx6hIBAIBkDHonnHCChg0bpmnTpmW5/MMPPwwu9/UAgHzk4ZsOfPPmSWPHRl0aAACQjPvo3XDDDZozZ466dOmiEiVKqFKlSlqxYoU2bNigRo0a6aabbiqYkgJAUVWxYrgwS58+Uv/+0vnnS8Xz/PENAACKkDy3FMqXLx/soefVNz///HP99ddfwXDOww8/XMcdd5yKFctzJyEAYHu81YIXu5o/X/JiWBdfHHWJAABAHNuhQ8IOcy1btgxO2aWnp+e4UAsAYCdUqBBuot6jhzRggNSunVSiRNSlAgAAyRT0pkyZok8//VTr168Pgp355+rVqzV79mx98MEH+V1OAMA110j33OPlj6XRo6XLL4+6RAAAIFmC3ogRI4KTh2t6Xp7n6RUvXlzLly8PevrOO++8gikpABR15cpJPXtKngs9cKB00UVSyZJRlwoAAMShPE+oe+mll4L98tyjd+mllwbDNz/++GNNmDAhWJjlgAMOKJiSAgCkzp2lqlWln36SRo2KujQAACBZgt5vv/2mM844I5iHV7duXX355ZfB5QcffLA6d+6s8ePHF0Q5AQDmvUp79w7P3367tG5d1CUCAADJEPS8IXpssZXq1atr8eLFWrt2bfC7g59/BwAUoE6dpH32kfx5+/jjUZcGAAAkQ9Br0KCBJk2aFJyvWbOmUlNTNWPGjOD3hQsXqiTzRQCgYJUuHe6pZ4MGSWvWRF0iAACQ6EHPwzO96qZ/OtSdeeaZ6tGjh7p27arBgwerefPmBVNSAMBmHTtK1apJS5ZII0dGXRoAAJDoQe+II44IFl455ZRTgt9vu+02nXTSSfrhhx908skn65ZbbimIcgIAMitVSop93t55p/TPP1GXCAAAJPL2Cg899FAQ7M4666zg91KlSmmgl/kGABSuSy8NQ96PP0oPPyx16xZ1iQAAQKL26I0cOZIFVwAgHpQo4WEV4fnBg6VVq6IuEQAASNSgV7t2bf3oo8f5ZNWqVerbt28wt69Jkybq1q2bli1blnG9F3o555xz1KhRo2Bo6OTJk7f7mM8++6xatWqlhg0b6sILL9TcuXPzrbwAEFc6dPAHs/S//0kjRkRdGgAAkKhDN71B+r333qsPP/xQderUCbZbyMxbL1xzzTW5frzrrrsuWK1z0KBB2nvvvXX//ffr4osvDjZm//nnn3XVVVfpsssu091336333ntP3bt312677aamTZvm+Hi+35AhQ4LhpPXq1dOjjz4a3P/1118P7gcASaV4calvX+mii6S775auvlraZZeoSwUAABIt6I34/yPGH330UXDKLi9Bb968eZo+fboee+wxHXvsscFlDmnHHXdc0HPnzdgdJm+44Ybguv333z/onXv88ce3GvQeeeQRdejQIVgN1O644w61bt062MjdoREAkk67duHm6d99Jw0bJt16a9QlAgAAiRb0vv3223z744sWLQp+Hn744RmXlStXLtiI/dNPP9VXX30VhLTMjjrqqKD3Lz09PWPj9hgP+fRjZg6BxYsXDx7/s88+I+gBSE6pqVK/fmHgGzpU6tpVqlQp6lIBAIBECnr5aY899gh+LlmyJOits40bN2rp0qWqXLly8LNq1apb3GfNmjX6888/txiK6dvbXnvttcV9djagOliuXr1aUfNzz/wTSCTU3wJ0+ukqXbeuis2bp7QhQ5TGVjf5jvqLREb9RSKLt/qbnkOHU1IEvV69em33Nnd6ue9caNCggWrVqhUsxjJ06FBVrFhRw4cPD0JcWlqa1q5dG2zKnlns9/Xr12/xeLF/fvb7eAuIdevWaWe4PB5qGi9ivaFAIqL+FoxKl16q/Xv0ULHhwzW/dWttrFgx6iIlJeovEhn1F4ksnupvyWx5IymC3ieffLLFZe7pWrFihSpVqhSEt7y8QJ7z5wVWPEevRIkSOuOMM4IFX4oVKxYEtOyBLvZ7mTJltni80qVLZ7lNjENeTrfPC5fNK45GzWHWlbxGjRo7/ZyAwkb9LWB16mjTmDFKnTNH9adOVZqHcyLfUH+RyKi/SGTxVn8XLFigRJDnoPfuu+/meLlXzrz22mt19tln5+nxPGRz4sSJQVD0fLry5curbdu2wVw8D8H8/fffs9zev3ulzwoVKmzxWLEhm75NbCho7Pc999xTO8Pds9lXGI2SK3k8lQfIC+pvARo4UDr7bJV46CGVuPlmqUqVqEuUdKi/SGTUXySyeKm/KQkwbHOH9tHbGgerrl27ZqzKmds99LxCpufPuTfQIc+bsXtlzWbNmgWLqHhRlsxmzpypxo0bBz1+2XleX82aNbP0Om7YsEGzZs3SEUccsZPPEAASgFccbtxY+uefcLsFAABQJOVb0DMHtV9++SVPt/dkRq+i+f3332vOnDnq0qVL0JvnlTMvuugiff3117rnnnuCHsMnn3xSU6dO1RVXXJHxGO4J9CmmY8eOGjVqVLCfnrtVe/fuHcz1cy8hACQ9H2UcMCA87wNvv/0WdYkAAEAiDN389ddft7jMK2X+9ttvwUIqmYdM5oY3X/fm5u3atQvm7J144om62cONJB1wwAF66KGHgs3Sn376ae27777B+czbJ7gX0Z555png5/nnn6+VK1cGG687AB588MFB8GOzdABFxqmnSkce6UnV0uDB/qCNukQAACDeg97xxx+f47hU98x5MZS8DN00z53b1n28SEtsM/WcxAJeZpdffnlwAoAi3at30knSww9L3bpJe+8ddakAAEA8B7077rhji6Dn3z0M88gjj8xxkRQAQCE74QSpWTPpo4+ku+6Shg+PukQAAKAQ5TnonXPOOdq0aZPmz5+vgw46KLjsjz/+CBZQiYflTgEAmXr1WrWSRo6UPCR+v/2iLhUAAIjXxVg8F++ss84KtlKIcci76qqrghU0My+MAgCIUMuWUosW3lzUwzGiLg0AAIjnoDdkyJBgQ3KvhBnTokULvfjii0HIGzp0aH6XEQCwo716/fuH5594Qlq0KOoSAQCAeA16H3/8sbp166ZDDjkky+X16tXTddddp2nTpuVn+QAAO8M9eh6+mZYmDRoUdWkAAEC8Bj335qWmpuZ4nefo/eNNegEA8SO2r96oUdLChVGXBgAAxGPQa9SoUbAvXZqPDmeyYcMGjR49Wg0bNszP8gEAdtbRR0snn+xNT6Xbb4+6NAAAIB5X3fz3v/+tiy66SK1atQr2t6tcubKWL1+ujz76SMuWLctxXzsAQMQ8V2/qVGn0aKlXL+nAA6MuEQAAiKcePc/NGzt2bPDzvffe0xNPPKG3335b9evX1wsvvECPHgDEoyZNpNNPlzZt2jyUEwAAJK089+jFFl657777MubqrVmzJhi6yWbpABDnvXqvvSY995zUp49Ut27UJQIAAPHSo+e5eX379tX555+fcdmXX36ppk2bavDgwcFm6gCAONS4sdSmjZSevnnbBQAAkJTyHPQeeOABvfLKKzrttNOy9PB5y4Vx48bp8ccfz+8yAgDyS79+4c9x46Q5c6IuDQAAiJeg9+qrr6pHjx7q2LFjxmWVKlXSpZdeqhtuuEETJkzI7zICAPKL51Gfdx69egAAJLk8B70///xT++23X47X1apVS0uXLs2PcgEACkrfvlJKijRxojR7dtSlAQAA8RD0HObeeOONHK979913Vb169fwoFwCgoNSvL11wwebQBwAAkk6eV928+OKL1bNnT61YsUKtW7fO2Edv2rRpev3113XnnXcWTEkBAPnHAW/sWOmVV6RZs6TDD4+6RAAAIMqgd/bZZ+uff/7RQw89pDfffDPj8l133VW33XabzjrrrPwsHwCgINSpI3XoEG6g7tA3eXLUJQIAAFEO3bT27dtr+vTpmjJlip577jm99tprmjRpkpYtW6bjjz8+P8sHACgot94qeT/UKVOkmTOjLg0AAIg66FlKSkowX8+9e948vVWrVhoxYkTGJuoAgDhXu7Z0ySXheebqAQBQtIdumufkeRsF75v3yy+/qHz58mrTpk0wbPNw5nkAQOK45ZZw+KaH4k+fLjVvHnWJAABAYQe9mTNnauzYsXr77be1ceNGHXbYYUHQe/DBB9WkSZP8KA8AoDDVrCl5X9RHH5Vuu83LJ0ddIgAAUFhDN5966imdcsopwaboc+fO1dVXXx1speCAl56eHgzjBAAkqD59pJIlpWnTwhMAACgaQe+uu+5SyZIlNXr06GAPvS5duqhq1aoEPABIBtWqSZ06bZ6rl54edYkAAEBhBL3TTjtNP/30k6666qqgN++tt97Shg0bdvZvAwDiRa9eUqlS0ocfSu+8E3VpAABAYczRGzp0qFatWqVXX31VL774orp27Rrsm+cN092rR88eACS4ffaROneWhg0L5+q1auXllaMuFQAAKOjtFbyyZrt27TR+/Pgg8HmFTc/T8xy93r17a9iwYVqwYMGOlgMAELWePaUyZaQZM6SpU6MuDQAAKOx99A444AD17NlT77//vh544IFgP73HHntMZ5xxhs4888ydKQ8AICpVq0pXXx2ed68ec/UAACh6G6Zb8eLFdcIJJ+iRRx7Re++9pxtvvJG5ewCQyLp3l8qVk2bNkl57LerSAACAKIJeZrvvvrs6deqkKVOm5NdDAgAK2x57SF27hufp1QMAIGHlW9ADACSJbt08MVuaPVuaNCnq0gAAgB1A0AMAZFW5snT99Zv31du0KeoSAQCAPCLoAQC2dOON0i67SHPmSBMmRF0aAACQRwQ9AMCWdt01DHvWr5+0cWPUJQIAAHlA0AMA5MzDNx345s2Txo6NujQAACAPCHoAgJxVrBguzGL9+0tsnwMAQMIg6AEAts5bLXhxlvnzpeeei7o0AAAglwh6AICtq1Ah3ETdBgyQ0tKiLhEAAMgFgh4AYNuuuUaqUkVauFAaPTrq0gAAgFwg6AEAtq1cOalnz/D8wIHS+vVRlwgAAMR70NuwYYOGDRumli1b6tBDD1X79u01e/bs4LqLLrpIderUyfE0adKkrT7mZZddtsXt/VgAgB3UubNUtar000/SqFFRlwYAAGxHcUXs4Ycf1vjx43XXXXdpv/3202OPPaYrrrhCU6ZM0QMPPKC0TPNB0tPTdcMNN+ivv/7SCSecsNXH/O6779SvXz+1bt0647ISJUoU+HMBgKRVtqzUu7f0739Lt98uXXqpVKpU1KUCAADx2qP39ttv6/TTT1fz5s1VvXp19ezZUytXrgx69SpVqqQqVapknN588019/fXXGj58uMp5KFEOli1bFpwaNWqU5b5+LADATujUSdpnH2nxYunxx6MuDQAAiOegV7lyZU2bNk2LFy/Wxo0bNXbsWJUsWVIHHXRQltstX75c999/v7p06aJatWpt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PWhXlCtXTp06dVIim5CHtlR+KTJBz1/IVapUyXJZ9erV9c033wRfxMka9HJ63vHADSvsvLJly2b5/+63337BEd3TTjst+MK+4YYb8uXv7Lbbbrm+baVKlVSQMj/fWJCNxzqO+PXcc88FYa5t27YZl9WqVUu//fabRo8erWuvvVY1atTQq6++ukXQmz17thYtWqShQ4dmXObe9IULF+rdd9/VmWeemXG5g9+HH34YhEmgoPighXuj7733Xp100klZvg9c/zx6ZnsHVfNyWySvvBwczt6Oc3tzZw+0Zm/T+Lw/jz/99NOgrZ7oQW+3PLSl8kuRGLq5LU7WxYtvzru//vpr0Dhu2rSp6tevH3zJ+whZrMvYR8ncjRz76S/4c845R59//nmWI2M9evQIhkIcddRRGjVq1BZ/98svv9TFF18cNACOPPLIYKiQhzNm7t599NFHdeWVV6pRo0bB72+//XZw8ge5hxtdfvnlGV3mO8pHbp566qngMRs0aBD89FHBmE8++UT16tULyuJy+rn6tXCDyK+Tn6Mv97AlN35iXK5///vfwXXuer/ggguCN6r5CJC7+/37znTzI2d77713UDc9ZCxznbz11luD+ug657o3Z86cLPfzl/y//vWvoL653t93330ZR/YyDzfwUGcf9WvWrFlQZ84++2y9+eabWx26mZu6/sQTT6hr167BEDrfxkenN2zYsMOvgcvg5+sGi+uoh+nZxIkTg15u10n/fPrpp7MMB9levUZyKFasWFAv//rrryyX+/PWPeHmIOjP2+xD+/3ZddBBBwWf/THu0WvVqpWmTp2a5ba+vz/j3IgGCoJHNDjkDR8+PEvIM3+OuR7fddddWrJkyTYfJy+3RfzyaAMfbPJ3nL9bH3rooSw9dP/973+DsOTvWve+uX0aa9PmNHTT382tW7cOPu/8eA8++GAQ8Nw29Pe6+bPP988+dPOff/7RwIED1bx58+DveaSRO1fyo62emzaND9SdcsopQTvFbQEfxMtcPp/3+6Zly5ZBGf1dv379+qDN79fGZfbw/swjhvxa+voWLVoEr8nJJ5+cpc28rbZvTkM333vvveBv+G+5DB5l6NFZmcvoXsBLL700eDzfZsSIEXl67Yps0HMj0i/wyy+/rLPOOivj8i5dugQVyJXfX9ruQnbPiI/UxvhD0B+s/mf7S9/DIP3miB3d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",
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#compare all accuracy graphically for these accuracies 0.93574, 0.94378, 0.93976, 0.93574, 0.94000, 0.92965 use line plot\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.figure(figsize=(10,5))\n",
    "model=['Random Forest','Decision Tree','SVM', 'KNN', 'Logistic Regression']\n",
    "accuracy=[98.54 , 98.54, 81.59203980099502, 85.71428571428571, 81.41592920353983]\n",
    "sns.lineplot(x=model, y=accuracy,marker='*', markersize=15 ,  color='red')\n",
    "plt.title('Accuracy Comparison')\n",
    "plt.xlabel('Model')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
