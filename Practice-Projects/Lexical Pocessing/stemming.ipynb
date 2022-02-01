{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Stemming"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# import libraries\n",
    "import pandas as pd\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "from nltk.stem.snowball import SnowballStemmer"
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
      "Very orderly and methodical he looked, with a hand on each knee, and a loud watch ticking a sonorous sermon under his flapped newly bought waist-coat, as though it pitted its gravity and longevity against the levity and evanescence of the brisk fire.\n"
     ]
    }
   ],
   "source": [
    "text = \"Very orderly and methodical he looked, with a hand on each knee, and a loud watch ticking a sonorous sermon under his flapped newly bought waist-coat, as though it pitted its gravity and longevity against the levity and evanescence of the brisk fire.\"\n",
    "print(text)"
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
      "['very', 'orderly', 'and', 'methodical', 'he', 'looked', ',', 'with', 'a', 'hand', 'on', 'each', 'knee', ',', 'and', 'a', 'loud', 'watch', 'ticking', 'a', 'sonorous', 'sermon', 'under', 'his', 'flapped', 'newly', 'bought', 'waist-coat', ',', 'as', 'though', 'it', 'pitted', 'its', 'gravity', 'and', 'longevity', 'against', 'the', 'levity', 'and', 'evanescence', 'of', 'the', 'brisk', 'fire', '.']\n"
     ]
    }
   ],
   "source": [
    "tokens = word_tokenize(text.lower())\n",
    "print(tokens)"
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
      "['veri', 'orderli', 'and', 'method', 'he', 'look', ',', 'with', 'a', 'hand', 'on', 'each', 'knee', ',', 'and', 'a', 'loud', 'watch', 'tick', 'a', 'sonor', 'sermon', 'under', 'hi', 'flap', 'newli', 'bought', 'waist-coat', ',', 'as', 'though', 'it', 'pit', 'it', 'graviti', 'and', 'longev', 'against', 'the', 'leviti', 'and', 'evanesc', 'of', 'the', 'brisk', 'fire', '.']\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "47"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "stemmer = PorterStemmer()\n",
    "porter_stemmed = [stemmer.stem(token) for token in tokens]\n",
    "print(porter_stemmed)\n",
    "len(porter_stemmed)"
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
      "['veri', 'order', 'and', 'method', 'he', 'look', ',', 'with', 'a', 'hand', 'on', 'each', 'knee', ',', 'and', 'a', 'loud', 'watch', 'tick', 'a', 'sonor', 'sermon', 'under', 'his', 'flap', 'newli', 'bought', 'waist-coat', ',', 'as', 'though', 'it', 'pit', 'it', 'graviti', 'and', 'longev', 'against', 'the', 'leviti', 'and', 'evanesc', 'of', 'the', 'brisk', 'fire', '.']\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "47"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# snowball stemmer\n",
    "stemmer = SnowballStemmer(\"english\")\n",
    "snowball_stemmed = [stemmer.stem(token) for token in tokens]\n",
    "print(snowball_stemmed)\n",
    "len(snowball_stemmed)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "df = pd.DataFrame({'token': tokens, 'porter_stemmed': porter_stemmed, 'snowball_stemmed': snowball_stemmed})\n",
    "df = df[['token', 'porter_stemmed', 'snowball_stemmed']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>token</th>\n",
       "      <th>porter_stemmed</th>\n",
       "      <th>snowball_stemmed</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>very</td>\n",
       "      <td>veri</td>\n",
       "      <td>veri</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>orderly</td>\n",
       "      <td>orderli</td>\n",
       "      <td>order</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>methodical</td>\n",
       "      <td>method</td>\n",
       "      <td>method</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>looked</td>\n",
       "      <td>look</td>\n",
       "      <td>look</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>ticking</td>\n",
       "      <td>tick</td>\n",
       "      <td>tick</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>sonorous</td>\n",
       "      <td>sonor</td>\n",
       "      <td>sonor</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>his</td>\n",
       "      <td>hi</td>\n",
       "      <td>his</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>flapped</td>\n",
       "      <td>flap</td>\n",
       "      <td>flap</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>newly</td>\n",
       "      <td>newli</td>\n",
       "      <td>newli</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>32</th>\n",
       "      <td>pitted</td>\n",
       "      <td>pit</td>\n",
       "      <td>pit</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>33</th>\n",
       "      <td>its</td>\n",
       "      <td>it</td>\n",
       "      <td>it</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>34</th>\n",
       "      <td>gravity</td>\n",
       "      <td>graviti</td>\n",
       "      <td>graviti</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>36</th>\n",
       "      <td>longevity</td>\n",
       "      <td>longev</td>\n",
       "      <td>longev</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39</th>\n",
       "      <td>levity</td>\n",
       "      <td>leviti</td>\n",
       "      <td>leviti</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>41</th>\n",
       "      <td>evanescence</td>\n",
       "      <td>evanesc</td>\n",
       "      <td>evanesc</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          token porter_stemmed snowball_stemmed\n",
       "0          very           veri             veri\n",
       "1       orderly        orderli            order\n",
       "3    methodical         method           method\n",
       "5        looked           look             look\n",
       "18      ticking           tick             tick\n",
       "20     sonorous          sonor            sonor\n",
       "23          his             hi              his\n",
       "24      flapped           flap             flap\n",
       "25        newly          newli            newli\n",
       "32       pitted            pit              pit\n",
       "33          its             it               it\n",
       "34      gravity        graviti          graviti\n",
       "36    longevity         longev           longev\n",
       "39       levity         leviti           leviti\n",
       "41  evanescence        evanesc          evanesc"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[(df.token != df.porter_stemmed) | (df.token != df.snowball_stemmed)]"
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
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
