{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Tokenisation\n",
    "\n",
    "The notebook contains three types of tokenisation techniques:\n",
    "1. Word tokenisation\n",
    "2. Sentence tokenisation\n",
    "3. Tweet tokenisation\n",
    "4. Custom tokenisation using regular expressions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Word tokenisation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "At nine o'clock I visited him myself. It looks like religious mania, and he'll soon think that he himself is God.\n"
     ]
    }
   ],
   "source": [
    "document = \"At nine o'clock I visited him myself. It looks like religious mania, and he'll soon think that he himself is God.\"\n",
    "print(document)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Tokenising on spaces using python"
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
      "['At', 'nine', \"o'clock\", 'I', 'visited', 'him', 'myself.', 'It', 'looks', 'like', 'religious', 'mania,', 'and', \"he'll\", 'soon', 'think', 'that', 'he', 'himself', 'is', 'God.']\n"
     ]
    }
   ],
   "source": [
    "print(document.split())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Tokenising using nltk word tokeniser"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from nltk.tokenize import word_tokenize\n",
    "words = word_tokenize(document)"
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
      "['At', 'nine', \"o'clock\", 'I', 'visited', 'him', 'myself', '.', 'It', 'looks', 'like', 'religious', 'mania', ',', 'and', 'he', \"'ll\", 'soon', 'think', 'that', 'he', 'himself', 'is', 'God', '.']\n"
     ]
    }
   ],
   "source": [
    "print(words)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "NLTK's word tokeniser not only breaks on whitespaces but also breaks contraction words such as he'll into \"he\" and \"'ll\". On the other hand it doesn't break \"o'clock\" and treats it as a separate token."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Sentence tokeniser"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Tokenising based on sentence requires you to split on the period ('.'). Let's use nltk sentence tokeniser."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from nltk.tokenize import sent_tokenize\n",
    "sentences = sent_tokenize(document)"
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
      "[\"At nine o'clock I visited him myself.\", \"It looks like religious mania, and he'll soon think that he himself is God.\"]\n"
     ]
    }
   ],
   "source": [
    "print(sentences)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Tweet tokeniser"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A problem with word tokeniser is that it fails to tokeniser emojis and other complex special characters such as word with hashtags. Emojis are common these days and people use them all the time."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "message = \"i recently watched this show called mindhunters:). i totally loved it 😍. it was gr8 <3. #bingewatching #nothingtodo 😎\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['i', 'recently', 'watched', 'this', 'show', 'called', 'mindhunters', ':', ')', '.', 'i', 'totally', 'loved', 'it', '😍', '.', 'it', 'was', 'gr8', '<', '3', '.', '#', 'bingewatching', '#', 'nothingtodo', '😎']\n"
     ]
    }
   ],
   "source": [
    "print(word_tokenize(message))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The word tokeniser breaks the emoji '<3' into '<' and '3' which is something that we don't want. Emojis have their own significance in areas like sentiment analysis where a happy face and sad face can salone prove to be a really good predictor of the sentiment. Similarly, the hashtags are broken into two tokens. A hashtag is used for searching specific topics or photos in social media apps such as Instagram and facebook. So there, you want to use the hashtag as is.\n",
    "\n",
    "Let's use the tweet tokeniser of nltk to tokenise this message."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from nltk.tokenize import TweetTokenizer\n",
    "tknzr = TweetTokenizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['i',\n",
       " 'recently',\n",
       " 'watched',\n",
       " 'this',\n",
       " 'show',\n",
       " 'called',\n",
       " 'mindhunters',\n",
       " ':)',\n",
       " '.',\n",
       " 'i',\n",
       " 'totally',\n",
       " 'loved',\n",
       " 'it',\n",
       " '😍',\n",
       " '.',\n",
       " 'it',\n",
       " 'was',\n",
       " 'gr8',\n",
       " '<3',\n",
       " '.',\n",
       " '#bingewatching',\n",
       " '#nothingtodo',\n",
       " '😎']"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tknzr.tokenize(message)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As you can see, it handles all the emojis and the hashtags pretty well.\n",
    "\n",
    "Now, there is a tokeniser that takes a regular expression and tokenises and returns result based on the pattern of regular expression.\n",
    "\n",
    "Let's look at how you can use regular expression tokeniser."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from nltk.tokenize import regexp_tokenize\n",
    "message = \"i recently watched this show called mindhunters:). i totally loved it 😍. it was gr8 <3. #bingewatching #nothingtodo 😎\"\n",
    "pattern = \"#[\\w]+\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['#bingewatching', '#nothingtodo']"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "regexp_tokenize(message, pattern)"
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
