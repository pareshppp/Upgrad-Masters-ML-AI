{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import re\n",
    "from collections import Counter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# function to tokenise words\n",
    "def words(document):\n",
    "    \"Convert text to lower case and tokenise the document\"\n",
    "    return re.findall(r'\\w+', document.lower())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# create a frequency table of all the words of the document\n",
    "all_words = Counter(words(open('big.txt').read()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# check frequency of a random word, say, 'chair'\n",
    "all_words['chair']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# look at top 10 frequent words\n",
    "all_words.most_common(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def edits_one(word):\n",
    "    \"Create all edits that are one edit away from `word`.\"\n",
    "    alphabets    = 'abcdefghijklmnopqrstuvwxyz'\n",
    "    splits     = [(word[:i], word[i:])                   for i in range(len(word) + 1)]\n",
    "    deletes    = [left + right[1:]                       for left, right in splits if right]\n",
    "    inserts    = [left + c + right                       for left, right in splits for c in alphabets]\n",
    "    replaces   = [left + c + right[1:]                   for left, right in splits if right for c in alphabets]\n",
    "    transposes = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right)>1]\n",
    "    return set(deletes + inserts + replaces + transposes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def edits_two(word):\n",
    "    \"Create all edits that are two edits away from `word`.\"\n",
    "    return (e2 for e1 in edits_one(word) for e2 in edits_one(e1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def known(words):\n",
    "    \"The subset of `words` that appear in the `all_words`.\"\n",
    "    return set(word for word in words if word in all_words)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def possible_corrections(word):\n",
    "    \"Generate possible spelling corrections for word.\"\n",
    "    return (known([word]) or known(edits_one(word)) or known(edits_two(word)) or [word])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def prob(word, N=sum(all_words.values())): \n",
    "    \"Probability of `word`: Number of appearances of 'word' / total number of tokens\"\n",
    "    return all_words[word] / N"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(len(set(edits_one(\"monney\"))))\n",
    "print(edits_one(\"monney\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(known(edits_one(\"monney\")))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's look at words that are two edits away\n",
    "print(len(set(edits_two(\"monney\"))))\n",
    "print(known(edits_one(\"monney\")))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's look at possible corrections of a word\n",
    "print(possible_corrections(\"monney\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's look at probability of a word\n",
    "print(prob(\"money\"))\n",
    "print(prob(\"monkey\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def spell_check(word):\n",
    "    \"Print the most probable spelling correction for `word` out of all the `possible_corrections`\"\n",
    "    correct_word = max(possible_corrections(word), key=prob)\n",
    "    if correct_word != word:\n",
    "        return \"Did you mean \" + correct_word + \"?\"\n",
    "    else:\n",
    "        return \"Correct spelling.\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# test spell check\n",
    "print(spell_check(\"monney\"))"
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
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
