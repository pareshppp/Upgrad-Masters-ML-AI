{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Levenshtein Edit Distance\n",
    "The levenshtein distance calculates the number of steps (insertions, deletions or substitutions) required to go from source string to target string."
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
    "def lev_distance(source='', target=''):\n",
    "    \"\"\"Make a Levenshtein Distances Matrix\"\"\"\n",
    "    \n",
    "    # get length of both strings\n",
    "    n1, n2 = len(source), len(target)\n",
    "    \n",
    "    # create matrix using length of both strings - source string sits on columns, target string sits on rows\n",
    "    matrix = [ [ 0 for i1 in range(n1 + 1) ] for i2 in range(n2 + 1) ]\n",
    "    \n",
    "    # fill the first row - (0 to n1-1)\n",
    "    for i1 in range(1, n1 + 1):\n",
    "        matrix[0][i1] = i1\n",
    "    \n",
    "    # fill the first column - (0 to n2-1)\n",
    "    for i2 in range(1, n2 + 1):\n",
    "        matrix[i2][0] = i2\n",
    "    \n",
    "    # fill the matrix\n",
    "    for i2 in range(1, n2 + 1):\n",
    "        for i1 in range(1, n1 + 1):\n",
    "            \n",
    "            # check whether letters being compared are same\n",
    "            if (source[i1-1] == target[i2-1]):\n",
    "                value = matrix[i2-1][i1-1]               # top-left cell value\n",
    "            else:\n",
    "                value = min(matrix[i2-1][i1]   + 1,      # left cell value     + 1\n",
    "                            matrix[i2][i1-1]   + 1,      # top cell  value     + 1\n",
    "                            matrix[i2-1][i1-1] + 1)      # top-left cell value + 1\n",
    "            \n",
    "            matrix[i2][i1] = value\n",
    "    \n",
    "    # return bottom-right cell value\n",
    "    return matrix[-1][-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lev_distance('cat', 'cta')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Levenshtein distance in nltk library"
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
    "# import library\n",
    "from nltk.metrics.distance import edit_distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "edit_distance(\"apple\", \"appel\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Damerau-Levenshtein Distance\n",
    "The Damerau-Levenshtein distance allows transpositions (swap of two letters which are adjacent to each other) as well."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "edit_distance(\"apple\", \"appel\", transpositions=False, )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
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
