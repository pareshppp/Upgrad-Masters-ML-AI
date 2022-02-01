{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import re"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q1.\n",
    "Write a regular expression to match all the files that have either .exe, .xml or .jar extensions. A valid file name can contain any alphabet, digit and underscore followed by the extension."
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
      "['employees.xml', 'calculator.jar', 'nfsmw.exe']\n"
     ]
    }
   ],
   "source": [
    "files = ['employees.xml', 'calculator.jar', 'nfsmw.exe', 'bkgrnd001.jpg', 'sales_report.ppt']\n",
    "\n",
    "pattern = \"^.+\\.(xml|jar|exe)$\"\n",
    "\n",
    "result = []\n",
    "\n",
    "for file in files:\n",
    "    match = re.search(pattern, file)\n",
    "    if match !=None:\n",
    "        result.append(file)\n",
    "\n",
    "# print result - result should only contain the items that match the pattern\n",
    "print(result)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q2\n",
    "Write a regular expression to match all the addresses that have Koramangala embedded in them.\n",
    "\n",
    "Strings that should match:\n",
    "* 466, 5th block, Koramangala, Bangalore\n",
    "* 4th BLOCK, KORAMANGALA - 560034\n",
    "\n",
    "Strings that shouldn't match:\n",
    "* 999, St. Marks Road, Bangalore\n"
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
      "['466, 5th block, Koramangala, Bangalore', '4th BLOCK, KORAMANGALA - 560034']\n"
     ]
    }
   ],
   "source": [
    "addresses = ['466, 5th block, Koramangala, Bangalore', '4th BLOCK, KORAMANGALA - 560034', '999, St. Marks Road, Bangalore']\n",
    "\n",
    "pattern = \"^[\\w\\d\\s,-]*koramangala[\\w\\d\\s,-]*$\"\n",
    "\n",
    "result = []\n",
    "\n",
    "for address in addresses:\n",
    "    match = re.search(pattern, address, re.I)\n",
    "    if match !=None:\n",
    "        result.append(address)\n",
    "\n",
    "# print result - result should only contain the items that match the pattern\n",
    "print(result)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q3. \n",
    "Write a regular expression that matches either integer numbers or floats upto 2 decimal places.\n",
    "\n",
    "Strings that should match: \n",
    "* 2\n",
    "* 2.3\n",
    "* 4.56\n",
    "* .61\n",
    "\n",
    "Strings that shoudln't match:\n",
    "* 4.567\n",
    "* 75.8792\n",
    "* abc\n"
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
      "['2', '2.3', '4.56', '.61']\n"
     ]
    }
   ],
   "source": [
    "numbers = ['2', '2.3', '4.56', '.61', '4.567', '75.8792', 'abc']\n",
    "\n",
    "pattern = \"^[0-9]*(\\.[0-9]{,2})?$\"\n",
    "\n",
    "result = []\n",
    "\n",
    "for number in numbers:\n",
    "    match = re.search(pattern, number)\n",
    "    if match != None:\n",
    "        result.append(number)\n",
    "\n",
    "# print result - result should only contain the items that match the pattern\n",
    "print(result)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q4. \n",
    "Write a regular expression to match the model names of smartphones which follow the following pattern: \n",
    "\n",
    "mobile company name followed by underscore followed by model name followed by underscore followed by model number\n",
    "\n",
    "Strings that should match:\n",
    "* apple_iphone_6\n",
    "* samsung_note_4\n",
    "* google_pixel_2\n",
    "\n",
    "Strings that shouldn’t match:\n",
    "* apple_6\n",
    "* iphone_6\n",
    "* google\\_pixel\\_\n"
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
      "['apple_iphone_6', 'samsung_note_4', 'google_pixel_2']\n"
     ]
    }
   ],
   "source": [
    "phones = ['apple_iphone_6', 'samsung_note_4', 'google_pixel_2', 'apple_6', 'iphone_6', 'google_pixel_']\n",
    "\n",
    "pattern = \"^.*_.*_\\d$\"\n",
    "\n",
    "result = []\n",
    "\n",
    "for phone in phones:\n",
    "    match = re.search(pattern, phone)\n",
    "    if match !=None:\n",
    "        result.append(phone)\n",
    "\n",
    "# print result - result should only contain the items that match the pattern\n",
    "print(result)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q5. \n",
    "Write a regular expression that can be used to match the emails present in a database. \n",
    "\n",
    "The pattern of a valid email address is defined as follows:\n",
    "The '@' character can be preceded either by alphanumeric characters, period characters or underscore characters. The length of the part that precedes the '@' character should be between 4 to 20 characters.\n",
    "\n",
    "The '@' character should be followed by a domain name (e.g. gmail.com). The domain name has three parts - a prefix (e.g. 'gmail'), the period character and a suffix (e.g. 'com'). The prefix can have a length between 3 to 15 characters followed by a period character followed by either of these suffixes - 'com', 'in' or 'org'.\n",
    "\n",
    "\n",
    "Emails that should match:\n",
    "* random.guy123@gmail.com\n",
    "* mr_x_in_bombay@gov.in\n",
    "\n",
    "Emails that shouldn’t match:\n",
    "* 1@ued.org\n",
    "* @gmail.com\n",
    "* abc!@yahoo.in\n",
    "* sam_12@gov.us\n",
    "* neeraj@"
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
      "['random.guy123@gmail.com', 'mr_x_in_bombay@gov.in']\n"
     ]
    }
   ],
   "source": [
    "emails = ['random.guy123@gmail.com', 'mr_x_in_bombay@gov.in', '1@ued.org',\n",
    "          '@gmail.com', 'abc!@yahoo.in', 'sam_12@gov.us', 'neeraj@']\n",
    "\n",
    "pattern = \"^[a-z_.0-9]{4,20}@[a-z]{3,15}\\.(com|in|org)$\"\n",
    "\n",
    "result = []\n",
    "\n",
    "for email in emails:\n",
    "    match = re.search(pattern, email, re.I)\n",
    "    if match !=None:\n",
    "        result.append(email)\n",
    "\n",
    "# print result - result should only contain the items that match the pattern\n",
    "print(result)"
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
