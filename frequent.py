W= input('Please enter a string ')
def most_frequent(string):
    d = dict()
    for key in string:
        if key not in d:
            d[key] = 1
        else:
            d[key] += 1
    return d

print (most_frequent(W))


Output:
  Please enter a string Mississippi
{'M': 1, 'i': 4, 's': 4, 'p': 2}