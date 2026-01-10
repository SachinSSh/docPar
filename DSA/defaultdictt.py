from collections import defaultdict

# Group elements by their first character
words = ["apple", "ant", "banana", "ball", "cat"]
groups = defaultdict(list)

for word in words:
    groups[word[0]].append(word)
    
# Result: {'a': ['apple', 'ant'], 'b': ['banana', 'ball'], 'c': ['cat']}
