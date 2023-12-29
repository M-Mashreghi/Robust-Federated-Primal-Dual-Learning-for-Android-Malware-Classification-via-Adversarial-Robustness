integers = [1, 2, 3]
letters = ['a', 'b', 'c']
floats = [4.0, 5.0, 6.0]
zipped = zip(integers,zip(letters, floats) )  # Three input iterables
for i in zipped:
    print(i)
print(zipped)
list(zipped)

print(zipped)