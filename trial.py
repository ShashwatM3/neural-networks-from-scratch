layers = [1,2,3,4]
for i in range(len(layers)):
  for j in reversed(range(i+1, len(layers))):
    print(j)
  break