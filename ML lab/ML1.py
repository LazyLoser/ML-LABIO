import csv
def loadCsv(filename):
 lines = csv.reader(open(filename, "r"))
 dataset = list(lines)
 for i in range(len(dataset)):
  dataset[i] = dataset[i]
 return dataset
attributes = ['Sky','Temp','Humidity','Wind','Water','Forecast']
print('Attributes =',attributes)
num_attributes = len(attributes)
filename = "finds.csv"
dataset = loadCsv(filename)
print(dataset)
hypothesis=['0'] * num_attributes
print("Intial Hypothesis")
print(hypothesis)
print("The Hypothesis are")
for i in range(0, len(dataset)):
    if dataset[i][num_attributes] == 'yes':
        print ("\nInstance ", i+1, "is", dataset[i], " and is Positive Instance")
        for j in range(0, num_attributes):
            if hypothesis[j] == '0' or hypothesis[j] == dataset[i][j]:
                hypothesis[j] = dataset[i][j]
            else:
                hypothesis[j] = '?'
        print(i+1, hypothesis, "\n")

    if dataset[i][num_attributes] == 'no':
        print ("\nInstance ", i+1, "is", dataset[i], " and is Negative Instance Hence Ignored")
        print("The hypothesis for the training instance", i+1, " is: " , hypothesis, "\n")

print("\nThe Maximally specific hypothesis for the training instance is ", hypothesis)
