import csv

with open('source.csv','a') as source:
    with open('new.csv','rb') as new:
        writer = csv.writer(source)
        reader = csv.reader(new)
        for row in reader:
            writer.writerow(row)
