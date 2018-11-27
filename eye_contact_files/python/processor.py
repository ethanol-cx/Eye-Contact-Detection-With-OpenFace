import csv
with open('1.csv','rb') as annotations: #annotations csv goes here
    with open('cutesttalkingtoddler_combined.csv', 'w') as out:
        writer = csv.writer(out)
        annotReader = csv.reader(annotations)
        for row in annotReader:
            start = float(row[3])
            end = float(row[6])
            id = row[11]
            with open('2.csv') as of: #open face csv goes here
                ofReader = csv.reader(of)
                next(ofReader)
                for r in ofReader:
                    timestamp = float(r[2])
                    if(timestamp > start) and (timestamp < end):
                        writer.writerow(r + [id])
