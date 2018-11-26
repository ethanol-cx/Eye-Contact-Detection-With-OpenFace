import socket
import sys
import pickle
import csv

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 8102      # The port used by the server

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    args = sys.argv
    frame = int(sys.argv[1])
    del sys.argv[1]

    data = pickle.dumps(args)
    s.send(data)
    data = s.recv(1024)
    prediction = pickle.loads(data)

    # # Uncomment this blob if you want the annotations exported to an output.csv
    ## Make sure you make an empty output.csv file in this folder before running. The script adds rows to existing csv's.
    # with open('output.csv', 'a') as fd:
    #     writer = csv.writer(fd, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(str(prediction))
    ## end of blob

    # # Comment out this blob if running with realtime video.
    # # This is for checking against annotations while running faceLandMarkVid on a video file.
    # with open('009-annotations.csv') as csvfile:
    #     readCSV = list(csv.reader(csvfile, delimiter=','))
    #     if(readCSV[frame][0] != ''):
    #         if(int(readCSV[frame][0]) == 0 or int(readCSV[frame][0]) == 1):
    #             if(int(prediction) == int(readCSV[frame][0])):
    #                 print (bcolors.OKGREEN + "Matches annotation"  + bcolors.ENDC)
    #             else:
    #                 print (bcolors.FAIL + "Doesn't match annotation"  + bcolors.ENDC)
    #     else:
    #         print (bcolors.OKBLUE + "No annotations found for this frame" + bcolors.ENDC)
    # # end of blob

    if(prediction == 0):
        print ("MODEL PREDICTION " + bcolors.WARNING +"NOT LOOKING" +  bcolors.ENDC)
    else:
        print ("MODEL PREDICTION " + bcolors.OKGREEN +"LOOKING" +  bcolors.ENDC)
    print ("–––––––––––––––––––––––––––––––")
    s.close()
    exit()
