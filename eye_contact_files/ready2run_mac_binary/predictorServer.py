from sklearn.externals import joblib
import sys
import numpy as np
import socket
import pickle
import csv

def  predict(args,model,conn):
    args.pop(0) #removes name of script
    argsArray = np.array(args) #convert to array for passing as features
    argsArray.reshape(1,-1) #convert to correct shape
    argsArray = argsArray.astype(np.float64) #convert to float64

    # pred =[0, 0]
    pred = model.predict([argsArray])  #make prediction from arguments
    # print("predict reached", pred[0])

    data2 = pickle.dumps(pred[0])
    conn.send(data2)

    return pred[0] #return prediction

model = joblib.load('dt.joblib')#load sci kit model

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 8102   # Port to listen on (non-privileged ports are > 1023)


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(100)
    while True:
        conn, addr = s.accept()
        try:
            data = conn.recv(1024)
            args = pickle.loads(data)
            predict(args,model,conn)
            # s.close()
        except:
            # s.close()
            print("")
