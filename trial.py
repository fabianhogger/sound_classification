import socket
import time
from ssh2.session import Session
from ssh2.sftp import LIBSSH2_FXF_READ, LIBSSH2_SFTP_S_IRUSR
import sys
import make_prediction
#recording audio with shell command arecord -D sysdefault:CARD=1 -d 10 -f cd -t wav f2.wav
#take image sudo fswebcam -r 1280x720 --no-banner image3.jpg
host="192.168.1.76"
user="pi"
password="kostas"
socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
socket.connect((host,22))
session=Session()
session.handshake(socket)
session.userauth_password(user,password)
#memory=open("mem.txt","a+")
channel=session.open_session()
#test=open("P:\\text.txt","w+")
#test.write("success")
#test.close()

command="arecord -D sysdefault:CARD=1 -d"+sys.argv[2]+" -f cd -t wav "+sys.argv[1]+ ".wav\n"#add to the command the time in seconds and the filename to be created
channel.shell()
channel.write(command)
time.sleep(10)#wait forthe recording to take place
#channel.write("ls\n")
time.sleep(2)
#size,data=channel.read()
#print(data.decode())

#memory.write(data.decode())

channel.close()
print("exit status{0}".format(channel.get_exit_status()))
memory.close()
name=sys.argv[1]+'.wav' #making the string with the filename to be passed to make_pred()
make_prediction.make_pred(name,sys.argv[2])
