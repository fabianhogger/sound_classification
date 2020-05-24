import socket
import time
from ssh2.session import Session
from ssh2.sftp import LIBSSH2_FXF_READ, LIBSSH2_SFTP_S_IRUSR
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
memory=open("mem.txt","a+")
channel=session.open_session()
test=open("P:\\text.txt","w+")
test.write("success")
test.close()
'''
sftp = session.sftp_init()

with sftp.open("/home/pi/Desktop/test.wav",LIBSSH2_FXF_READ, LIBSSH2_SFTP_S_IRUSR) as remote_fh,\
    open("C:/Users/30697/Desktop/file", 'wb') as local_fh:
    for size, data in remote_fh:
        local_fh.write(data)
'''
channel.shell()
channel.write("arecord -D sysdefault:CARD=1 -d 5 -f cd -t wav f2.wav\n")
time.sleep(10)
channel.write("ls\n")
time.sleep(2)
size,data=channel.read()
#print(data.decode())

memory.write(data.decode())

channel.close()
print("exit status{0}".format(channel.get_exit_status()))
memory.close()
