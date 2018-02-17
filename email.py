# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:56:18 2018

@author: User
"""
import zipfile
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders
import os

fromaddr = "ankmi.ip@gmail.com"
toaddr = "rashmidabir@gmail.com"
 
msg = MIMEMultipart()
 
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "ALERT"
 
body = "Check this"
 
msg.attach(MIMEText(body, 'plain'))

#vid_zip = zipfile.ZipFile('output.zip', 'w')
#vid_zip.write('output.avi', compress_type=zipfile.ZIP_DEFLATED)
 
#vid_zip.close()
 
filename = "output.avi"
attachment = open("C:/Users/User/PycharmProjects/face_detect/output.avi", "rb")
 
part = MIMEBase('application', 'octet-stream')
part.set_payload((attachment).read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
 
msg.attach(part)
 
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(fromaddr, "ankmi.ip2018")
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
server.quit()

        