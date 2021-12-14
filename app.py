from flask import Flask, request, render_template, session, flash, redirect, url_for
from flask_mail import Mail, Message
from flask_mysqldb import MySQL 
from functools import wraps
from flask_session import Session
import re

app = Flask(__name__)

# Session storage for login
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = 'abcdef'

# Configs for MYSQLdb login
app.config['MYSQL_USER'] = 'sql6458642'
app.config['MYSQL_PASSWORD'] = 'fpla5W97zw'
app.config['MYSQL_HOST'] = 'sql6.freemysqlhosting.net'
app.config['MYSQL_DB'] = 'sql6458642'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)


# Email module
def mailer(anamoly):
    mail = Mail(app) 

    # configuration of mail
    app.config['MAIL_SERVER']='smtp.gmail.com'
    app.config['MAIL_PORT'] = 465
    app.config['MAIL_USERNAME'] = 'capstone.ivs@gmail.com'
    app.config['MAIL_PASSWORD'] = 'capstone123'
    app.config['MAIL_USE_TLS'] = False
    app.config['MAIL_USE_SSL'] = True

    mail = Mail(app)
    # message object mapped to a particular URL ‘/’
    msg = Message(
                'ALERT',
                sender ="capstone.ivs@gmail.com",
                recipients = ["shivamrawat2000@gmail.com"]
                )
    msg.body = anamoly;
    mail.send(msg)
    print('Mail sent')

# Logout of the page
@app.route("/logout")
def logout():
    session["name"] = None
    return redirect("/")

# Login page : First page of the app (auth)
@app.route('/')
def login():
    return render_template('login.html')

# Verify login credentials and redirect to homepage if valid
@app.route('/',methods=['POST'])
def authenticate():
    username = request.form['email']
    password = request.form['pass']
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM user WHERE username='"+username+"' AND password='"+password+"'")
    data=cur.fetchone()
    if data is None:
        return "Username or Password is wrong"
    else:
        session["name"]=username
        return render_template('index.html')

# Homepage of the app
@app.route('/index')
def index():
    if not session.get("name"):
    # if not there in the session then redirect to the login page
        return redirect(url_for('login'))
    return render_template('index.html')

# Contact us page
@app.route('/ContactUs')
def ContactUs():
    # if not session.get("name"):
    #     return redirect(url_for('login'))
    return render_template('ContactUs.html')
    
# Fall detection model page
@app.route('/fall')
def Fall():
    # if not session.get("name"):
    #     return redirect(url_for('login'))
    return render_template('fall.html')

# Fall detection model code
@app.route('/DetectFall')
def DetectFall():
    # if not session.get("name"):
    #     return redirect(url_for('login'))
    
    import cv2
    import time

    fitToEllipse = False
    cap = cv2.VideoCapture('Videos/Fall/fall.mp4')
    time.sleep(2)

    fgbg = cv2.createBackgroundSubtractorMOG2()
    j = 0
    fall_flag=0
    while(1):
        ret, frame = cap.read()
        
        #Convert each frame to gray scale and subtract the background
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(gray)
            
            #Find contours
            contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
            
                # List to hold all areas
                areas = []

                for contour in contours:
                    ar = cv2.contourArea(contour)
                    areas.append(ar)
                
                max_area = max(areas, default = 0)

                max_area_index = areas.index(max_area)

                cnt = contours[max_area_index]

                M = cv2.moments(cnt)
                
                x, y, w, h = cv2.boundingRect(cnt)

                cv2.drawContours(fgmask, [cnt], 0, (255,255,255), 3, maxLevel = 0) #removed
                
                if h < w:
                    j += 1
                    
                if j > 10:
                    fall_flag=1
                    break;
                    # Uncomment below statements to alert on imshow window
                    # cv2.putText(fgmask, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 2) #removed
                    # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) #removed

                if h > w:
                    j = 0 
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                cv2.imshow('video', frame)
            
                if cv2.waitKey(33) == 27:
                    break
        except Exception as e:
            break
    cv2.destroyAllWindows()
    if(fall_flag==1):
        # mailer("Fall Detected !")
        # print('mail sent')
        return render_template("alert.html",type_of_anamoly="fall")
    else:
        return render_template('normal.html',type_of_anamoly="fall") 

# Social distancing model page
@app.route('/SocialDistancing')
def SocialDistancing():
    # if not session.get("name"):
    #     return redirect(url_for('login'))
    return render_template('SocialDistancing.html')    

# Abandoned object detection model page
@app.route('/ObjectDetection')
def ObjectDetection():
    # if not session.get("name"):
    #     return redirect(url_for('login'))
    return render_template('ObjectDetection.html')

# Social distancing model code
@app.route('/DetectSocial')
def DetectSocial():
    # if not session.get("name"):
    #     return redirect(url_for('login'))

    from Utility_Files import social_distancing_config as config
    from Utility_Files.detection import detect_people
    from scipy.spatial import distance as dist
    import numpy as np
    import argparse
    import imutils
    import cv2
    import os

    labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    if config.USE_GPU:
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture('Videos/SocialDist/pedestrians.mp4')
    social_flag = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
            personIdx=LABELS.index("person"))

        violate = set()

        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < config.MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)
                        social_flag=1

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # if(social_flag==1):
        #     # mailer("Social distancing norm violated !")
        #     # print('mail sent')
        #     return render_template("alert.html",type_of_anamoly="Social Distancing breach")
        # else:
        #     return render_template('normal.html',type_of_anamoly="Social Distancing breach") 

@app.route('/Survilliance')
def Survilliance():
    # if not session.get("name"):
    #     return redirect(url_for('login'))
    # importing required libraries
    import numpy as np
    import cv2

    #reading input video frame by frame by opencv
    cap = cv2.VideoCapture('Videos/Abandoned/video1.avi')

    #saving first frame of the video as background image
    _, BG= cap.read()
    BG=cv2.cvtColor(BG,cv2.COLOR_BGR2GRAY)            #changing backgroung image to gray scale
    cv2.equalizeHist(BG)                              #increasing contrast of the image
    BG=cv2.GaussianBlur(BG,(7,7),0)                   #bluring the edges of the image 
    cv2.imshow('BG', BG)
    #circular dictionary initiaization which has frame no. as key and list centroids of blobs as value 
    fgcnts={}
    frameno=0
    flag=0
    while (cap.isOpened()):
        # reading frame from video one by one
        ret, frame = cap.read()
    
        if ret==0:                    #break the if it is last frame
            break
        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)        #change frame to gray scale
        cv2.equalizeHist(gray)                             # increasing cntrast of frame
        gray=cv2.GaussianBlur(gray,(7,7),0)                #bluring the edges 
    
        # taking absolute difference of background image with current frame
        fgmask=cv2.absdiff(gray.astype(np.uint8), BG.astype(np.uint8))
    
        #applying threshold on subtacted image
        rt,fgmask=cv2.threshold(fgmask.astype(np.uint8), 25, 255, cv2.THRESH_BINARY)
  
        #applying mrphological operation both erosion and dilation  
        kernel2 = np.ones((8,8),np.uint8)   #higher the kernel, eg (10,10), more will be eroded or dilated
        thresh2 = cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE, kernel2,iterations=3)
        #applying edge detector after morphological operation 
        edged = cv2.Canny(thresh2, 30,50)
    
        #finding boundaries of all blobs of the frame
        contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
        #initialising list of centroids of blobs
        fgcnts[frameno%1000]=[]
        for contour in contours:
            #finding the centroid of each blob
             M = cv2.moments(contour)
             if not M['m00'] == 0: 
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centre=(cx,cy) 
                #appending the centroid to list
                fgcnts[frameno%1000].append(centre)
           
            #checking for unattended object
                if frameno>200:
                    if cv2.contourArea(contour) in range(200,15000) and centre in fgcnts[(frameno-190)%1000] and fgcnts[(frameno-100)%1000] and fgcnts[(frameno-50)%1000]:
                        flag=1
                        (x,y,w,h) = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
                        cv2.putText(frame,'Alert', (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                        break
        
        #background update: to remove the wrong prediction press key 'a'    
        if(cv2.waitKey(1) & 0xFF == ord('a')):
            BG[y:y+h, x:x+w]=gray[y:y+h, x:x+w] 
        
        frameno+=1
        frame=cv2.resize(frame, (1500,720))          #final frame is resized to a value
        cv2.imshow('original',frame)                #final frame is shown

    #to
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    #when everything is done, release cap
    cap.release()
    cv2.destroyAllWindows()
    if(flag==1):
        # mailer("Abandoned object detected !")
        # print('mail sent')
        return render_template("alert.html",type_of_anamoly="suspicious object")
    else:
        return render_template('normal.html',type_of_anamoly="suspicious object")  

@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form :
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM user WHERE username = % s', (username, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers !'
        elif not username or not password :
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO user VALUES (%s,%s)', (username, password, ))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
