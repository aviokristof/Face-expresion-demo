
# Face expresion measuring demo version
# Krzysztof Ragan

from email.mime import image
import glob
from imghdr import what
from pickle import FRAME
from posixpath import split
from statistics import mean
from importlib.util import set_loader
from logging import captureWarnings, root
from msilib.schema import ListView
from tkinter.tix import Tk
from tracemalloc import stop
from turtle import width
from PIL import Image,ImageTk
import cv2
import imutils
import tkinter
import numpy as np
import mediapipe as mp
import time
#przekazanie do  GUI
# wartosc do przekazania img,fps,id,x,y,blue,green,red,resized

class GUI(object):
    def __init__(self) -> None:

        self.root = tkinter.Tk()
        self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.mediapipe.python.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=1)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)
        self.lblVideo = tkinter.Label(self.root)
        self.frame=0
        self.zero =0


        self.buffer_x = np.zeros((100,468))
        self.buffer_y = np.zeros((100,468))
        self.diff_x = np.zeros((100,468))
        self.diff_y = np.zeros((100,468))
        self.mean_value_x = np.zeros(10)
        self.mean_value_y = np.zeros(10)
        self.flag_mean = 0
        self.flag_show = 0
        self.i=0
        self.k=0
        self.pTime=0
        self.dim = (800,600)
        x = 0
        y = 0
        self.mouth_flag=0
        self.left_eye_flag=0
        self.right_eye_flag = 0
        self.forhead_flag = 0
        self.left_chick_flag = 0
        self.right_chick_flag = 0
        self.nose_flag = 0
        self.chin_flag = 0
        self.flag_all = 1
        
        #self.main_loop()
        
        

    def display(self):


  
        

        def mean_value(i,id,diff_x,diff_y,mean_value_x,mean_value_y):
            '''
            Function  calculating mean value from ten next difference values of facemesh points
            '''

            # zmienne
            mean_value_x[i] = mean(diff_x[:(10),id])
            mean_value_y[i] = mean(diff_y[:(10),id])
        
        def forhead(self,img,id,x,y,mean_value_x,mean_value_y,flag_show,i):
            '''
            Function visualise points changing colors if muscles on forhead moves
            '''
            mean_value_x = abs(mean_value_x)
            mean_value_y = abs(mean_value_y) 

            if (mean_value_x[i]>1 or mean_value_y[i]>1):
                blue = 255
                green = 255
                red = 125
            else:
                blue = 0
                green = 255
                red = 0
            forhead = np.array([54,68,63,53,52,105,104,103,65,66,69,67,109,108,107,55,9,151,10,338,337,336,285,295,296,299,297,332,333,334,282,283,293,298,284])
            #sorted = sort(left_eye_points)
            if np.any(forhead == id):
                cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(blue,green,red),1)
            pass
        def left_eye(self,img,id,x,y,mean_value_x,mean_value_y,flag_show,i):

            '''
            Function visualise points changing colors if muscles around left eye moves
            '''

            mean_value_x = abs(mean_value_x)
            mean_value_y = abs(mean_value_y) 

            if (mean_value_x[i]>1 or mean_value_y[i]>1):
                blue = 125
                green = 125
                red = 255
            else:
                blue = 0
                green = 255
                red = 0
            left_eye_points = np.array([226,113,31,130,25,223,247,33,228,225,229,7,246,161,30,224,163,110,29,160,144,24,27,159,145,23,230,231,22,153,158,28,222,
                                        221,56,157,154,26,232,233,112,155,173,190,180,133,243,244,245,128])
            #sorted = sort(left_eye_points)
            if np.any(left_eye_points == id):
                cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(blue,green,red),1)
        def right_eye(self,img,id,x,y,mean_value_x,mean_value_y,flag_show,i):
            '''
            Function visualise points changing colors if muscles around right eye moves
            '''
            mean_value_x = abs(mean_value_x)
            mean_value_y = abs(mean_value_y) 

            if (mean_value_x[i]>1 or mean_value_y[i]>1):
                blue = 255
                green = 0
                red = 0
            else:
                blue = 0
                green = 255
                red = 0
            right_eye_points = np.array([465,413,464,357,441,453,463,414,286,258,257,259,260,467,359,263,466,388,387,386,385,384,398,362,341,256,252,253,254,339
                                        ,255, 249,390,373,374,380,381,382,442,443,444,445,342,446,451,450,449,448,261])
            #sorted = sort(left_eye_points)
            if np.any(right_eye_points == id):
                cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(blue,green,red),1)
        def left_chcik(self,img,id,x,y,mean_value_x,mean_value_y,flag_show,i):
            '''
            Function visualise points changing colors if muscles of left chick moves
            '''
            mean_value_x = abs(mean_value_x)
            mean_value_y = abs(mean_value_y) 

            if (mean_value_x[i]>1 or mean_value_y[i]>1):
                blue = 200
                green = 155
                red = 255
            else:
                blue = 0
                green = 255
                red = 0
            left_chick = np.array([251,301300,276,353,265,340,383,368,389,372,264,356,454,447,345,346,352,366,323,361,401,376,411,280,347,288,435,433,416,367,
                                    397,365,364,434,427,425,266,423,426,436,432,430,394,379])
            #sorted = sort(left_eye_points)
            if np.any(left_chick == id):
                cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(blue,green,red),1)
            pass
        def right_chick(self,img,id,x,y,mean_value_x,mean_value_y,flag_show,i):
            '''
            Function visualise points changing colors if muscles of right chick moves
            '''
            mean_value_x = abs(mean_value_x)
            mean_value_y = abs(mean_value_y) 

            if (mean_value_x[i]>1 or mean_value_y[i]>1):
                blue = 30
                green = 70
                red = 255
            else:
                blue = 0
                green = 255
                red = 0
            chick = np.array([150,169,210,212,216,206,36,100,120,119,101,205,207,214,135,136,172,138,192,187,50,118,117,111,116,123,147,213,215,58,132,177,137,227,234,
                                93,143,34,127,156,139,162,21,71,70])
            #sorted = sort(left_eye_points)
            if np.any(chick == id):
                cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(blue,green,red),1)
            pass
        def nose(self,img,id,x,y,mean_value_x,mean_value_y,flag_show,i):
            '''
            Function visualise points changing colors if nose muscles moves
            '''
            mean_value_x = abs(mean_value_x)
            mean_value_y = abs(mean_value_y) 

            if (mean_value_x[i]>1 or mean_value_y[i]>1):
                blue = 0
                green = 255
                red = 255
            else:
                blue = 0
                green = 255
                red = 0
            nose = np.array([2,326,97,327,371,98,129,142,1,4,19,94,275,274,354,370,461,457,440,363,462,458,459,438,344,360,250,309,328,290,392,439,278,
                             279,429,289,305,455,460,294,331,97,98,129,142,141,125,44,45,51,242,241,237,220,134,99,20,238,238,6079,218,115,131,75,166,59,219,235,
                             240,64,48,102,49,209,5,281,51,142,209,198,236,3,195,248,456,480,429,371,329,437,399,419,197,196,174,217,126,100,243,412,351,6,122,
                             168,114,128,245,244,233,243,112,133,155,173,190,189,193,168,417,413,414,398,343,412,351,357,465,464,453,463,341,362,382])
            #sorted = sort(left_eye_points)
            if np.any(nose == id):
                cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(blue,green,red),1)
            pass
        def chin(self,img,id,x,y,mean_value_x,mean_value_y,flag_show,i):
            '''
            Function visualise points changing colors if muscles of chin moves
            '''
            mean_value_x = abs(mean_value_x)
            mean_value_y = abs(mean_value_y) 

            if (mean_value_x[i]>1 or mean_value_y[i]>1):
                blue = 10
                green = 100
                red = 190
            else:
                blue = 0
                green = 255
                red = 0
            chin = np.array([149,170,211,204,106,182,194,32,140,176,148,171,208,201,83,18,200,199,175,152,377,396,428,421,313,406,418,262,369,400,378,395,431,424,335])
            #sorted = sort(left_eye_points)
            if np.any(chin == id):
                cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(blue,green,red),1)
            pass
        def sort(numbers):
            '''
            Function sorting  points on face
            '''

            for j in enumerate(numbers):
                for i,val in enumerate(numbers):
                    #print(i)
                    if i< (numbers.size-1) and (numbers[i]>numbers[(i+1)]):
                        x=numbers[i]
                        numbers[i] = numbers[i+1]
                        numbers[i+1] = x
            return(numbers)
        def mouth(self,img,id,x,y,mean_value_x,mean_value_y,flag_show,i):
            '''
            Function visualise points changing colors if muscles of mouth moves
            '''
           
            mean_value_x = abs(mean_value_x)
            mean_value_y = abs(mean_value_y) 

            if (mean_value_x[i]>1 or mean_value_y[i]>1):
                blue = 255
                green = 0
                red = 255
            else:
                blue = 0
                green = 255
                red = 0

            duppa = np.array([  0,  11,  12,  13,  14,  15,  16,  17,  37,  38,  39,  40,  41,  42,  61,  62,  72,  73,
                        74,  76,  77,  78,  80,  81,  82,  84,  85,  86,  87,  88,  89, 90,  91,  95,  96, 146,
                        178, 179, 180, 181, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 291, 292, 302, 303,
                        304, 306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 324, 325, 375,
                        402, 403, 404, 405, 406, 407, 409, 415])
            if np.any(duppa == id):
                cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(blue,green,red),1)            

        def mouth_flag():
            """
            Function used to set self.mouth_flag value to 1 and other flags to 0 after pressing mouth button in GUI 
            """
            self.mouth_flag = 1
            self.left_eye_flag = 0
            self.right_eye_flag = 0
            self.nose_flag = 0
            self.left_chick_flag = 0
            self.forhead_flag = 0
            self.chin_flag = 0
            self.right_chick_flag = 0
            self.flag_all = 0       
        def left_eye_flag():
            """
            Function used to set self.left_eye value to 1 and other flags to 0 after pressing mouth button in GUI 
            """
            self.mouth_flag = 0
            self.left_eye_flag = 1
            self.right_eye_flag = 0
            self.nose_flag = 0
            self.left_chick_flag = 0
            self.forhead_flag = 0
            self.chin_flag = 0
            self.right_chick_flag = 0
            self.flag_all = 0   
        def right_eye_flag():
            """
            Function used to set self.right_eye value to 1 and other flags to 0 after pressing mouth button in GUI 
            """
            self.mouth_flag = 0
            self.left_eye_flag = 0
            self.right_eye_flag = 1
            self.nose_flag = 0
            self.left_chick_flag = 0
            self.forhead_flag = 0
            self.chin_flag = 0
            self.right_chick_flag = 0
            self.flag_all = 0
        def nose_flag():
            """
            Function used to set self.nose_flag value to 1 and other flags to 0 after pressing mouth button in GUI 
            """
            self.mouth_flag = 0
            self.left_eye_flag = 0
            self.right_eye_flag = 0
            self.nose_flag = 1
            self.left_chick_flag = 0
            self.forhead_flag = 0
            self.chin_flag = 0
            self.right_chick_flag = 0
            self.flag_all = 0
        def left_chick_flag():
            """
            Function used to set self.left_chick_flag value to 1 and other flags to 0 after pressing mouth button in GUI 
            """
            self.mouth_flag = 0
            self.left_eye_flag = 0
            self.right_eye_flag = 0
            self.nose_flag = 0
            self.left_chick_flag = 1
            self.forhead_flag = 0
            self.chin_flag = 0
            self.right_chick_flag = 0
            self.flag_all = 0
        def forhead_flag():
            """
            Function used to set self.forhead_flag value to 1 and other flags to 0 after pressing mouth button in GUI 
            """
            self.mouth_flag = 0
            self.left_eye_flag = 0
            self.right_eye_flag = 0
            self.nose_flag = 0
            self.left_chick_flag = 0
            self.forhead_flag = 1
            self.chin_flag = 0
            self.right_chick_flag = 0
            self.flag_all = 0
        def chin_flag():
            """
            Function used to set self.chin_flag value to 1 and other flags to 0 after pressing mouth button in GUI 
            """
            self.mouth_flag = 0
            self.left_eye_flag = 0
            self.right_eye_flag = 0
            self.nose_flag = 0
            self.left_chick_flag = 0
            self.forhead_flag = 0
            self.chin_flag = 1
            self.right_chick_flag = 0
            self.flag_all = 0
        def right_chick_flag():
            """
            Function used to set self.right_chick_flag value to 1 and other flags to 0 after pressing mouth button in GUI 
            """
            self.mouth_flag = 0
            self.left_eye_flag = 0
            self.right_eye_flag = 0
            self.nose_flag = 0
            self.left_chick_flag = 0
            self.forhead_flag = 0
            self.chin_flag = 0
            self.right_chick_flag = 1
            self.flag_all = 0
                        

                

            
        #wyświetlanie działa zmienić wyświetlanie
        def result_show(self,img,id,x,y,mean_value_x,mean_value_y,flag_show,i):


            
            #Wyświetl dane, dane potrzrbne do przekazania:
            # img
            # id
            # x
            # y
            # mean_x
            # mean_y
            mean_value_x = abs(mean_value_x)
            mean_value_y = abs(mean_value_y) 

            if (mean_value_x[i]>1 or mean_value_y[i]>1):
                blue = 0
                green = 0
                red = 255
            else:
                blue = 0
                green = 255
                red = 0

            cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(blue,green,red),1)# wartość do przekazania

        def grid_tk():
            '''
            Function used for creating user interface via tkinter library
            Frames, buttons, real time video broadcasting 
            '''
            #main_frame = tkinter.Frame(self.root,bd=2,relief = tkinter.SOLID)
            
            left_frame = tkinter.Frame(self.root,bd=2,relief=tkinter.SOLID,width=100,height=600)#wymiary okna video teraz są na sztywno zmienić na zmienną
            right_frame = tkinter.Frame(self.root,bd=2,relief=tkinter.SOLID,width=200,height=600)
            middle_frame = tkinter.Frame(self.root,bd=2,relief=tkinter.SOLID,width=810,height=610)
            
            self.lblVideo = tkinter.Label(middle_frame)
            #self.lblVideo.grid(column=0,row=0)
            
            left_frame.grid(row=0, column=0,sticky="nsew")
            middle_frame.grid(row=0, column=1,sticky="nsew")
            right_frame.grid(row=0, column=2, columnspan=2, sticky="n")
            
            #left_frame.place(relx=0,relheight=1,relwidth=split)
            #what = tkinter.Label(left_frame,self.lblVideo).grid(row=0,column=0)

            
            #right_frame.place(relx=split,relheight=1,relwidth=1.0-split)


            button_init = tkinter.Button(right_frame, text='Init', width=45, command=my_loop)
            button_stop = tkinter.Button(right_frame,text='nose',width=45,command=nose_flag)
            button_mean_2 = tkinter.Button(right_frame,text='mouth',width=45,command=mouth_flag)
            button_mean_3 = tkinter.Button(right_frame,text='left_eye',width=45,command=left_eye_flag)
            button_mean_4 = tkinter.Button(right_frame,text='right_eye',width=45,command=right_eye_flag)
            button_mean_5 = tkinter.Button(right_frame,text='left_chick',width=45,command=left_chick_flag)
            button_mean_6 = tkinter.Button(right_frame,text='forhead',width=45,command=forhead_flag)
            button_mean_7 = tkinter.Button(right_frame,text='chin',width=45,command=chin_flag)
            button_mean_8 = tkinter.Button(right_frame,text='right_chick',width=45,command=right_chick_flag)
            #lblVideo = tkinter.Label(self.root)
            button_init.grid(column=0,row=0,padx=5,pady=5)
            button_mean_2.grid(column=0,row=1,padx=5,pady=5)
            button_mean_3.grid(column=0,row=2,padx=5,pady=5)
            button_mean_4.grid(column=0,row=3,padx=5,pady=5)
            button_stop.grid(column=0,row=4,padx=5,pady=5)
            button_mean_5.grid(column=0,row=5,padx=5,pady=5)
            button_mean_6.grid(column=0,row=6,padx=5,pady=5)
            button_mean_7.grid(column=0,row=7,padx=5,pady=5)
            button_mean_8.grid(column=0,row=8,padx=5,pady=5)


        def my_loop():

            '''
            Main loop of the program
            Capturing video from camera
            Putting 468 points on face using Facemesh
            Calculating difference of Facemesh points positions between frames of video
            Calls grid_tk function responsible for visualising face muscles movement



            '''

            success,img = self.cap.read()
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = self.faceMesh.process(imgRGB)
            if results.multi_face_landmarks:
                for faceLms in results.multi_face_landmarks:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpec,self.drawSpec)
                    for id,lm in enumerate(faceLms.landmark):
                        ih,iw,ic = img.shape
                        x,y = int(lm.x*iw),int(lm.y*ih)
                        self.buffer_x [self.i,id] = x
                        self.buffer_y [self.i,id] = y
                        self.diff_x [self.i,id] = self.buffer_x [self.i-1,id] - self.buffer_x [self.i,id]
                        self.diff_y [self.i,id] = self.buffer_y [self.i-1,id] - self.buffer_y [self.i,id]
                            
                            #Wywołanie funkcji obliczającej średnią arytmetyczną
                            #komentarz: przekazać tablice czy tylko x i y?
                            #najpierw przekaże x i y
                        if self.flag_mean != 0:
                            mean_value(self.i,id,self.diff_x,self.diff_y,self.mean_value_x,self.mean_value_y)
                        if self.flag_mean != 0 and self.flag_all == 1:
                            result_show(self,img,id,x,y,self.mean_value_x,self.mean_value_y,self.flag_show,self.i)

                        if self.flag_mean !=0 and self.mouth_flag == 1:
                            mouth(self,img,id,x,y,self.mean_value_x,self.mean_value_y,self.flag_show,self.i)
                        if self.flag_mean !=0 and self.left_eye_flag == 1:
                            left_eye(self,img,id,x,y,self.mean_value_x,self.mean_value_y,self.flag_show,self.i)
                        if self.flag_mean !=0 and self.right_eye_flag == 1:
                            right_eye(self,img,id,x,y,self.mean_value_x,self.mean_value_y,self.flag_show,self.i)
                        if self.flag_mean !=0 and self.nose_flag == 1:
                            nose(self,img,id,x,y,self.mean_value_x,self.mean_value_y,self.flag_show,self.i)
                        if self.flag_mean !=0 and self.left_chick_flag == 1:
                            left_chcik(self,img,id,x,y,self.mean_value_x,self.mean_value_y,self.flag_show,self.i)
                        if self.flag_mean !=0 and self.forhead_flag == 1:
                            forhead(self,img,id,x,y,self.mean_value_x,self.mean_value_y,self.flag_show,self.i)
                        if self.flag_mean !=0 and self.chin_flag == 1:
                            chin(self,img,id,x,y,self.mean_value_x,self.mean_value_y,self.flag_show,self.i)
                        if self.flag_mean !=0 and self.right_chick_flag == 1:
                            right_chick(self,img,id,x,y,self.mean_value_x,self.mean_value_y,self.flag_show,self.i)


            self.i=self.i+1
            self.k = self.k+1
            if self.k == 100:
                self.flag_show = 1
                self.k=0
            if self.i ==10:
                self.flag_mean = 1
                self.i=-1 #bardzo ważny moment i=-1 !!!!!!!!!!!! ???tab[i-1,id]-tab[i,id]???
                
            resized = cv2.resize(img,self.dim, interpolation=cv2.INTER_AREA)
            #frame = imutils.resize(resized,width=640)
            frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)    
            img = ImageTk.PhotoImage(image=im)
            self.lblVideo.configure(image=img)
            self.lblVideo.image = img
            self.zero = 1
            self.lblVideo.grid(column=0,row=0) #wartość z funkcji grid żeby ustalać def szerokość anie taki wąski pasek
            self.lblVideo.after(10,my_loop)

        

        grid_tk()

        self.root.mainloop()

        



if __name__ == "__main__":

    
    app =GUI()
    app.display()
