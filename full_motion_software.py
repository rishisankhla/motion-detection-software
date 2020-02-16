import cv2
import datetime
import pandas as pd
from bokeh.models import HoverTool
from bokeh.plotting import *
from bokeh.plotting import show, figure
f_frame = None
status_list = [None, None]
t = []
video = cv2.VideoCapture(0)
df = pd.DataFrame(columns=['start', 'end'])
while True:#here we are running loop under loop first loop shows every frame in the window and the inner loop mark every motion in the video by rectangle
    c, frame = video.read()#read means read the current position
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)#it convert all the sharpen values in the image to smoothen value,21,21 is the level of intensity
    if f_frame is None:
        f_frame = gray
        continue
    d_frame = cv2.absdiff(f_frame, gray)
    t_d = cv2.threshold(d_frame, 30, 255, cv2.THRESH_BINARY)[1]#here instead of [1] & [0] are the two value it returns [0]= second parameter i.e. 30 try print [0], 30 is the intensity ,THRESH_BINARY is the format type of thresh hold we want , 255 is the pixel intensity (it will convert white color of motion to little bit black if we use 10 here)
    t_d = cv2.dilate(t_d, None, iterations=0)# delta frame (which is a comparision frame) all the noise is cleared and converted in to black and white [initially its black gets converted white where motion is found ],iteration = 0 try 50 it will make less partition in in the whole motion object , here none stands for  kernel which is an array as we increase the size of array it makes the image more blur , but here we don't want because its not real image its converted into threshold(i.e. black and white)
    (cnts, _) = cv2.findContours(t_d.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # it returns that frame in which motion is detected in the form of list of arrays , here we use .copy() because if not then this find function will make the changes in real file too (here i.e. our t_d) other wise it makes a copy and use that copy and makes changes in that copy file only.#cv2.CHAIN_APPROX_SIMPLE this method return all the 4 points there our other method too , suppose some of will return whole square line instead of points ,cv2.RETR_EXTERNAL it is a mode of this function here we use this mode because some time there is a shape inside a shape (like child standing in front of elder [in front of camera] means two object) here this 'cv2.RETR_EXTERNAL' mode returns the point of rectangle external object i.e. adult but there r some other mode which will give rectangle points for both the objects
    for c in cnts:#this loop runs only that much amount of time as many rectangles can be made in that frame (means as many element in list)
        if cv2.contourArea(c) < 10000:# level of zoom you want to be detect here we are skipping those element of list whose value is smaller than condition.eg list has 18 element but only 2 rectangle is printed then 16 are skied here (because we can derive coordinate all the 18 but we want only of pixel more that 1000[eg1000])
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(c)#this function recognize the pattern in the list element and returns the coordinates value
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    status_list.append(status)
    status_list = status_list[-2:]
    if status_list[-1] == 1 and status_list[-2] == 0:
        t.append(datetime.datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        t.append(datetime.datetime.now())
    cv2.imshow("original frame", frame)
    cv2.imshow("thrash frame", t_d)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
for i in range(0, len(t), 2):
    df = df.append({"start": t[i], "end": t[i+1]}, ignore_index=True)
ce = ColumnDataSource(df)#some places we need this , that's why we use @column_name in hovertool other wise we uses to mark = df["column_name"]
p = figure(x_axis_type='datetime', height=600, width=1200, title="my motion graph")#if you want x axis showing date time format then we have to specify type
p.yaxis.minor_tick_line_color = None # it removes the small small line partition on y-axis between 0 and 1
p.ygrid[0].ticker.desired_num_ticks = 1 #here we can set the number of grid lines we need between 0-1 on y-axis, why [0] because p.ygrid/xgrid is a list and contain only one element(i.e. our grid) i mean we have to specify on which axis we want the grid
h = HoverTool(tooltips=[("yo_start", "@start"), ("yo_end", "@end")])
p.add_tools(h)
p.quad(left="start", right="end", bottom=0, top=1, color="red", source=ce)#it is a quadrant(which is only use to study single axis) so we have to specify bottom and top of y-axis
show(p)
