from tkinter import *
from tkinter import ttk
from threading import Thread
import tkinter.font as tkfont
import tobii_research as tr
from PIL import Image, ImageTk
import numpy as np
import math
import os
import json
import base64
from io import BytesIO
from datetime import datetime
import random
import time


''' INITIALIZED HYPERPARAMETERS: edit variables below to your preferences '''

#folder_path = r"C:\Users\marta\tfg\spdocvqa_images" #folder path that contains all set of images to be shown

#print the last item in the file
PARTICIPANT_IDS_FILE = "participant_IDs.txt"
with open(PARTICIPANT_IDS_FILE, "r") as f:
    user_id_n = f.readlines()[-1]
    user_id = user_id_n.strip()
    #import pdb; pdb.set_trace()

user_file = f"{user_id}_experiment_data.json" #file to store all data collected for given user

#indicates whether images are only vertical or landscape or a mix of both. 
#if they are landscape set to False
#if they are mixed set to False
#otherwise, set to True (meaning there are just vertical documents)
only_vertical = False 


if not only_vertical:
    folder_path =  r"Human-Eye-Tracking-for-Driving-Explainability-in-DocVQA\dataset_landscape"  #folder path that contains all set of images to be shown
    #OCR DATA -- contains data on each of the images of DocVQA dataset Task 1
    npy_filtered_landscape_path = r"Human-Eye-Tracking-for-Driving-Explainability-in-DocVQA\filtered_imdb_train_landscape.npy" 
    ocr_data = np.load(npy_filtered_landscape_path,allow_pickle=True)

    #reduced proportion: the maximum proportion of display height/width that image will take up in experiment
    #only hyperparameter recommended to not change... all others should be at least considered by the experimenter
    reduced_prop=.8

    split_layout=False

else:
    
    folder_path = r"C:\Users\marta\tfg\1. datasets\dataset_vertical" #folder path that contains all set of images to be shown
    #OCR DATA -- contains data on each of the images of DocVQA dataset Task 1
    ocr_data = np.load("C:/Users/marta/tfg/filtered_imdb_train_vertical.npy",allow_pickle=True)
    
    #reduced proportion: the maximum proportion of display height/width that image will take up in experiment
    #only hyperparameter recommended to not change... all others should be at least considered by the experimenter
    reduced_prop=.95

    split_layout=True

rand_image = True #indicates whether image will be randomly fetched from folder (if False, images are fetched in order)
total_num_images = 10 #number of images to be shown in experiment

latest_eye_tracker_timestamp = None

'''INITIALIZED GLOBAL VARIABLES: each used in one or more functions below'''

#PIXEL DIMENSIONS USED TO CALCULATE CORRESPONDING LOCATION ON IMAGE FOR GIVEN GAZE COORDINATE
widget_width,widget_height=0,0
left_x, top_y = 0,0

#INFORMATION ABOUT IMAGE TO HELP DATA ANALYSIS/VISUALIZATION
#NOTE: replace folder_path with whatever path contains desired images to be shown in experiment
image_number = None
image_id = None
resized_image = None
file_names=[filename for filename in os.listdir(folder_path)]

#BOOLEANS TO INDICATE PROGRESS ALONG DIFFERENT POINTS OF EXPERIMENT 
collecting_data=False
done_collecting=False
experiment_done=False
first_time=True
num_images_sofar=0

on_intermediate_page=False

user_answer = ""
key_sequence = []

#DateTime to use as marker to organize data collection
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

current_trial_data = {formatted_now:{}}

key_pressed_data = {formatted_now:{}} #dictionary to hold all key press data for each image

#RETRIEVE ALL EXPERIMENT DATA
current_dir = os.path.dirname(os.path.abspath(__file__))
PARTICIPANT_EYE_DATA_DIR = "participant_eye_data"
participant_eye_data_path = os.path.join(current_dir, PARTICIPANT_EYE_DATA_DIR)
if not os.path.exists(participant_eye_data_path):
    os.makedirs(participant_eye_data_path)


#if file existst open if not create
filepaths = os.listdir(participant_eye_data_path)
if user_file not in filepaths:
    with open(os.path.join(participant_eye_data_path, user_file), 'w') as json_file:
        json.dump({}, json_file)

#LOAD EXPERIMENT DATA
json_path = os.path.join(participant_eye_data_path, user_file)
#json_path = os.path.join(current_dir, 'experiment_data.json')

with open(json_path, 'r') as json_file:
    experiment_data = json.load(json_file)


shown_images = set()
if experiment_data:  #check if the dictionary not empty
    for session in experiment_data.values():  #iterate over sessions
        shown_images.update(session.keys())
    shown_images = {key + ".png" for key in shown_images}  #add .png extension to each image name 
        


def save_data():
    global experiment_data, current_trial_data, key_pressed_data, formatted_now
    
    if formatted_now not in experiment_data:
        experiment_data[formatted_now] = {}

    #combined_data = current_trial_data.copy()

    for img_id, trial_data in current_trial_data[formatted_now].items():
        if img_id not in experiment_data[formatted_now]:
            experiment_data[formatted_now][img_id] = {}
        
        experiment_data[formatted_now][img_id].update(trial_data)
    
        if formatted_now in key_pressed_data and img_id in key_pressed_data[formatted_now]:
            experiment_data[formatted_now][img_id].update(key_pressed_data[formatted_now][img_id])

            #print("key pressed data:", key_pressed_data[formatted_now][img_id])
            #print("experiment data:", experiment_data[formatted_now][img_id]["keypressed_data"])
    #experiment_data[formatted_now][img_id].update(trial_data)
    #experiment_data[formatted_now]=current_trial_data[formatted_now]
    #experiment_data[formatted_now] = combined_data
    #print("current trial data:", current_trial_data[formatted_now][image_id]["keypressed_data"])
            

    
    with open(json_path,'w') as json_file:
        json.dump(experiment_data, json_file, indent=4)

def get_file_question(id):
    '''
    Given an Image ID, returns the question associated with the image

    Parameters:
        id (str): Image ID. Found within the image_next() function

    Returns:
        str: string that is the question corresponding to given image (in the context of DocVQA dataset Task 1)
    '''
    
    for file in ocr_data[1:]:
        if file['image_id']==id:
            return file['question']
    
    raise(Exception(f'Image ID {id} not found'))

def get_file_number(id):
    '''
    Given an Image ID, returns associated index of that image in the DocVQA dataset
    
    Parameters:
        id (str): Image ID. Found within the image_next() function

    Returns:
        int: integer that is used to index into DocVQA dataset to access the given image's data information
    '''
 
    for index, file in enumerate(ocr_data[1:]):
        if file['image_id']==id:
            return index+1
        

def fit(filename):
    '''
    Returns a new image that is at most reduced_prop size of the screen in both height and width
    
    Parameters:
        filename (str): the filename for the image
        reduced_prop (float): a float between 0 and 1 that indicates that maximal proportion of the screen 
                                you want the image to take up in both height & width. Default value is .8

    Returns:
        Image: a Pillow Image object that is resized to fit the screen
    '''
    
    image = Image.open(filename)
    #this block is resizing the image until it is smaller than constraints defined by reduced_prop
    while image.size[0]>reduced_prop*screensize[0] or image.size[1]>reduced_prop*screensize[1]:
        image = image.resize((math.floor(.95*image.size[0]),math.floor(.95*image.size[1])))
    return image

#move to next page function
def nextpage():
    '''
    Changes the tkinter screen to the next IMAGE/QUESTION page. 

    '''
    global collecting_data, first_time

    collecting_data = True #boolean to indicate data collection to begin -- see function run_eye_tracker()

    if first_time: #transition out of Welcome page

        #get rid of home_page widgets
        welcome.grid_remove()
        instr.grid_remove()
        instr_head.grid_remove()
        press_enter_msg.grid_remove()
        centerframe['padding']=0

        if split_layout:
            #image on left, question and response on right
            document.grid(column=2, row=0, rowspan=2, sticky=(N, W))
            #question.grid(column=1, row=0, sticky=(N, W, S), padx=5, pady=20)
            question.grid(column=0, row=0, columnspan=2, sticky=(W), padx=5, pady=1)
            #text_response.grid(column=1, row=1, sticky=(W), padx=10, pady=5)
            text_response.grid(column=0, row=1, sticky=(E), padx=5, pady=1)
            response.grid(column=1, row=1, sticky=(W, E), padx=5, pady=1)
            root.columnconfigure(2, weight=1)
        else:
            #Dafault: initialize experiment widgets
            document.grid(column=0,row=1,columnspan=2)
            question.grid(column=0,row=0,columnspan=2,pady=header_len)
            text_response.grid(column=0,row=2)
            response.grid(column=1,row=2)
            

        first_time=False #no longer first time
    else:
        next_question.grid_remove()  #remove Next Question button

        #reset and re-visualize user_response entry
        user_response.set('')
        text_response.grid(column=0,row=2)
        response.grid(column=1,row=2)
        
    

    #call function that handles fetching of next image/question pair
    image_next()

    

def image_next():
    '''
    Handles the actual fetching of the next image/question pair from the given folder.

    '''
    global resized_image, image_number, image_id, rand_image, num_images_sofar
    global widget_height,widget_width,left_x,top_y, experiment_done
    global user_answer, key_sequence

    available_images = [img for img in file_names if img not in shown_images]
    print("num available images:", len(available_images))
    #import pdb; pdb.set_trace()
    
    if num_images_sofar < total_num_images and available_images: #ensure we still have more images to show
        if rand_image:
            new_image_name = random.choice(available_images)
        else:
            #new_image_name = file_names.pop(0)
            new_image_name = available_images.pop(0)
        new_image_path = os.path.join(folder_path,new_image_name)

        shown_images.add(new_image_name) #add image to shown images so it won't be shown again
        print("num shown images:", len(shown_images))

        image_id = new_image_name[:-4] #IMAGE ID -- ONLY valid for 3-letter image files (PNG,JPG,etc) -- otherwise will require modification


        resized_image=fit(new_image_path) #Resize and format image to prepare for visualization in GUI
        formatted_img=ImageTk.PhotoImage(resized_image)


        #add image as widget for next page
        document['image']=formatted_img
        document.image=formatted_img

        if split_layout:
            #image on the left, question & response on the right
            document.grid(column=2, row=0, rowspan=2, sticky=(N, W))
            #question.grid(column=1, row=0, sticky=(N, W, S), padx=5, pady=20)
            question.grid(column=0, row=0, columnspan=2, sticky=(W), padx=5, pady=1)
            #text_response.grid(column=1, row=1, sticky=(W), padx=10, pady=5)
            text_response.grid(column=0, row=1, sticky=(E), padx=5, pady=1)
            response.grid(column=1, row=1, sticky=(W, E), padx=5, pady=1)
            root.columnconfigure(2, weight=1)

        else:
            document.grid(column=0,row=1,columnspan=2)
            question.grid(column=0, row=0, columnspan=2, pady=header_len)
            text_response.grid(column=0,row=2)
            response.grid(column=1,row=2)
            document.grid(column=0,row=1,columnspan=2)

        #focus mouse on user entry    
        response.focus_set()
        
        #set document question
        image_number = get_file_number(image_id)
        doc_question.set(ocr_data[image_number]['question'])

        #increment # of images shown by 1
        num_images_sofar+=1

        #get widget width and length
        root.update()
        widget_width , widget_height = document.winfo_width(),document.winfo_height()

        #calculate display dimensions, necessary for accurate calibration between 
        # gaze data point and corresponding image location
        left_x = leftframe.winfo_width()+(centerframe.winfo_width()-widget_width)//2
        top_y = question.winfo_height()+2*header_len

        user_answer = ""
        key_sequence = []

    else: #once we have no more images to show for the trial
        #remove everything, set experiment_done Boolean to true and save trial data
        document.grid_remove()
        response.grid_remove()
        text_response.grid_remove()
        experiment_done = True
        doc_question.set('                All done! Yay! \n\n\nPress the escape key to quit the app')
        
        #center the question label
        question.grid(column=0, row=0, columnspan=2) 
        question.place(relx=0.5, rely=0.5, anchor=CENTER)  #to center within the grid cell
        save_data()


def on_key_press(event):
    """log each key press with timestamp"""
    global image_id, key_pressed_data, formatted_now, latest_eye_tracker_timestamp
    global user_answer, key_sequence

    current_time = time.time()
    if image_id not in key_pressed_data[formatted_now]:
        key_pressed_data[formatted_now][image_id] = {"keypressed_data": []}
        #print("imageid:", image_id)
        #print(current_trial_data[formatted_now])
        
    key = event.keysym #get key pressed 
    key_sequence.append(key)

    possible_key = {"minus":"-", "space": " ", "comma": ",", "period":".", "parenright":")", "parenleft":"(", "equal":"=", "percent":"%", "dollar": "$", "slash":"/", "question":"?", "quotedbl":"\"", "apostrophe":"'", "exclam":"!", "ampersand":"&", "colon": ":"}
    
    if len(key) == 1:
        user_answer += key
    elif key in possible_key:
        user_answer += possible_key[key]
    elif key == "BackSpace":
        user_answer = user_answer[:-1]         
    

    key_pressed_data[formatted_now][image_id]["keypressed_data"].append({"key": key, "system_timestamp": current_time, "eye_tracker_timestamp": latest_eye_tracker_timestamp})
    #print(key_pressed_data[formatted_now][image_id]["keypressed_data"])
    #print(key_pressed_data[formatted_now])
 
    #if "keypressed_data" not in current_trial_data[formatted_now][image_id]:
    #   current_trial_data[formatted_now][image_id]["keypressed_data"] = []
        
    #current_trial_data[formatted_now][image_id]["keypressed_data"].append((timestamp, key))



def threading(): 
    # if experiment still going
    # initialize new thread that will run eye tracker during a given question/image answer process
    if not experiment_done: 
        t1=Thread(target=run_eye_tracker) 
        t1.start()
    else: #do nothing if experiment over
        pass
   
def run_eye_tracker():
    '''
    Runs eye tracker during question/image process, terminates when an answer is submitted via <Enter>
    Then stores eye-tracking data, along with image properties, to current_trial_data, which will be dumped into JSON

    '''
    #get list of available eye trackers
    found_eyetrackers = tr.find_all_eyetrackers()
    my_eyetracker = found_eyetrackers[0]

    ### PRINT info about eye tracker if desired
    # print("Address: " + my_eyetracker.address)
    # print("Model: " + my_eyetracker.model)
    # print("Name: " + my_eyetracker.device_name)
    # print("Serial number: " + my_eyetracker.serial_number)

    global collecting_data, done_collecting
    global user_answer, key_sequence

    current_question_eye_data={}
    done_collecting=False

    def gaze_data_callback(gaze_data):
        '''
        Function that adds gaze data from eye tracker to global dictionary

        Parameters: 
            gaze_data (dict): a data point from a single gaze point

        Returns:
            None: but modifies global dictionary to include current gaze data point
                    in the form of {timestamp: (left_eye_coordinate, right_eye_coordinate)}
        '''
        global latest_eye_tracker_timestamp
        nonlocal current_question_eye_data
        current_question_eye_data[gaze_data['system_time_stamp']]=(gaze_data['left_gaze_point_on_display_area'],
                                                        gaze_data['right_gaze_point_on_display_area'])
        
        latest_eye_tracker_timestamp = gaze_data['system_time_stamp']

    # this block continuously subscribes to the gaze data and inputs it into gaze_data_callback function 
    # until is unsubscribed from (determined by Boolean collecting_data which is determined by status of question/image progress)
    # the gaze data is collected at a rate of 60 Hz, according to Tobii spark website

    my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)  
    while collecting_data:  
        pass

    my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)




    #locally create new dict to hold all info about current question/answer
    current_question_answer_info={}

    def image_to_base64(image):
        buffered = BytesIO()
        format = image.format if image.format else 'PNG'  # Default to 'PNG' if format is None
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    current_question_answer_info['gaze_data']=current_question_eye_data
    current_question_answer_info['edited_image']=image_to_base64(resized_image)
    current_question_answer_info['widget_width']=widget_width
    current_question_answer_info['widget_height']=widget_height
    current_question_answer_info['left_x']=left_x
    current_question_answer_info['top_y']=top_y
    current_question_answer_info['screensize']=screensize
    current_question_answer_info['image_number']=image_number
    current_question_answer_info['user_answer']=user_answer
    #print("user answer:", user_answer)
    current_question_answer_info['key_sequence'] = key_sequence
    # current_question_answer_info['user_answer']=user_response.get()
    # print("user response:", user_response.get())
    #current_question_answer_info['question']=doc_question.get()

    current_trial_data[formatted_now][image_id]=current_question_answer_info

    done_collecting = True


    
    
def question_answered():
    '''
    Displays page following question/image. Serves as a transition for user to prepare for next question/image
    
    '''
    document.grid_remove()
    text_response.grid_remove()
    response.grid_remove()
    

    #doc_question.set('Good job, click below for next question/image pair')
    doc_question.set('                         Good job!\n\n\n\nPress ENTER to go to the next question.')

    #next_question.grid(column=0,row=1,columnspan=2) #display button that will take to next image/question pair

    #center the question label
    question.grid(column=0, row=0, columnspan=2)  
    question.place(relx=0.5, rely=0.5, anchor=CENTER)  #to center within grid cell




def stop_data_collection():
    '''
    Sets collecting_data to false to indicate end of image/question process
    '''
    global collecting_data, on_intermediate_page 
    
    if collecting_data:
        collecting_data  = False


    while not done_collecting: #ensure question/answer data is stored before moving on
        pass


    question_answered()
    on_intermediate_page = True

def handle_enter(event):
    '''
    Handles <Enter> key press.
    - First press shows intermediate page  
    - Second press moves to the next question/image pair
    '''
    global on_intermediate_page, first_time
    if first_time:
        nextpage()
        threading()
    elif on_intermediate_page:
        on_intermediate_page = False
        nextpage()
        threading()
    else:
        stop_data_collection()





def quit_app(event):
    '''
    Pressing <esc> allows to quit tkinter app
    '''
    root.destroy()
    
    
if __name__ == "__main__": 
    #root
    root = Tk()
    root.attributes('-fullscreen',True)
    root.title("Document Analysis Experiment")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # 'Enter' and 'esc' key functions
    #root.bind('<Return>',lambda event: [stop_data_collection()])
    root.bind('<Return>', handle_enter)
    root.bind('<Escape>', quit_app)

    root.bind('<KeyPress>', on_key_press)
    
    # size of display screen -- needed for later calculations
    screensize = root.winfo_screenwidth(), root.winfo_screenheight()

    #outermost mainframe
    mainframe=ttk.Frame(root)
    mainframe.grid(column=0,row=0,columnspan=3,rowspan=1,sticky=(N, W, E, S))
    mainframe.rowconfigure(0,weight=1)
    for col in [0,2]:
        mainframe.columnconfigure(col,weight=1)

    #three column frames
    #centerframe will contain all content
    centerframe = ttk.Frame(mainframe)
    centerframe.grid(column=1, row=0, sticky=(N, W, E, S))
    centerframe['relief']='sunken'
    centerframe['padding']=(5,0)

    leftframe = ttk.Frame(mainframe)
    leftframe.grid(column=0, row=0, sticky=(N, W, E, S))
    leftframe['relief']='groove'


    rightframe = ttk.Frame(mainframe)
    rightframe.grid(column=2, row=0, sticky=(N, W, E, S))
    rightframe['relief']='groove'


    #creating new fonts and styles
    # also establishing size "header_len" that is proportional to the display screen
    header_font = tkfont.Font(family="Consolas", size=28, weight="bold")
    header_len = header_font.measure("m")
    instr_head_font = tkfont.Font(family="Consolas", size=22, weight="bold", underline=1, slant="italic")
    button_style = ttk.Style()
    button_style.configure("start_style.TButton", foreground="black", background="green", font=("Times New Roman",25))

    #welcome page widgets
    # Welcome header, instructions, & start button
    welcome=ttk.Label(centerframe, text="Welcome to Eye Tracking Experiment!",font=header_font)
    welcome.grid(column=0,row=0,pady=header_len*2)

    instr_head=ttk.Label(centerframe,text='Instructions',justify='left',font=instr_head_font)
    instr_head.grid(column=0,row=1,pady=header_len)     


    instructions='You will be shown a sequence of documents, along with a question at the top of the page corresponding to each image. \n\nAnalyze the document and type your answer to the question in the field that will appear at the bottom of the screen, and then press Enter.'
    instr=ttk.Label(centerframe,text=instructions,justify="center", font=16, wraplength=650)
    instr.grid(column=0,row=2,pady=(0,10*header_len))

    press_enter_msg = ttk.Label(centerframe, text="Press Enter to Start", font=("Times New Roman", 17, "bold"))
    press_enter_msg.grid(column=0, row=3, pady=(10, 0))  #below the instructions

    #experiment widgets
    #question text, document image, user response entry

    doc_question = StringVar()
    question_font = tkfont.Font(family="Arial", size=18, weight="bold")
    question = ttk.Label(centerframe,textvariable = doc_question, font=question_font, foreground="blue", wraplength=470)

    document = ttk.Label(centerframe)

    text_response = ttk.Label(centerframe, text = "Answer:", font=("Times New Roman", 14))
    user_response = StringVar()
    response = ttk.Entry(centerframe,textvariable = user_response, font=("Arial", 14), width=30)

    #intermediate button between question/image pairs

    next_question = ttk.Button(centerframe,style='start_style.TButton',text='Next Question',command=lambda: [nextpage(),threading()])

    
    root.mainloop()
