import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button
import hdbscan
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
import random
import json
import base64
from PIL import Image, ImageDraw, ImageFont 

from io import BytesIO
import os

from mpl_toolkits.mplot3d import Axes3D

from difflib import SequenceMatcher
import traceback
import datetime
import textwrap 
import matplotlib.cm as cm
import argparse


'''
INITIALIZED HYPERPARAMETERS: edit variables below to your preferences
    More information on recommendations found in their respective 
    function (nearest_ocr() & ocr_cluster()) documentations
'''
fixation_radius = 82 #72 
min_cluster_sample = 15 #experiment with this val #12

# loads in data necessary for OCR designation
# NOTE: this is from SQVA Task 1 (https://rrc.cvc.uab.es/?ch=17&com=downloads) training set
#       the current state of code can only analyze images from this dataset 
#ocr_data = np.load("C:/path/to/your/OCR/imdb/dataset.npy",allow_pickle=True) #load in ocr data
#ocr_data = np.load("C:/Users/marta/tfg/filtered_imdb_train_horizontal.npy",allow_pickle=True) #load in ocr data
filtered_data_npy_path = r"D:\tfg\TFG_FINAL\filtered_imdb_train_landscape.npy" #"C:/Users/marta/tfg/filtered_imdb_train_landscape.npy"
ocr_data = np.load(filtered_data_npy_path,allow_pickle=True) #load in ocr data

errors_file = "error_file.txt"
wrong_answ_file = "wrong_answ_file.txt"

SKIPPED_LOG_FILE = "skipped_heatmaps_log.json"


def gaussian_2d(x, y, x0, y0, A, sigma):
    '''
    Calculates value of (x,y) coordinate following Gaussian_2D distribution from a focal point (x0, y0)
    Used with assumption that gaze_point is not 100% accurate and roughly follows a gaussian distribution outwards
    Solely used for heatmap visualization purposes; not actual data analysis

    Parameters:
        x, y: coordinates of point whose value we desire based on Gaussian distribution
        x0, y0: focal point based on gaze data
        A: value given to focal point
        sigma: standard deviation of distribution

        NOTE: when we use this in the heatpoint() function, we use A=100 and sigma=2 as these 
                were found empirically to depict relatively clean and appealing heatmaps. 
                These values could be played with though

    Returns:
        float: value associated with (x,y) given that A was associated with (x0,y0)
    '''
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))


def heatpoint(heatmap,x,y):
    '''
    Modifies heatmap to increment (x,y) by A with radially decreasing
    incrementations according to gaussian distribution

    Parameters:
        heatmap (2D np array): the heatmap associated with gaze data
        x&y coordinates (ints): pixel coordinates corresponding to image location that was being gazed upon
    
    Returns: 
        None: modifies the 2D heatmap array to include increased value on given pixel coordinate 
    '''
    A = 100
    sigma = 2
    #all pixels within 3 units (horizontally & laterally) are also incremented according to gaussian distribution
    for y_diff in range(-3,4,1):
        for x_diff in range(-3,4,1):
            heatmap[y+y_diff][x+x_diff] = heatmap[y+y_diff][x+x_diff] + gaussian_2d(x+x_diff,y+y_diff,x,y,A,sigma)

def screen_to_image_pixels(x,y,x_min,x_max,y_min,y_max,screen_width,screen_height):
    '''
    Converts screen proportion coordinates to pixel value coordinates 
    (relative to image -- i.e (0,0) is top-left of image), primarily used in heatmap construction

    Parameters:  
        x,y (floats): screen coordinates in units that are proportion of screen width/height
        x_min,x_max,y_min,y_max (ints): pixel coordinates that bound the document being displayed
        screen_width, screen_height (ints): pixel values of screensize

    Returns: 
        adjusted_x, adjusted_y (ints): pixel value coordinates that correspond to inputted proportional coordinates 
                                    ONLY if coordinates lie on the document being displayed. Else, None is returned.

    '''
    try:
        x = screen_width*x
        y = screen_height*y
        if x_min<=x<=x_max and y_min<=y<=y_max: #ensures pixel coordinates are only returned if they correspond to spot on displayed image
            adjusted_x = x - x_min
            adjusted_y = y - y_min
            return (adjusted_x,adjusted_y)
        else:
            return None, None
    except: # usually occurs when a cluster is located to be off the screen; 
            # likely a result of faulty calibration of the eye tracker
        raise(Exception)
    

    
def image_prop_to_image_pixels(x,y,x_min,x_max,y_min,y_max):
    '''
    Converts image proportion coordinates to image pixel coordinates (relative to image)
    Used to find nearest OCRs in ocr_cluster()

    Parameters:
        x,y: coordinates in units of proportion of image width & height
        x_min,x_max: pixel value that bounds left and right side of image
        y_min,y_max: pixel values that bound top and bottom of image

    Returns:
        relative_x, relative_y: coordinates in units of pixel values relative to image
    '''
    img_width = x_max-x_min
    img_height = y_max-y_min
    relative_x , relative_y = x*img_width , y*img_height
    return relative_x, relative_y

#COLOR FUNCTIONS for cluster visualization purposes

def generate_random_color(threshold=0.2):
    '''Generate a random hexadecimal color that is not too close to white or black.'''

    def rgb_to_hex(r, g, b):
        '''Convert RGB to Hexadecimal.'''
        return f'#{r:02x}{g:02x}{b:02x}'

    def calculate_lightness(r, g, b):
        '''Calculate the lightness of a color given RGB values.'''
        r_prime = r / 255.0
        g_prime = g / 255.0
        b_prime = b / 255.0
        c_max = max(r_prime, g_prime, b_prime)
        c_min = min(r_prime, g_prime, b_prime)
        lightness = (c_max + c_min) / 2.0
        return lightness

    while True:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        lightness = calculate_lightness(r, g, b)
            
        # Ensure the lightness is not too close to black (0) or white (1)
        if threshold < lightness < (1 - threshold):
            return rgb_to_hex(r, g, b)


def nearest_ocr(ocr_info,gaze_location,kdtree,rad=fixation_radius):
    '''
    Parameters: 
    ocr_info (list of dicts): this is gathered from imdb data. Each dict is an OCR in the image, containing its location
    avg_gaze_location (tuple): the avg coordinates of the cluster being inquired. Units are proportion of file
    kdtree (class KDTree): scipy class which allows for efficient nearest neighbors searching given coordinates of data
    rad (float): radial distance, in units of screen proportion, that indicates the region of OCRs to be captured for a given gaze point
    NOTE: due to difference in height/width of display, this will not result in a perfect circle around a gaze point

    Returns: 
    (str): nearest OCR tokens as a string
    '''

    #NOTE: rad is important parameter. Determines the radius around fixation point that catches all the OCRs. 
    # The current default value is 72 pixels. This was calculated based on several assumptions.
    # In perceptual span for reading, width of single fixation is about 18 letter spaces under 12 pt font (14-15 to the right, 3-4 to the left)
    # Assuming character width to be about half of character height (rule of thumb used in estimation, but another assumption)
    # This equates to 108 pt width of fixation point, equating to a radius of 72 pixels
    # The fixation is more akin to an off-centered ellipse though, so still not an entirely accurate model
    # RECOMMENDATION FOR FUTURE: create some tool that draws an elliptical region around a fixation point and use this to catch OCRs

    indeces = kdtree.query_ball_point(gaze_location,rad) #scipy module

    nearest_tokens = [ocr_info[index] for index in indeces]

    return([token['word'] for token in nearest_tokens])



def ocr_cluster(eye_data,image_number,screensize,reduced_width,reduced_height,widget_width,widget_height,left_x,top_y,min_cluster_size=min_cluster_sample):
    '''
    Takes in tracking data and image, clusters data by time and space, and returns information regarding cluster graph and ocr designation.

    Parameters:
        eye_data (dict): eye-tracking data for a given image/question
        image information: already outlined screen_to_image_coords
        min_cluster_size (int): minimum size for data to be considered a cluster. See subsequent "NOTE" in function comments for more info

    Returns:
        x, y, z (lists): time, x-location, y-location of the tracking data
        list_colors (list of strings): list containing the colors for each data point in the data
        cluster_segment_info (list of tuples of floats): list containing information regarding each cluster, used for temporally visualization eye tracking on heatmap using arrows
        ocr_designation (dict): dict containing each cluster and corresponding nearest OCRs within given screen proportion radius
    '''
    x = []
    y = []
    z = []

    #prepare data into a 2D numpy array
    #rows = data point
    #columns = features
    for time in eye_data:
        try:
            x.append(float(time))
            y.append((eye_data[time][0][0]+eye_data[time][1][0])/2) #averaging between right and left eye points
            z.append((eye_data[time][0][1]+eye_data[time][1][1])/2)
        except:
            print(f'Subject looked away from screen at {eye_data[time]}') 

    data = np.column_stack((x,y,z))

    #standardize the data then put into clusters and obtain corresponding labels
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)


    #NOTE: min_cluster_size parameter is important. Default value is 5 as this will catch shortest of fixations
    # Eye-tracking data samples at 60Hz, meaning 1 every 16.67 ms
    # Avg reading fixation duration 200-250 ms, often in 125-175 ms range, can be as low as 75 ms
    # 
    # Given these numbers, min_cluster_size of 5 will catch even the smallest of fixations, but avg cluster size should be ~12-15
    # This logic and implementation can be tested and played around with for further adjustment as needed
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size)


    labels = clusterer.fit_predict(standardized_data) #get arbitrary cluster labels

    
    #prepare number of colors needed to mark all clusters
    colors=[generate_random_color() for _ in range(1+max(labels))]

    #initalize list of len(data_points)
    #each element corresponding to color of that point
    list_colors=[]

    #initialize dict to keep track of average locations for each cluster
    # has format {cluster_pt:[avg_time,avg_x,avg_y]}
    cluster_info={pt:[0,0,0] for pt in range(max(labels)+1)}

    for i, pt in enumerate(labels):
        #if a point doesn't belong to cluster, designate as grey
        if pt<0:
            list_colors.append('grey')
        #otherwise, designate as corresponding cluster color
        else:
            list_colors.append(colors[pt])
            cluster_info[pt][0]+=float(data[i][0])
            cluster_info[pt][1]+=float(data[i][1])
            cluster_info[pt][2]+=float(data[i][2])


    #finalize cluster avg locations
    for pt in cluster_info:
        freq=list(labels).count(pt)
        cluster_info[pt][0]=cluster_info[pt][0]/freq
        cluster_info[pt][1]=cluster_info[pt][1]/freq
        cluster_info[pt][2]=cluster_info[pt][2]/freq
 
    cluster_info=sorted(cluster_info.items(), key= lambda item: item[1][0]) #sort clusters by chronological order // changes cluster_info to list format
    
    cluster_segment_info=[] # initialize list that will contain locations for each cluster appearing on the file // used later for visualization purposes

    #pixel values that bound the document image
    x_max = left_x + widget_width - (widget_width-reduced_width)//2
    y_max = top_y + widget_height - (widget_height-reduced_height)//2
    x_min = left_x + (widget_width-reduced_width)//2
    y_min = top_y + (widget_height-reduced_height)//2


    #this next section prepares the OCR data and builds KD tree
    #this data and tree will be used in nearest OCR calculations
   
    ocr_info = ocr_data[image_number]['ocr_info'] #ocr info for this specific image

    ocr_designation = {} # going to have each color cluster with corresponding OCRs

    ocr_prep = [] #preparing location data for each ocr

    for ocr in ocr_info: 
        #as of now we are using center of ocr box location to calculate nearest OCRs
        #this could be played with; could potentially use any edge of box to calculate nearest OCRs
        avg_x = ocr['bounding_box']['topLeftX']+(ocr['bounding_box']['width']/2)
        avg_y = ocr['bounding_box']['topLeftY']+(ocr['bounding_box']['height']/2)
        x_pixel, y_pixel = image_prop_to_image_pixels(avg_x,avg_y,x_min,x_max,y_min,y_max)
        ocr_prep.append((x_pixel,y_pixel))

    ocr_locations = np.array(ocr_prep)

    tree = KDTree(ocr_locations) #using KDTree for more efficient searching


    #this next block calculates nearest OCRs for each cluster

    for cluster in cluster_info:
        #convert display proportion coordinates to image proportion coordinates
        final_x, final_y = screen_to_image_pixels(cluster[1][1],cluster[1][2],x_min,x_max,y_min,y_max,screensize[0],screensize[1])
            
        #if cluster lies on screen, find nearest OCR and add to ocr_designation
        if final_x and final_y:
            near_ocr = nearest_ocr(ocr_info,(final_x,final_y),tree)
            cluster_segment_info.append((cluster[1][0],cluster[1][1],cluster[1][2]))
            ocr_designation[colors[cluster[0]]]=near_ocr
        else:
            ocr_designation[colors[cluster[0]]]=None




    #return data to be displayed on cluster graph
    return (x,y,z,list_colors,cluster_segment_info,ocr_designation)


def anls(image_number, user_answer):
    """
    Function to compare user answer with the real answer and return the similarity between them
    Params:
        - image_number 
        - user_answer (str): answer given by the user

    Returns:
        - max_similarity (float): max similarity between user answer and real answer
        - answer_matched (str): real answer that matched the user answer
        - user_answer
        - real_answers
    """
    real_answers = ocr_data[image_number]["answers"]
    if isinstance(real_answers, str):
        real_answers_for_anls = [real_answers] #put in a list if it is a single str
    else:
        real_answers_for_anls = real_answers
        
    max_similarity = 0.0

    answer_matched = None

    if not real_answers: 
        return 0.0
    
    for correct_answer in real_answers_for_anls: 
        similarity = SequenceMatcher(None, user_answer.lower(), correct_answer.lower()).ratio()
        if similarity > max_similarity: 
            max_similarity = similarity
            answer_matched = correct_answer
            
    if not answer_matched: 
        answer_matched = real_answers_for_anls[0]

    return max_similarity, answer_matched, user_answer, real_answers_for_anls


def analyze_data(gaze_data,edited_image,widget_width,widget_height,left_x,top_y,screensize,image_number, user_answer, keystrokes=[], return_fig=None):
    '''
    Primary function that executes entire analysis of gaze data. Creates and visualizes heatmap corresponding to gaze data 
    overylaying the file image, creates and visualizes clusters which represent fixation points, and returns nearest OCRs within
    a given screen proportion radius of each fixation points. 

    Parameters:
        gaze_data (dict): eye tracking data
        edited_image (Pillow Image): image being gazed upon
        widget_width (int): width of widget in pixels
        widget_height (int): height of widget in pixels
        left_x (int): # of pixels to the left of widget containing the image
        top_y (int): # of pixels above the widget containing the image
        screensize (tuple): size of display being used for experiment
        image_number (int): used to locate image information in dataset
        user_answer (str): the answer given by the user
        keystrokes (list): list containing time when keystrokes ocurred
        return_fig: to save the plots 

    Visualizes:
        heatmap of eye-tracking gaze data, with arrows at each cluster which indicated the chronological order upon which they were gazed at
        scatterplot showing the actual data of the clusters
        
    Returns:
        ocr_designation (dict): dictionary containing the color of a cluster (HEX format) as keys, and all of its nearby
                                OCRs as its values
    '''

    
    #(height,width) of image and screensize
    height=edited_image.size[1]
    width=edited_image.size[0]
    screen_width, screen_height = screensize[0],screensize[1]

    #initialize array representing image and overlaying heatmap
    image_array = np.array(edited_image)
    heatma=np.zeros((height,width))

    #calculate pixel values that bound the document image
    x_max = left_x + widget_width - (widget_width-width)//2
    y_max = top_y + widget_height - (widget_height-height)//2
    x_min = left_x + (widget_width-width)//2
    y_min = top_y + (widget_height-height)//2

    #add each data point to the heatmap
    for stamp in gaze_data:
        for eye in [0,1]: #to account for both left and right eye data
            try:
                #convert screen proportion to image pixel values and add to heatmap
                adjusted_x,adjusted_y = screen_to_image_pixels(gaze_data[stamp][eye][0],gaze_data[stamp][eye][1],x_min,x_max,y_min,y_max,screen_width,screen_height)
                heatpoint(heatma,round(adjusted_x),round(adjusted_y))
            except:
                #if eye-tracking value == ('nan','nan')
                #this just means user was not looking at screen
                pass
 

    #initialize plots
    fig,ax = plt.subplots(1,2)

    #create first plot; heatmap
    ax[0].imshow(image_array,cmap='gray')
    ax[0].imshow(heatma,cmap='jet',alpha=.5,aspect='auto',vmin=0,vmax=np.max(heatma))
    ax[0].set_title("Heatmap")


    #retrieve data for cluster plot, arrow segments, and ocr designations
    (x,y,z,list_colors,clusters,ocr_designation) = ocr_cluster(gaze_data,image_number,screensize,width,height,widget_width,widget_height,left_x,top_y)

    #this block creates the visual arrow segments that indicate the temporal progress of eye gazing
    segments=[]
    for index, avg_cluster in enumerate(clusters):
        if index==len(clusters)-1:
            continue
        start_x,start_y = screen_to_image_pixels(clusters[index][1],clusters[index][2],x_min,x_max,y_min,y_max,screen_width,screen_height)
        end_x,end_y = screen_to_image_pixels(clusters[index+1][1],clusters[index+1][2],x_min,x_max,y_min,y_max,screen_width,screen_height)
        segments.append([[start_x,start_y],[end_x,end_y]])

    # Add the arrows
    arrows = []
    for seg in segments:
        arrow = FancyArrowPatch((seg[0][0], seg[0][1]),(seg[1][0],seg[1][1]),
                                arrowstyle='->', mutation_scale=15, color='red')
        ax[0].add_patch(arrow)
        arrows.append(arrow)

    # Function to toggle arrow visibility
    def toggle_arrows(event):
        for arrow in arrows:
            arrow.set_visible(not arrow.get_visible())
        plt.draw()

    # Add a button to toggle arrow visibility
    ax_toggle = plt.axes([0.8, 0.01, 0.1, 0.05])
    btn_toggle = Button(ax_toggle, 'Toggle Arrows')
    btn_toggle.on_clicked(toggle_arrows)

    #create second plot; clustered scatterplot
    ax[1]  = fig.add_subplot(122, projection='3d')
    ax[1].scatter(x, y, z, c=list_colors)
    
    #add keystrokes to scatterplot
    if keystrokes:
        keystrokes = np.array(keystrokes) #/1000000000 #convert to seconds      
    
        for key_time in keystrokes:
            ylims = ax[1].get_ylim()
            zlims = ax[1].get_zlim()
            #import pdb; pdb.set_trace()

            ax[1].plot(xs=(key_time, key_time), ys=(ylims[0], ylims[1]), zs=(zlims[0], zlims[0]), color='r', linestyle='solid', linewidth=1.5, label="Keystroke Event")

    # Set labels for scatterplot
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('x-location')
    ax[1].set_zlabel('y-location (inverted)')

    ### if you want to be able to see OCR designations alongside data visualization
    for color,ocrs in ocr_designation.items(): 
        print(f"{color} cluster was near the following OCRs: {ocrs}")

    if return_fig:
        #return the plots to later save them
        return fig, ocr_designation
    else:
        plt.show()

        max_similarity, answer_matched, user_answer, real_answers = anls(image_number, user_answer)
    
    print(f"\nMax similarity: {max_similarity}")
    print(f"Answer matched: {answer_matched}")  
    print(f"User answer: {user_answer}\n")
    if not answer_matched: 
        print(f"Correct answer: {real_answers}")

    return ocr_designation

    

def base64_to_image(base64_str): 
    '''
    Converts base64 string into Pillow image object
    '''
    # Decode the base64 string to bytes
    img_data = base64.b64decode(base64_str)
    
    # Load the bytes into a BytesIO object
    buffered = BytesIO(img_data)
    
    # Open the image using Pillow
    return Image.open(buffered)

def analyze_single_image(image_id):
    '''
    Performs data analysis on a single image for all experiments that have been conducted
    NOTE: this could result in a long execution time if this image has been collected data from many times
            the code will continuously be interrupted by matplotlib, the script only continues after each plot is closed by the user

    Parameters:
        image_id (str): the string ID that identifies the desired image to be analyzed
    
    Visualizes:
        For each image analyzed, heatmap of eye-tracking data and corresponding cluster scatterplot for each fixation point
        in a one-by-one fashion via Matplotlib

    Returns:
        ocr_designations (dicts): all of the ocr_designations for each image analyzed. See analyze_data() for more info
        NOTE: uncomment out the print statement directly before plt.show() in analyze_data() function 
        for ocr_designation alongside heatmap/cluster visualization
    '''
    #path_data = "C:/Users/ljdde/Downloads/CVC/test1/experiment_data.json"
    #path_data = "C:/Users/marta/tfg/Eye-Tracking/experiment_data.json"
    with open(PARTICIPANTS_IDs_FILE, "r") as f:
        user_id_n = f.readlines()[-1]
        user_id = user_id_n.strip()
        print("user_id:", user_id)
        
    user_file = f"{user_id}_experiment_data.json"

    path_data = f"{PARTICIPANTS_EYE_DATA_DIR}/{user_file}"
    with open(path_data, 'r') as json_file:
        experiment_data = json.load(json_file)

    #this first block is inefficient; for the purposes of giving user an idea of how many images will be analyzed
    #if analysis is only being done by computer the code can easily be simplified into a single FOR loop
    num_analyses=0
    applicable_trials=[]
    for trial,images in experiment_data.items():
        if image_id in images:
            num_analyses+=1
            applicable_trials.append(trial)

    print(f'Number of analyses: {num_analyses}') #to help user know how long execution will take
    for trial in applicable_trials:
            desired_img=experiment_data[trial][image_id]
            # print("-------------------")
            # print("hola aqui entra també")
            # keypressed_data = desired_img['keypressed_data']
            # list_times = []
            # for key_data in keypressed_data:
            #     time = key_data['eye_tracker_timestamp']
            #     list_times.append(time)
            # print(list_times)
            # print("-------------------")
            # #list_times = None
            # analyze_data(desired_img['gaze_data'],base64_to_image(desired_img['edited_image']),desired_img['widget_width'],desired_img['widget_height'],desired_img['left_x'],desired_img['top_y'],desired_img['screensize'],desired_img['image_number'], list_times)

def log_to_file(message):
  """Appends a timestamped message to the specified log file."""
  with open(errors_file, "a") as f:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"[{timestamp}] {message}\n")

def log_wrong_answers(message_wrong): 
    """To log wrong answers in a separated .txt file"""
    with open(wrong_answ_file, "a") as f: 
        f.write(f"{message_wrong}\n")

def log_skipped_heatmap(user_id, image_name, reason):
    """
    Reads the existing log, appends a new entry for a skipped heatmap,
    and writes it back to the JSON file.
    """
    log_entry = {
        "user_id": user_id,
        "image_name": image_name,
        "reason": reason,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    try:
        #try open and read the existing file
        with open(SKIPPED_LOG_FILE, 'r+') as f:
            try:
                #load existing data
                data = json.load(f)
            except json.JSONDecodeError:
                #file empty/corrupt --> start new list
                data = []
            
            data.append(log_entry)
            f.seek(0) 
            f.truncate() 
            json.dump(data, f, indent=4)
            
    except FileNotFoundError:
        #create file with first entry
        with open(SKIPPED_LOG_FILE, 'w') as f:
            json.dump([log_entry], f, indent=4)


def analyze_single_trial(date_time, experiment_data, user_id, save_plots=False, analysis_mode="original_heatmaps"):
    trial_data = experiment_data.get(date_time, {})
    print(f'\nAnalyzing Trial: {date_time} for user {user_id} (Mode: {analysis_mode})')

    for img_name, desired_img in trial_data.items():
        try:
            if not desired_img.get('gaze_data'):
                log_to_file(f"{user_id}: No gaze data for {img_name}, skipping.")
                continue

            #print(f"--- Processing image: {img_name} ---")
            
            max_similarity, answer_matched, user_answer, _ = anls(desired_img['image_number'], desired_img["user_answer"])
            if max_similarity > 0.9: 
                category = "correct_answers_users"
            elif max_similarity > 0.5:
                category = "semi_correct_answers_users"
            else:
                category = "wrong_answers_users"
                mess = f"Img - {img_name}: {user_id} answer - {user_answer}, real answer - {answer_matched}, similarity - {max_similarity} "
                log_wrong_answers(mess)


            filename_base = os.path.splitext(img_name)[0]

            if analysis_mode == "bbox_heatmaps":
                keystroke_times = [k['eye_tracker_timestamp'] for k in desired_img.get('keypressed_data', []) if k.get('eye_tracker_timestamp')]
                first_keystroke_time = min(keystroke_times) if keystroke_times else None
                
                (_, _, _, _, _, pre_typing_events, typing_events, last_fixation_events) = calculate_human_attention(
                    desired_img['gaze_data'], first_keystroke_time, desired_img['image_number'],
                    desired_img['screensize'], base64_to_image(desired_img['edited_image']).size[0],
                    base64_to_image(desired_img['edited_image']).size[1], desired_img['widget_width'],
                    desired_img['widget_height'], desired_img['left_x'], desired_img['top_y']
                )
                
                raw_attention_data_for_plot = {
                    "pre_typing_attention": pre_typing_events,
                    "last_fixation_attention": last_fixation_events,
                    "typing_attention": typing_events
                }
                main_dir = "./2_bounding_boxes_heatmaps"
                os.makedirs(main_dir, exist_ok=True)
                plot_output_dir = f"{main_dir}/plots/{category}/{user_id}"
                os.makedirs(plot_output_dir, exist_ok=True)
                plot_path = os.path.join(plot_output_dir, f"{filename_base}_human_heatmap.png")

                #to create the plots
                create_token_heatmap_visualization(
                    pil_image=base64_to_image(desired_img['edited_image']),
                    attention_data=raw_attention_data_for_plot,
                    image_number=desired_img['image_number'],
                    question_text=ocr_data[desired_img['image_number']]['question'],
                    pred_answer=user_answer,
                    correct_answer=answer_matched,
                    similarity=max_similarity,
                    output_path=plot_path,
                    user_id=user_id,
                    img_name=img_name
                )
                
                ocr_info_for_image = ocr_data[desired_img['image_number']]['ocr_info']
                consolidated_attention_for_json = {
                    "image_number": desired_img['image_number'],
                    "user_answer": user_answer,
                    "correct_answer": answer_matched,
                    "last_fixation_attention": _format_attention_map_for_json(last_fixation_events, ocr_info_for_image),
                    "pre_typing_attention": _format_attention_map_for_json(pre_typing_events, ocr_info_for_image),
                    "typing_attention": _format_attention_map_for_json(typing_events, ocr_info_for_image)
                }
                
                attention_output_dir = f"{main_dir}/json/{user_id}/{category}"
                os.makedirs(attention_output_dir, exist_ok=True)
                attention_path = os.path.join(attention_output_dir, f"{filename_base}_human_attention.json")
                with open(attention_path, 'w') as f:
                    json.dump(consolidated_attention_for_json, f, indent=4)
                #print(f"Saved consolidated attention map: {attention_path}")

            elif analysis_mode == "original_heatmaps":
                old_filename_base = f"{img_name}"
                keystroke_times = [k['eye_tracker_timestamp'] for k in desired_img.get('keypressed_data', []) if k.get('eye_tracker_timestamp')]
                if save_plots: 
                    fig, _ = analyze_data(
                        desired_img['gaze_data'], base64_to_image(desired_img['edited_image']),
                        desired_img['widget_width'], desired_img['widget_height'],
                        desired_img['left_x'], desired_img['top_y'], desired_img['screensize'],
                        desired_img['image_number'], user_answer, keystrokes=keystroke_times, return_fig=True
                    )

                    output_dir = f"./1_heatmaps_plots/{category}/{user_id}"
                    os.makedirs(output_dir, exist_ok=True)
                
                    plot_path = os.path.join(output_dir, f"{old_filename_base}_heatmap.png")
                    fig.savefig(plot_path)
                    plt.close(fig)
                    #print(f"Saved original_heatmaps-style heatmap plot: {plot_path}")
                else: 
                    analyze_data(
                        desired_img['gaze_data'], base64_to_image(desired_img['edited_image']),
                        desired_img['widget_width'], desired_img['widget_height'],
                        desired_img['left_x'], desired_img['top_y'], desired_img['screensize'],
                        desired_img['image_number'], user_answer, keystrokes=keystroke_times, return_fig=False
                    )
                

        except Exception as e:
            log_message = f"FATAL Error analyzing {img_name} for user {user_id}: {e}"
            print(log_message)
            traceback.print_exc()
            log_to_file(log_message)

def generate_question_log():
    """
    Analyzes all experiment data to create a log of unique questions that appeared.
    The number next to each question corresponds to its appearing order in the source ocr_data. 
    Output:
        'image_questions_log.txt'.
    """
    print("\n" + "="*50)
    print("Generating Question Log")

    
    #info about each image
    #{ 'img_name': {'image_numbers': {set_of_indices}, 'questions_asked': {set_of_questions}}, ... }
    image_info = {}

    all_trials = []
    try:
        with open(PARTICIPANTS_IDs_FILE, "r") as f:
            user_ids = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"FATAL ERROR: The participant IDs file was not found at '{PARTICIPANTS_IDs_FILE}'")
        return

    for user_id in user_ids:
        user_file = f"{user_id}_experiment_data.json"
        path_data = os.path.join(PARTICIPANTS_EYE_DATA_DIR, user_file)
        try:
            with open(path_data, 'r') as json_file:
                experiment_data = json.load(json_file)
                for trial_content in experiment_data.values():
                    all_trials.append(trial_content)
        except Exception:
            continue #skip files that can't be opened

    #process collected trial data 
    for trial_content in all_trials:
        for img_name, img_data in trial_content.items():
            if "image_number" not in img_data:
                continue

            image_number = img_data["image_number"]
            question = str(ocr_data[image_number]["question"])

            if img_name not in image_info:
                image_info[img_name] = {
                    "image_numbers": set(),
                    "questions_asked": set()
                }
            
            image_info[img_name]["image_numbers"].add(image_number)
            image_info[img_name]["questions_asked"].add(question)

    #Write results to log file
    output_filename = "image_questions_log.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("Log of Questions That Appeared in Experiments\n")
        f.write("="*46 + "\n\n")

        sorted_image_names = sorted(image_info.keys())

        for img_name in sorted_image_names:
            f.write(f"Image: {img_name}\n")
            data = image_info[img_name]
            
            #order: get indices each image, sort them and create a list of questions in that order.
            sorted_indices = sorted(list(data["image_numbers"]))
            master_question_list = [str(ocr_data[i]["question"]) for i in sorted_indices]

            questions_to_log = {}
            for q_asked in data["questions_asked"]:
                try:
                    order = master_question_list.index(q_asked) + 1
                    questions_to_log[q_asked] = order
                except ValueError:
                    #prevent errors
                    questions_to_log[q_asked] = "N/A"

            #sort the questions 
            items_to_sort = sorted(questions_to_log.items(), key=lambda x: x[1] if isinstance(x[1], int) else float('inf'))

            if not items_to_sort:
                f.write("  No questions recorded for this image.\n")
            else:
                for q_text, order in items_to_sort:
                    f.write(f"  {order}. {q_text}\n")
            f.write("-" * 30 + "\n")

    print(f"\nSuccessfully created question log: '{output_filename}'")
    print("="*50)


def _get_fixations_from_gaze_data(gaze_data_dict):
    """
    Extracts a list of fixation events from dictionary of gaze points.
    """
    if not gaze_data_dict:
        return []

    gaze_points = []
    for time, coords in gaze_data_dict.items():
        try:
            avg_x = (coords[0][0] + coords[1][0]) / 2
            avg_y = (coords[0][1] + coords[1][1]) / 2
            gaze_points.append([float(time), avg_x, avg_y])
        except (TypeError, IndexError):
            continue
            
    if len(gaze_points) < min_cluster_sample:
        return []

    data = np.array(gaze_points)
    spatial_data = data[:, 1:3]
    
    fixation_info = []
    try:
        
        standardized_spatial_data = StandardScaler().fit_transform(spatial_data)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_sample, gen_min_span_tree=True)
        labels = clusterer.fit_predict(standardized_spatial_data)

        fixations = {}
        for i, label in enumerate(labels):
            if label != -1:
                if label not in fixations:
                    fixations[label] = {"points": [], "timestamps": []}
                fixations[label]["points"].append(data[i, 1:3])
                fixations[label]["timestamps"].append(data[i, 0])
        
        for label, values in fixations.items():
            if not values["timestamps"]: continue
            duration = max(values["timestamps"]) - min(values['timestamps'])
            center = np.mean(values["points"], axis=0)
            start_time = min(values["timestamps"])
            fixation_info.append({"center": center, "duration": duration, "start_time": start_time})

    except ValueError:
        #all points as a single fixation
        print("Info: All gaze points in this phase were identical. Treating as a single fixation.")
        duration = max(data[:, 0]) - min(data[:, 0])
        center = np.mean(data[:, 1:3], axis=0)
        start_time = min(data[:, 0])
        fixation_info.append({"center": center, "duration": duration, "start_time": start_time})
        
    return fixation_info

def _process_gaze_phase(phase_eye_data, image_number, screensize, reduced_width, reduced_height, widget_width, widget_height, left_x, top_y):
    """
    Return: detailed list of attention events including:
        - token index
        - score of 1
        - fixation timestamp.
    """
    fixation_info = _get_fixations_from_gaze_data(phase_eye_data)
    if not fixation_info:
        return []

    ocr_info = ocr_data[image_number]["ocr_info"]
    x_max = left_x + widget_width - (widget_width - reduced_width) // 2
    y_max = top_y + widget_height - (widget_height - reduced_height) // 2
    x_min = left_x + (widget_width - reduced_width) // 2
    y_min = top_y + (widget_height - reduced_height) // 2
    
    ocr_pixel_locations = []
    for ocr in ocr_info:
        box = ocr["bounding_box"]
        avg_x_prop = box["topLeftX"] + (box["width"] / 2)
        avg_y_prop = box["topLeftY"] + (box["height"] / 2)
        x_pixel, y_pixel = image_prop_to_image_pixels(avg_x_prop, avg_y_prop, x_min, x_max, y_min, y_max)
        ocr_pixel_locations.append((x_pixel, y_pixel))

    ocr_tree = KDTree(ocr_pixel_locations)
    
    #list of individual attention events
    phase_attention_events = []
    for fixation in fixation_info:
        fixation_center_pixels = screen_to_image_pixels(fixation["center"][0], fixation["center"][1], x_min, x_max, y_min, y_max, screensize[0], screensize[1])
        if fixation_center_pixels[0] is None: continue
        
        nearby_indices = ocr_tree.query_ball_point(fixation_center_pixels, r=fixation_radius)
        for token_idx in nearby_indices:
            #create a record of the event
            phase_attention_events.append({
                "token_idx": token_idx,
                "score": 1, 
                "timestamp": fixation["start_time"]
            })
            
    return phase_attention_events

def calculate_human_attention(eye_data, first_keystroke_time, image_number, screensize, reduced_width, reduced_height, widget_width, widget_height, left_x, top_y):
    """
    isolate SINGLE closest token for the last fixation.
    """
    #split gaze data into Phases 
    pre_typing_eye_data, typing_eye_data = {}, {}
    if first_keystroke_time is None:
        pre_typing_eye_data = eye_data
    else:
        for time_str, coords in eye_data.items():
            if float(time_str) < first_keystroke_time:
                pre_typing_eye_data[time_str] = coords
            else:
                typing_eye_data[time_str] = coords
    
    #process peneral phases
    common_args = (image_number, screensize, reduced_width, reduced_height, widget_width, widget_height, left_x, top_y)
    pre_typing_attention_events = _process_gaze_phase(pre_typing_eye_data, *common_args)
    typing_attention_events = _process_gaze_phase(typing_eye_data, *common_args)


    #get all fixations occurred before typing
    fixations_before_typing = _get_fixations_from_gaze_data(pre_typing_eye_data)
    last_fixation_attention_events = []
    
    if fixations_before_typing:
        #fixation with latest start time
        last_fixation = max(fixations_before_typing, key=lambda f: f["start_time"])
        
        #KD-Tree of OCR locations 
        ocr_info = ocr_data[image_number]["ocr_info"]
        x_max = left_x + widget_width - (widget_width - reduced_width) // 2
        y_max = top_y + widget_height - (widget_height - reduced_height) // 2
        x_min = left_x + (widget_width - reduced_width) // 2
        y_min = top_y + (widget_height - reduced_height) // 2
        ocr_pixel_locations = []
        for ocr in ocr_info:
            box = ocr["bounding_box"]
            avg_x_prop = box["topLeftX"] + (box["width"] / 2)
            avg_y_prop = box["topLeftY"] + (box["height"] / 2)
            x_pixel, y_pixel = image_prop_to_image_pixels(avg_x_prop, avg_y_prop, x_min, x_max, y_min, y_max)
            ocr_pixel_locations.append((x_pixel, y_pixel))
        ocr_tree = KDTree(ocr_pixel_locations)

        #find the nearest token to last fixation point
        fixation_center_pixels = screen_to_image_pixels(last_fixation["center"][0], last_fixation["center"][1], x_min, x_max, y_min, y_max, screensize[0], screensize[1])
        if fixation_center_pixels[0] is not None:
            distance, closest_idx = ocr_tree.query(fixation_center_pixels)
            
            #only include it if within desired radius
            if distance <= fixation_radius:
                last_fixation_attention_events.append({
                    "token_idx": closest_idx,
                    "score": 1,
                    "timestamp": last_fixation["start_time"]
                })

    (x, y, z, list_colors, clusters, _) = ocr_cluster(eye_data, image_number, screensize, reduced_width, reduced_height, widget_width, widget_height, left_x, top_y)
    
    return (x, y, z, list_colors, clusters, pre_typing_attention_events, typing_attention_events, last_fixation_attention_events)



def _format_attention_map_for_json(attention_events_list, ocr_info_for_image):
    """
    Takes a list of attention events, aggregates the scores for each token,
    and formats the result into a list of dictionaries for JSON serialization.
    """
    if not attention_events_list:
        return []

    #aggregate scores from list of events into dictionary
    aggregated_scores = {}
    for event in attention_events_list:
        token_idx = event["token_idx"]
        score = event["score"]
        aggregated_scores[token_idx] = aggregated_scores.get(token_idx, 0.0) + score

    #normalize scores
    if not aggregated_scores:
        return []
    max_score = max(aggregated_scores.values(), default=1.0)
    
    #final output list
    output_list = []
    for token_idx, agg_score in aggregated_scores.items():
        try:
            token_info = ocr_info_for_image[token_idx]
            output_list.append({
                "word": token_info.get("word", ""),
                "bounding_box": token_info.get("bounding_box", {}),
                "attention_score": agg_score / max_score
            })
        except IndexError:
            log_to_file(f"Warning: token_idx {token_idx} out of range for current image's ocr_info.")
            continue
            
    output_list.sort(key=lambda x: x["attention_score"], reverse=True)
    return output_list


def create_token_heatmap_visualization(pil_image, attention_data, image_number, question_text, pred_answer, correct_answer, similarity, output_path, user_id, img_name):
    """
    multi-layered heatmap 
    """
    #get attention data 
    pre_typing_events = attention_data.get("pre_typing_attention", [])
    last_fixation_events = attention_data.get("last_fixation_attention", [])
    typing_events = attention_data.get("typing_attention", [])

    if not any([pre_typing_events, last_fixation_events, typing_events]):
        reason = "No valid fixations were detected in any phase (pre-typing, typing, or last)."
        print(f"No attention data to visualize for {img_name}. Skipping plot generation.")
        log_skipped_heatmap(user_id, img_name, reason)
        return

    #font-header/footer Setup 
    image_width, image_height = pil_image.size
    header_font_size = max(18, min(45, int(image_width / 30)))
    footer_font_size = max(15, int(header_font_size * 0.85))
    try:
        header_font = ImageFont.truetype("arial.ttf", size=header_font_size)
        footer_font = ImageFont.truetype("arial.ttf", size=footer_font_size)
    except IOError:
        header_font = ImageFont.load_default()
        footer_font = ImageFont.load_default()

    #header
    char_width_header = header_font.getbbox("a")[2] if hasattr(header_font, "getbbox") else 10
    header_wrap_width = int((image_width - 30) / char_width_header) 
    header_text = textwrap.fill(f"Question: {question_text}", width=header_wrap_width)
    temp_draw = ImageDraw.Draw(Image.new('RGB', (0,0)))
    header_text_bbox = temp_draw.multiline_textbbox((0, 0), header_text, font=header_font, spacing=5)
    header_height = (header_text_bbox[3] - header_text_bbox[1]) + 40
    

    #footer
    char_width_footer = footer_font.getbbox("a")[2] if hasattr(footer_font, "getbbox") else 10
    footer_wrap_width = int((image_width - 1) / char_width_footer) 
    pred_answer_line = textwrap.fill(f"Predicted Answer: {pred_answer}", width=footer_wrap_width)
    correct_answer_line = textwrap.fill(f"Correct Answer: {correct_answer}", width=footer_wrap_width)
    similarity_line = f"Similarity: {similarity:.3f}"
    
    spacing = 5
    line_height = footer_font.getbbox("A")[3] - footer_font.getbbox("A")[1]
    
    #measure height needed for answers and similarity
    answer_text_block = f"{pred_answer_line}\n{correct_answer_line}\n{similarity_line}"
    answer_text_bbox = temp_draw.multiline_textbbox((0, 0), answer_text_block, font=footer_font, spacing=spacing)
    answer_height = answer_text_bbox[3] - answer_text_bbox[1]
    
    # height needed for legend 
    legend_height = (line_height + spacing) * 4 
    footer_height = answer_height + legend_height + 40 

    #final canvas + header
    final_canvas = Image.new("RGB", (image_width, image_height + header_height + footer_height), "white")
    canvas_draw = ImageDraw.Draw(final_canvas)
    canvas_draw.multiline_text((15, 20), header_text, font=header_font, fill="black", spacing=spacing)
    
    #+ footer 
    footer_y_start = header_height + image_height + 20
    canvas_draw.multiline_text((15, footer_y_start), answer_text_block, font=footer_font, fill="black", spacing=spacing)
    
    current_y = footer_y_start + answer_height + (spacing * 2)
    canvas_draw.text((15, current_y), "--- Heatmap Legend ---", font=footer_font, fill="black")
    current_y += line_height + spacing

    rect_y = current_y + (line_height / 4)
    canvas_draw.rectangle([20, rect_y, 20 + 25, rect_y + (line_height / 2)], outline="red", width=2)
    canvas_draw.text((20 + 35, current_y), "Final glance before typing", font=footer_font, fill="black")
    current_y += line_height + spacing

    rect_y = current_y + (line_height / 4)
    canvas_draw.rectangle([20, rect_y, 20 + 25, rect_y + (line_height / 2)], fill="#a8c9f0") 
    canvas_draw.text((20 + 35, current_y), "Reading before typing", font=footer_font, fill="black")
    current_y += line_height + spacing

    rect_y = current_y + (line_height / 4)
    canvas_draw.rectangle([20, rect_y, 20 + 25, rect_y + (line_height / 2)], fill="#a3d9a5") 
    canvas_draw.text((20 + 35, current_y), "Looking while typing", font=footer_font, fill="black")


    #heatmap layers 
    pre_typing_overlay = Image.new("RGBA", pil_image.size, (255, 255, 255, 0))
    typing_overlay = Image.new("RGBA", pil_image.size, (255, 255, 255, 0))
    last_fixation_overlay = Image.new("RGBA", pil_image.size, (255, 255, 255, 0))

    #pre-typing --> BLUE
    if pre_typing_events:
        pre_typing_draw = ImageDraw.Draw(pre_typing_overlay)
        colormap_pre = cm.get_cmap("Blues")
        timestamps = [event["timestamp"] for event in pre_typing_events]
        min_time, max_time = min(timestamps), max(timestamps)
        for event in pre_typing_events:
            token_info = ocr_data[image_number]["ocr_info"][event["token_idx"]]
            box_data = token_info["bounding_box"]
            pixel_box = [int(box_data["topLeftX"] * image_width), int(box_data["topLeftY"] * image_height),
                         int((box_data["topLeftX"] + box_data["width"]) * image_width), int((box_data["topLeftY"] + box_data["height"]) * image_height)]
            time_range = max_time - min_time
            time_intensity = (event["timestamp"] - min_time) / time_range if time_range > 0 else 1.0
            color_float = colormap_pre(time_intensity)
            fill_color = (int(color_float[0]*255), int(color_float[1]*255), int(color_float[2]*255), 100)
            pre_typing_draw.rectangle(pixel_box, fill=fill_color)

    #typing --> GREEN
    if typing_events:
        typing_draw = ImageDraw.Draw(typing_overlay)
        colormap_typing = cm.get_cmap("Greens")
        typing_scores = {}
        for event in typing_events:
            typing_scores[event["token_idx"]] = typing_scores.get(event["token_idx"], 0) + 1
        if typing_scores:
            max_score = max(typing_scores.values())
            for token_idx, score in typing_scores.items():
                normalized_score = score / max_score if max_score > 0 else 0
                token_info = ocr_data[image_number]["ocr_info"][token_idx]
                box_data = token_info["bounding_box"]
                pixel_box = [int(box_data["topLeftX"] * image_width), int(box_data["topLeftY"] * image_height),
                             int((box_data["topLeftX"] + box_data["width"]) * image_width), int((box_data["topLeftY"] + box_data["height"]) * image_height)]
                color_float = colormap_typing(normalized_score)
                fill_color_rgba = (int(color_float[0]*255), int(color_float[1]*255), int(color_float[2]*255), 130)
                typing_draw.rectangle(pixel_box, fill=fill_color_rgba)
            
    #last fixation --> RED BOX
    if last_fixation_events:
        last_fixation_draw = ImageDraw.Draw(last_fixation_overlay)
        for event in last_fixation_events:
            token_info = ocr_data[image_number]["ocr_info"][event["token_idx"]]
            box_data = token_info["bounding_box"]
            pixel_box = [int(box_data["topLeftX"] * image_width), int(box_data["topLeftY"] * image_height),
                         int((box_data["topLeftX"] + box_data["width"]) * image_width), int((box_data["topLeftY"] + box_data["height"]) * image_height)]
            last_fixation_draw.rectangle(pixel_box, outline=(255, 0, 0, 255), width=3)

    #composite layers 
    final_image = pil_image.copy().convert("RGBA")
    final_image = Image.alpha_composite(final_image, pre_typing_overlay)
    final_image = Image.alpha_composite(final_image, typing_overlay)
    final_image = Image.alpha_composite(final_image, last_fixation_overlay)
    
    final_canvas.paste(final_image, (0, header_height))
    final_canvas.save(output_path)
    #print(f"Saved new bbox_heatmap: {output_path}")


if __name__ == "__main__": 
    pass
    ### ENTER YOUR CODE BELOW TO ANALYZE DATA
    
    ### EXAMPLE: run the following line to analyze data from a single trial run (data initialized in JSON)
    ### analyze_single_trial("2024-07-30 14:26:06")
    #analyze_single_trial("2025-03-07 11:22:10")

    # TO RUN THE SCRIPT: 
    #MODE
    #single: python main_data_analysis.py --mode single
    #full: python main_data_analysis.py --mode full
    #questions: python main_data_analysis.py --mode full

    #ANALYSIS-MODE
    #default (regular heatmaps)
    #bounding boxes heatmaps: python main_data_analysis.py --mode full --analysis-mode bbox_heatmaps

    PARTICIPANTS_IDs_FILE = "participant_IDs.txt"
    #directory where all participants' eye data is stored 
    PARTICIPANTS_EYE_DATA_DIR = r"D:\tfg\TFG_FINAL\Eye-Tracking\participant_eye_data" #"C:/Users/marta/tfg/Eye-Tracking/participant_eye_data/" 

    
    parser = argparse.ArgumentParser(description="Run data analysis for the DocVQA Eye-Tracking experiment.")
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=["single", "full", "questions"],
        help="The analysis mode to run. 'single': analyze the last trial of the last user. 'full': analyze all data. 'questions': generate the question log file."
    )

    parser.add_argument(
        '--save-plots',
        action='store_true', 
        help="If included, the analysis plots in 'original_heatmaps' mode will be saved to disk."
    )
    #args = parser.parse_args()

    parser.add_argument(
        '--analysis-mode',
        type=str,
        default='original_heatmaps',
        choices=['original_heatmaps', 'bbox_heatmaps'],
        help="Specifies the clustering analysis mode. 'bbox_heatmaps': heatmaps using bounding boxes (saves plots by default). Defaults to 'original_heatmaps'."
    )
    args = parser.parse_args()

    if args.mode == "single": 
        print("--- Running in SINGLE ANALYSIS mode ---")
        with open(PARTICIPANTS_IDs_FILE, "r") as f:
            user_id_n = f.readlines()[-1]
            user_id = user_id_n.strip()
            print("user_id:", user_id)
            
        user_file = f"{user_id}_experiment_data.json"

        path_data = f"{PARTICIPANTS_EYE_DATA_DIR}/{user_file}"
        with open(path_data, 'r') as json_file:
            experiment_data = json.load(json_file)

        date_time = list(experiment_data.keys())[-1]
        print(date_time)

        analyze_single_trial(date_time, experiment_data, user_id, save_plots=args.save_plots, analysis_mode=args.analysis_mode)

    elif args.mode == "full":
        print(f"--- Running in FULL ANALYSIS mode (mode: {args.analysis_mode}) ---")
        with open(PARTICIPANTS_IDs_FILE, "r") as f:
            user_ids = []
            for line in f.readlines():
                user_id = line.strip()
                user_ids.append(user_id)
           
        
        for user_id in user_ids: 
            user_file = f"{user_id}_experiment_data.json"
            path_data = f"{PARTICIPANTS_EYE_DATA_DIR}/{user_file}"

            print(f"Analyzing {user_id}")
        
            try: 
                with open(path_data, 'r') as json_file:
                    experiment_data = json.load(json_file)
                
                for date_time in experiment_data.keys():
                    analyze_single_trial(date_time, experiment_data, user_id, save_plots=False, analysis_mode=args.analysis_mode)
            
            except Exception as e: 
                log_message = f"Error analyzing data for {user_id}: {e}"
                print(log_message)
                log_to_file(log_message)

    elif args.mode == 'questions':
        print("--- Running in GENERATE QUESTIONS mode ---")
        generate_question_log()
        print("--- Question log generation complete ---")

    
    #print(date_time)
    #analyze_single_trial(date_time, experiment_data)

    