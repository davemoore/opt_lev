import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import os


#define a class with all of the attributes and methods necessary for processing a single data file to  
class data:
    #This is a class with all of the attributes and methods for a single data file.

    def __init__(self):
        self.fname = "Filename not assigned."
        #self.path = "Directory not assigned." #Assuming directory in filename
        self.data = "bead position data not loaded"
        self.cant_data = "cantilever position data no loaded"
        self.separation = "separation not entered"
        self.Fsamp = "Fsamp not loaded"
        self.Time = "Time not loaded"
        self.temps = "temps not loaded"
        self.pressures = "pressures not loaded"
        self.synth_setting = "Synth setting not loaded"
        self.dc_supply_setting = "DC supply settings not loaded"
        self.electrode_data = "electrode data not loaded yet"
        self.electrode_settings = "Electrode settings not loaded"
        self.electrode_dc_vals = "Electrode potenitals not loaded"
        self.stage_settings = "Stage setting not loaded yet"
    
    def load(self, fstr, sep):
        #Methods to load the attributes from a single data file. 
        dat, attribs, f = bu.getdata(fstr)
        self.fname = fstr
        self.data = np.transpose(dat[:, 0:2]) #x, y, z bead position
        self.cant_data = np.transpose(dat[:, 17:19]) # x, y, z, cantilever monitor 
        self.separation = sep #Manually entreed distance of closes approach
        self.Fsamp = attribs["Fsamp"] #Sampling frequency of the data
        self.Time = bu.labview_time_to_datetime(attribs["Time"]) #Time of end of file
        self.temps = attribs["temps"] #Vector of thermocouple temperatures 
        self.pressures = attribs["pressures"] #Vector of chamber pressure readings [pirani, cold cathode]
        self.synth_settings = attribs["synth_settings"] #Synthesizer fron pannel settings
        self.dc_supply_settings = attribs["dc_supply_settings"] #DC power supply front pannel testings.
        self.electrode_data = dat[:, 8:15] #Record of voltages on the electrodes
        self.electrode_settings = attribs["electrode_settings"] #Electrode front pannel settings for all files in the directory 
        self.electrode_dc_vals = attribs["electrode_dc_vals"] #Front pannel settings applied to this particular file
        self.stage_settings = attribs['stage_settings'] #Front pannel settings for the stage for this particular file.
        
