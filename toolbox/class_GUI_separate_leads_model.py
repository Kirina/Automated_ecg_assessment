import tkinter as tk
from  tkinter import ttk
import pandas as pd
from scipy.io.wavfile import read
import wfdb.io
# import PIL.ImageGrab as ImageGrab
import pathlib
from AlgorithmsV5_k_model import processing
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from parameters_gui import parameters_gui as Parameters

class GUI_app(tk.Frame):
    def __init__(self, master, sampling_frequency, screen_width, screen_height, font, colour_process_button, colour_import_button, 
                    colour_background, SNR_threshold, signal_freq_band, window_length, heart_rate_limits, max_loss_passband, 
                    min_loss_stopband, length_recording, path_model, name_model):
        super().__init__(master)
        self.master = master
        self.sampling_frequency = sampling_frequency
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.font = font
        self.buttons_width = int(self.screen_width // 9.5)
        self.buttons_height = int(self.screen_height // 10)
        self.lead_selection_height = self.buttons_height + (self.screen_height // 16)
        self.font_size = self.screen_height // 37
        self.plot_distance_from_buttons = self.screen_height // 6
        self.table_width = self.screen_width - (2 * self.buttons_width)
        self.column_width = int(self.table_width // 16)
        
        self.SNR_threshold = SNR_threshold
        self.signal_freq_band = signal_freq_band
        self.window_length = window_length
        self.heart_rate_limits = heart_rate_limits
        self.max_loss_passband = max_loss_passband
        self.min_loss_stopband = min_loss_stopband
        self.length_recording = length_recording
        
        self.path_model = path_model
        self.name_model = name_model
        
        self.colour_process_button = colour_process_button
        self.colour_import_button = colour_import_button 
        self.colour_background = colour_background

        self.lead_selection = None
        self.import_data_button = None
        self.process_button = None
        self.fig = None
        self.canvas = None
        self.button_color_wrap = None
        self.num_leads = None
        self.ECG = None
        
        self.column_number = False
        
        self.init_plot()
        self.gui_window()  
        
        self.master.bind('<Key-F11>', self.make_fullscreen)
        self.master.bind('<Escape>', self.exit_fullscreen)
        
    def get_num_leads(self, new_ecg):
        if new_ecg.ndim > 1:
            if self.column_number: # check if column number
                self.num_leads = len(new_ecg) - 1
            else:
                self.num_leads = len(new_ecg)
        elif new_ecg.ndim == 1: 
            self.num_leads = 1
        else:
            self.num_leads = -1
        
    def get_table_list(self):
        self.table_list = [["Lead-number",
                            "Heart rate found",
                            "Flat line check",
                            "Noise ratio",
                            "CNN-LSTM model",
                            "Overall Result"]]

        for lead in range(1, self.num_leads + 1):
                self.table_list.append([lead, "", "", "", "", ""])
        self.table_list = np.array(self.table_list).T.tolist()
           
    def make_fullscreen(self, event):
        self.master.attributes("-fullscreen", True)
        
    def exit_fullscreen(self, event):
        self.master.attributes("-fullscreen", False)
    
    def check_column_number(self, new_ecg):
        if new_ecg.ndim > 1:
            self.column_number = np.array_equal(new_ecg[0], np.arange(len(new_ecg[0])).astype(int))
        
    def init_plot(self):
        self.fig = Figure(figsize=(15, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().place(x=0, y=self.plot_distance_from_buttons)

    def gui_window(self):
        self.master.attributes("-fullscreen", True) # make GUI fullscreen
        self.master["bg"] = "white" # set background white 
        ttk.Style().configure('Treeview', rowheight=(int(self.screen_height//27)), font=(self.font, self.font_size), background=self.colour_background) # height+font of table rows
        ttk.Style().configure('Treeview.Heading', rowheight=(int(self.screen_height//27)), font=(self.font, int(self.font_size*1.10)), background=self.colour_background) # height+font of table headers
        self.plot_figure(self.ECG) # initialize plot
        self.button_create() # create import data button and process button

    def button_create(self):
        self.import_data_button = tk.Button(self.master, text="Import Data", command=self.import_data, bg=self.colour_import_button,
                                            font=(self.font, self.font_size))
        self.import_data_button.place(x=0, y=0, width=self.buttons_width, height=self.buttons_height)

        self.process_button = tk.Button(self.master, text="Process", command=self.process_ecg, bg=self.colour_process_button, 
                                        font=(self.font, self.font_size))
        self.process_button.place(x = self.buttons_width, y=0, width=self.buttons_width, height=self.buttons_height)
                                  
    def button_create_lead(self):
        self.lead_selection = tk.Scale(self.master, from_=1, to_=self.num_leads, command=self.plot_figure, 
                                       bd=0, orient="horizontal", bg=self.colour_background, highlightthickness=0, 
                                       font=(self.font, self.font_size))
        self.lead_selection.place(x=0, y=self.lead_selection_height, width=self.buttons_width * 2)
        self.lead_selection.set(1)
        self.lead_label = tk.Label(self.master, text="Lead: ", font=(self.font, self.font_size), bg=self.colour_background)
        self.lead_label.place(x=0, y=self.buttons_height, width=self.buttons_width * 2)
             
    def import_data(self):
        file_path = tk.filedialog.askopenfilename()
        file_extension = pathlib.Path(file_path).suffix
        if file_extension == '.txt':
            new_ecg = np.loadtxt(file_path, delimiter=",", dtype="int")
            new_ecg = np.transpose(new_ecg)
        elif file_extension == '.csv':
            new_ecg = np.genfromtxt(file_path, delimiter=",", dtype="int")
            new_ecg = np.transpose(new_ecg)
        elif file_extension == '.hea' or file_extension == '.xws' or file_extension == '.dat' or file_extension == '.atr':
            new_ecg, _ = wfdb.io.rdsamp(file_path.replace(file_extension, ''))
            new_ecg = np.transpose(new_ecg)
            new_ecg = new_ecg * 1000
            new_ecg = new_ecg.astype(int)
        elif file_extension == '.xls':
            new_ecg = pd.read_excel(file_path, dtype = "int").to_numpy()
            new_ecg = np.transpose(new_ecg)
        elif file_extension == '.xlsx':
            new_ecg = pd.read_excel(file_path, dtype = "int").to_numpy()
            new_ecg = np.transpose(new_ecg)
        elif file_extension == '.wav':
            new_ecg = read(file_path)[1]
            new_ecg = np.array(new_ecg).astype(int)
        else:
            print('No file selected or file extension not accepted')
            print('Please only use files with extension: -.txt -.csv -.hea -.xws -.dat -.atr -.xls -.xlsx -.wav')
            return        
        self.ECG = np.array([])
        self.column_number = False
        try:
            # check if the provided data has a column with datapoint numbers
            # if it does not, add the datapoint number column
            self.check_column_number(new_ecg)
            self.get_num_leads(new_ecg)
            if not self.column_number:
                self.ECG = np.arange(len(new_ecg)) if self.num_leads == 1 else np.arange(len(new_ecg[0])).astype(int)
                self.ECG = np.append([self.ECG], [new_ecg], axis=0)
            else:
                self.ECG = new_ecg
        except: 
            print('The provided data is in the wrong format')
            return
        
        self.button_create_lead() # data might have different lead number, so create button again
        self.plot_figure(self.ECG) # plot ECG
        
    def plot_figure(self, _):
        if (self.ECG is not None and not len(self.ECG) == 0):
            self.master.wm_title("Embedding in Tk")
            self.fig.clear()
            ax = self.fig.add_subplot()

            ax.plot(self.ECG[self.lead_selection.get()])
            ax.set_xlabel("Time [s]", fontsize=self.font_size, font=self.font)
            ax.set_ylabel("mV", fontsize=self.font_size, font=self.font)
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', linewidth='0.4', color='red')
            ax.grid(which='minor', linestyle='-', linewidth='0.4', color=(1, 0.7, 0.7))
            
            length_ecg_data = int(self.ECG[0][-1] + 1)
            original_stepsize = length_ecg_data // 5
            
            original_x_axis = np.arange(0, length_ecg_data + 1, original_stepsize) 

            step_size = int(self.length_recording / (len(original_x_axis) - 1))
            ax.set_xticks(original_x_axis, labels=list(range(0, self.length_recording + step_size, step_size)))
            self.canvas.draw()            
            
    def process_ecg(self):
        if not self.ECG.size:
            return
        else:
            table_list_result = (processing(self.ECG, self.num_leads, self.sampling_frequency, self.SNR_threshold, self.signal_freq_band,
                                            self.window_length, self.heart_rate_limits, self.max_loss_passband, self.min_loss_stopband,
                                            self.sampling_frequency, self.path_model, self.name_model, self.length_recording))
            
            self.get_table_list()
            for x in range(0,  len(table_list_result)):
                self.table_list[x+1][1:self.num_leads+1] = table_list_result[x][0:self.num_leads+1]
            self.table_fill()

    def table_fill(self):
        # place table labels
        space_before_table = int(self.column_width//5)
        table_labels = ttk.Treeview(self.master, height=len(self.table_list) - 1)
        table_labels['columns'] = self.table_list[0][0]
        table_labels.column('#0', width=0, stretch=tk.NO)
        table_labels.column(self.table_list[0][0], anchor=tk.CENTER, width=4*self.column_width)
        table_labels.heading(self.table_list[0][0], text=self.table_list[0][0], anchor=tk.CENTER)
        
        for i in range(1, len(self.table_list)):
            table_labels.insert(parent='',index='end', tag=f'id_{i}', text='', values=(self.table_list[i][:1]))
        table_labels.tag_configure(f'id_{len(self.table_list)-1}', background='#DFDFDF')
        table_labels.place(x = self.buttons_width * 2 + space_before_table, y=0)
        
        # place table results 
        table = ttk.Treeview(self.master, height=len(self.table_list) - 1)
        table['columns'] = self.table_list[0][1:]
        table.column('#0', width=0, stretch=tk.NO)
        for col in self.table_list[0]: 
            # column layout
            if col == self.table_list[0][0]:
                pass
            else: 
                table.column(col, anchor=tk.CENTER, width=self.column_width)
                table.heading(col, text=col, anchor=tk.CENTER)
        for i in range(1, len(self.table_list)):
            table.insert(parent='',index='end', text='', tag=f'id_{i}', values=(self.table_list[i][1:]))
        table.tag_configure(f'id_{len(self.table_list)-1}', background='#DFDFDF')
        table.place(x = self.buttons_width * 2 + 4*self.column_width + space_before_table, y = 0)

def get_curr_screen_geometry():
    '''
    gets screen size
    
    return: screen width, screen height
    '''
    root = tk.Tk()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    geometry = root.winfo_geometry()
    width = int(geometry.split('x')[0])
    height = int(geometry.split('x')[1].split('+')[0])
    root.destroy()
    return width, height

def main():
    # parameters for GUI
    parameters = Parameters()
    sampling_frequency = parameters['sampling_frequency']        # Hz
    max_loss_passband = parameters['max_loss_passband']     # dB
    min_loss_stopband = parameters['min_loss_stopband']      # dB
    SNR_threshold = parameters['SNR_threshold']
    signal_freq_band = parameters['signal_freq_band']      # from .. to .. in Hz
    heart_rate_limits = parameters['heart_rate_limits']       # from ... to ... in beats per minute
    length_recording = parameters['length_recording']       # seconds
    window_length = parameters['window_length']  

    path_model = parameters['path_model']
    name_model = parameters['name_model']

    screen_width, screen_height = get_curr_screen_geometry()
    init_gui_width = int(screen_width // 1.2) 
    init_gui_height = int(screen_height // 1.2)
    
    colour_process_button = '#3BA4FB'
    colour_import_button = '#E75E53'
    colour_background = '#FFFFFF'
    font = 'Calibri'
    
    root = tk.Tk()
    root.geometry(f"{init_gui_width}x{init_gui_height}")
    
    # To create a screenshot, needs ImageGrab import
    # def save_canvas():
    #     img = ImageGrab.grab()
    #     img.save('screenshot.png')
    # root.after(20000, save_canvas)

    # create the GUI object from the GUI_app class
    myapp = GUI_app(root, sampling_frequency, screen_width, screen_height, font, colour_process_button, colour_import_button, 
                    colour_background, SNR_threshold, signal_freq_band, window_length, heart_rate_limits, max_loss_passband, 
                    min_loss_stopband, length_recording, path_model, name_model)
    myapp.mainloop()

if __name__ == "__main__":
    main()


