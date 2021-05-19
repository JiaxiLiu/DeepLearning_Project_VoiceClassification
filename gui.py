import tkinter as Tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from numpy.core.numeric import identity

import pyaudio 
import wave
import struct
import numpy as np
from matplotlib import pyplot as plt
from encoder.inference import *
from encoder.audio import *
import librosa
import librosa.display
from pathlib import Path
import umap

# for display
import sys
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')

colormap = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=np.float) / 255 

class UI_encoder:
    def __init__(self):
        self.root = Tk.Tk()
        self.root.wm_title("Encoder Model Toolbox")
        self.ioFrame = Tk.Frame(self.root)
        self.buttonFrame = Tk.Frame(self.ioFrame)
        self.embed_list = []
        self.filename_list = []
        self.speaker_list = []
        self.classify_speaker_list = []
        self.classify_embed_list = []
        load_model(Path('./encoder/saved_models/my_run.pt'))
        self.speaker_name = Tk.StringVar()
        self.speaker_name.set('user01')
        self.classify_speaker_name = Tk.StringVar()
        self.classify_speaker_name.set('Classified as Speaker:')
        self.entry_speaker = Tk.Entry(self.ioFrame, 
                                        textvariable = self.speaker_name,
                                        width=10)
        self.label_speaker = Tk.Label(self.ioFrame, text = 'Speaker Name')
        self.label_classify_speaker = Tk.Label(self.ioFrame, textvariable = self.classify_speaker_name)
        self.button_rec = Tk.Button(self.buttonFrame, 
                                    bg = 'white', 
                                    activebackground='red',
                                    text = 'record', 
                                    command = self.record_wav)
        self.button_load = Tk.Button(self.buttonFrame, bg = 'white', text = 'load sample', command = self.load_wav)
        self.button_classify = Tk.Button(self.ioFrame, bg = 'white', text = 'Identify', command = self.identify)
        self.umap_hot = False

        # matplotlib canvas
        self.fig_org_wav,self.ax_org_wav = plt.subplots(2,1,figsize=(7,3))
        #self.fig_org_wav.suptitle('Input Wav file')
        self.ax_org_wav[0].set_title('Input Wav file')
        self.canv_org_wav = FigureCanvasTkAgg(self.fig_org_wav, master=self.root)
        self.canv_org_wav.draw()
        
        self.fig_prc_wav,self.ax_prc_wav = plt.subplots(2,1,figsize=(7,3))
        #self.fig_prc_wav.suptitle('After preprocessing')
        self.ax_prc_wav[0].set_title('After preprocessing')
        self.canv_prc_wav = FigureCanvasTkAgg(self.fig_prc_wav, master=self.root)
        self.canv_prc_wav.draw()

        self.fig_embed = plt.figure(figsize=(4,4))
        self.fig_embed.suptitle('Embedding')
        self.canv_embed = FigureCanvasTkAgg(self.fig_embed, master=self.root)

        self.fig_umap,self.ax_umap = plt.subplots(figsize=(4,4))
        self.fig_umap.suptitle('UMAP')
        self.canv_umap = FigureCanvasTkAgg(self.fig_umap, master=self.root)

        # packing:
        # self.label_speaker.grid(row=1,column=1)
        # self.entry_speaker.grid(row=2,column=1)
        # self.button_rec.grid(row=1,column=2)
        # self.button_load.grid(row=1,column=2)

        self.canv_org_wav.get_tk_widget().pack(side=Tk.TOP, fill=Tk.X, expand=1)
        self.canv_prc_wav.get_tk_widget().pack(side=Tk.TOP, fill=Tk.X, expand=1)
        self.canv_embed.get_tk_widget().pack(side=Tk.LEFT, expand=1)
        self.canv_umap.get_tk_widget().pack(side=Tk.LEFT, expand=1)
    
        self.button_rec.pack(side=Tk.LEFT, expand=True)
        self.button_load.pack(side=Tk.LEFT, expand=True)
        self.label_speaker.pack(side=Tk.TOP, expand=True)
        self.entry_speaker.pack(side=Tk.TOP, expand=True)
        self.buttonFrame.pack(side=Tk.TOP, expand=True)
        self.button_classify.pack()
        self.label_classify_speaker.pack()
        self.ioFrame.pack(side=Tk.RIGHT, expand=True)

        


    def sample_update(self, wf_str):
        _filename = wf_str.split('/')[-1]
        wav_orig,fs = librosa.load(str(wf_str), sr=16000)
        wav_orig = normalize_volume(wav_orig, -30, increase_only=True)
        mel_orig = wav_to_mel_spectrogram(wav_orig)
        wav = preprocess_wav(wf_str, source_sr=fs)
        mel = wav_to_mel_spectrogram(wav)
        # original signal plot
        self.ax_org_wav[0].clear()
        self.ax_org_wav[1].clear()
        self.ax_org_wav[0].set_title('Input Wav file: \'%s\''%_filename)
        t_orig = np.arange(len(wav_orig))/fs
        self.ax_org_wav[0].plot(t_orig,wav_orig)
        self.ax_org_wav[0].autoscale(enable=True, axis='x', tight=True)
        librosa.display.specshow(librosa.power_to_db(mel_orig.T,ref=np.max),ax=self.ax_org_wav[1],y_axis='mel',x_axis='time',sr=fs)
        self.canv_org_wav.draw()
        # processed signal plot
        self.ax_prc_wav[0].clear()
        self.ax_prc_wav[1].clear()
        self.ax_prc_wav[0].set_title('After Prepreprocessing')
        t = np.arange(len(wav))/fs
        self.ax_prc_wav[0].plot(t,wav)
        self.ax_prc_wav[0].autoscale(enable=True, axis='x', tight=True)
        librosa.display.specshow(librosa.power_to_db(mel.T,ref=np.max),ax=self.ax_prc_wav[1],y_axis='mel',x_axis='time',sr=fs)
        self.canv_prc_wav.draw()
        # Embedding plot
        embed = embed_utterance(wav)
        self.fig_embed.clear()
        self.fig_embed.suptitle('Embedding')
        ax_embed = self.fig_embed.add_subplot()
        plot_embedding_as_heatmap(embed, ax=ax_embed)
        self.canv_embed.draw()
        self.embed_list.append(embed)
        self.speaker_list.append(self.speaker_name.get())
        self.filename_list.append(_filename)
        self.plot_umap()

    def plot_umap(self):
        min_umap_points = 5
        embeds = self.embed_list + self.classify_embed_list
        embeds_id = [0]*len(self.embed_list) + [1]*len(self.classify_embed_list)
        self.ax_umap.clear()
        speakers = np.unique(self.speaker_list)
        colors = {speaker_name: colormap[i] for i, speaker_name in enumerate(speakers)}
        if len(self.embed_list) < min_umap_points:
            self.ax_umap.text(.5, .5, "Add %d more points to\ngenerate the projections" % 
                              (min_umap_points - len(self.embed_list)), 
                              horizontalalignment='center', fontsize=15)
        else:
            if not self.umap_hot:
                print("Drawing UMAP projections for the first time, this will take a few seconds.")
                self.umap_hot = True
            reducer = umap.UMAP(int(np.ceil(np.sqrt(len(embeds)))), metric="cosine")
            projections = reducer.fit_transform(embeds)
            speakers_done = set()
            for projection, speakername, identity in zip(projections, self.speaker_list+self.classify_speaker_list, embeds_id):
                color = colors[speakername]
                if identity==0:
                    mark = "."
                else:
                    mark = "x"
                label = None if speakername in speakers_done else speakername
                speakers_done.add(speakername)
                self.ax_umap.scatter(projection[0], projection[1], c=[color], marker=mark,
                                     label=label)
            
            self.ax_umap.legend(prop={'size': 10})
            self.ax_umap.set_aspect("equal", "datalim")
            self.ax_umap.set_xticks([])
            self.ax_umap.set_yticks([])
        self.canv_umap.draw()

    def record_wav(self):
        self.button_rec.config(bg = 'red')
        self.root.update()
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 16000
        seconds = 8
        speaker = self.speaker_name.get()
        i = 0
        while (Path.exists(Path(speaker+'_'+str(i).zfill(2)+'.wav'))):
            i += 1
        filename = speaker+'_'+str(i).zfill(2)+'.wav'
        p = pyaudio.PyAudio()
        print('Recording...')
        stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=fs,
                input=True)
        rec_sound = stream.read(fs*seconds)
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(rec_sound)
        wf.close
        print('Done...File: %s'%filename)
        self.button_rec.config(bg = 'white')
        self.root.update()
        self.sample_update(filename)
    
    def load_wav(self):
        filename = Tk.filedialog.askopenfilename()
        if filename=='':
            return
        self.sample_update(filename)

    def identify(self):
        wf_str = Tk.filedialog.askopenfilename()
        if wf_str=='':
            return
        _filename = wf_str.split('/')[-1]
        wav_orig,fs = librosa.load(str(wf_str), sr=16000)
        wav_orig = normalize_volume(wav_orig, -30, increase_only=True)
        mel_orig = wav_to_mel_spectrogram(wav_orig)
        wav = preprocess_wav(wf_str, source_sr=fs)
        mel = wav_to_mel_spectrogram(wav)
        # original signal plot
        self.ax_org_wav[0].clear()
        self.ax_org_wav[1].clear()
        self.ax_org_wav[0].set_title('Input Wav file: \'%s\''%_filename)
        t_orig = np.arange(len(wav_orig))/fs
        self.ax_org_wav[0].plot(t_orig,wav_orig)
        self.ax_org_wav[0].autoscale(enable=True, axis='x', tight=True)
        librosa.display.specshow(librosa.power_to_db(mel_orig.T,ref=np.max),ax=self.ax_org_wav[1],y_axis='mel',x_axis='time',sr=fs)
        self.canv_org_wav.draw()
        # processed signal plot
        self.ax_prc_wav[0].clear()
        self.ax_prc_wav[1].clear()
        self.ax_prc_wav[0].set_title('After Prepreprocessing')
        t = np.arange(len(wav))/fs
        self.ax_prc_wav[0].plot(t,wav)
        self.ax_prc_wav[0].autoscale(enable=True, axis='x', tight=True)
        librosa.display.specshow(librosa.power_to_db(mel.T,ref=np.max),ax=self.ax_prc_wav[1],y_axis='mel',x_axis='time',sr=fs)
        self.canv_prc_wav.draw()
        # Embedding plot
        embed = embed_utterance(wav)
        self.fig_embed.clear()
        self.fig_embed.suptitle('Embedding')
        ax_embed = self.fig_embed.add_subplot()
        plot_embedding_as_heatmap(embed, ax=ax_embed)
        self.canv_embed.draw()
        # calculating distance 
        distance_list = []
        for old_embed in self.embed_list:
            distance_list.append(np.mean(np.abs(embed-old_embed)))
        id_name_i = np.argmin(distance_list)
        self.classify_embed_list.append(embed)
        self.classify_speaker_list.append(self.speaker_list[id_name_i])
        self.classify_speaker_name.set('Input file:\n%s\n\nClassified as Speaker: \n%s'%(_filename,self.speaker_list[id_name_i]))
        self.filename_list.append(_filename)
        self.plot_umap()



ui_ = UI_encoder()
# autoloading
ui_.speaker_name.set('sp01')
ui_.sample_update('./sample_data/19/19-198-0010.flac')
ui_.sample_update('./sample_data/19/19-198-0011.flac')
ui_.sample_update('./sample_data/19/19-198-0012.flac')
ui_.speaker_name.set('sp02')
ui_.sample_update('./sample_data/40/40-222-0012.flac')
ui_.sample_update('./sample_data/40/40-222-0013.flac')
ui_.sample_update('./sample_data/40/40-222-0014.flac')
ui_.speaker_name.set('sp03')
ui_.sample_update('./sample_data/89/89-218-0016.flac')
ui_.sample_update('./sample_data/89/89-218-0017.flac')
ui_.sample_update('./sample_data/89/89-218-0018.flac')

ui_.root.mainloop()
