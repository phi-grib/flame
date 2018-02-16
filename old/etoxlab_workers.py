#!/usr/bin/env python
# -*- coding: utf-8 -*-

##    Description    eTOXlab simple GUI
##                   
##    Authors:       Ines Martinez and Manuel Pastor (manuel.pastor@upf.edu) 
##
##    Copyright 2014 Manuel Pastor
##
##    This file is part of eTOXlab.
##
##    eTOXlab is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    eTOXlab is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with eTOXlab.  If not, see <http://www.gnu.org/licenses/>.
    
from Tkinter import *  # Importing the Tkinter (tool box) library 
import Tkconstants
import ttk

import tkMessageBox
import tkFileDialog
import os
import subprocess
import shutil
import Queue

from threading import Thread
from utils import wkd
from utils import VERSION
from utils import removefile
from utils import cleanSandbox
import tarfile
from PIL import ImageTk, Image
import glob


################################################################
### MANAGE
################################################################
    
'''
Creates a new object to execute manage.py commands in new threads
'''
class manageLauncher:
    
    def __init__(self, parent, command, seeds, q, gui):       
        self.model = parent
        self.command = command
        self.seeds = seeds
        self.queue = q
        self.gui = gui
        
        self.dest =''

    def processJob(self):
        p = processWorker(self.command, self.seeds, self.queue, self.dest, self.gui)                                                                       
        p.compute()                      

    def process(self):       
        self.seeds = []        
        self.seeds.append (self.model.selEndpoint())
        self.seeds.append (self.model.selVersion())

        reqFile = ['--get=series', '--get=model', '--export']
                            
        if self.command in reqFile:
            self.dest=tkFileDialog.askdirectory(initialdir='.',title="Choose a directory...")
            if self.dest=='':
                return
        elif self.command == '--import':
            importfile = self.gui.importTar.get()

            if importfile == None or importfile == '':
                tkMessageBox.showerror("Error Message", "No suitable packed model selected")
                return

            endpoint = importfile[:-4]
            
            if os.path.isdir (wkd+'/'+endpoint.split('/')[-1]):
                tkMessageBox.showerror("Error Message", "This endpoint already exists")
                return

            self.gui.importTar.delete(0, END)
            
            self.seeds = []
            self.seeds.append (endpoint)

        elif self.command == '--expose':
            pubversion = self.gui.pubver.get()
            self.seeds.append(pubversion)

        self.gui.addBackgroundProcess()
                
        t = Thread(target=self.processJob)
        t.start()
        
class processWorker: 

    def __init__(self, command, seeds, queue, dest, gui):
        self.command = command
        self.seeds = seeds
        self.q = queue
        self.dest = dest
        self.gui = gui

    def compute(self):

        endpoint = self.seeds[0]
        mycommand = [wkd+'/manage.py','-e', endpoint, self.command]
        
        if self.command=='--get=series' or self.command=='--get=model':
            version = self.seeds[1]
            mycommand.append ('-v')
            mycommand.append (version)
            os.chdir(self.dest)
            
        elif self.command=='--expose':
            mycommand = [wkd+'/manage.py','-e', endpoint]
            version = self.seeds[1]
            mycommand.append ('-v')
            mycommand.append (version)
            mycommand.append ('--expose='+self.seeds[2])
            
        elif self.command=='--export':
            os.chdir(self.dest)
        
        try:
            proc = subprocess.Popen(mycommand,stdout=subprocess.PIPE)
            self.q.put ('PROCESS '+str(proc.pid))
            
        except:
            self.q.put ('ERROR: Manage process failed')
            return
 
        for line in iter(proc.stdout.readline,''):
            line = line.rstrip()
            if line.startswith('ERROR:'):
                self.q.put (line)
                return

        if proc.wait() == 1 :
            self.q.put ('ERROR: Unknown error')
            return
                
        if self.command in ['--publish','--expose','--remove', '--import'] :
            self.q.put ('update '+endpoint)

        self.q.put('Manage completed OK')


################################################################
### VIEW
################################################################
        
'''
Creates an object to execute view.py command in a new thread
'''
class viewLauncher:
    
    def __init__(self, parent, seeds, q, gui):        
        self.model = parent 
        self.seeds = seeds
        self.queue = q
        self.gui   = gui

    def viewJob(self):
        view = viewWorker(self.seeds, self.queue, self.vtype, self.molecules, self.background, self.refname, self.refver)                                                                       
        view.compute()                      

    def view(self):        
        self.seeds = [] 
        self.seeds.append(self.model.selEndpoint())
        self.seeds.append(self.model.selVersion())

        # Call new thread to visualize the series       
        self.gui.viewButton1.configure(state='disable')
        self.gui.addBackgroundProcess()

        self.vtype   = self.gui.viewTypeCombo.get()
        self.refname = self.gui.referEndpointCombo.get()
        self.refname = self.refname.strip()

        if self.refname=='None':
            self.refname = ''
            
        refverstr = self.gui.referVersionCombo.get()
        refverstr = refverstr.strip()
        
        try:            
            self.refver  = int(refverstr)
        except:
            self.refver = 0     
       
        self.background = (self.gui.viewBackground.get() == '1')
        self.molecules = ''
        
        t = Thread(target=self.viewJob)
        t.start()

    def viewQuery(self):
        self.seeds = [] 
        self.seeds.append(self.model.selEndpoint())
        self.seeds.append(self.model.selVersion())

        self.vtype       = self.gui.viewTypeComboQuery.get()
        self.refname     = self.seeds[0]
        self.refver      = self.seeds[1]
        self.molecules   = self.gui.eviewQuery1.get()      
        self.background  = (self.gui.viewBackgroundQuery.get() == '1')

        if self.molecules == None or self.molecules=='':
            tkMessageBox.showerror("Error Message", "Enter a query series file")
            return

        # Call new thread to visualize the series       
        self.gui.viewButton2.configure(state='disable')
        self.gui.addBackgroundProcess()
        
        t = Thread(target=self.viewJob)
        t.start()

    def viewModel(self):
        self.seeds = [] 
        self.seeds.append(self.model.selEndpoint())
        self.seeds.append(self.model.selVersion())

        # Call new thread to visualize the series       
        self.gui.viewButton2.configure(state='disable')
        self.gui.addBackgroundProcess()
        
        self.vtype       = 'model'
        self.refname     = None
        self.refver      = None
        self.molecules   = None      
        self.background  = None
        
        t = Thread(target=self.viewJob)
        t.start()
            

class viewWorker: 

    def __init__(self, seeds, queue, vtype, molecules, background, refname, refver):
        self.seeds = seeds
        self.q = queue
        self.vtype = vtype
        self.molecules = molecules
        self.background = background
        self.refname = refname
        if refver==None:
            self.refver = None
        else:
            try:
                self.refver = int(refver)
            except:
                self.refver = None
            
    def compute(self):        
        name    = self.seeds[0]
        version = self.seeds[1]

        mycommand=[wkd+'/view.py','-e',name,'-v',version,
                   '--type='+self.vtype]

        if self.molecules != None and len(self.molecules)>0:
            mycommand.append('-f')
            mycommand.append(self.molecules)

        if self.refname != None and len(self.refname)>0:
            mycommand.append('--refname=' +self.refname)

        if self.refver != None:
            mycommand.append('--refver=%d' %self.refver)
            
        if self.background :
            mycommand.append('--background')
        
        try:
            proc = subprocess.Popen(mycommand,stdout=subprocess.PIPE)
            self.q.put ('PROCESS '+str(proc.pid))
        except:
            self.q.put ('ERROR: View process failed')
            return

        message = 'View completed OK '+ name + ' ' + version
        
        for line in iter(proc.stdout.readline,''):
            line = line.rstrip()
            if line.startswith('ERROR:'):
                self.q.put (line)
                return
            else:
                message += ' '+line

##        if proc.wait() == 1 :
##            self.q.put ('ERROR: Unknown error')
##            return

        self.q.put(message)


################################################################
### BUILD
################################################################
'''
Creates an object to execute build.py command in a new thread
'''
class buildLauncher:
    
    def __init__(self, parent, seeds, q, gui):        
        self.model = parent
        self.seeds = seeds
        self.queue = q
        self.gui   = gui
          
    def buildJob(self):
        job = buildWorker(self.seeds, self.queue)
        job.rebuild()

    def build(self):        
        name    = self.model.selEndpoint()
        version = self.model.selVersion()
        series  = self.gui.buildSeries.get()
        model   = self.gui.buildModel.get()

        origDir = self.model.selDir()+'/'
        destDir = origDir[:-5]+'0000/'
        
        # clean sandbox
        if origDir != destDir:
            cleanSandbox(destDir)

        # If 'series' starts with '<series' then copy training.sdf, tdata.pkl, info.pkl to the sandbox            
        if series.startswith('<series'):

            series = ''  # never use '<series i>' as a filename
            
            if version != '0' :
                files = ['training.sdf',
                         'tstruct.sdf',
                         'tdata.pkl']
                try:
                    for i in files:
                        if os.path.isfile(origDir+i):
                            shutil.copy(origDir+i,destDir)
                except:
                    tkMessageBox.showerror("Error Message", "Unable to copy series")
                    return

            if not os.path.isfile(destDir+'training.sdf'):
                tkMessageBox.showerror("Error Message", "No series found")
                return

        # If 'model' starts with '<edited' the file imodel.py has been already copied. Else, copy it    
        if not model.startswith('<edited'):
            if version != '0' :
                try:
                    shutil.copy(self.model.selDir()+'/imodel.py',wkd+'/'+name+'/version0000/')
                except:
                    tkMessageBox.showerror("Error Message", "Unable to copy imodel.py")
                    return
        
        # Add argument to build list 
        self.seeds = [] 
        self.seeds.append(name)
        self.seeds.append('0')      # model and series have been copied to sandbox
        self.seeds.append(series)   # '' for existing series or a filename for copied ones
    
        # Call new thread to build the model       
        self.gui.buildButton.configure(state='disable')
        #self.gui.pb.start(100)
        self.gui.addBackgroundProcess()
    
        t = Thread(target=self.buildJob)
        t.start()


class buildWorker: 

    def __init__(self, seeds, queue):
        self.seeds = seeds
        self.q = queue

    def rebuild(self):      
        name    = self.seeds[0]
        version = self.seeds[1]
        series  = self.seeds[2]

        mycommand = [wkd+'/build.py','-e',name,'-v',version]

        if series and series !='':
            mycommand.append ('-f')
            mycommand.append (series)

        try:
            proc = subprocess.Popen(mycommand,stdout=subprocess.PIPE)
            self.q.put ('PROCESS '+str(proc.pid))
            
        except:
            self.q.put ('ERROR: Building process failed')
            return
            
        for line in iter(proc.stdout.readline,''):
            line = line.rstrip()
            if line.startswith('ERROR:'):
                self.q.put (line)
                return

            if line.startswith('LOCAL MODEL'):
                self.q.put (line)
                
            if "Model OK" in line:
                self.q.put('Building completed OK'+name)
                return
        
        if proc.wait() == 1 :
            self.q.put ('ERROR: Unknown error')
            return

        self.q.put('update '+name)
        self.q.put('ERROR: building process aborted')
    


################################################################
### PREDICT
################################################################
'''
Creates an object to execute build.py command in a new thread
'''
class predictLauncher:
    
    def __init__(self, parent, seeds, q, gui):        
        self.model = parent
        self.seeds = seeds
        self.queue = q
        self.gui   = gui
          
    def PredictJob(self):
        job = predictWorker(self.seeds, self.queue, self.series)
        job.predict()

    def predict(self):        
        name    = self.model.selEndpoint()
        version = self.model.selVersion()
        series  = self.gui.predictSeries.get()

        if series == '':
            tkMessageBox.showerror("Error Message", "Please enter the name of the series")
            return

        # Add argument to build list 
        self.seeds = [] 
        self.seeds.append(name)
        self.seeds.append(version)      # model and series have been copied to sandbox
        self.series = series    # '' for existing series or a filename for copied ones
    
        # Call new thread to predict the series       
        self.gui.predictButton.configure(state='disable')
##        self.gui.pb.start(100)
        self.gui.addBackgroundProcess()
    
        t = Thread(target=self.PredictJob)
        t.start()


class predictWorker: 

    def __init__(self, seeds, queue, series):
        self.seeds = seeds
        self.q = queue
        self.series = series
            
    def predict(self):        
        name    = self.seeds[0]
        version = self.seeds[1]
        series  = self.series

        removefile ('/var/tmp/results.txt')
        
        mycommand=[wkd+'/predict.py','-e',name,'-v',version,'-f', series, '-g']
            
        try:
            proc = subprocess.Popen(mycommand,stdout=subprocess.PIPE)
            self.q.put ('PROCESS '+str(proc.pid))
            
        except:
            self.q.put ('ERROR: Predict process failed')
            return
 
        for line in iter(proc.stdout.readline,''):
            
            line = line.rstrip()
            if line.startswith('ERROR:'):
                self.q.put (line)
                return

        if proc.wait() == 1 :
            self.q.put ('ERROR: Unknown error')
            return
            
        self.q.put('Predict completed OK '+ name + ' ' + version + ' ' + series)
