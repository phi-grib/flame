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
import signal
import shutil
import Queue
import re
import fcntl

from threading import Thread
from utils import wkd
from utils import VERSION
from utils import removefile
from utils import cleanSandbox
import tarfile
from PIL import ImageTk, Image
import glob
from etoxlab_workers import *
from etoxlab_viewers import *


################################################################
### TREEVIEW CLASS
################################################################

'''
Creates a TreeView to shows general information about the existing models
'''
class modelViewer (ttk.Treeview):

    def __init__(self, parent):
              
        scrollbar_tree = ttk.Scrollbar(root)
        
        self.tree=ttk.Treeview.__init__(self, parent, columns = ('a','b','c','d','e'),
                                        selectmode='browse', yscrollcommand = scrollbar_tree.set)
               
        self.column("#0",minwidth=0,width=150, stretch=NO)
        self.column ('a', width=5)
        self.column ('b', width=30)
        self.column ('c', width=80)
        self.column ('d', width=20)
        self.column ('e', width=200)
        self.heading ('a', text='#')
        self.heading ('b', text='MD')
        self.heading ('c', text='mod')
        self.heading ('d', text='mol')
        self.heading ('e', text='quality')         
        
        self.datos= []

        scrollbar_tree.pack(side="left", fill=Y)
        self.pack(side="top", fill="both",expand=True,ipadx=100)

        # Move scrollbar 
        scrollbar_tree.config(command = self.yview)
        self.chargeData()
        self.updateFocus()
        
    def clearTree(self):
        for i in self.get_children():
            self.delete(i)

    def selEndpoint(self):       
        return self.focus().split()[0]

    def selVersion(self):
        d = self.set(self.focus()).get('a')
        
        if d:
            #d=d.replace('@',' ')
            d=d[:-2]
            d=d.strip()
            if d=='*':
                return ('0')
            else:
                return d
        else:
            return ('0')

    def selDir (self):
        e = self.selEndpoint()
        v = self.selVersion()
        return (wkd + '/' + e +'/version%0.4d'%int(v))
      
    # Charges general information about models.    
    def chargeData(self):
        self.clearTree()
        version = []

        process = subprocess.Popen([wkd+'/manage.py','--info=short'], stdout=subprocess.PIPE)
        output, err = process.communicate()

        output=output.split('------------------------------------------------------------------------------')
        
        for line in output[1:]:
            line=line.split("\n")
            self.datos.append(line)
            name=line[1].split("[")[0]            
            self.insert('', 'end', '%-9s'%(name), text='%-9s'%(name))
            count=0                            
            for x in line[2:-1]:
                if x.startswith ("All requested models"):
                    continue
                version=self.chargeDataDetails(x)
                
                if len(version)>4:
                    if 'confident' in version[-1]:
                        ctag = ('confident',)
                    else:
                        ctag = ('normal',)

                    if version[0][5]!=' ':
                        ctag = (ctag[0],'web-service')
                        
                    self.insert('%-9s'%(name), 'end', values=(version[0],version[1],version[2],version[3],version[4]),
                                iid='%-9s'%(name)+str(count), tags=ctag)

                    # The list of child is unfold in treeView
                    self.item('%-9s'%(name), open=True)                    
                count+=1
                
        self.tag_configure('web-service', foreground='red')
        #self.tag_configure('confident', font='sans 9 italic')
        #self.tag_configure('confident', background='#00C0C0')
        self.tag_configure('confident', background='orange')
                    
        self.maxver = 1
        for child in self.get_children():
            iver= len(self.get_children(child))
            if iver > self.maxver : self.maxver=iver

    def updateFocus(self):
        # Focus in first element of the TreeView
        self.selection_set(self.get_children()[0:1])
        self.focus(self.get_children()[0:1][0])


    def setFocus(self, elabel, ever):
        if len(elabel)<10:
            iid = '%-8s'%elabel
        else:
            iid = elabel
        iid = iid+' '+ever

        try:
            self.selection_set((iid,))
            self.focus(iid)
        except:
            self.selection_set(self.get_children()[0:1])
            self.focus(self.get_children()[0:1][0])


    # Charges detailed information about each version of a given model(line).   
    def chargeDataDetails(self,line):       
        y = []

        isWeb = (line[5] != ' ')

        line=line.replace(':','\t')        
        l=line.split('\t')
        #print line

        y.append('%-7s' %l[0][:7])
        
        if 'no model info available' in line:
            y.append ('na') #MD
            y.append ('na') #mod
            y.append ('na') #mol
            y.append ('no info available') # quality
            return y

        try:
            y.append ('%-10s'%l[1][:10])         # MD
            y.append ('%-8s'%l[2][:18])          # regression method

            try:
                numMol = int (l[3][:4])
                strMol = str(numMol)
            except:
                strMol = 'na'
                
            y.append ('%-6s'%strMol)             # num mol

            if 'R2' in l[3]:     # quantitative

                try:
                    r2 = float(l[4][:5])
                    cache1 = 'R2:%5.2f'%r2
                except:
                    cache1 = 'R2:     '
                try:
                    q2 = float(l[5][:5])
                    cache2 = cache1 + '     Q2:%5.2f'%q2
                except:
                    cache2 = cache1 + '     Q2:     '
                    
                try:
                    sdep = float(l[6][:5])
                    cache3 = cache2 + '     SDEP:%5.2f'%sdep
                except:
                    cache3 = cache2 + '     SDEP:     '

                #print cache
                
                y.append( cache3 )         
                    
            elif 'spe' in l[4]:  # qualitative
                try:
                    sen = float(l[4][:5])
                    spe = float(l[5][:5])
                    mcc = float(l[6][:5])
                    y.append( 'sen:%5.2f'%sen+'    spe:%5.2f'%spe+'    MCC:%5.2f'%mcc)
                except:
                    y.append( 'na')
            elif 'SSX' in l[3]:  # PCA
                try:
                    SSX = float(l[4][:5])
                    y.append( 'SSX:%5.2f'%SSX)
                except:
                    y.append( 'na')                 
            else:                # fallback
                y.append ('not recognized')

            #print l
            
            if 'confident' in l[-1]:
                y.append ('confident')
                
        except:
            y = []
            y.append('%-7s' %l[0])
            y.append ('na') #MD
            y.append ('na') #mod
            y.append ('na') #mol
            y.append ('no info available') # quality
            
        return y

     
###################################################################################
### MAIN GUI CLASS
###################################################################################   
'''
GUI class
'''
class etoxlab:


    def __init__(self, master):
        self.seeds  = []
        self.q      = Queue.Queue()
        self.master = master
        self.myfont = 'Courier New'
        self.skipUpdate = False
        self.backgroundCount = 0
        self.processList = []

        # create a toplevel menu
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Refresh", command=self.updateGUI, accelerator="Ctrl+R")
        filemenu.add_command(label="Kill process", command=self.killProcess)
        filemenu.add_command(label="Exit", command= lambda: self._quit(event=None), accelerator="Ctrl+Q")
        helpmenu = Menu(menubar, tearoff=0)   
        helpmenu.add_command(label="About eTOXlab", command= lambda: visualizeHelp().showAbout() )
        menubar.add_cascade(label="File", menu=filemenu)
        menubar.add_cascade(label="Help", menu=helpmenu)
        
        root.bind("<Control-q>", self._quit)
        root.bind("<Control-r>", self.updateGUI)
        root.config(menu=menubar)
             
        i1 = Frame(root) # frame for tree (common to all tabs)
        i2 = Frame(root) # frame for notebook
        
        ## Treeview
        t1 = Frame(i1)
        self.models = modelViewer (t1)
        self.models.bind('<<TreeviewSelect>>', self.selectionChanged)
     
        # Main container is a Notebook
        n = ttk.Notebook (i2)
        f1 = Frame(n)
        f2 = Frame(n)
        f3 = Frame(n)
        f4 = Frame(n)

        self.pb = ttk.Progressbar(i2, orient='horizontal', mode='indeterminate', value=0)
        
        f1.pack(side="top", fill='x', expand=False)
        f2.pack(side="top", fill='x', expand=False)
        f3.pack(side="top", fill='x', expand=False)
        f4.pack(side="top", fill='x', expand=False)
    
        n.add (f1, text='manage')
        n.add (f2, text='build')
        n.add (f3, text='view')
        n.add (f4, text='predict')
        n.pack (side="top", fill="both", expand=True)

        self.pb.pack(side="top", fill='x', expand=False)
        
        ## MANAGE Frame        
        f12 = Frame(f1)
    
        fnew = LabelFrame(f12, text='endpoint')
        
        fnewi = Frame(fnew)
        fnew1 = Frame(fnewi)
        fnew2 = Frame(fnewi)
        fnewj = Frame(fnew)

        vcmd1 = (root.register(self.validateEndpoint), '%S', '%P')
        
        Label(fnew1, width = 10, anchor='e', text='name').pack(side="left")       
        self.enew1 = Entry(fnew1, bd =1, validate = 'key', validatecommand = vcmd1 )
        self.enew1.pack(side="left")               # field containing the new endpoint name

        vcmd2 = (root.register(self.validateTag), '%S', '%P')
        
        Label(fnew2, width = 10, anchor='e', text='tag').pack(side="left")
        self.enew2 = Entry(fnew2, bd =1, validate = 'key', validatecommand = vcmd2)
        self.enew2.pack(side="left")               # field containing the new endpoint tag
       
        Label(fnewj, text='creates a new endpoint').pack(side="left", padx=5, pady=5)       
        Button(fnewj, text ='new', command = self.new, width=5).pack(side="right", padx=5, pady=5)

        fnew1.pack(fill='x')
        fnew2.pack(fill='x')

        fnewi.pack(fill="x")
        fnewj.pack(fill="x")        
        fnew.pack(fill="x", padx=5, pady=2)
        
        fkill = Label(fnew)
        Label(fkill, text='removes endpoint').pack(side="left", padx=5, pady=5)
        Button(fkill, text ='kill', command = self.kill, width=5).pack(side="right", padx=5, pady=5)        
        fkill.pack(fill="x")

        fmodel = LabelFrame(f12, text='model')
        
        self.publish=manageLauncher(self.models,'--publish', self.seeds, self.q, self) 

        fpublish = Label(fmodel)
        Label(fpublish, text='clone sandbox as a new version').pack(side="left",padx=5, pady=5)
        Button(fpublish, text ='publish', command = self.publish.process, width=5).pack(side="right", padx=5, pady=5)
        fpublish.pack(fill='x')

        finfo = Label(fmodel)
        Label(finfo, text='shows complete model information').pack(side="left", padx=5, pady=5)
        Button(finfo, text ='info', command = self.seeDetails, width=5).pack(side="right", padx=5, pady=5)        
        finfo.pack(fill="x")
        
        self.remove=manageLauncher(self.models,'--remove', self.seeds, self.q, self)
        
        frem = Label(fmodel)
        Label(frem, text='removes last model version').pack(side="left",padx=5, pady=5)
        Button(frem, text ='remove', command = self.remove.process, width=5).pack(side="right", padx=5, pady=5)
        frem.pack(fill='x')
        
        
        fexpi = Frame(fmodel)
        fexpj = Frame(fmodel)

        vcmd3 = (root.register(self.validateVersion), '%S', '%P')
        
        Label(fexpi, width = 10, anchor='e', text='public ver').pack(side="left")
        self.pubver = Entry(fexpi, bd =1, validate = 'key', validatecommand = vcmd3)
        self.pubver.pack(side='left')               # field containing the new endpoint tag

        self.expose=manageLauncher(self.models,'--expose', self.seeds, self.q, self)
        
        fexpose = Label(fexpj)
        Label(fexpose, text='exposes as web service').pack(side="left",padx=5, pady=5)
        Button(fexpose, text ='expose', command = self.expose.process, width=5).pack(side="right", padx=5, pady=5)
        fexpose.pack(fill='x')

        fexpi.pack(fill='x')
        fexpj.pack(fill='x')

        fmodel.pack(fill='x', padx=5, pady=2)



        fget = LabelFrame(f12, text='get')     

        self.gseries=manageLauncher(self.models,'--get=series', self.seeds, self.q, self)
        fgets = Label(fget)
        Label(fgets, text='saves training series').pack(side="left", padx=5, pady=5)
        Button(fgets, text ='series', command = self.gseries.process, width=5).pack(side="right", padx=5, pady=5)
        fgets.pack(fill='x')

        self.gmodel=manageLauncher(self.models,'--get=model', self.seeds, self.q, self)

        fgetm = Label(fget)
        Label(fgetm, text='saves model definition file').pack(side="left", padx=5, pady=5)
        Button(fgetm, text ='model', command = self.gmodel.process, width=5).pack(side="right", padx=5, pady=5)
        fgetm.pack(fill='x')

        fget.pack(fill='x', padx=5, pady=2)        
        
        fexp_imp = LabelFrame(f12, text='import/export')

        self.importa=manageLauncher(self.models,'--import',self.seeds,self.q, self)
        fimp = Label(fexp_imp)
        fimp0 = Frame(fimp)
        fimp1 = Frame(fimp)

        Label(fimp0, width = 10, anchor='e', text='pack').pack(side='left')
        self.importTar = Entry(fimp0, bd =1)
        self.importTar.pack(side='left')        
        Button(fimp0, text ='...', width=2, command = lambda : self.selectFile (self.importTar,('Packs','*.tgz'))).pack(side='left') 
        
        Label(fimp1, text='imports packed endpoint').pack(side="left", padx=5)        
        Button(fimp1, text ='import', command = self.importa.process, width=5).pack(side="right", padx=5)

        fimp0.pack(fill='x')
        fimp1.pack(fill='x')        
        fimp.pack(fill='x')

        self.export=manageLauncher(self.models,'--export',self.seeds,self.q, self)
        fexp = Label(fexp_imp)
        Label(fexp, text='packs selected endpoint').pack(side="left",padx=5, pady=10)
        Button(fexp, text ='export', command = self.export.process, width=5).pack(side="right", padx=5, pady=10)
        fexp.pack(fill='x')
        
        fexp_imp.pack(fill="x", padx=5, pady=2)
        
        f12.pack(fill='x')
       
        ## BUILD Frame        
        self.bmodel=buildLauncher(self.models, self.seeds, self.q, self) 
        
        f22 = Frame(f2)

        fbuild = LabelFrame(f22, text='build model')

        fbuild0 = Frame(fbuild)
        fbuild1 = Frame(fbuild)
        fbuild2 = Frame(fbuild)
        
        Label(fbuild0, width = 10, anchor='e', text='series').pack(side='left')       
        self.buildSeries = Entry(fbuild0, bd =1)
        self.buildSeries.pack(side='left')      
        Button(fbuild0, text ='...', width=2, command = lambda : self.selectFile (self.buildSeries,('Series','*.sdf'))).pack(side='left')
        
        Label(fbuild1, width = 10, anchor='e', text='model').pack(side='left')       
        self.buildModel = Entry(fbuild1, bd =1)
        self.buildModel.pack(side='left')      
        Button(fbuild1, text ='...', width=2, command = self.editModelFile).pack(side='left')

        Label(fbuild2, text='build model  in sandbox with above components').pack(side="left", padx=5, pady=5)
        self.buildButton = Button(fbuild2, text = 'OK', command = self.bmodel.build, width=5)
        self.buildButton.pack(side="right", padx=5, pady=5)

        fbuild0.pack(fill='x')
        fbuild1.pack(fill='x')
        fbuild2.pack(fill='x')
        
        fbuild.pack(fill='x', padx=5, pady=5)    

        f22.pack(side="top", fill="x", expand=False)
 
        ## VIEW Frame
        self.view=viewLauncher(self.models,self.seeds, self.q, self)

        f32 = Frame(f3)

        fview = LabelFrame(f32, text='view model')
                
        fview0 = Frame(fview)
        fview1 = Frame(fview)
        fview2 = Frame(fview)
        fview3 = Frame(fview)
        fviewi = Frame(fview)

        # frame 0: combo-box for seletig view type
        Label (fview0, width = 10, anchor='e', text='type').pack(side='left')
        self.viewTypeCombo = StringVar()
        self.cboCombo = ttk.Combobox(fview0, values=('pca','property','project','model'),
                                     textvariable=self.viewTypeCombo, state='readonly')
        
        self.cboCombo.current(0)
        self.cboCombo.pack()

        # frame 1: entry field for selecting reference endpoint        
        Label (fview1, width = 10, anchor='e', text='refername').pack(side='left')
        self.referEndpointCombo = StringVar ()
        comboValues=("None",) + self.models.get_children()
           
        self.eview1 = ttk.Combobox(fview1, values=comboValues, textvariable=self.referEndpointCombo, state='readonly')
        self.eview1.current(0)
        self.eview1.pack()

        # frame 2: entry field for selecting reference version
        Label(fview2, width = 10, anchor='e', text='refver').pack(side='left')
        self.referVersionCombo = StringVar ()
        
        comboVersions = ()
        for i in range(self.models.maxver):
            comboVersions=comboVersions+(str(i),)  # this is updated by updateGUI method
           
        self.eview2 = ttk.Combobox(fview2, values=comboVersions, textvariable=self.referVersionCombo, state='readonly')
        self.eview2.current(0)
        self.eview2.pack()

        self.eview1.configure(state="disable")
        self.eview2.configure(state="disable")

        # frame 3: check button for showing background
        Label(fview3, width = 10, anchor='e', text='   ').pack(side='left')
        self.viewBackground = StringVar()
        self.checkBackground = ttk.Checkbutton(fview3, text='show background', variable=self.viewBackground, command = lambda: self.updateBack(True))
        
        self.viewBackground.set(0)
        self.checkBackground.pack()
        
        self.cboCombo.bind('<<ComboboxSelected>>', self.updateBack)
        
        # frame button 
        Label(fviewi, text='represents graphically the series or model').pack(side="left", padx=5, pady=5)        
        self.viewButton1 = Button(fviewi, text ='OK', width=5, command = self.view.view)
        self.viewButton1.pack(side="right", padx=5, pady=5)

        fview0.pack(anchor='w')

        fview1.pack(anchor='w')
        fview2.pack(anchor='w')
        fview3.pack(anchor='w')
        fviewi.pack(fill='x')

        fview.pack(fill="x", padx=5, pady=5)

        fviewQuery = LabelFrame(f32, text='view query')
                
        fviewQuery0 = Frame(fviewQuery)
        fviewQuery1 = Frame(fviewQuery)
        fviewQuery2 = Frame(fviewQuery)
        fviewQueryi = Frame(fviewQuery)

        # frame 0: combo-box for selecting view type
        Label (fviewQuery0, width = 10, anchor='e', text='type').pack(side='left')
        self.viewTypeComboQuery = StringVar()
        self.cboComboQuery = ttk.Combobox( fviewQuery0, values=('pca','property','project'), textvariable=self.viewTypeComboQuery, state='readonly')
        self.cboComboQuery.current(0)
        self.cboComboQuery.pack(anchor ='w')

        # frame 1: entry field for selecting reference endpoint
        Label(fviewQuery1, width = 10, anchor='e', text='query').pack(side='left')      
        self.eviewQuery1 = Entry(fviewQuery1, bd =1,width=22)               # field containing the new endpoint name
        self.eviewQuery1.pack(side="left")
        Button(fviewQuery1, text ='...', width=2, command = lambda : self.selectFile (self.eviewQuery1,('Series','*.sdf'))).pack(side="right") 

        # frame 2: check button for showing background
        Label (fviewQuery2, width = 10, anchor='e', text='   ').pack(side='left')
        self.viewBackgroundQuery = StringVar()
        self.checkBackgroundQuery = ttk.Checkbutton(fviewQuery2, text='show background', variable=self.viewBackgroundQuery)
        self.viewBackgroundQuery.set(0)
        self.checkBackgroundQuery.pack(anchor='w')

        # frame button 
        Label(fviewQueryi, anchor = 'w', text='represents graphically a query series').pack(side="left", padx=5, pady=5)        
        self.viewButton2 = Button(fviewQueryi, text ='OK', width=5, command = self.view.viewQuery)
        self.viewButton2.pack(side="right", padx=5, pady=5)

        fviewQuery0.pack(anchor='w')
        fviewQuery1.pack(anchor='w')
        fviewQuery2.pack(anchor='w')
        fviewQueryi.pack(fill='x')

        fviewQuery.pack(fill="x", padx=5, pady=5)


        ## PREDICT Frame        
        self.predict=predictLauncher(self.models, self.seeds, self.q, self) 
        
        f41 = Frame(f4)

        fpredict = LabelFrame(f41, text='predict series')

        fpredict0 = Frame(fpredict)
        fpredict1 = Frame(fpredict)
        
        Label(fpredict0, width = 10, anchor='e', text='query').pack(side='left')       
        self.predictSeries = Entry(fpredict0, bd =1)
        self.predictSeries.pack(side='left')      
        Button(fpredict0, text ='...', width=2, command = lambda : self.selectFile (self.predictSeries,('Series','*.sdf'))).pack(side='left')

        Label(fpredict1, text='predict series using selected model').pack(side="left", padx=5, pady=5)
        self.predictButton = Button(fpredict1, text = 'OK', command = self.predict.predict, width=5)
        self.predictButton.pack(side="right", padx=5, pady=5)

        fpredict0.pack(fill='x')
        fpredict1.pack(fill='x')
        
        fpredict.pack(fill='x', padx=5, pady=5)    

        f41.pack(side="top", fill="x", expand=False)  

        # TABS packing
        f32.pack(side="top",fill='x', expand=False)
        
        t1.pack(side="left", fill="both",expand=True)
        i1.pack(side="left", fill="both",expand=True)
        i2.pack(side="right", fill="both",expand=False)
        
        # Start queue listener
        self.periodicCall()

    def selectionChanged (self, event):
        # copy this to build series and models
        # print self.models.selEndpoint (), ' ver ', self.models.selVersion()

        if self.skipUpdate:
            self.skipUpdate = False
            return
        
        v = self.models.selVersion()
        self.buildSeries.delete(0, END)
        self.buildSeries.insert(0, '<series ver '+v+'>')

        self.buildModel.delete (0, END)
        self.buildModel.insert (0, '<model ver '+v+'>')

    def updateBack(self, event):
        enableType = (self.viewTypeCombo.get() == 'project' )
        enableBack = (self.viewBackground.get() == '1')

        if enableType or enableBack:
            self.eview1.configure(state="enable")
            self.eview2.configure(state="enable")
        else:
            self.eview1.configure(state="disable")
            self.eview1.current(0)
            self.eview2.configure(state="disable")
            self.eview2.current(0)

    def selectFile (self, myEntry, myType):
        selection=tkFileDialog.askopenfilename(parent=root, filetypes=( myType, ("All files", "*.*")) )
        if selection:
            myEntry.delete(0, END)
            myEntry.insert(0,selection)

    def editModelFile(self):
        # copy imodel.py of the selected version to the sandbox
        e = self.models.selEndpoint()
        v = self.models.selVersion()
        vdir = wkd + '/' + e + '/version%0.4d/'%int(v)
        zdir = wkd + '/' + e + '/version0000/'

        if (vdir!=zdir):
            try:
                shutil.copy(vdir+'imodel.py', zdir)   # copy imodel.py to the sandbox. This will be the base version for build 
            except:
                tkMessageBox.showerror("Error Message", "Unable to access source imodel.py")
                return

            removefile (zdir+'info.pkl')  # remove imodel.py from the sandbox to force model rebuilding
            
            self.skipUpdate=True
            self.models.chargeData()
            self.model.setFocus(e,v)

        # launch idle with the imodel.py of the sandbox
        try:
            subprocess.Popen(['/usr/bin/idle',zdir+'imodel.py'])
        except:
            tkMessageBox.showerror("Error Message", "Unable to edit imodel.py")
            pass
            
        self.buildModel.delete(0, END)
        self.buildModel.insert(0, '<edited model (save first)>')
            
    def validateEndpoint(self, char, entry_value):
        for c in char:
            if c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890_-' :
                return False
        return True
    
    def validateTag(self, char, entry_value):
        for c in char:
            if c not in ' /ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890_-' :
                return False
        return True

    def validateVersion(self, char, entry_value):
        for c in char:
            if c not in '1234567890' :
                return False
        return True

    def addBackgroundProcess(self):
        self.backgroundCount+=1
        if self.backgroundCount == 1:
            self.pb.start(100)

    def removeBackgroundProcess(self):
        if self.backgroundCount>0:
            self.backgroundCount-=1
        if self.backgroundCount == 0:
            self.pb.stop()
            self.processList = []

    def _quit(self,event):
        fcntl.lockf(fp, fcntl.LOCK_UN)
        root.destroy()         

    def updateGUI (self,newVersions=False):
        self.models.chargeData()
        self.models.updateFocus()

        if newVersions:
            comboVersions = ()
            for i in range(self.models.maxver):
                comboVersions=comboVersions+(str(i),)
            self.eview2['values'] = comboVersions

    def killProcess (self):
        for pid in self.processList:
            try:
                os.kill(pid, signal.SIGQUIT)
            except:
                pass
            
        self.processList = []
            
        

    ##################
    ### DIRECT TASKS
    ##################

    '''
    Creates a new endpoint.
    '''
    def new(self):    
        endpoint = self.enew1.get()
        tag = self.enew2.get()
                
        if not endpoint:
            tkMessageBox.showerror("Error Message", "Please enter the name of the endpoint")
            return

        elif not tag:
            tkMessageBox.showerror("Error Message", "Please enter the name of the tag")
            return

        #tag format imposed by API2
        p = re.compile('^(/[^/]+)+/\\d+$')
        if not p.match(tag):
            tkMessageBox.showerror("Error Message", 'Valid tags must have a format like this:\n"/toxicity type/endpoint name/3"')
            return

        for line in self.models.get_children():
            labels = line.split()
            
            if endpoint == labels[0]:
                tkMessageBox.showerror("Error Message", "This endpoint already exists!")
                return

        try:
            mycommand = [wkd+'/manage.py', '--new', '-e', endpoint, '-t', tag]
            result = subprocess.call(mycommand)
        except:
            self.q.put ('ERROR: Unable to execute manage command')
            return
        
        if result == 1 :
            self.q.put ('ERROR: Failed to create new endpoint')
            return

        self.enew1.delete(0,END)
        self.enew2.delete(0,END)

        self.updateGUI(True)
        
        tkMessageBox.showinfo("Info Message",'New endpoint created')    


    '''
    Kills an endpoint.
    '''
    def kill(self):    
        name = self.models.selEndpoint()

        if not tkMessageBox.askyesno('Verify', 'Do you really want to remove '
                                     'the whole endpoint '+name+' and all '
                                     'associated models?'):
            return
        
        try:
            mycommand = [wkd+'/manage.py', '--kill', '-e', name]
            result = subprocess.call(mycommand)
        except:
            self.q.put ('ERROR: Unable to execute manage command')
            return
        
        if result == 1 :
            self.q.put ('ERROR: Failed to remove endpoint')
            return

        self.updateGUI()
        
        tkMessageBox.showinfo("Info Message",'Endpoint '+name+' removed')  

    '''
    Presents information about the model defined by the endpoint
    If no version is specified (general endpoint), shows all the model versions.
    '''    
    def seeDetails(self):

        name = self.models.selEndpoint()
        version = self.models.selVersion()

        # Obtain information about the model
        mycommand = [wkd+'/manage.py', '-e', name, '-v', version, '--info=long']
        try:
            process = subprocess.Popen(mycommand, stdout=subprocess.PIPE)
            output, err = process.communicate()
        except:
            tkMessageBox.showerror("Error Message", "Unable to obtain information")
            return

        outputlist = output.split('\n')
        outputlist = outputlist [1:-2]         
        
        output = ''
        for l in outputlist: output+= l+'\n'

        visualizeDetails().showDetails (name+' ver '+version, output)

        

    '''
    Handle all the messages currently in the queue (if any)
    and run events depending of its information
    '''
    def periodicCall(self):
    
        while self.q.qsize():
        
            try:
                msg = self.q.get(0)

                ## post BUILDING OK
                if 'Manage completed OK' in msg:

                    self.removeBackgroundProcess()
                    
                    tkMessageBox.showinfo("Info Message", msg)

                    
                ## post BUILDING OK
                if 'Building completed OK' in msg:
                    endpointName = msg[21:]
                    msg = msg[:21]
                    
                    endpointDir = wkd + '/' + endpointName + '/version0000'
                    files = glob.glob(endpointDir+"/*.png")
                    if len(files) > 0 :
                        files.sort()
                        self.win=visualizewindow('model: '+ endpointName +' ver 0')
                        self.win.viewFiles(files)
                    
                    self.models.chargeData()
                    self.models.setFocus(endpointName,'0')
                    
                    self.buildButton.configure(state='normal') # building
                    #self.pb.stop()
                    self.removeBackgroundProcess()
                    
                    tkMessageBox.showinfo("Info Message", msg)

                ## post VIEWING OK
                elif 'View completed OK' in msg:
                    self.viewButton1.configure(state='normal') # view OK
                    self.viewButton2.configure(state='normal') # view OK

                    msglist = msg.split()[3:]
                    
                    self.viewButton1.configure(state='normal')
                    self.viewButton2.configure(state='normal')
                    #self.pb.stop()
                    self.removeBackgroundProcess()
                    
                    if len(msglist)<3:

                        tkMessageBox.showerror("Error Message",'Abnormal termination')
                        
                    else:                                        
                        self.win=visualizewindow('series: '+msglist[0]+' ver '+msglist[1])
                        files = msglist[2:]
                        self.win.viewFiles(files)

                ## post PREDICTING OK
                elif 'Predict completed OK' in msg:

                    self.predictButton.configure(state='normal') # predict
                    self.removeBackgroundProcess()
                    
                    msglist = msg.split()[3:]
                    
                    if len(msglist)<3:
                        tkMessageBox.showerror("Error Message",'Abnormal termination')
                      
                    elif not os.path.isfile('/var/tmp/results.txt'):
                        
                        tkMessageBox.showerror("Error Message",'Results not found')
                        
                    else:  
                        try:
                            self.predWin.deiconify()
                        except:
                            self.predWin=visualizePrediction()
                        
                        self.predWin.show(msglist[0], msglist[1], msglist[2])
                    
                # ANY ERROR
                elif msg.startswith ('ERROR:'):
                    self.viewButton1.configure(state='normal') # view OK
                    self.viewButton2.configure(state='normal') # view OK
                    self.buildButton.configure(state='normal') # building
                    self.predictButton.configure(state='normal') # predict
                    #self.pb.stop()
                    self.removeBackgroundProcess()
                    tkMessageBox.showerror("Error Message", msg)

                elif msg.startswith ('LOCAL MODEL'):
                    tkMessageBox.showinfo("Model progress", msg)

                elif msg.startswith ('PROCESS '):
                    self.processList.append(int(msg[8:]))
                    
                elif 'update' in msg:
                    self.models.chargeData()
                    self.models.setFocus(msg[7:],'')

            except Queue.Empty:
                pass

        self.master.after(500, self.periodicCall) # re-call after 500ms


if __name__ == "__main__":

    def quitCallback():
        if tkMessageBox.askokcancel("Quit", "Do you really wish to quit?"):
            fcntl.lockf(fp, fcntl.LOCK_UN)
            root.destroy()

    root = Tk()
    root.title("etoxlab GUI ("+VERSION+")")
    
    # check if there are another instances of eTOXlab GUI already running
    pid_file = '/var/tmp/etoxlab.pid'
    fp = open(pid_file, 'w')
    try:
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        
    except IOError:
        f = Frame(root)
        msg = Message (f,text="There is another instance of eTOXlab running!", width=200)
        msg.config(bg='yellow', justify=CENTER, font=("sans",12))
        msg.pack(fill='x', expand=True)
        f.pack()
        
    else:
        root.protocol("WM_DELETE_WINDOW", quitCallback)
        app = etoxlab(root)
        
    root.mainloop()
