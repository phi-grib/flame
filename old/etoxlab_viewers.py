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
from rdkit import Chem

from threading import Thread
from utils import wkd
from utils import VERSION
from utils import removefile
from utils import cleanSandbox
import tarfile
from PIL import ImageTk, Image
import glob


#####################################################################################################
### visualizeHelp
#####################################################################################################        

class visualizeHelp (Toplevel):
    def __init__(self):        
        Toplevel.__init__(self)
        self.title ('About..')

    def showAbout (self):
        f = Frame(self)
        msg = Message (f,text="An eTOXlab simple GUI\n\n"+
                    "Manuel Pastor (manuel.pastor@upf.edu)\n"+
                    "Copyright 2014-2017 Manuel Pastor", width=600)
        msg.config(bg='white', justify=CENTER, font=("sans",14))
        msg.pack(fill='x', expand=True)

        if os.path.isfile(wkd+'/logoeTOX.png'):
            self.image = ImageTk.PhotoImage(Image.open(wkd+'/logoeTOX.png'))
            self.logo = Label (f, image=self.image,bg='white' )
            self.logo.pack(fill='x', expand=True)

        ops = Message (f,text="\n\neTOXlab is free software: you can redistribute it and/or modify "+
                    "it under the terms of the GNU General Public License as published by "+
                    "the Free Software Foundation version 3.", width=600)
        ops.config(bg='white', justify=LEFT, font=("sans",10))
        ops.pack(fill='x', expand=True)
        f.pack()

#####################################################################################################
### visualizeDetails
#####################################################################################################   

class visualizeDetails (Toplevel):
    def __init__(self):        
        Toplevel.__init__(self)

    def showDetails (self, model, output):
        self.title (model)

        scrollbar = Scrollbar(self,orient=VERTICAL)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        text = Text(self, wrap=WORD, font=('Courier New',10), yscrollcommand=scrollbar.set)
        text.insert(INSERT, output)
        text.config(state=DISABLED)
        text.pack(side="top", fill="both", expand=True)
        
        scrollbar.config(command=text.yview)      
        

#####################################################################################################
### visualize graphics 
##################################################################################################### 

'''
Creates a new window that displays one or more plots given as a list of png files
'''
class visualizewindow(Toplevel):
    
    def __init__(self, vtitle='graphic viewer'):        
        Toplevel.__init__(self)
        self.title (vtitle)

    def viewFiles (self, fnames):
        #if not fnames : return
        
        if len(fnames)<2:
            self.viewSingle (fnames[0])
        else:
            self.viewMultiple (fnames)
            
    def viewSingle(self, fname):
        
        if fname==None or fname=='':
            self.destroy()
            return

        if not os.path.isfile (fname):
            self.destroy()
            return
        
        f = Frame(self)
        self.i = ImageTk.PhotoImage(Image.open(fname))
        ttk.Label(f,image=self.i).pack()        
        f.pack()

    def viewMultiple (self, fnames):
        
        if fnames==None or len(fnames)==0:
            self.destroy()
            return
        
        self.note_view = ttk.Notebook(self)
        self.note_view.pack()

        self.i=[]
        for t in fnames:
            if not os.path.isfile (t) : continue
            self.i.append (ImageTk.PhotoImage(Image.open(t)))
            
            f = Frame(self)
            self.note_view.add(f,text=os.path.splitext(os.path.basename(t))[0])
            ttk.Label(f,image=self.i[-1]).pack()

        if not len(self.i):
            self.destroy()
            return
        
        self.note_view.pack()

#####################################################################################################
### visualizePrediction
#####################################################################################################        
        
'''
Creates a new window that displays one or more plots given as a list of png files
'''
class visualizePrediction (Toplevel):
    
    def __init__(self):   
        Toplevel.__init__(self)
        self.title ('Prediction results')

        f0 = Frame (self)
        scrollbar_tree = ttk.Scrollbar(f0)
        self.tree = ttk.Treeview (f0, columns = ('m','a','b','c'), selectmode='browse',yscrollcommand = scrollbar_tree.set)
        self.tree.column ("m", width=120, anchor='w' )
        self.tree.column ('a', width=120, anchor='e')
        self.tree.column ('b', width=50, anchor='center')
        self.tree.column ('c', width=120, anchor='e')
        self.tree.heading ('m', text='mol')
        self.tree.heading ('a', text='value')
        self.tree.heading ('b', text='AD')
        self.tree.heading ('c', text='CI')

        scrollbar_tree.pack(side="left", fill=Y)
        scrollbar_tree.config(command = self.tree.yview)
        
        self.tree.pack(side='top', expand=True, fill='both')

        b0 = Frame (f0)
        Button(b0, text ='Quit', command = lambda: self.destroy(), width=10).pack(side='right')
        Button(b0, text ='Export CSV', command = self.exportCSV, width=10).pack(side='right')
        Button(b0, text ='Export SDF', command = self.exportSDF, width=10).pack(side='right')
        b0.pack(side='right')
        
        f0.pack(side="top", expand=True, fill='both')


    def exportCSV(self):

        sel = self.tree.focus()
        
        if self.tree.parent(sel) != '':
            sel = self.tree.parent(sel)

        f = tkFileDialog.asksaveasfile(parent=self, filetypes=( ('CSV file', '*.csv'), ("All files", "*.*")) )
        
        for i in self.tree.get_children(sel):
            line = self.tree.set(i,'m')+'\t' + self.tree.set(i,'a')+'\t'+ self.tree.set(i,'b')+'\t'+self.tree.set(i,'c')
            f.write (line+'\n')
            
        f.close()
            

    def exportSDF(self):
        sel = self.tree.focus()
        
        if self.tree.parent(sel) != '':
            sel = self.tree.parent(sel)

        f = tkFileDialog.asksaveasfile(parent=self, filetypes=( ('SDFile', '*.sdf'), ("All files", "*.*")) )

        sdf = sel.split(':')[2]
        suppl=Chem.SDMolSupplier(sdf)

        for i in self.tree.get_children(sel):
            try:
                mi = suppl.next()
            except:
                continue
            mb = Chem.MolToMolBlock(mi)
            f.write(mb)
            f.write('>  <'+sel.split(':')[0]+'>\n'+self.tree.set(i,'a').strip()+'\n\n')
            f.write('>  <AD>\n'+self.tree.set(i,'b').strip()+'\n\n')
            f.write('>  <CI>\n'+self.tree.set(i,'c').strip()+'\n\n')
            f.write('$$$$\n')
        
        f.close()

        
    def show (self, endpoint, version, series):
        ## check if endpoint+version already exists
        eID = endpoint+':'+version+':'+series
        
        if eID in self.tree.get_children():
            self.tree.delete(eID)

        plabel = endpoint+' '+version+' ['+os.path.basename(series)+']'

        self.tree.insert ('','end', eID, text=plabel, open=True )
        try:
            f = open ('/var/tmp/results.txt','r')
        except:
            tkMessageBox.showerror("Error Message", 'no result generated')
            return

        # get molecule names from supplied SDFile
        molNames = []
        
        suppl=Chem.SDMolSupplier(series)

        idlist = ['name','Name', 'NAME', '_Name']
        count = 0
        while True:
            name = 'mol%0.4d'%count
            
            try:
                mi = suppl.next()
            except:
                break
            
            if mi:
                for idi in idlist:
                    if mi.HasProp (idi):
                        try:
                            name = mi.GetProp(idi)
                            name = name.decode('utf-8') # needed to handle names with extrange unicode chars
                            name = name.encode('ascii','ignore') # use 'replace' to insert '?'
                        except:
                            name = 'mol%0.4d'%count # failsafe just in case
                        break
                    
            count +=1
            
            molNames.append(name)
        
        count = 0
        for line in f:
            result = line.split('\t')

            if result[0]=='0' and 'ERROR:' in result[1]:
                tkMessageBox.showerror("Error Message", result[1])
                f.close()
                return

            if len(result) < 6:
                continue
                
            value = 'na'
            AD    = 'na'
            CI    = 'na'
                
            if result[0]!='0':
                try:
                    v = float(result[1])
                    value = '%10.3f'%v
                except:
                    value = result[1]

            if result[2]!='0':        
                AD = result[3]

            if result[4]!='0':
                try:
                    c = float(result[5])
                    CI = '%10.3f'%c
                except:
                    CI = result[5]

            if count>=len(molNames):
                mname = 'mol'
            else:
                mname = molNames[count]
            
            self.tree.insert(eID, 'end', values=(mname,value,AD,CI), iid=eID+str(count))
            count+=1
            
        f.close()

        sel=self.tree.get_children()
        
        self.tree.selection_set(sel[0])
        self.tree.focus(sel[0])


if __name__ == "__main__":

    root = Tk()
    root.title("test ("+VERSION+")")    

    v = visualizePrediction()
    v.show('AAA','0','/home/modeler/workspace/test.sdf')
    
    root.mainloop()
    
