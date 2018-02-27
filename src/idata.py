#! -*- coding: utf-8 -*-

##    Description    Flame flInput class
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
##    Copyright 2018 Manuel Pastor
##
##    This file is part of Flame
##
##    Flame is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    Flame is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with Flame. If not, see <http://www.gnu.org/licenses/>.

import os
import hashlib
from rdkit import Chem
import multiprocessing as mp

class Idata:

    def __init__ (self, control, ifile):

        self.control = control
        self.ifile = ifile

    def countmol (self, ifile):
        # estimate number of molecules inside the SDFile

        nobj = []
        temp_files = []
        
        # RdKit version
        try:
            suppl = Chem.SDMolSupplier(ifile)
        except:
            return False, 'unable to open molfile'
        
        nmol = len(suppl)

        if nmol == 0:
            return False, "no molecule found in file"+ifile

        if self.control.numCPUs > 1 :
            index = []
            chunksize = nmol//self.control.numCPUs
            for a in range (nmol):
                index.append(a//chunksize)
            
            moli = 0      # molecule counter in next loop
            chunki = 0    # chunk counter in next toolp

            filename, fileext = os.path.splitext(ifile)
            chunkname = filename + '_%d' %chunki + fileext
            try:
                with open (ifile,'r') as fi:
                    ofile = open (chunkname,"w")
                    moli_chunk = 0      # molecule counter inside the chunk
                    for line in fi:
                        ofile.write(line)

                        # end of molecule
                        if line.startswith('$$$$'):
                            moli += 1
                            moli_chunk += 1 

                            # if we reached the end of the file...
                            if (moli >= nmol):
                                ofile.close()
                                temp_files.append(chunkname)
                                nobj.append(moli_chunk)

                            # ...otherwyse
                            elif (index[moli] > chunki):
                                ofile.close()
                                temp_files.append(chunkname)
                                nobj.append(moli_chunk)

                                chunki+=1
                                chunkname = filename + '_%d' %chunki + fileext
                                moli_chunk=0
                                ofile = open (chunkname,"w")
            except:
                return False, "error splitting: "+ifile

        else :
            nobj.append(nmol)
            temp_files.append(ifile)

        return True, (nobj, temp_files)

    def extractAnotations (self, ifile):

        # returns a list of names, biological anotations and experimental values
        # TODO: make it more flexible
        #  
        results = [None, None, None]

        return True, results

    def standardize (self, ifile):

        return True, "debug dummy"

    def ionize (self, ifile):

        return True, "debug dummy"

    def convert3D (self, ifile):

        return True, "debug dummy"

    def computeMD (self, ifile):

        return True, "debug dummy"

    def consolidate (self, tfiles, tnames):

        return True, "debug dummy"

    def save (self, results):

        print (self.control.md5stamp())
        # pickle results + stamp in ifile.pickle
        # return True

        return True

    def workflow (self, ifile):
        ''' Executes in sequence methods required to generate MD, starting from a single molecular file
            input : ifile, a molecular file in SDFile format
            output: results is a numpy bidimensional array containing MD '''

        tfile = ifile
        # normalize chemical
        if self.control.normalize_method != None:
            success, results = self.standardize (tfile)
            if not success:
                return False, "input error: chemical standardization failed: "+str(results)
            else:
                tfile = results

        # ionize molecules
        if self.control.ionize_method != None:
            success, results = self.ionize (tfile)
            if not success:
                return False, "input error: molecule ionization error at position: "+str(results)
            else:
                tfile = results

        # generate a 3D structure
        if self.control.convert3D_method != None:
            success, results = self.convert3D (tfile)
            if not success:
                return False, "input error: 3D conversion error at position: "+str(results)
            else:
                tfile = results

        # compute MD
        success, results = self.computeMD (tfile)
        if not success:
            return False, "input error: failed computing MD: "+str(results)

        return success, results


    def run (self):
        ''' process input file to obtain metadata (size, type, number of objects, name of objects, etc.) as well
            as for generating MD
            
            The results are saved in a MD5 stamped pickle, to avoid recomputing model input from the same input
            file
            
            This methods supports multiprocessing, splitting original files in a chunck per CPU
            '''

        # TODO: check for presence of pickle file
        # if true, extract MD5 stamp, compute control MD5 stamp and if both are coincident extract results and exit

        # open file
        if not os.path.isfile (self.ifile):
            return False, "input error: file not found"
        
        # processing for diverse molecule type
        if (self.control.input_type == 'molecule'):

            # extract useful information from file
            success, results = self.extractAnotations (self.ifile)
            if not success:
                return False, "input error: annotation extraction failed: "+str(results)
            else:
                self.obj_nam = results[0]
                self.obj_bio = results[1]
                self.obj_exp = results[2]

            # count number of molecules and split in chuncks for multiprocessing if necessary
            success, results = self.countmol (self.ifile)
            if not success:
                return False, "input error: no molecule recognized: "+str(results)
            else:
                nobj   = results[0]  # list with nobj of each piece
                tfiles = results[1]  # list with filename of pieces

            print (nobj, tfiles)

            # execute the workflow in 1 or n CPUs
            if len(tfiles) > 1 :
                print ('multi CPU')
                pool = mp.Pool(len(tfiles))
                results = pool.map(self.workflow, tfiles)
            else:
                print ('single CPU')
                success, results = self.workflow (tfiles[0])

            # check the results and make sure there are no missing objects
            # reassemble results for parallel computing results

            success, results = self.consolidate(results,nobj) 

            if not success:
                return False, str(results)


        elif (self.control.input_type == 'data'):

            #   test and obtain dimensions
            #   normalize data

            print ("data")

        else:

            print ("unknown input format")


        # save and stamp
        success = self.save (results)

        # runner class? will split in chunks and run every chunck in a thread, then reassembling the results
        # at the end
        # the same class will take care of situations where the loop execution fails and fallback to compound 
        # per compound mode

        results = self.ifile + '_i'

        return success, results

