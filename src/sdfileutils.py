#! -*- coding: utf-8 -*-

##    Description    SDFile tools class
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
from rdkit import Chem

def nummols (ifile):
    try:
        suppl = Chem.SDMolSupplier(ifile)
    except:
        return False, 'unable to open molfile'
    
    return True, len(suppl)

def splitSDFile (ifile, num_mols, num_chunks):
    ''' 
    
    Splits the input SDfile in num_chunks SDfiles files of balanced size. Every file is named "filename_0.sdf", "filename_1.sdf", ...
    
    Argument:
        ifile       : input SDfile
        num_mols    : number of molecules in ifile (warning! this methods does not check if this value is correct)
        num_chunks  : number of pieces the SDfile is split into

    Output:

        success     : Boolean
        results     : a tuple of two lists, nobj (the number of mols inside each chunk) and temp_files (the chunk filenames)

    '''
    
    index = []
    nobj = []
    temp_files = []

    chunksize = num_mols // num_chunks
    for a in range (num_mols):
        index.append(a//chunksize)
    
    moli = 0      # molecule counter in next loop
    chunki = 0    # chunk counter in next toolp

    filename, fileext = os.path.splitext(ifile)
    chunkname = filename + '_%d' %chunki + fileext
    try:
        with open (ifile,'r') as fi:
            fo = open (chunkname,"w")
            moli_chunk = 0      # molecule counter inside the chunk
            for line in fi:
                fo.write(line)

                # end of molecule
                if line.startswith('$$$$'):
                    moli += 1
                    moli_chunk += 1 

                    # if we reached the end of the file...
                    if (moli >= num_mols):
                        fo.close()
                        temp_files.append(chunkname)
                        nobj.append(moli_chunk)

                    # ...otherwyse
                    elif (index[moli] > chunki):
                        fo.close()
                        temp_files.append(chunkname)
                        nobj.append(moli_chunk)

                        chunki+=1
                        chunkname = filename + '_%d' %chunki + fileext
                        moli_chunk=0
                        fo = open (chunkname,"w")
    except:
        return False, "error splitting: "+ifile

    return True, (nobj, temp_files)