from scipy.io import loadmat
import numpy as np
import os


#wrapper of Matlab struct with additional useful methods
class Record():

    #record: file name of .mat file
    def __init__(self,record):
        self._wrapped=loadmat(record,squeeze_me=True,struct_as_record=False)['dataRaw']
        self.filename=record

    #easyest way to retriev fields of the wrapped matlab struct
    def __getattr__(self,attr):
        return self._wrapped.__dict__[attr]

    #easyest way to set them
    def __setattr__(self,attr,value):
        if attr!="_wrapped": self._wrapped.__dict__[attr]=value
        else: self.__dict__["_wrapped"]=value

    #return formatted time duration in the specifed format (sec/min/hh)
    def fmtTime(self,fmt="sec"):

        sec=int(len(self.ecgRaw[0])/self.fsEdf)
        fmtTime="Time length in {0}: ".format(fmt)

        if fmt=="sec": fmtTime+=str(sec)
    
        minutes=sec//60
        sec%=60
    
        if fmt=="min": fmtTime+=str(minutes)+":"+str(sec)
    
        hh=minutes//60
        minutes%=60
    
        if fmt=="hh": fmtTime+=str(hh)+":"+str(minutes)+":"+str(sec)

        return fmtTime
    
    
    #pretty-print of object fields
    def __str__(self):
        
        fmt="*"*40+"\n"
        fmt+=self.fmtTime("hh")+"\n"
        fmt+=self.fmtTime("sec")+"\n"*2

        for key,value in self._wrapped.__dict__.items():
            
            if key=="_fieldnames": continue

            fmt+="\t"+key+"\n"
            fmt+="\t"*2+"Type: {0} ".format(type(value))+"\n"

            if type(value)==np.ndarray: fmt+="\t"*2+"Shape: {0} ".format(value.shape)+"\n"*2

            elif key=="labels" or key=="labelsRaw": fmt+="\t"*2+"Length: {0} ".format(len(value))+"\n"*2

            else: fmt+="\t"*2+"Value: {0} ".format(value)+"\n"*2

        return fmt[:-2]
    


#iter records from the specified directory.
def iterRecords(directory,max_iter=None,verb=False):
    count=0
    for record in os.listdir(directory):
        if not max_iter or count<max_iter:
            
            if verb:
                print("\n"*2+"-"*20+"   READING RECORD {0}   ".format(count)+"-"*20+"\n")
            
            yield Record(directory+"/"+record)
            count+=1

    if verb:
        print("{0} records read".format(count))

